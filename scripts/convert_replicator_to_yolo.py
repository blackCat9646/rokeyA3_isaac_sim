#!/usr/bin/env python3
import argparse
import json
import random
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _frame_id(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def _intruder_key(prim_path: str) -> str | None:
    match = re.search(r"(/World/Intruders/Intruder_\d+)", prim_path)
    return match.group(1) if match else None


def _union_boxes(boxes):
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return (x1, y1, x2, y2, area)


def _clip_box(box, width: int, height: int):
    x1, y1, x2, y2, _area = box
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return (x1, y1, x2, y2, area)


def _boxes_for_class(
    npy_path: Path,
    labels_path: Path,
    prim_paths_path: Path,
    class_name: str,
    min_area: int,
    group_intruders: bool,
):
    labels = _load_json(labels_path)
    rows = np.load(npy_path, allow_pickle=True)
    prim_paths = _load_json(prim_paths_path)

    boxes = []
    grouped_boxes = {}
    for row, prim_path in zip(rows, prim_paths):
        semantic_id = str(int(row["semanticId"]))
        label = labels.get(semantic_id, {}).get("class", "")
        if label != class_name:
            continue

        if group_intruders:
            intruder_key = _intruder_key(prim_path)
            if intruder_key is None:
                continue

        x1 = int(row["x_min"])
        y1 = int(row["y_min"])
        x2 = int(row["x_max"])
        y2 = int(row["y_max"])
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area < min_area:
            continue

        box = (x1, y1, x2, y2, area)
        if group_intruders:
            grouped_boxes.setdefault(intruder_key, []).append(box)
        else:
            boxes.append(box)

    if group_intruders:
        boxes = [_union_boxes(group) for group in grouped_boxes.values()]
        boxes = [box for box in boxes if box[4] >= min_area]
    return boxes


def _yolo_line(box, image_width: int, image_height: int, class_id: int = 0) -> str | None:
    x1, y1, x2, y2, area = _clip_box(box, image_width, image_height)
    if area <= 0 or x2 <= x1 or y2 <= y1:
        return None

    x_center = ((x1 + x2) * 0.5) / image_width
    y_center = ((y1 + y2) * 0.5) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    if width <= 0.0 or height <= 0.0:
        return None
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def _write_data_yaml(output_dir: Path, class_name: str) -> None:
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir}",
                "train: images/train",
                "val: images/val",
                "names:",
                f"  0: {class_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def convert_dataset(args) -> None:
    source_dir = args.source_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    image_paths = sorted(source_dir.glob("rgb_*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No rgb_*.png files found in {source_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(image_paths)
    val_count = max(1, int(round(len(image_paths) * args.val_ratio))) if len(image_paths) > 1 else 0
    val_frames = set(image_paths[:val_count])

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_count = 0
    labeled_image_count = 0
    box_count = 0
    missing_count = 0

    for image_path in sorted(image_paths):
        frame = _frame_id(image_path)
        npy_path = source_dir / f"bounding_box_2d_tight_{frame}.npy"
        labels_path = source_dir / f"bounding_box_2d_tight_labels_{frame}.json"
        prim_paths_path = source_dir / f"bounding_box_2d_tight_prim_paths_{frame}.json"
        if not npy_path.exists() or not labels_path.exists() or not prim_paths_path.exists():
            missing_count += 1
            continue

        split = "val" if image_path in val_frames else "train"
        target_image = output_dir / "images" / split / image_path.name
        target_label = output_dir / "labels" / split / f"{image_path.stem}.txt"

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        boxes = _boxes_for_class(
            npy_path,
            labels_path,
            prim_paths_path,
            args.class_name,
            args.min_area,
            group_intruders=not args.no_group_intruders,
        )
        lines = []
        for box in boxes:
            line = _yolo_line(box, image_width, image_height)
            if line is not None:
                lines.append(line)

        shutil.copy2(image_path, target_image)
        target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        image_count += 1
        if lines:
            labeled_image_count += 1
        box_count += len(lines)

    _write_data_yaml(output_dir, args.class_name)
    train_count = len(list((output_dir / "images" / "train").glob("rgb_*.png")))
    val_count = len(list((output_dir / "images" / "val").glob("rgb_*.png")))
    print(f"YOLO dataset written to: {output_dir}")
    print(f"Images: {image_count} total, {train_count} train, {val_count} val")
    print(f"Labeled images: {labeled_image_count}, boxes: {box_count}, missing frames skipped: {missing_count}")
    print(f"Data yaml: {output_dir / 'data.yaml'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path, help="Replicator output directory containing rgb_*.png and bbox files.")
    parser.add_argument("output_dir", type=Path, help="YOLO dataset output directory.")
    parser.add_argument("--class-name", default="person")
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument(
        "--no-group-intruders",
        action="store_true",
        help="Export every person-labeled prim separately instead of merging by /World/Intruders/Intruder_N.",
    )
    args = parser.parse_args()
    convert_dataset(args)


if __name__ == "__main__":
    main()
