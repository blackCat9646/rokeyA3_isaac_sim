#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def _nms(boxes, threshold):
    selected = []
    for box in sorted(boxes, key=lambda item: item[4], reverse=True):
        if all(_iou(box[:4], kept[:4]) < threshold for kept in selected):
            selected.append(box)
    return selected


def _load_labels(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _frame_id(path):
    return path.stem.rsplit("_", 1)[-1]


def _intruder_key(prim_path):
    match = re.search(r"(/World/Intruders/Intruder_\d+)", prim_path)
    return match.group(1) if match else None


def _union_boxes(boxes):
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return (x1, y1, x2, y2, area)


def _boxes_for_class(npy_path, labels_path, prim_paths_path, class_name, min_area, group_intruders):
    labels = _load_labels(labels_path)
    rows = np.load(npy_path, allow_pickle=True)
    with prim_paths_path.open("r", encoding="utf-8") as file:
        prim_paths = json.load(file)

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


def _draw_boxes(image_path, boxes, output_path, class_name):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for x1, y1, x2, y2, _area in boxes:
        draw.rectangle((x1, y1, x2, y2), outline=(255, 40, 40), width=3)
        text = class_name
        text_width, text_height = draw.textsize(text, font=font)
        draw.rectangle((x1, max(0, y1 - (text_height + 4)), x1 + text_width + 6, y1), fill=(255, 40, 40))
        draw.text((x1 + 3, max(0, y1 - (text_height + 2))), text, fill=(255, 255, 255), font=font)
    image.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--class-name", default="person")
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument(
        "--no-group-intruders",
        action="store_true",
        help="Draw every person-labeled prim separately instead of merging by /World/Intruders/Intruder_N.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser()
    output_dir = args.output_dir.expanduser() if args.output_dir else dataset_dir / "preview_boxes"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    box_count = 0
    for image_path in sorted(dataset_dir.glob("rgb_*.png")):
        frame = _frame_id(image_path)
        npy_path = dataset_dir / f"bounding_box_2d_tight_{frame}.npy"
        labels_path = dataset_dir / f"bounding_box_2d_tight_labels_{frame}.json"
        prim_paths_path = dataset_dir / f"bounding_box_2d_tight_prim_paths_{frame}.json"
        if not npy_path.exists() or not labels_path.exists() or not prim_paths_path.exists():
            continue
        boxes = _boxes_for_class(
            npy_path,
            labels_path,
            prim_paths_path,
            args.class_name,
            args.min_area,
            group_intruders=not args.no_group_intruders,
        )
        if args.no_group_intruders:
            boxes = _nms(boxes, args.nms_iou)
        _draw_boxes(image_path, boxes, output_dir / image_path.name, args.class_name)
        frame_count += 1
        box_count += len(boxes)

    print(f"Wrote {frame_count} overlay images to {output_dir}")
    print(f"Kept {box_count} {args.class_name} boxes after filtering")


if __name__ == "__main__":
    main()
