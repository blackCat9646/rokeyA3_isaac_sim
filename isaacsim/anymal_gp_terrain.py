# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import json
import carb
import numpy as np
import omni
import omni.appwindow  # Contains handle to keyboard
import omni.client
import omni.graph.core as og
import omni.kit.commands
import omni.replicator.core as rep
import usdrt.Sdf
from pathlib import Path
from isaacsim.core.api import World
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.semantics import add_labels
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.policy.examples.robots import AnymalFlatTerrainPolicy
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GUARD_TOWER_ASSET = PROJECT_ROOT / "assets/props/guard_tower/Guard_Tower_Free_Asset.usdz"
DEFAULT_CHAINLINK_FENCE_ASSET = PROJECT_ROOT / "assets/props/chainlink_fence/chainlink_fence_tileable.usdz"
INSPECTION_COMMAND_FILE = Path("/tmp/dmz_sentry_inspection_command.json")
INTRUDER_STATES_FILE = Path("/tmp/dmz_sentry_intruder_states.json")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--terrain-amplitude",
    type=float,
    default=0.28,
    help="Fence-side berm/ditch roughness in meters. Keep low for the flat ANYmal policy.",
)
parser.add_argument("--terrain-size", type=float, default=80.0, help="Square terrain width/depth in meters.")
parser.add_argument(
    "--terrain-resolution",
    type=float,
    default=0.5,
    help="Horizontal spacing between terrain height samples in meters.",
)
parser.add_argument(
    "--terrain-grid-step",
    type=float,
    default=0.0,
    help="Spacing in meters for the dark visual grid drawn on top of the terrain. Set <= 0 to hide it.",
)
parser.add_argument("--trail-width", type=float, default=5.0, help="Width of the smoother patrol trail in meters.")
parser.add_argument(
    "--fence-y",
    type=float,
    default=16.0,
    help="Y position of the forward fence line. Interior is y < fence-y, river is beyond it.",
)
parser.add_argument("--river-width", type=float, default=18.0, help="Visual river width beyond the fence in meters.")
parser.add_argument(
    "--terrain-texture",
    type=str,
    default="",
    help="Optional PNG/JPG satellite, orthophoto, or PBR albedo texture to map over the terrain.",
)
parser.add_argument(
    "--terrain-normal-texture",
    type=str,
    default="",
    help="Optional PBR normal map for the terrain material.",
)
parser.add_argument(
    "--terrain-roughness-texture",
    type=str,
    default="",
    help="Optional PBR roughness map for the terrain material.",
)
parser.add_argument(
    "--terrain-texture-scale",
    type=float,
    default=1.0,
    help="UV tiling scale for terrain textures. Use 1.0 for orthophoto/satellite images, larger values for tiled PBR materials.",
)
parser.add_argument("--no-ros2-sensors", action="store_true", help="Disable ROS 2 camera, LiDAR, TF, and clock publishers.")
parser.add_argument("--no-ros2-lidar", action="store_true", help="Disable only the ROS 2 RTX LiDAR publisher.")
parser.add_argument("--no-ros2-cmd-vel", action="store_true", help="Disable ROS 2 /cmd_vel control subscriber.")
parser.add_argument("--no-ros2-odom", action="store_true", help="Disable ROS 2 /odom publishing.")
parser.add_argument("--cmd-vel-topic", type=str, default="cmd_vel", help="ROS 2 Twist topic used to command ANYmal.")
parser.add_argument(
    "--cmd-vel-timeout",
    type=float,
    default=0.35,
    help="Reserved for future Python-side /cmd_vel timeout handling.",
)
parser.add_argument(
    "--cmd-vel-linear-scale",
    type=float,
    default=1.0,
    help="Scale applied to Twist linear.x and linear.y before feeding the locomotion policy.",
)
parser.add_argument(
    "--cmd-vel-yaw-scale",
    type=float,
    default=1.0,
    help="Scale applied to Twist angular.z before feeding the locomotion policy.",
)
parser.add_argument(
    "--lidar-mount",
    choices=("follow", "robot", "world"),
    default="world",
    help="Use world for a fixed RTX LiDAR debug pose, robot for direct child mount, or follow for experimental ANYmal tracking.",
)
parser.add_argument("--no-intruder", action="store_true", help="Disable the moving intruder scenario.")
parser.add_argument("--intruder-count", type=int, default=3, help="Number of moving intruder targets to spawn.")
parser.add_argument("--intruder-speed", type=float, default=0.45, help="Intruder approach speed in meters per second.")
parser.add_argument("--intruder-seed", type=int, default=31, help="Seed for deterministic intruder spawn positions.")
parser.add_argument(
    "--intruder-visual",
    choices=("auto", "isaac-human", "primitive"),
    default="auto",
    help="Intruder visual source. auto tries Isaac human USD assets and falls back to primitive.",
)
parser.add_argument(
    "--intruder-human-usd",
    type=str,
    default="",
    help="Optional explicit human USD path or URL for intruder visuals.",
)
parser.add_argument(
    "--intruder-yaw-deg",
    type=float,
    default=0.0,
    help="Yaw rotation applied to the intruder visual in degrees. Use 180 if the human asset faces backward.",
)
parser.add_argument("--no-gp-props", action="store_true", help="Disable lightweight GP visual props.")
parser.add_argument("--no-external-props", action="store_true", help="Disable external USD/USDZ prop assets.")
parser.add_argument(
    "--guard-tower-asset",
    type=str,
    default=str(DEFAULT_GUARD_TOWER_ASSET),
    help="Optional USD/USDZ guard tower asset to place near the fence.",
)
parser.add_argument(
    "--guard-tower-scale",
    type=float,
    default=1.0,
    help="Uniform scale for the external guard tower asset.",
)
parser.add_argument(
    "--fence-asset",
    type=str,
    default=str(DEFAULT_CHAINLINK_FENCE_ASSET),
    help="Optional USD/USDZ chainlink fence tile asset to repeat along the fence line.",
)
parser.add_argument(
    "--fence-asset-scale",
    type=float,
    default=1.0,
    help="Uniform scale for the external chainlink fence tile asset.",
)
parser.add_argument(
    "--fence-asset-count",
    type=int,
    default=12,
    help="Number of external fence tiles repeated along the main fence line.",
)
parser.add_argument(
    "--no-ground-detail",
    action="store_true",
    help="Disable procedural dirt, gravel, and grass overlays. Use this with real orthophoto or tiled PBR terrain textures.",
)
parser.add_argument("--terrain-seed", type=int, default=7, help="Seed for deterministic terrain noise.")
parser.add_argument(
    "--replicator-dataset",
    action="store_true",
    help="Generate a small Replicator RGB + 2D bbox dataset preview, then exit.",
)
parser.add_argument(
    "--dataset-frames",
    type=int,
    default=60,
    help="Number of synthetic frames to capture when --replicator-dataset is enabled.",
)
parser.add_argument(
    "--dataset-output-dir",
    type=str,
    default=str(PROJECT_ROOT / "datasets/person_preview_replicator"),
    help="Output directory for Replicator preview data.",
)
parser.add_argument("--dataset-width", type=int, default=640, help="Replicator dataset image width.")
parser.add_argument("--dataset-height", type=int, default=480, help="Replicator dataset image height.")
parser.add_argument("--dataset-seed", type=int, default=20260517, help="Seed for dataset camera randomization.")
parser.add_argument(
    "--dataset-profile",
    choices=("fence", "clear", "hard", "mixed", "calibration"),
    default="fence",
    help=(
        "Camera sampling profile for Replicator data. calibration mixes clear person views, "
        "fence-occluded views, and hard long-angle views."
    ),
)
args, unknown = parser.parse_known_args()

if not args.no_ros2_sensors or not args.no_ros2_cmd_vel or not args.no_ros2_odom:
    enable_extension("isaacsim.ros2.bridge")
    enable_extension("isaacsim.sensors.rtx")
    simulation_app.update()

if not args.no_intruder and args.intruder_visual in ("auto", "isaac-human"):
    enable_extension("omni.anim.people")
    simulation_app.update()

FRONT_CAMERA_NAME = "SentryFrontCamera"
INSPECTION_CAMERA_NAME = "SentryInspectionCamera"
LIDAR_NAME = "SentryLidar"
FRONT_CAMERA_TRANSLATION = (0.95, 0.0, 0.64)
INSPECTION_CAMERA_LOCAL_OFFSET = (0.35, 0.0, 1.18)
INSPECTION_CAMERA_DEFAULT_FOCAL_LENGTH = 35.0
INSPECTION_CAMERA_MIN_FOCAL_LENGTH = 18.0
INSPECTION_CAMERA_MAX_FOCAL_LENGTH = 90.0
INSPECTION_CAMERA_PAN_STEP_DEG = 8.0
INSPECTION_CAMERA_TILT_STEP_DEG = 5.0
INSPECTION_CAMERA_MAX_TILT_DEG = 70.0
# USD camera local -Z looks forward. This quaternion points it along robot +X
# while keeping robot +Z as image up, so the viewport is not rolled sideways.
FRONT_CAMERA_ORIENTATION_IJKR = (0.5, -0.5, -0.5, 0.5)
DEFAULT_INTRUDER_HUMAN_ASSETS = (
    "original_female_adult_business_02",
    "original_male_adult_business_02",
    "original_male_adult_construction_05",
    "original_male_adult_construction_02",
)


class _OptionalRosStringPublisher:
    def __init__(self, node_name: str, topic_name: str) -> None:
        self._rclpy = None
        self._string_type = None
        self._node = None
        self._publisher = None
        self._owns_context = False

        try:
            import rclpy
            from std_msgs.msg import String
        except Exception as exc:
            carb.log_warn(f"ROS 2 Python publisher unavailable for {topic_name}: {exc}")
            return

        try:
            if not rclpy.ok():
                rclpy.init(args=None)
                self._owns_context = True
            self._node = rclpy.create_node(node_name)
            self._publisher = self._node.create_publisher(String, topic_name, 10)
            self._rclpy = rclpy
            self._string_type = String
            print(f"ROS 2 string publisher enabled: {topic_name}")
        except Exception as exc:
            carb.log_warn(f"Failed to create ROS 2 Python publisher for {topic_name}: {exc}")

    def publish(self, data: str) -> None:
        if self._publisher is None or self._string_type is None:
            return
        msg = self._string_type()
        msg.data = data
        self._publisher.publish(msg)
        if self._rclpy is not None and self._node is not None:
            self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def shutdown(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._owns_context and self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()


class _OptionalRosStringSubscriber:
    def __init__(self, node_name: str, topic_name: str) -> None:
        self._rclpy = None
        self._node = None
        self._subscription = None
        self._messages = []
        self._owns_context = False

        try:
            import rclpy
            from std_msgs.msg import String
        except Exception as exc:
            carb.log_warn(f"ROS 2 Python subscriber unavailable for {topic_name}: {exc}")
            return

        try:
            if not rclpy.ok():
                rclpy.init(args=None)
                self._owns_context = True
            self._node = rclpy.create_node(node_name)
            self._subscription = self._node.create_subscription(String, topic_name, self._on_message, 10)
            self._rclpy = rclpy
            print(f"ROS 2 string subscriber enabled: {topic_name}")
        except Exception as exc:
            carb.log_warn(f"Failed to create ROS 2 Python subscriber for {topic_name}: {exc}")

    def _on_message(self, msg) -> None:
        self._messages.append(msg.data)
        if len(self._messages) > 10:
            self._messages = self._messages[-10:]

    def poll(self) -> list[str]:
        if self._rclpy is not None and self._node is not None:
            self._rclpy.spin_once(self._node, timeout_sec=0.0)
        messages = self._messages
        self._messages = []
        return messages

    def shutdown(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._owns_context and self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()


class _FileStringMailbox:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._last_sequence = None

    def poll(self) -> list[str]:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except Exception as exc:
            carb.log_warn(f"Failed to read inspection command file {self._path}: {exc}")
            return []

        sequence = payload.get("sequence")
        if sequence is not None and sequence == self._last_sequence:
            return []
        self._last_sequence = sequence

        data = payload.get("data")
        if isinstance(data, str):
            return [data]
        if isinstance(data, dict):
            return [json.dumps(data)]
        return []

    def shutdown(self) -> None:
        return


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _bilinear_resample(coarse_values: np.ndarray, samples: int) -> np.ndarray:
    coarse_y = np.linspace(0.0, samples - 1, coarse_values.shape[0])
    coarse_x = np.linspace(0.0, samples - 1, coarse_values.shape[1])
    target = np.arange(samples)
    rows = np.array([np.interp(target, coarse_x, row) for row in coarse_values])
    return np.array([np.interp(target, coarse_y, rows[:, col]) for col in range(samples)]).T


def _get_fence_y(size: float, fence_y: float) -> float:
    return float(np.clip(fence_y, -size * 0.15, size * 0.35))


def _generate_heightfield(
    size: float,
    resolution: float,
    amplitude: float,
    seed: int,
    fence_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = max(9, int(round(size / resolution)) + 1)
    if samples % 2 == 0:
        samples += 1

    x_values = np.linspace(-size * 0.5, size * 0.5, samples)
    y_values = np.linspace(-size * 0.5, size * 0.5, samples)
    xx, yy = np.meshgrid(x_values, y_values)

    rng = np.random.default_rng(seed)
    detail_samples = max(9, samples // 5)
    detail_noise = _bilinear_resample(rng.uniform(-1.0, 1.0, size=(detail_samples, detail_samples)), samples)

    fence_y = _get_fence_y(size, fence_y)
    fence_band = np.exp(-((yy - fence_y) / 5.5) ** 2)
    interior_falloff = 1.0 - _smoothstep((yy - (fence_y - 11.0)) / 7.0)
    interior_micro_noise = interior_falloff * (0.012 * detail_noise + 0.008 * np.sin(0.23 * xx))

    berm = 0.65 * amplitude * np.exp(-((yy - (fence_y - 1.1)) / 1.3) ** 2)
    ditch = -0.38 * amplitude * np.exp(-((yy - (fence_y + 1.3)) / 1.6) ** 2)
    fence_noise = 0.35 * amplitude * detail_noise * fence_band
    river_bank_drop = -0.42 * _smoothstep((yy - (fence_y + 3.2)) / 8.0)

    patrol_path = np.exp(-(yy / 5.5) ** 2)
    heights = interior_micro_noise + berm + ditch + fence_noise + river_bank_drop
    heights *= 1.0 - 0.65 * patrol_path

    center_idx = samples // 2
    heights -= heights[center_idx, center_idx]

    # Keep the initial spawn area almost flat; blend the noise in outside it.
    radius = np.sqrt(xx * xx + yy * yy)
    spawn_flat_radius = 2.3
    spawn_blend_width = 2.5
    spawn_blend = _smoothstep((radius - spawn_flat_radius) / spawn_blend_width)
    heights *= spawn_blend
    heights[center_idx, center_idx] = 0.0
    return x_values, y_values, heights


def _add_terrain_uvs(mesh: UsdGeom.Mesh, x_values: np.ndarray, y_values: np.ndarray, texture_scale: float) -> None:
    x_min = float(x_values[0])
    x_range = float(x_values[-1] - x_values[0])
    y_min = float(y_values[0])
    y_range = float(y_values[-1] - y_values[0])
    texture_scale = max(0.001, float(texture_scale))
    texcoords = [
        Gf.Vec2f(float((x - x_min) / x_range * texture_scale), float((y - y_min) / y_range * texture_scale))
        for y in y_values
        for x in x_values
    ]
    primvars_api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    st = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(texcoords)


def _bind_terrain_texture(
    stage,
    terrain_prim,
    texture_path: str,
    normal_texture_path: str,
    roughness_texture_path: str,
) -> None:
    if not texture_path and not normal_texture_path and not roughness_texture_path:
        return

    material = UsdShade.Material.Define(stage, Sdf.Path("/World/Looks/GP_TerrainPBRMaterial"))
    shader = UsdShade.Shader.Define(stage, Sdf.Path("/World/Looks/GP_TerrainPBRMaterial/PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.92)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.17, 0.18, 0.12))

    st_reader = UsdShade.Shader.Define(stage, Sdf.Path("/World/Looks/GP_TerrainPBRMaterial/StReader"))
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    if texture_path:
        texture = UsdShade.Shader.Define(stage, Sdf.Path("/World/Looks/GP_TerrainPBRMaterial/AlbedoTexture"))
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_path))
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")

    if roughness_texture_path:
        roughness_texture = UsdShade.Shader.Define(stage, Sdf.Path("/World/Looks/GP_TerrainPBRMaterial/RoughnessTexture"))
        roughness_texture.CreateIdAttr("UsdUVTexture")
        roughness_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(roughness_texture_path))
        roughness_texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
        roughness_texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        roughness_texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        roughness_texture.CreateOutput("r", Sdf.ValueTypeNames.Float)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(roughness_texture.ConnectableAPI(), "r")

    if normal_texture_path:
        normal_texture = UsdShade.Shader.Define(stage, Sdf.Path("/World/Looks/GP_TerrainPBRMaterial/NormalTexture"))
        normal_texture.CreateIdAttr("UsdUVTexture")
        normal_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(normal_texture_path))
        normal_texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
        normal_texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        normal_texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        normal_texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(normal_texture.ConnectableAPI(), "rgb")

    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(terrain_prim).Bind(material)


def _add_terrain_grid(stage, x_values: np.ndarray, y_values: np.ndarray, heights: np.ndarray, grid_step: float) -> None:
    if grid_step <= 0.0:
        return

    resolution = float(abs(x_values[1] - x_values[0]))
    stride = max(1, int(round(grid_step / resolution)))
    samples = heights.shape[0]
    points = []
    curve_vertex_counts = []

    for row in range(0, samples, stride):
        curve_vertex_counts.append(samples)
        for col in range(samples):
            points.append(Gf.Vec3f(float(x_values[col]), float(y_values[row]), float(heights[row, col] + 0.012)))

    for col in range(0, samples, stride):
        curve_vertex_counts.append(samples)
        for row in range(samples):
            points.append(Gf.Vec3f(float(x_values[col]), float(y_values[row]), float(heights[row, col] + 0.012)))

    curves = UsdGeom.BasisCurves.Define(stage, Sdf.Path("/World/GP_TerrainGrid"))
    curves.CreateTypeAttr(UsdGeom.Tokens.linear)
    curves.CreateCurveVertexCountsAttr(curve_vertex_counts)
    curves.CreatePointsAttr(points)
    curves.CreateWidthsAttr([0.018])
    curves.CreateDisplayColorAttr([Gf.Vec3f(0.035, 0.045, 0.03)])


def _sample_height(x_values: np.ndarray, y_values: np.ndarray, heights: np.ndarray, x: float, y: float) -> float:
    x = float(np.clip(x, x_values[0], x_values[-1]))
    y = float(np.clip(y, y_values[0], y_values[-1]))
    col = int(np.searchsorted(x_values, x) - 1)
    row = int(np.searchsorted(y_values, y) - 1)
    col = int(np.clip(col, 0, len(x_values) - 2))
    row = int(np.clip(row, 0, len(y_values) - 2))

    x0 = x_values[col]
    x1 = x_values[col + 1]
    y0 = y_values[row]
    y1 = y_values[row + 1]
    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

    h00 = heights[row, col]
    h10 = heights[row, col + 1]
    h01 = heights[row + 1, col]
    h11 = heights[row + 1, col + 1]
    return float((1.0 - tx) * (1.0 - ty) * h00 + tx * (1.0 - ty) * h10 + (1.0 - tx) * ty * h01 + tx * ty * h11)


def _normalize_vector(vector: np.ndarray, fallback) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return np.asarray(fallback, dtype=float)
    return np.asarray(vector, dtype=float) / norm


def _quat_wxyz_from_rotation_matrix(matrix: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    quat = np.asarray([w, x, y, z], dtype=float)
    quat /= max(1e-9, float(np.linalg.norm(quat)))
    return float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])


def _camera_look_at_quat(camera_position: np.ndarray, target_position: np.ndarray) -> tuple[float, float, float, float]:
    forward = _normalize_vector(target_position - camera_position, (1.0, 0.0, 0.0))
    z_axis = -forward
    world_up = np.asarray((0.0, 0.0, 1.0), dtype=float)
    if abs(float(np.dot(z_axis, world_up))) > 0.98:
        world_up = np.asarray((0.0, 1.0, 0.0), dtype=float)
    x_axis = _normalize_vector(np.cross(world_up, z_axis), (0.0, -1.0, 0.0))
    y_axis = _normalize_vector(np.cross(z_axis, x_axis), (0.0, 0.0, 1.0))
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    return _quat_wxyz_from_rotation_matrix(rotation)


def _set_xform(prim, translate, scale=None, rotate_xyz=None) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.AddTranslateOp().Set(Gf.Vec3f(*translate))
    if rotate_xyz is not None:
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(*rotate_xyz))
    if scale is not None:
        xformable.AddScaleOp().Set(Gf.Vec3f(*scale))


def _set_display_color(gprim, color) -> None:
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _enable_static_collision(prim) -> None:
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
    else:
        collision_api = UsdPhysics.CollisionAPI(prim)
    collision_api.CreateCollisionEnabledAttr(True)


def _add_cube(stage, path: str, center, scale, color, rotate_xyz=None):
    cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
    cube.CreateSizeAttr(1.0)
    _set_xform(cube.GetPrim(), center, scale, rotate_xyz)
    _set_display_color(cube, color)
    _enable_static_collision(cube.GetPrim())
    return cube.GetPrim()


def _add_cylinder(stage, path: str, center, radius: float, height: float, color):
    cylinder = UsdGeom.Cylinder.Define(stage, Sdf.Path(path))
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    _set_xform(cylinder.GetPrim(), center)
    _set_display_color(cylinder, color)
    _enable_static_collision(cylinder.GetPrim())
    return cylinder.GetPrim()


def _apply_class_label(prim, label: str) -> None:
    add_labels(prim, [label], instance_name="class")


def _add_visual_cube(stage, path: str, center, scale, color, label: str | None = None, rotate_xyz=None):
    cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
    cube.CreateSizeAttr(1.0)
    _set_xform(cube.GetPrim(), center, scale, rotate_xyz)
    _set_display_color(cube, color)
    if label is not None:
        _apply_class_label(cube.GetPrim(), label)
    return cube.GetPrim()


def _add_visual_cylinder(stage, path: str, center, radius: float, height: float, color, label: str | None = None):
    cylinder = UsdGeom.Cylinder.Define(stage, Sdf.Path(path))
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    _set_xform(cylinder.GetPrim(), center)
    _set_display_color(cylinder, color)
    if label is not None:
        _apply_class_label(cylinder.GetPrim(), label)
    return cylinder.GetPrim()


def _add_visual_sphere(stage, path: str, center, radius: float, color, label: str | None = None):
    sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(path))
    sphere.CreateRadiusAttr(radius)
    _set_xform(sphere.GetPrim(), center)
    _set_display_color(sphere, color)
    if label is not None:
        _apply_class_label(sphere.GetPrim(), label)
    return sphere.GetPrim()


def _add_curve_lines(stage, path: str, lines, width: float, color) -> None:
    points = []
    counts = []
    for line in lines:
        counts.append(len(line))
        points.extend([Gf.Vec3f(*point) for point in line])

    curves = UsdGeom.BasisCurves.Define(stage, Sdf.Path(path))
    curves.CreateTypeAttr(UsdGeom.Tokens.linear)
    curves.CreateCurveVertexCountsAttr(counts)
    curves.CreatePointsAttr(points)
    curves.CreateWidthsAttr([width])
    curves.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _add_sphere_light(stage, path: str, center, radius: float, intensity: float, color) -> None:
    light = UsdLux.SphereLight.Define(stage, Sdf.Path(path))
    light.CreateRadiusAttr(radius)
    light.CreateIntensityAttr(intensity)
    light.CreateColorAttr(Gf.Vec3f(*color))
    _set_xform(light.GetPrim(), center)


def _add_concertina_wire(
    stage,
    path: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    heights: np.ndarray,
    x_min: float,
    x_max: float,
    center_y: float,
    center_z_offset: float,
    radius: float,
    loops: int,
    width: float,
    color,
) -> None:
    points = []
    sample_count = max(80, loops * 18)
    for idx, x in enumerate(np.linspace(x_min, x_max, sample_count)):
        angle = 2.0 * np.pi * loops * idx / max(1, sample_count - 1)
        ground_z = _sample_height(x_values, y_values, heights, x, center_y)
        y = center_y + radius * np.cos(angle)
        z = ground_z + center_z_offset + radius * np.sin(angle)
        points.append((float(x), float(y), float(z)))
    _add_curve_lines(stage, path, [points], width=width, color=color)


def _add_ground_surface_variation(
    stage,
    x_values: np.ndarray,
    y_values: np.ndarray,
    heights: np.ndarray,
    terrain_size: float,
    fence_y: float,
) -> None:
    rng = np.random.default_rng(20260517)
    stage.DefinePrim("/World/GP_GroundDetail", "Xform")

    x_limit = terrain_size * 0.43
    interior_y_min = -terrain_size * 0.42
    interior_y_max = fence_y - 1.5
    fence_y_max = fence_y + 2.8

    dirt_palette = (
        (0.19, 0.16, 0.10),
        (0.23, 0.19, 0.12),
        (0.15, 0.13, 0.09),
        (0.27, 0.23, 0.15),
    )
    grass_palette = (
        (0.11, 0.16, 0.07),
        (0.15, 0.20, 0.08),
        (0.08, 0.13, 0.06),
        (0.20, 0.22, 0.10),
    )
    gravel_palette = (
        (0.28, 0.27, 0.24),
        (0.20, 0.20, 0.18),
        (0.35, 0.33, 0.29),
    )

    for idx in range(34):
        x = float(rng.uniform(-x_limit, x_limit))
        y = float(rng.uniform(interior_y_min, fence_y_max))
        if abs(y) < 3.2 and abs(x) < 8.0:
            y += 5.5
        z = _sample_height(x_values, y_values, heights, x, y)
        sx = float(rng.uniform(1.4, 4.8))
        sy = float(rng.uniform(0.7, 2.4))
        color = dirt_palette[idx % len(dirt_palette)]
        _add_visual_cube(
            stage,
            f"/World/GP_GroundDetail/BareSoilPatch_{idx}",
            (x, y, z + 0.018),
            (sx, sy, 0.018),
            color,
            rotate_xyz=(0.0, 0.0, float(rng.uniform(-35.0, 35.0))),
        )

    for idx in range(24):
        x = float(rng.uniform(-x_limit, x_limit))
        y = float(rng.uniform(interior_y_min, interior_y_max))
        z = _sample_height(x_values, y_values, heights, x, y)
        color = gravel_palette[idx % len(gravel_palette)]
        _add_visual_cube(
            stage,
            f"/World/GP_GroundDetail/GravelPatch_{idx}",
            (x, y, z + 0.024),
            (float(rng.uniform(0.45, 1.2)), float(rng.uniform(0.25, 0.75)), 0.020),
            color,
            rotate_xyz=(0.0, 0.0, float(rng.uniform(-45.0, 45.0))),
        )

    for idx in range(72):
        x = float(rng.uniform(-x_limit, x_limit))
        y = float(rng.uniform(interior_y_min, fence_y + 3.0))
        if -6.5 < y < -3.4:
            continue
        z = _sample_height(x_values, y_values, heights, x, y)
        height = float(rng.uniform(0.12, 0.42))
        color = grass_palette[idx % len(grass_palette)]
        _add_visual_cube(
            stage,
            f"/World/GP_GroundDetail/GrassClump_{idx}",
            (x, y, z + height * 0.5),
            (float(rng.uniform(0.06, 0.14)), float(rng.uniform(0.04, 0.10)), height),
            color,
            rotate_xyz=(float(rng.uniform(-6.0, 6.0)), float(rng.uniform(-5.0, 5.0)), float(rng.uniform(0.0, 180.0))),
        )


def _add_river(
    stage,
    x_values: np.ndarray,
    y_values: np.ndarray,
    heights: np.ndarray,
    terrain_size: float,
    fence_y: float,
    river_width: float,
) -> None:
    stage.DefinePrim("/World/GP_River", "Xform")
    river_width = max(4.0, min(river_width, terrain_size * 0.38))
    river_start_y = fence_y + 5.0
    river_center_y = min(terrain_size * 0.5 - river_width * 0.5 - 0.4, river_start_y + river_width * 0.5)
    river_sample_y = min(y_values[-1], river_start_y + 1.0)
    river_z = _sample_height(x_values, y_values, heights, 0.0, river_sample_y) + 0.035
    _add_cube(
        stage,
        "/World/GP_River/Water",
        (0.0, river_center_y, river_z),
        (terrain_size * 0.92, river_width, 0.035),
        (0.05, 0.19, 0.30),
    )
    _add_cube(
        stage,
        "/World/GP_River/FarBank",
        (0.0, river_center_y + river_width * 0.5 + 0.8, river_z + 0.12),
        (terrain_size * 0.92, 1.2, 0.24),
        (0.12, 0.18, 0.10),
    )


def _add_external_prop_reference(
    stage,
    prim_path: str,
    asset_path: str,
    center,
    scale: float,
    rotate_xyz=(0.0, 0.0, 0.0),
    label: str | None = None,
) -> bool:
    if not asset_path:
        return False

    resolved_path = str(Path(asset_path).expanduser())
    if not Path(resolved_path).exists():
        carb.log_warn(f"External prop asset not found: {resolved_path}")
        return False

    try:
        root_prim = add_reference_to_stage(usd_path=resolved_path, prim_path=prim_path)
    except Exception as exc:
        carb.log_warn(f"Failed to add external prop '{resolved_path}': {exc}")
        return False

    if root_prim is None or not root_prim.IsValid():
        carb.log_warn(f"External prop did not create a valid prim: {resolved_path}")
        return False

    xformable = UsdGeom.Xformable(root_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3f(*[float(value) for value in center]))
    xformable.AddRotateXYZOp().Set(Gf.Vec3f(*[float(value) for value in rotate_xyz]))
    xformable.AddScaleOp().Set(Gf.Vec3f(float(scale), float(scale), float(scale)))
    if label:
        _label_prim_tree(root_prim, label)
    print(f"Loaded external prop: {resolved_path} -> {prim_path}")
    return True


def _enable_alpha_cutout_for_prop(root_prim, threshold: float = 0.43) -> None:
    alpha_texture = None
    preview_shaders = []

    for prim in Usd.PrimRange(root_prim):
        if prim.GetTypeName() != "Shader":
            continue
        shader = UsdShade.Shader(prim)
        shader_id = shader.GetIdAttr().Get()
        if shader_id == "UsdPreviewSurface":
            preview_shaders.append(shader)
        elif shader_id == "UsdUVTexture":
            file_input = shader.GetInput("file")
            texture_file = file_input.Get() if file_input else None
            texture_file_str = str(texture_file).lower() if texture_file is not None else ""
            if "basecolor" in texture_file_str or "cutoff" in texture_file_str or "alpha" in texture_file_str:
                alpha_texture = shader

    if alpha_texture is not None:
        alpha_texture.CreateOutput("a", Sdf.ValueTypeNames.Float)

    for shader in preview_shaders:
        shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(float(threshold))
        if alpha_texture is not None:
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(
                alpha_texture.ConnectableAPI(),
                "a",
            )


def _add_external_gp_props(
    stage,
    x_values: np.ndarray,
    y_values: np.ndarray,
    heights: np.ndarray,
    terrain_size: float,
    fence_y: float,
    guard_tower_asset: str,
    guard_tower_scale: float,
    fence_asset: str,
    fence_asset_scale: float,
    fence_asset_count: int,
) -> None:
    stage.DefinePrim("/World/GP_ExternalProps", "Xform")
    fence_y = _get_fence_y(terrain_size, fence_y)

    for tower_idx, tower_x in enumerate((-terrain_size * 0.31, terrain_size * 0.31)):
        tower_y = fence_y - 6.0
        tower_z = _sample_height(x_values, y_values, heights, tower_x, tower_y)
        _add_external_prop_reference(
            stage,
            f"/World/GP_ExternalProps/GuardTower_{tower_idx}",
            guard_tower_asset,
            (float(tower_x), float(tower_y), float(tower_z)),
            guard_tower_scale,
            rotate_xyz=(0.0, 0.0, 0.0 if tower_idx == 0 else 180.0),
            label="guard_tower",
        )

    count = max(0, int(fence_asset_count))
    if count > 0:
        x_min = -terrain_size * 0.40
        x_max = terrain_size * 0.40
        for tile_idx, x in enumerate(np.linspace(x_min, x_max, count)):
            tile_y = fence_y
            tile_z = _sample_height(x_values, y_values, heights, x, tile_y)
            _add_external_prop_reference(
                stage,
                f"/World/GP_ExternalProps/ChainlinkFenceTile_{tile_idx}",
                fence_asset,
                (float(x), float(tile_y), float(tile_z)),
                fence_asset_scale,
                rotate_xyz=(0.0, 0.0, 0.0),
                label="fence",
            )
            fence_prim = stage.GetPrimAtPath(f"/World/GP_ExternalProps/ChainlinkFenceTile_{tile_idx}")
            if fence_prim.IsValid():
                _enable_alpha_cutout_for_prop(fence_prim)


def _label_prim_tree(prim, label: str) -> None:
    if not prim.IsValid():
        return
    for child in Usd.PrimRange(prim):
        _apply_class_label(child, label)


def _join_asset_url(root: str, relative_path: str) -> str:
    return root.rstrip("/") + "/" + relative_path.lstrip("/")


def _find_first_usd_in_asset_folder(folder_url: str) -> str | None:
    result, items = omni.client.list(folder_url)
    if result != omni.client.Result.OK:
        carb.log_warn(f"Unable to read Isaac character folder: {folder_url}")
        return None

    for item in items:
        if item.relative_path.lower().endswith(".usd"):
            return _join_asset_url(folder_url, item.relative_path)
    carb.log_warn(f"No USD file found in Isaac character folder: {folder_url}")
    return None


def _resolve_intruder_human_usd(character_or_path: str) -> str | None:
    if not character_or_path:
        return None

    local_path = Path(character_or_path).expanduser()
    if local_path.exists():
        return str(local_path)

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_warn("Isaac asset root not found; using primitive intruder visual.")
        return None

    if character_or_path.lower().endswith(".usd"):
        if character_or_path.startswith("/Isaac/"):
            return _join_asset_url(assets_root_path, character_or_path)
        return character_or_path

    character_folder = _join_asset_url(assets_root_path, f"/Isaac/People/Characters/{character_or_path}")
    return _find_first_usd_in_asset_folder(character_folder)


def _get_default_intruder_human_usd() -> str | None:
    for character_name in DEFAULT_INTRUDER_HUMAN_ASSETS:
        character_usd = _resolve_intruder_human_usd(character_name)
        if character_usd is not None:
            return character_usd
    carb.log_warn("No built-in Isaac people assets were found; using primitive intruder visual.")
    return None


def _create_isaac_human_intruder_visual(stage, path: str, human_usd_path: str, yaw_deg: float) -> dict | None:
    if not human_usd_path:
        return None
    try:
        root_prim = add_reference_to_stage(usd_path=human_usd_path, prim_path=path)
    except Exception as exc:
        carb.log_warn(f"Failed to add human intruder USD '{human_usd_path}': {exc}")
        return None

    if root_prim is None or not root_prim.IsValid():
        carb.log_warn(f"Human intruder USD did not create a valid prim: {human_usd_path}")
        return None

    _label_prim_tree(root_prim, "person")
    xformable = UsdGeom.Xformable(root_prim)
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    # Keep the root transform simple so the same motion controller can move
    # both referenced human USDs and the primitive fallback.
    xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, float(yaw_deg)))
    print(f"Using Isaac human intruder asset: {human_usd_path}")
    return {"prim": root_prim, "translate_op": translate_op, "visual_type": "isaac-human"}


def _create_primitive_intruder_visual(stage, path: str) -> dict:
    root = UsdGeom.Xform.Define(stage, Sdf.Path(path))
    root_prim = root.GetPrim()
    _apply_class_label(root_prim, "person")

    clothing = (0.20, 0.19, 0.15)
    pack = (0.13, 0.10, 0.07)
    skin = (0.48, 0.35, 0.26)
    hair = (0.035, 0.030, 0.025)
    pants = (0.07, 0.075, 0.08)

    _add_visual_cube(stage, f"{path}/Torso", (0.0, 0.0, 0.95), (0.38, 0.22, 0.62), clothing, "person")
    _add_visual_cube(stage, f"{path}/Backpack", (0.0, 0.12, 1.00), (0.36, 0.12, 0.48), pack, "person")
    _add_visual_cube(stage, f"{path}/ChestBag", (0.0, -0.115, 1.02), (0.24, 0.045, 0.28), pack, "person")
    _add_visual_sphere(stage, f"{path}/Head", (0.0, 0.0, 1.45), 0.16, skin, "person")
    _add_visual_sphere(stage, f"{path}/Hair", (0.0, 0.0, 1.57), 0.15, hair, "person")
    _add_visual_cube(stage, f"{path}/LeftArm", (-0.31, 0.0, 0.96), (0.11, 0.12, 0.55), clothing, "person", rotate_xyz=(0.0, 0.0, -8.0))
    _add_visual_cube(stage, f"{path}/RightArm", (0.31, 0.0, 0.96), (0.11, 0.12, 0.55), clothing, "person", rotate_xyz=(0.0, 0.0, 8.0))
    _add_visual_cube(stage, f"{path}/LeftLeg", (-0.12, 0.0, 0.36), (0.13, 0.13, 0.68), pants, "person")
    _add_visual_cube(stage, f"{path}/RightLeg", (0.12, 0.0, 0.36), (0.13, 0.13, 0.68), pants, "person")

    xformable = UsdGeom.Xformable(root_prim)
    translate_op = xformable.AddTranslateOp()
    return {"prim": root_prim, "translate_op": translate_op, "visual_type": "primitive"}


def _create_intruder_visual(stage, path: str, visual_mode: str, human_usd_path: str, yaw_deg: float) -> dict:
    if visual_mode in ("auto", "isaac-human"):
        resolved_human_usd_path = (
            _resolve_intruder_human_usd(human_usd_path) if human_usd_path else _get_default_intruder_human_usd()
        )
        handles = _create_isaac_human_intruder_visual(stage, path, resolved_human_usd_path, yaw_deg)
        if handles is not None:
            return handles
        if visual_mode == "isaac-human":
            carb.log_warn("Requested Isaac human intruder visual, but loading failed. Falling back to primitive.")

    return _create_primitive_intruder_visual(stage, path)


class IntruderScenario:
    def __init__(
        self,
        stage,
        x_values: np.ndarray,
        y_values: np.ndarray,
        heights: np.ndarray,
        terrain_size: float,
        fence_y: float,
        river_width: float,
        count: int,
        speed: float,
        seed: int,
        visual_mode: str,
        human_usd_path: str,
        yaw_deg: float,
    ) -> None:
        self._stage = stage
        self._x_values = x_values
        self._y_values = y_values
        self._heights = heights
        self._terrain_size = terrain_size
        self._fence_y = _get_fence_y(terrain_size, fence_y)
        self._river_width = river_width
        self._speed = max(0.0, float(speed))
        self._rng = np.random.default_rng(seed)
        self._intruders = []
        self._visual_mode = visual_mode
        self._human_usd_path = human_usd_path
        self._yaw_deg = float(yaw_deg)
        self._visual_types = set()

        stage.DefinePrim("/World/Intruders", "Xform")
        for index in range(max(0, int(count))):
            path = f"/World/Intruders/Intruder_{index}"
            handles = _create_intruder_visual(stage, path, self._visual_mode, self._human_usd_path, self._yaw_deg)
            self._visual_types.add(handles.get("visual_type", "unknown"))
            state = {
                "path": path,
                "translate_op": handles["translate_op"],
                "visual_type": handles.get("visual_type", "unknown"),
                "phase": float(self._rng.uniform(0.0, np.pi * 2.0)),
                "time": 0.0,
            }
            self._respawn_intruder(state)
            self._intruders.append(state)

        if self._intruders:
            print(
                f"Intruder scenario enabled: count={len(self._intruders)}, "
                f"speed={self._speed:.2f} m/s, visual={','.join(sorted(self._visual_types))}, label=person"
            )

    def _respawn_intruder(self, state: dict) -> None:
        x_limit = self._terrain_size * 0.30
        start_y_min = min(self._y_values[-1] - 2.0, self._fence_y + max(8.0, self._river_width * 0.45))
        start_y_max = min(self._y_values[-1] - 1.0, self._fence_y + max(14.0, self._river_width * 0.78))
        if start_y_max <= start_y_min:
            start_y_max = start_y_min + 0.5

        x = float(self._rng.uniform(-x_limit, x_limit))
        y = float(self._rng.uniform(start_y_min, start_y_max))
        target_y = float(self._fence_y + 1.25 + self._rng.uniform(-0.25, 0.55))
        state["position"] = np.array([x, y, 0.0], dtype=float)
        state["target_y"] = target_y
        state["arrived"] = False
        state["hold_time"] = 0.0
        self._set_intruder_pose(state)

    def randomize_for_dataset(self) -> list[np.ndarray]:
        positions = []
        if not self._intruders:
            return positions

        x_limit = self._terrain_size * 0.34
        start_y_min = min(self._y_values[-1] - 2.0, self._fence_y + 1.6)
        start_y_max = min(self._y_values[-1] - 1.0, self._fence_y + max(4.2, self._river_width * 0.30))
        if start_y_max <= start_y_min:
            start_y_max = start_y_min + 0.5

        for state in self._intruders:
            x = float(self._rng.uniform(-x_limit, x_limit))
            y = float(self._rng.uniform(start_y_min, start_y_max))
            state["position"] = np.array([x, y, 0.0], dtype=float)
            state["target_y"] = float(self._fence_y + 1.0)
            state["arrived"] = False
            state["hold_time"] = 0.0
            state["time"] = float(self._rng.uniform(0.0, 4.0))
            self._set_intruder_pose(state)
            positions.append(state["position"].copy())
        return positions

    def get_intruder_reports(self) -> list[dict]:
        reports = []
        for index, state in enumerate(self._intruders):
            position = state["position"]
            ground_z = _sample_height(self._x_values, self._y_values, self._heights, position[0], position[1])
            reports.append(
                {
                    "id": index,
                    "label": "person",
                    "path": state["path"],
                    "visual_type": state.get("visual_type", "unknown"),
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(ground_z),
                    "arrived": bool(state["arrived"]),
                    "target_y": float(state["target_y"]),
                }
            )
        return reports

    def _set_intruder_pose(self, state: dict) -> None:
        position = state["position"]
        ground_z = _sample_height(self._x_values, self._y_values, self._heights, position[0], position[1])
        bob = 0.025 * np.sin(state["time"] * 5.0 + state["phase"]) if not state["arrived"] else 0.0
        state["translate_op"].Set(Gf.Vec3d(float(position[0]), float(position[1]), float(ground_z + bob)))

    def update(self, step_size: float) -> None:
        if not self._intruders:
            return

        for state in self._intruders:
            state["time"] += float(step_size)
            if state["arrived"]:
                state["hold_time"] += float(step_size)
                if state["hold_time"] > 8.0:
                    self._respawn_intruder(state)
                else:
                    self._set_intruder_pose(state)
                continue

            state["position"][1] = max(state["target_y"], state["position"][1] - self._speed * float(step_size))
            if state["position"][1] <= state["target_y"] + 1e-3:
                state["arrived"] = True
            self._set_intruder_pose(state)


def _add_gp_props(
    stage,
    x_values: np.ndarray,
    y_values: np.ndarray,
    heights: np.ndarray,
    terrain_size: float,
    fence_y: float,
    river_width: float,
    skip_watchtowers: bool = False,
) -> None:
    stage.DefinePrim("/World/GP_Props", "Xform")
    fence_y = _get_fence_y(terrain_size, fence_y)
    x_min = -terrain_size * 0.44
    x_max = terrain_size * 0.44
    post_spacing = 3.5
    main_post_height = 2.55

    wire_lines = []
    fence_specs = [
        ("Main", fence_y, main_post_height, 0.055, (0.10, 0.12, 0.10)),
        ("Inner", fence_y - 2.4, 1.55, 0.045, (0.12, 0.12, 0.10)),
    ]
    for fence_name, current_fence_y, post_height, radius, color in fence_specs:
        xs = np.arange(x_min, x_max + 0.001, post_spacing)
        for post_idx, x in enumerate(xs):
            z = _sample_height(x_values, y_values, heights, x, current_fence_y)
            _add_cylinder(
                stage,
                f"/World/GP_Props/Fence_{fence_name}_Post_{post_idx}",
                (float(x), float(current_fence_y), z + post_height * 0.5),
                radius=radius,
                height=post_height,
                color=color,
            )

        wire_heights = (0.65, 1.15, 1.65, 2.15) if fence_name == "Main" else (0.55, 1.05, 1.42)
        for wire_height in wire_heights:
            line = []
            for x in np.linspace(x_min, x_max, 80):
                z = _sample_height(x_values, y_values, heights, x, current_fence_y)
                line.append((float(x), float(current_fence_y), z + wire_height))
            wire_lines.append(line)

    _add_curve_lines(stage, "/World/GP_Props/FenceWires", wire_lines, width=0.025, color=(0.05, 0.06, 0.05))

    mesh_lines = []
    for mesh_idx, x in enumerate(np.arange(x_min, x_max - post_spacing, post_spacing)):
        for base_height, top_height in ((0.55, 1.15), (1.15, 1.75), (1.75, 2.25)):
            z0 = _sample_height(x_values, y_values, heights, x, fence_y)
            z1 = _sample_height(x_values, y_values, heights, x + post_spacing, fence_y)
            if mesh_idx % 2 == 0:
                mesh_lines.append(
                    (
                        (float(x), float(fence_y - 0.015), z0 + base_height),
                        (float(x + post_spacing), float(fence_y - 0.015), z1 + top_height),
                    )
                )
            else:
                mesh_lines.append(
                    (
                        (float(x), float(fence_y - 0.015), z0 + top_height),
                        (float(x + post_spacing), float(fence_y - 0.015), z1 + base_height),
                    )
                )
    _add_curve_lines(stage, "/World/GP_Props/FenceDiamondMesh", mesh_lines, width=0.012, color=(0.035, 0.045, 0.035))

    _add_concertina_wire(
        stage,
        "/World/GP_Props/MainFenceTopConcertina",
        x_values,
        y_values,
        heights,
        x_min,
        x_max,
        fence_y,
        2.35,
        0.23,
        34,
        0.025,
        (0.055, 0.065, 0.055),
    )
    _add_concertina_wire(
        stage,
        "/World/GP_Props/RiverSideGroundConcertina",
        x_values,
        y_values,
        heights,
        x_min,
        x_max,
        fence_y + 1.55,
        0.35,
        0.28,
        32,
        0.023,
        (0.055, 0.060, 0.052),
    )
    _add_concertina_wire(
        stage,
        "/World/GP_Props/InteriorGroundConcertina",
        x_values,
        y_values,
        heights,
        x_min,
        x_max,
        fence_y - 3.15,
        0.30,
        0.22,
        28,
        0.021,
        (0.065, 0.060, 0.050),
    )

    for rail_idx, (rail_name, rail_y, rail_height, rail_thickness, rail_color) in enumerate(
        (
            ("Main", fence_y, 1.15, 0.12, (0.06, 0.075, 0.06)),
            ("MainTop", fence_y, 1.95, 0.08, (0.06, 0.075, 0.06)),
            ("Inner", fence_y - 2.4, 0.95, 0.10, (0.075, 0.075, 0.06)),
        )
    ):
        rail_z = _sample_height(x_values, y_values, heights, 0.0, rail_y) + rail_height
        _add_cube(
            stage,
            f"/World/GP_Props/Fence_{rail_name}_CollisionRail_{rail_idx}",
            (0.0, rail_y, rail_z),
            (x_max - x_min, rail_thickness, rail_thickness),
            rail_color,
        )

    road_center_y = fence_y - 5.0
    road_z = _sample_height(x_values, y_values, heights, 0.0, road_center_y) + 0.018
    _add_visual_cube(
        stage,
        "/World/GP_Props/PatrolRoadPackedSoil",
        (0.0, road_center_y, road_z),
        (x_max - x_min, 2.0, 0.018),
        (0.18, 0.15, 0.10),
    )

    service_road_lines = []
    for road_offset in (-4.1, -5.7):
        line = []
        road_y = fence_y + road_offset
        for x in np.linspace(x_min, x_max, 90):
            z = _sample_height(x_values, y_values, heights, x, road_y)
            line.append((float(x), float(road_y), z + 0.025))
        service_road_lines.append(line)
    _add_curve_lines(stage, "/World/GP_Props/FenceServiceRoadEdges", service_road_lines, width=0.12, color=(0.22, 0.18, 0.11))

    if not skip_watchtowers:
        for tower_idx, tower_x in enumerate((-terrain_size * 0.30, terrain_size * 0.30)):
            tower_y = fence_y - 4.8
            tower_z = _sample_height(x_values, y_values, heights, tower_x, tower_y)
            for leg_idx, (dx, dy) in enumerate(((-0.8, -0.8), (0.8, -0.8), (-0.8, 0.8), (0.8, 0.8))):
                _add_cube(
                    stage,
                    f"/World/GP_Props/Watchtower_{tower_idx}_Leg_{leg_idx}",
                    (tower_x + dx, tower_y + dy, tower_z + 1.75),
                    (0.12, 0.12, 3.5),
                    (0.16, 0.12, 0.08),
                )
            _add_cube(stage, f"/World/GP_Props/Watchtower_{tower_idx}_Platform", (tower_x, tower_y, tower_z + 3.35), (2.25, 2.25, 0.18), (0.18, 0.15, 0.10))
            _add_cube(stage, f"/World/GP_Props/Watchtower_{tower_idx}_Cabin", (tower_x, tower_y, tower_z + 4.1), (1.75, 1.55, 1.15), (0.25, 0.29, 0.19))
            _add_cube(stage, f"/World/GP_Props/Watchtower_{tower_idx}_Roof", (tower_x, tower_y, tower_z + 4.78), (2.25, 1.95, 0.18), (0.09, 0.10, 0.08))
            _add_cube(stage, f"/World/GP_Props/Watchtower_{tower_idx}_SearchLightHousing", (tower_x, tower_y + 0.95, tower_z + 4.42), (0.42, 0.22, 0.22), (0.04, 0.045, 0.04))
            _add_sphere_light(
                stage,
                f"/World/GP_Props/Watchtower_{tower_idx}_SearchLight",
                (tower_x, tower_y + 1.08, tower_z + 4.42),
                0.16,
                2200.0,
                (1.0, 0.86, 0.60),
            )

    for bunker_idx, bunker_x in enumerate((-terrain_size * 0.16, terrain_size * 0.08, terrain_size * 0.24)):
        bunker_y = fence_y - 7.2 - 1.0 * (bunker_idx % 2)
        bunker_z = _sample_height(x_values, y_values, heights, bunker_x, bunker_y)
        _add_cube(stage, f"/World/GP_Props/Bunker_{bunker_idx}_Body", (bunker_x, bunker_y, bunker_z + 0.7), (3.6, 2.4, 1.4), (0.23, 0.24, 0.21))
        _add_cube(stage, f"/World/GP_Props/Bunker_{bunker_idx}_Roof", (bunker_x, bunker_y, bunker_z + 1.52), (4.0, 2.8, 0.28), (0.16, 0.18, 0.15))
        _add_cube(stage, f"/World/GP_Props/Bunker_{bunker_idx}_Slit", (bunker_x, bunker_y + 1.22, bunker_z + 0.92), (1.8, 0.07, 0.22), (0.02, 0.025, 0.02))

    for idx, x in enumerate(np.linspace(-terrain_size * 0.34, terrain_size * 0.34, 9)):
        y = fence_y - 3.25 - 0.35 * (idx % 2)
        z = _sample_height(x_values, y_values, heights, x, y)
        _add_cube(
            stage,
            f"/World/GP_Props/Concrete_Block_{idx}",
            (float(x), float(y), z + 0.28),
            (1.1, 0.35, 0.55),
            (0.34, 0.34, 0.31),
            rotate_xyz=(0.0, 0.0, 8.0 if idx % 2 == 0 else -8.0),
        )

    for marker_idx, x in enumerate(np.linspace(-terrain_size * 0.40, terrain_size * 0.40, 13)):
        y = fence_y - 2.0
        z = _sample_height(x_values, y_values, heights, x, y)
        color = (0.72, 0.62, 0.12) if marker_idx % 2 == 0 else (0.55, 0.08, 0.06)
        _add_cylinder(stage, f"/World/GP_Props/BoundaryMarker_{marker_idx}", (float(x), y, z + 0.42), 0.045, 0.84, color)

    for sign_idx, sign_x in enumerate(np.linspace(-terrain_size * 0.38, terrain_size * 0.38, 7)):
        sign_y = fence_y - 1.0
        sign_z = _sample_height(x_values, y_values, heights, sign_x, sign_y)
        _add_cylinder(stage, f"/World/GP_Props/WarningSign_{sign_idx}_Post", (sign_x, sign_y, sign_z + 0.65), 0.03, 1.3, (0.08, 0.08, 0.07))
        _add_cube(stage, f"/World/GP_Props/WarningSign_{sign_idx}_Plate", (sign_x, sign_y, sign_z + 1.23), (0.75, 0.05, 0.45), (0.85, 0.72, 0.18))
        _add_visual_cube(stage, f"/World/GP_Props/WarningSign_{sign_idx}_RedStripe", (sign_x, sign_y - 0.028, sign_z + 1.23), (0.72, 0.018, 0.08), (0.70, 0.08, 0.06))

    river_start_y = fence_y + 5.0
    for buoy_idx, x in enumerate(np.linspace(-terrain_size * 0.32, terrain_size * 0.32, 8)):
        y = river_start_y + 3.8 + 0.55 * (buoy_idx % 2)
        z = _sample_height(x_values, y_values, heights, x, y) + 0.17
        _add_visual_cylinder(stage, f"/World/GP_Props/RiverMarkerBuoy_{buoy_idx}", (float(x), y, z), 0.13, 0.34, (0.80, 0.22, 0.08))

    for reed_idx, x in enumerate(np.linspace(-terrain_size * 0.43, terrain_size * 0.43, 28)):
        y = river_start_y - 0.65 + 0.35 * np.sin(reed_idx * 1.7)
        z = _sample_height(x_values, y_values, heights, x, y)
        height = 0.38 + 0.18 * ((reed_idx * 37) % 11) / 10.0
        _add_visual_cube(
            stage,
            f"/World/GP_Props/RiverBankReed_{reed_idx}",
            (float(x), float(y), z + height * 0.5),
            (0.035, 0.035, height),
            (0.16, 0.20, 0.08),
            rotate_xyz=(0.0, 0.0, -9.0 + (reed_idx % 5) * 4.5),
        )

    for light_idx, x in enumerate(np.linspace(-terrain_size * 0.36, terrain_size * 0.36, 5)):
        y = fence_y - 6.8
        z = _sample_height(x_values, y_values, heights, x, y)
        _add_cylinder(stage, f"/World/GP_Props/PatrolLight_{light_idx}_Pole", (float(x), y, z + 1.7), 0.045, 3.4, (0.08, 0.08, 0.07))
        _add_cube(stage, f"/World/GP_Props/PatrolLight_{light_idx}_Head", (float(x), y + 0.18, z + 3.45), (0.45, 0.22, 0.18), (0.04, 0.045, 0.04))
        _add_sphere_light(stage, f"/World/GP_Props/PatrolLight_{light_idx}_Lamp", (float(x), y + 0.28, z + 3.45), 0.10, 1300.0, (1.0, 0.84, 0.55))


def _create_noise_terrain(
    stage,
    prim_path: str,
    size: float,
    resolution: float,
    amplitude: float,
    seed: int,
    grid_step: float,
    trail_width: float,
    fence_y: float,
    river_width: float,
    texture_path: str,
    normal_texture_path: str,
    roughness_texture_path: str,
    texture_scale: float,
    add_ground_detail: bool,
    add_gp_props: bool,
    add_external_props: bool,
    guard_tower_asset: str,
    guard_tower_scale: float,
    fence_asset: str,
    fence_asset_scale: float,
    fence_asset_count: int,
) -> float:
    x_values, y_values, heights = _generate_heightfield(size, resolution, amplitude, seed, fence_y)
    samples = heights.shape[0]

    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(prim_path))
    points = [
        Gf.Vec3f(float(x_values[col]), float(y_values[row]), float(heights[row, col]))
        for row in range(samples)
        for col in range(samples)
    ]

    indices = []
    for row in range(samples - 1):
        for col in range(samples - 1):
            p00 = row * samples + col
            p10 = p00 + 1
            p01 = (row + 1) * samples + col
            p11 = p01 + 1
            indices.extend([p00, p10, p11, p00, p11, p01])

    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr().Set("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.17, 0.18, 0.12)])
    _add_terrain_uvs(mesh, x_values, y_values, texture_scale)
    mesh.CreateExtentAttr(
        [
            Gf.Vec3f(float(x_values[0]), float(y_values[0]), float(np.min(heights))),
            Gf.Vec3f(float(x_values[-1]), float(y_values[-1]), float(np.max(heights))),
        ]
    )

    terrain_prim = mesh.GetPrim()
    if not terrain_prim.HasAPI(UsdPhysics.CollisionAPI):
        collision_api = UsdPhysics.CollisionAPI.Apply(terrain_prim)
    else:
        collision_api = UsdPhysics.CollisionAPI(terrain_prim)
    collision_api.CreateCollisionEnabledAttr(True)

    if not terrain_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(terrain_prim)
    else:
        mesh_collision_api = UsdPhysics.MeshCollisionAPI(terrain_prim)
    mesh_collision_api.CreateApproximationAttr().Set("none")

    _bind_terrain_texture(stage, terrain_prim, texture_path, normal_texture_path, roughness_texture_path)
    _add_terrain_grid(stage, x_values, y_values, heights, grid_step)
    _add_river(stage, x_values, y_values, heights, size, fence_y, river_width)
    if add_ground_detail:
        _add_ground_surface_variation(stage, x_values, y_values, heights, size, _get_fence_y(size, fence_y))
    if add_gp_props:
        external_guard_towers_enabled = (
            add_external_props and bool(guard_tower_asset) and Path(guard_tower_asset).expanduser().exists()
        )
        _add_gp_props(
            stage,
            x_values,
            y_values,
            heights,
            size,
            fence_y,
            river_width,
            skip_watchtowers=external_guard_towers_enabled,
        )
    if add_external_props:
        _add_external_gp_props(
            stage,
            x_values,
            y_values,
            heights,
            size,
            fence_y,
            guard_tower_asset,
            guard_tower_scale,
            fence_asset,
            fence_asset_scale,
            fence_asset_count,
        )

    height_min = float(np.min(heights))
    height_max = float(np.max(heights))
    print(f"Generated terrain height range: {height_min:.3f}m to {height_max:.3f}m")
    carb.log_info(
        f"Created GP noise terrain at {prim_path}: size={size:.2f}m, "
        f"resolution={resolution:.2f}m, amplitude={amplitude:.3f}m, seed={seed}"
    )
    return float(heights[samples // 2, samples // 2]), x_values, y_values, heights


def _add_scene_lighting(stage) -> None:
    dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome_light.CreateIntensityAttr(650.0)
    dome_light.CreateColorAttr(Gf.Vec3f(0.86, 0.92, 1.0))

    distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
    distant_light.CreateIntensityAttr(1800.0)
    distant_light.CreateAngleAttr(0.7)
    xformable = UsdGeom.Xformable(distant_light.GetPrim())
    xformable.AddRotateXYZOp().Set(Gf.Vec3f(-55.0, 0.0, 35.0))


def _make_child_path(parent_path: str, child_name: str) -> str:
    return f"{parent_path.rstrip('/')}/{child_name}"


def _find_anymal_sensor_mount(stage, robot_path: str = "/World/Anymal") -> str:
    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim.IsValid():
        carb.log_warn(f"ANYmal prim not found at {robot_path}; using robot path for sensors.")
        return robot_path

    preferred_paths = (
        f"{robot_path}/base",
        f"{robot_path}/base_link",
        f"{robot_path}/trunk",
        f"{robot_path}/body",
        f"{robot_path}/Body",
        f"{robot_path}/chassis",
    )
    for path in preferred_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            return path

    preferred_names = ("base", "base_link", "trunk", "body", "torso", "chassis")
    rigid_body_fallback = None
    for prim in Usd.PrimRange(robot_prim):
        if prim == robot_prim:
            continue
        prim_name = prim.GetName().lower()
        prim_path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and rigid_body_fallback is None:
            rigid_body_fallback = prim_path
        if prim_name in preferred_names or any(name in prim_name for name in preferred_names):
            return prim_path

    if rigid_body_fallback is not None:
        carb.log_warn(f"Using first rigid body prim as sensor mount: {rigid_body_fallback}")
        return rigid_body_fallback

    carb.log_warn(f"Could not find a moving body link under {robot_path}; using robot root for sensors.")
    return robot_path


def _create_front_camera(stage, camera_path: str) -> str:
    camera = UsdGeom.Camera.Define(stage, Sdf.Path(camera_path))
    xformable = UsdGeom.Xformable(camera.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*FRONT_CAMERA_TRANSLATION))
    xformable.AddOrientOp().Set(
        Gf.Quatf(
            FRONT_CAMERA_ORIENTATION_IJKR[3],
            Gf.Vec3f(
                FRONT_CAMERA_ORIENTATION_IJKR[0],
                FRONT_CAMERA_ORIENTATION_IJKR[1],
                FRONT_CAMERA_ORIENTATION_IJKR[2],
            ),
        )
    )
    camera.GetHorizontalApertureAttr().Set(21.0)
    camera.GetVerticalApertureAttr().Set(16.0)
    camera.GetProjectionAttr().Set("perspective")
    camera.GetFocalLengthAttr().Set(18.0)
    camera.GetFocusDistanceAttr().Set(20.0)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 250.0))
    return camera_path


def _create_inspection_camera(stage, camera_path: str) -> str:
    camera = UsdGeom.Camera.Define(stage, Sdf.Path(camera_path))
    xformable = UsdGeom.Xformable(camera.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 1.2))
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0)))
    camera.GetHorizontalApertureAttr().Set(21.0)
    camera.GetVerticalApertureAttr().Set(16.0)
    camera.GetProjectionAttr().Set("perspective")
    camera.GetFocalLengthAttr().Set(INSPECTION_CAMERA_DEFAULT_FOCAL_LENGTH)
    camera.GetFocusDistanceAttr().Set(20.0)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 300.0))
    return camera_path


def _add_sensor_visual_markers(stage, mount_path: str) -> None:
    _add_cube(
        stage,
        _make_child_path(mount_path, "SentryCameraHousing"),
        (0.88, 0.0, 0.60),
        (0.18, 0.10, 0.10),
        (0.03, 0.035, 0.04),
    )
    _add_cylinder(
        stage,
        _make_child_path(mount_path, "SentryLidarHousing"),
        (0.18, 0.0, 1.05),
        radius=0.13,
        height=0.10,
        color=(0.02, 0.025, 0.03),
    )


def _setup_ros2_camera_graph(
    camera_path: str,
    graph_path: str = "/ROS2_Sentry_Camera",
    viewport_name: str = "ROS2_Sentry_CameraViewport",
    viewport_id: int = 1,
    frame_id: str = FRONT_CAMERA_NAME,
    topic_prefix: str = "camera",
):
    keys = og.Controller.Keys
    (ros_camera_graph, _, _, _) = og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "push",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
        },
        {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnTick"),
                ("createViewport", "isaacsim.core.nodes.IsaacCreateViewport"),
                ("getRenderProduct", "isaacsim.core.nodes.IsaacGetViewportRenderProduct"),
                ("setCamera", "isaacsim.core.nodes.IsaacSetCameraOnRenderProduct"),
                ("cameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("cameraHelperInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                ("cameraHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            keys.CONNECT: [
                ("OnTick.outputs:tick", "createViewport.inputs:execIn"),
                ("createViewport.outputs:execOut", "getRenderProduct.inputs:execIn"),
                ("createViewport.outputs:viewport", "getRenderProduct.inputs:viewport"),
                ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
                ("getRenderProduct.outputs:renderProductPath", "setCamera.inputs:renderProductPath"),
                ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                ("setCamera.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
                ("setCamera.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
                ("getRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
                ("getRenderProduct.outputs:renderProductPath", "cameraHelperInfo.inputs:renderProductPath"),
                ("getRenderProduct.outputs:renderProductPath", "cameraHelperDepth.inputs:renderProductPath"),
            ],
            keys.SET_VALUES: [
                ("createViewport.inputs:name", viewport_name),
                ("createViewport.inputs:viewportId", int(viewport_id)),
                ("cameraHelperRgb.inputs:frameId", frame_id),
                ("cameraHelperRgb.inputs:topicName", f"{topic_prefix}/image_raw"),
                ("cameraHelperRgb.inputs:type", "rgb"),
                ("cameraHelperInfo.inputs:frameId", frame_id),
                ("cameraHelperInfo.inputs:topicName", f"{topic_prefix}/camera_info"),
                ("cameraHelperDepth.inputs:frameId", frame_id),
                ("cameraHelperDepth.inputs:topicName", f"{topic_prefix}/depth"),
                ("cameraHelperDepth.inputs:type", "depth"),
                ("setCamera.inputs:cameraPrim", [usdrt.Sdf.Path(camera_path)]),
            ],
        },
    )
    og.Controller.evaluate_sync(ros_camera_graph)
    return ros_camera_graph


def _setup_ros2_lidar_graph(
    render_product_path: str,
    topic_name: str = "lidar/points",
    frame_id: str = LIDAR_NAME,
    graph_path: str = "/ROS2_Sentry_Lidar",
):
    keys = og.Controller.Keys
    (ros_lidar_graph, _, _, _) = og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("PublishPointCloud", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishPointCloud.inputs:execIn"),
            ],
            keys.SET_VALUES: [
                ("PublishPointCloud.inputs:renderProductPath", render_product_path),
                ("PublishPointCloud.inputs:topicName", topic_name),
                ("PublishPointCloud.inputs:frameId", frame_id),
                ("PublishPointCloud.inputs:type", "point_cloud"),
                ("PublishPointCloud.inputs:showDebugView", True),
                ("PublishPointCloud.inputs:resetSimulationTimeOnStop", True),
            ],
        },
    )
    og.Controller.evaluate_sync(ros_lidar_graph)
    return ros_lidar_graph


def _create_lidar_and_ros2_writer(lidar_path: str, translation=(0.18, 0.0, 1.05)):
    _, sensor = omni.kit.commands.execute(
        "IsaacSensorCreateRtxLidar",
        path=lidar_path,
        parent=None,
        config="Example_Rotary",
        translation=translation,
        orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
    )

    hydra_texture = rep.create.render_product(
        sensor.GetPath(),
        resolution=(128, 128),
        render_vars=["GenericModelOutput", "RtxSensorMetadata"],
        name="SentryLidar",
    )
    render_product_path = hydra_texture.path
    lidar_graph = _setup_ros2_lidar_graph(render_product_path)

    debug_writer = rep.writers.get("RtxLidar" + "DebugDrawPointCloud")
    debug_writer.attach([render_product_path])

    return {
        "lidar_path": lidar_path,
        "render_product": render_product_path,
        "graph": lidar_graph,
        "writers": [debug_writer],
    }


def _setup_ros2_tf_clock_graph(target_paths=None, graph_path: str = "/ROS2_Sentry_TF_Clock"):
    if target_paths is None:
        target_paths = []

    keys = og.Controller.Keys
    create_nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
    ]
    connect = [
        ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
    ]
    set_values = [("PublishClock.inputs:topicName", "clock")]

    if target_paths:
        create_nodes.append(("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"))
        connect.extend(
            [
                ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
            ]
        )
        set_values.extend(
            [
                ("PublishTF.inputs:topicName", "tf"),
                ("PublishTF.inputs:targetPrims", [usdrt.Sdf.Path(path) for path in target_paths]),
            ]
        )

    (ros_tf_clock_graph, _, _, _) = og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: create_nodes,
            keys.CONNECT: connect,
            keys.SET_VALUES: set_values,
        },
    )
    og.Controller.evaluate_sync(ros_tf_clock_graph)
    return ros_tf_clock_graph


def _setup_ros2_static_sensor_tf_graph(
    parent_frame_id: str,
    graph_path: str = "/ROS2_Sentry_StaticSensorTF",
) -> dict:
    keys = og.Controller.Keys
    (sensor_tf_graph, _, _, _) = og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishCameraTF", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishCameraTF.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishCameraTF.inputs:timeStamp"),
            ],
            keys.SET_VALUES: [
                ("PublishCameraTF.inputs:topicName", "tf_static"),
                ("PublishCameraTF.inputs:parentFrameId", parent_frame_id),
                ("PublishCameraTF.inputs:childFrameId", FRONT_CAMERA_NAME),
                ("PublishCameraTF.inputs:translation", list(FRONT_CAMERA_TRANSLATION)),
                ("PublishCameraTF.inputs:rotation", list(FRONT_CAMERA_ORIENTATION_IJKR)),
                ("PublishCameraTF.inputs:staticPublisher", True),
            ],
        },
    )
    og.Controller.evaluate_sync(sensor_tf_graph)
    print(f"ROS 2 static sensor TF enabled: {parent_frame_id} -> {FRONT_CAMERA_NAME}")
    return {"graph": sensor_tf_graph}


def _setup_sentry_sensors(
    stage,
    robot_path: str = "/World/Anymal",
    lidar_mount: str = "robot",
    enable_lidar: bool = True,
) -> dict:
    mount_path = _find_anymal_sensor_mount(stage, robot_path)
    camera_path = _make_child_path(mount_path, FRONT_CAMERA_NAME)
    inspection_camera_path = f"/World/{INSPECTION_CAMERA_NAME}"
    _add_sensor_visual_markers(stage, mount_path)
    camera_path = _create_front_camera(stage, camera_path)
    camera_graph = _setup_ros2_camera_graph(camera_path)
    inspection_camera_path = _create_inspection_camera(stage, inspection_camera_path)
    inspection_camera_graph = _setup_ros2_camera_graph(
        inspection_camera_path,
        graph_path="/ROS2_Sentry_InspectionCamera",
        viewport_name="ROS2_Sentry_InspectionViewport",
        viewport_id=2,
        frame_id=INSPECTION_CAMERA_NAME,
        topic_prefix="inspection_camera",
    )

    if lidar_mount in ("follow", "world"):
        lidar_path = f"/World/{LIDAR_NAME}"
        lidar_translation = (0.0, -4.0, 1.6)
        if lidar_mount == "world":
            print("LiDAR debug mode: mounted at fixed world pose /World/SentryLidar")
        else:
            print("LiDAR follow mode: using /World/SentryLidar and updating it from ANYmal pose")
    else:
        lidar_path = _make_child_path(mount_path, LIDAR_NAME)
        lidar_translation = (0.18, 0.0, 1.05)

    if enable_lidar:
        lidar_handles = _create_lidar_and_ros2_writer(lidar_path, translation=lidar_translation)
        tf_clock_targets = [lidar_path] if lidar_mount == "world" else []
    else:
        lidar_handles = {}
        tf_clock_targets = []
        print("ROS 2 LiDAR disabled: camera, TF, and clock publishers remain enabled.")
    tf_clock_graph = _setup_ros2_tf_clock_graph(tf_clock_targets)
    simulation_app.update()
    sensor_topics = (
        "/camera/image_raw, /camera/depth, /camera/camera_info, "
        "/inspection_camera/image_raw, /inspection_camera/depth, /inspection_camera/camera_info, /tf, /clock"
    )
    if enable_lidar:
        sensor_topics += ", /lidar/points"
    print(f"ROS 2 sensors enabled: {sensor_topics}")
    print(f"Sentry sensors mounted on: {mount_path}")
    return {
        "mount_path": mount_path,
        "camera_path": camera_path,
        "camera_graph": camera_graph,
        "inspection_camera": {
            "camera_path": inspection_camera_path,
            "camera_graph": inspection_camera_graph,
        },
        "lidar": lidar_handles,
        "lidar_mount_mode": lidar_mount,
        "lidar_local_offset": np.array([0.18, 0.0, 1.05]),
        "tf_clock_graph": tf_clock_graph,
    }


def _setup_ros2_cmd_vel_graph(topic_name: str, graph_path: str = "/ROS2_Sentry_CmdVel") -> dict:
    keys = og.Controller.Keys
    (cmd_vel_graph, _, _, _) = og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("SubscribeTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "SubscribeTwist.inputs:execIn"),
            ],
            keys.SET_VALUES: [
                ("SubscribeTwist.inputs:topicName", topic_name),
            ],
        },
    )
    og.Controller.evaluate_sync(cmd_vel_graph)
    print(f"ROS 2 cmd_vel enabled via bridge graph: /{topic_name.lstrip('/')}")
    return {
        "graph": cmd_vel_graph,
        "linear_attr": f"{graph_path}/SubscribeTwist.outputs:linearVelocity",
        "angular_attr": f"{graph_path}/SubscribeTwist.outputs:angularVelocity",
    }


def _setup_ros2_odometry_graph(
    chassis_prim_path: str = "/World/Anymal",
    topic_name: str = "odom",
    odom_frame_id: str = "world",
    chassis_frame_id: str = "Anymal",
    graph_path: str = "/ROS2_Sentry_Odometry",
) -> dict:
    keys = og.Controller.Keys
    (odom_graph, _, _, _) = og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("ComputeOdometry", "isaacsim.core.nodes.IsaacComputeOdometry"),
                ("PublishOdometry", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                ("PublishBaseTF", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "ComputeOdometry.inputs:execIn"),
                ("ComputeOdometry.outputs:execOut", "PublishOdometry.inputs:execIn"),
                ("ComputeOdometry.outputs:execOut", "PublishBaseTF.inputs:execIn"),
                ("ComputeOdometry.outputs:position", "PublishOdometry.inputs:position"),
                ("ComputeOdometry.outputs:orientation", "PublishOdometry.inputs:orientation"),
                ("ComputeOdometry.outputs:linearVelocity", "PublishOdometry.inputs:linearVelocity"),
                ("ComputeOdometry.outputs:angularVelocity", "PublishOdometry.inputs:angularVelocity"),
                ("ComputeOdometry.outputs:position", "PublishBaseTF.inputs:translation"),
                ("ComputeOdometry.outputs:orientation", "PublishBaseTF.inputs:rotation"),
                ("ReadSimTime.outputs:simulationTime", "PublishOdometry.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishBaseTF.inputs:timeStamp"),
            ],
            keys.SET_VALUES: [
                ("ComputeOdometry.inputs:chassisPrim", [usdrt.Sdf.Path(chassis_prim_path)]),
                ("PublishOdometry.inputs:topicName", topic_name),
                ("PublishOdometry.inputs:odomFrameId", odom_frame_id),
                ("PublishOdometry.inputs:chassisFrameId", chassis_frame_id),
                ("PublishOdometry.inputs:publishRawVelocities", False),
                ("PublishBaseTF.inputs:topicName", "tf"),
                ("PublishBaseTF.inputs:parentFrameId", odom_frame_id),
                ("PublishBaseTF.inputs:childFrameId", chassis_frame_id),
            ],
        },
    )
    og.Controller.evaluate_sync(odom_graph)
    print(f"ROS 2 odometry enabled: /{topic_name.lstrip('/')} ({odom_frame_id} -> {chassis_frame_id})")
    return {"graph": odom_graph}


class Anymal_runner(object):
    def __init__(
        self,
        physics_dt,
        render_dt,
        terrain_amplitude,
        terrain_size,
        terrain_resolution,
        terrain_grid_step,
        trail_width,
        fence_y,
        river_width,
        terrain_texture,
        terrain_normal_texture,
        terrain_roughness_texture,
        terrain_texture_scale,
        add_ground_detail,
        add_gp_props,
        add_external_props,
        guard_tower_asset,
        guard_tower_scale,
        fence_asset,
        fence_asset_scale,
        fence_asset_count,
        terrain_seed,
        enable_ros2_sensors,
        enable_ros2_lidar,
        enable_ros2_cmd_vel,
        enable_ros2_odom,
        cmd_vel_topic,
        cmd_vel_timeout,
        cmd_vel_linear_scale,
        cmd_vel_yaw_scale,
        lidar_mount,
        enable_intruder,
        intruder_count,
        intruder_speed,
        intruder_seed,
        intruder_visual,
        intruder_human_usd,
        intruder_yaw_deg,
    ) -> None:
        """
        Creates the simulation world and places ANYmal on a generated GP-style noise terrain.

        Argument:
        physics_dt {float} -- Physics downtime of the scene.
        render_dt {float} -- Render downtime of the scene.

        """
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
        self._stage = simulation_app.context.get_stage()

        _add_scene_lighting(self._stage)
        terrain_center_height, terrain_x_values, terrain_y_values, terrain_heights = _create_noise_terrain(
            self._stage,
            prim_path="/World/GP_NoiseTerrain",
            size=terrain_size,
            resolution=terrain_resolution,
            amplitude=terrain_amplitude,
            seed=terrain_seed,
            grid_step=terrain_grid_step,
            trail_width=trail_width,
            fence_y=fence_y,
            river_width=river_width,
            texture_path=terrain_texture,
            normal_texture_path=terrain_normal_texture,
            roughness_texture_path=terrain_roughness_texture,
            texture_scale=terrain_texture_scale,
            add_ground_detail=add_ground_detail,
            add_gp_props=add_gp_props,
            add_external_props=add_external_props,
            guard_tower_asset=guard_tower_asset,
            guard_tower_scale=guard_tower_scale,
            fence_asset=fence_asset,
            fence_asset_scale=fence_asset_scale,
            fence_asset_count=fence_asset_count,
        )
        self._terrain_size = float(terrain_size)
        self._fence_y = _get_fence_y(terrain_size, fence_y)
        self._terrain_x_values = terrain_x_values
        self._terrain_y_values = terrain_y_values
        self._terrain_heights = terrain_heights
        self._intruder_scenario = (
            IntruderScenario(
                self._stage,
                terrain_x_values,
                terrain_y_values,
                terrain_heights,
                terrain_size=terrain_size,
                fence_y=fence_y,
                river_width=river_width,
                count=intruder_count,
                speed=intruder_speed,
                seed=intruder_seed,
                visual_mode=intruder_visual,
                human_usd_path=intruder_human_usd,
                yaw_deg=intruder_yaw_deg,
            )
            if enable_intruder
            else None
        )
        self._sim_time = 0.0
        self._last_intruder_state_publish_time = -1.0
        self._intruder_state_publisher = (
            _OptionalRosStringPublisher("dmz_sentry_intruder_state_publisher", "/intruder_states")
            if self._intruder_scenario is not None
            and (enable_ros2_sensors or enable_ros2_cmd_vel or enable_ros2_odom)
            else None
        )

        self._anymal = AnymalFlatTerrainPolicy(
            prim_path="/World/Anymal",
            name="Anymal",
            position=np.array([0.0, 0.0, terrain_center_height + 0.7]),
        )
        simulation_app.update()
        self._ros_sensor_handles = (
            _setup_sentry_sensors(self._stage, lidar_mount=lidar_mount, enable_lidar=enable_ros2_lidar)
            if enable_ros2_sensors
            else {}
        )
        self._inspection_camera_command_subscriber = _FileStringMailbox(INSPECTION_COMMAND_FILE) if enable_ros2_sensors else None
        self._inspection_camera_ros_subscriber = (
            _OptionalRosStringSubscriber("dmz_sentry_inspection_camera_command", "/inspection_camera/command")
            if enable_ros2_sensors
            else None
        )
        self._inspection_camera_translate_op = None
        self._inspection_camera_orient_op = None
        self._inspection_camera_focal_attr = None
        self._inspection_camera_target = None
        self._inspection_camera_focal_length = INSPECTION_CAMERA_DEFAULT_FOCAL_LENGTH
        self._inspection_camera_offset = np.asarray(INSPECTION_CAMERA_LOCAL_OFFSET, dtype=float)
        self._inspection_camera_manual_pan = 0.0
        self._inspection_camera_manual_tilt = 0.0
        if self._ros_sensor_handles.get("inspection_camera"):
            self._setup_inspection_camera_xform()
        self._chassis_prim_path = self._ros_sensor_handles.get("mount_path") or _find_anymal_sensor_mount(self._stage)
        self._chassis_frame_id = self._chassis_prim_path.rsplit("/", 1)[-1] or "Anymal"
        print(f"ANYmal odometry chassis prim: {self._chassis_prim_path}")
        self._static_sensor_tf_handles = None
        if enable_ros2_sensors:
            self._static_sensor_tf_handles = _setup_ros2_static_sensor_tf_graph(self._chassis_frame_id)
        self._lidar_follow_enabled = False
        self._lidar_follow_translate_op = None
        self._lidar_follow_orient_op = None
        self._lidar_follow_offset = np.array([0.18, 0.0, 1.05])
        if self._lidar_follow_enabled:
            self._setup_lidar_follow_xform()

        self._base_command = np.zeros(3)
        self._keyboard_command = np.zeros(3)
        self._cmd_vel_timeout = float(cmd_vel_timeout)
        self._cmd_vel_linear_scale = float(cmd_vel_linear_scale)
        self._cmd_vel_yaw_scale = float(cmd_vel_yaw_scale)
        self._cmd_vel_handles = None
        self._cmd_vel_topic = cmd_vel_topic
        if enable_ros2_cmd_vel:
            self._cmd_vel_handles = _setup_ros2_cmd_vel_graph(cmd_vel_topic)
        self._odom_handles = None
        if enable_ros2_odom:
            self._odom_handles = _setup_ros2_odometry_graph(
                self._chassis_prim_path,
                chassis_frame_id=self._chassis_frame_id,
            )

        # bindings for keyboard to command
        self._input_keyboard_mapping = {
            # forward command
            "NUMPAD_8": [1.0, 0.0, 0.0],
            "UP": [1.0, 0.0, 0.0],
            # back command
            "NUMPAD_2": [-1.0, 0.0, 0.0],
            "DOWN": [-1.0, 0.0, 0.0],
            # left command
            "NUMPAD_6": [0.0, -1.0, 0.0],
            "RIGHT": [0.0, -1.0, 0.0],
            # right command
            "NUMPAD_4": [0.0, 1.0, 0.0],
            "LEFT": [0.0, 1.0, 0.0],
            # yaw command (positive)
            "NUMPAD_7": [0.0, 0.0, 1.0],
            "N": [0.0, 0.0, 1.0],
            # yaw command (negative)
            "NUMPAD_9": [0.0, 0.0, -1.0],
            "M": [0.0, 0.0, -1.0],
        }
        self.needs_reset = False
        self.first_step = True

    def _setup_lidar_follow_xform(self) -> None:
        lidar_path = self._ros_sensor_handles["lidar"]["lidar_path"]
        lidar_prim = self._stage.GetPrimAtPath(lidar_path)
        if not lidar_prim.IsValid():
            carb.log_warn(f"LiDAR follow prim not found at {lidar_path}")
            self._lidar_follow_enabled = False
            return

        self._lidar_follow_offset = np.asarray(self._ros_sensor_handles["lidar_local_offset"], dtype=float)
        xformable = UsdGeom.Xformable(lidar_prim)
        xformable.ClearXformOpOrder()
        self._lidar_follow_translate_op = xformable.AddTranslateOp()
        self._lidar_follow_orient_op = xformable.AddOrientOp()

    def _update_lidar_follow_pose(self) -> None:
        if not self._lidar_follow_enabled:
            return

        position, orientation = self._anymal.robot.get_world_pose()
        # Experimental mode: keep disabled by default because some RTX LiDAR pipelines
        # can crash when the sensor prim is moved every frame.
        quat = np.asarray(orientation, dtype=float)
        w, x, y, z = quat
        rotation = np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ]
        )
        lidar_position = np.asarray(position, dtype=float) + rotation @ self._lidar_follow_offset

        self._lidar_follow_translate_op.Set(Gf.Vec3d(*[float(value) for value in lidar_position]))
        self._lidar_follow_orient_op.Set(
            Gf.Quatd(
                float(orientation[0]),
                Gf.Vec3d(float(orientation[1]), float(orientation[2]), float(orientation[3])),
            )
        )

    def _setup_inspection_camera_xform(self) -> None:
        camera_path = self._ros_sensor_handles["inspection_camera"]["camera_path"]
        camera_prim = self._stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsValid():
            carb.log_warn(f"Inspection camera prim not found at {camera_path}")
            return

        xformable = UsdGeom.Xformable(camera_prim)
        xformable.ClearXformOpOrder()
        self._inspection_camera_translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        self._inspection_camera_orient_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        self._inspection_camera_focal_attr = UsdGeom.Camera(camera_prim).GetFocalLengthAttr()

    def _rotation_matrix_from_quat_wxyz(self, orientation) -> np.ndarray:
        quat = np.asarray(orientation, dtype=float)
        w, x, y, z = quat
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ]
        )

    def _handle_inspection_camera_command(self, data: str) -> None:
        try:
            payload = json.loads(data)
        except Exception:
            carb.log_warn(f"Invalid inspection camera command: {data}")
            return

        action = str(payload.get("action", "look_at")).strip().lower()
        if action in ("look_at", "track", "select"):
            try:
                x = float(payload["x"])
                y = float(payload["y"])
                z = float(payload.get("z", _sample_height(self._terrain_x_values, self._terrain_y_values, self._terrain_heights, x, y) + 1.3))
            except Exception as exc:
                carb.log_warn(f"Inspection camera look_at command missing target coordinates: {exc}")
                return
            self._inspection_camera_target = np.asarray((x, y, z), dtype=float)
            print(f"Inspection camera tracking target: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        elif action in ("clear", "reset_target", "wide"):
            self._inspection_camera_target = None
            print("Inspection camera target cleared")
        elif action == "pan_left":
            self._inspection_camera_target = None
            self._inspection_camera_manual_pan += np.deg2rad(INSPECTION_CAMERA_PAN_STEP_DEG)
        elif action == "pan_right":
            self._inspection_camera_target = None
            self._inspection_camera_manual_pan -= np.deg2rad(INSPECTION_CAMERA_PAN_STEP_DEG)
        elif action == "tilt_up":
            self._inspection_camera_target = None
            self._inspection_camera_manual_tilt += np.deg2rad(INSPECTION_CAMERA_TILT_STEP_DEG)
            self._inspection_camera_manual_tilt = float(
                np.clip(
                    self._inspection_camera_manual_tilt,
                    -np.deg2rad(INSPECTION_CAMERA_MAX_TILT_DEG),
                    np.deg2rad(INSPECTION_CAMERA_MAX_TILT_DEG),
                )
            )
        elif action == "tilt_down":
            self._inspection_camera_target = None
            self._inspection_camera_manual_tilt -= np.deg2rad(INSPECTION_CAMERA_TILT_STEP_DEG)
            self._inspection_camera_manual_tilt = float(
                np.clip(
                    self._inspection_camera_manual_tilt,
                    -np.deg2rad(INSPECTION_CAMERA_MAX_TILT_DEG),
                    np.deg2rad(INSPECTION_CAMERA_MAX_TILT_DEG),
                )
            )
        elif action in ("center", "manual_center"):
            self._inspection_camera_target = None
            self._inspection_camera_manual_pan = 0.0
            self._inspection_camera_manual_tilt = 0.0
        elif action == "zoom_in":
            self._inspection_camera_focal_length = min(
                INSPECTION_CAMERA_MAX_FOCAL_LENGTH,
                self._inspection_camera_focal_length * 1.35,
            )
        elif action == "zoom_out":
            self._inspection_camera_focal_length = max(
                INSPECTION_CAMERA_MIN_FOCAL_LENGTH,
                self._inspection_camera_focal_length / 1.35,
            )
        elif action == "zoom_reset":
            self._inspection_camera_focal_length = INSPECTION_CAMERA_DEFAULT_FOCAL_LENGTH
        elif action == "set_zoom":
            self._inspection_camera_focal_length = float(
                np.clip(
                    float(payload.get("focal_length", self._inspection_camera_focal_length)),
                    INSPECTION_CAMERA_MIN_FOCAL_LENGTH,
                    INSPECTION_CAMERA_MAX_FOCAL_LENGTH,
                )
            )
        else:
            carb.log_warn(f"Unknown inspection camera command: {action}")

    def _poll_inspection_camera_commands(self) -> None:
        command_sources = (self._inspection_camera_command_subscriber, self._inspection_camera_ros_subscriber)
        for source in command_sources:
            if source is None:
                continue
            for message in source.poll():
                self._handle_inspection_camera_command(message)

    def _update_inspection_camera_pose(self) -> None:
        if self._inspection_camera_translate_op is None or self._inspection_camera_orient_op is None:
            return

        position, orientation = self._anymal.robot.get_world_pose()
        robot_position = np.asarray(position, dtype=float)
        robot_rotation = self._rotation_matrix_from_quat_wxyz(orientation)
        camera_position = robot_position + robot_rotation @ self._inspection_camera_offset

        if self._inspection_camera_target is None:
            pan = self._inspection_camera_manual_pan
            tilt = self._inspection_camera_manual_tilt
            forward_local = np.asarray(
                (
                    np.cos(tilt) * np.cos(pan),
                    np.cos(tilt) * np.sin(pan),
                    np.sin(tilt),
                ),
                dtype=float,
            )
            target_position = camera_position + robot_rotation @ (9.0 * forward_local)
        else:
            target_position = np.asarray(self._inspection_camera_target, dtype=float)

        w, x, y, z = _camera_look_at_quat(camera_position, target_position)
        self._inspection_camera_translate_op.Set(Gf.Vec3d(*[float(value) for value in camera_position]))
        self._inspection_camera_orient_op.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        if self._inspection_camera_focal_attr is not None:
            self._inspection_camera_focal_attr.Set(float(self._inspection_camera_focal_length))

    def _read_ros_cmd_vel_command(self) -> np.ndarray | None:
        if self._cmd_vel_handles is None:
            return None

        linear = np.asarray(og.Controller.get(og.Controller.attribute(self._cmd_vel_handles["linear_attr"])), dtype=float)
        angular = np.asarray(og.Controller.get(og.Controller.attribute(self._cmd_vel_handles["angular_attr"])), dtype=float)
        if linear.shape[0] < 2 or angular.shape[0] < 3:
            return None

        return np.array(
            [
                np.clip(linear[0] * self._cmd_vel_linear_scale, -1.0, 1.0),
                np.clip(linear[1] * self._cmd_vel_linear_scale, -1.0, 1.0),
                np.clip(angular[2] * self._cmd_vel_yaw_scale, -1.0, 1.0),
            ],
            dtype=float,
        )

    def _get_active_base_command(self) -> np.ndarray:
        ros_command = self._read_ros_cmd_vel_command()
        if ros_command is not None and np.linalg.norm(ros_command) > 1e-6:
            return ros_command
        return self._keyboard_command

    def shutdown(self) -> None:
        if self._inspection_camera_command_subscriber is not None:
            self._inspection_camera_command_subscriber.shutdown()
        if self._inspection_camera_ros_subscriber is not None:
            self._inspection_camera_ros_subscriber.shutdown()
        if self._intruder_state_publisher is not None:
            self._intruder_state_publisher.shutdown()

    def _publish_intruder_states(self, force: bool = False) -> None:
        if self._intruder_scenario is None:
            return
        if not force and self._sim_time - self._last_intruder_state_publish_time < 0.2:
            return

        payload = {
            "stamp": self._sim_time,
            "frame_id": "world",
            "intruders": self._intruder_scenario.get_intruder_reports(),
        }
        try:
            INTRUDER_STATES_FILE.write_text(json.dumps(payload), encoding="utf-8")
        except Exception as exc:
            carb.log_warn(f"Failed to write intruder state file {INTRUDER_STATES_FILE}: {exc}")
        if self._intruder_state_publisher is not None:
            self._intruder_state_publisher.publish(json.dumps(payload))
        self._last_intruder_state_publish_time = self._sim_time

    def setup(self) -> None:
        """
        Set up keyboard listener and add physics callback

        """
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        self._world.add_physics_callback("anymal_forward", callback_fn=self.on_physics_step)

    def on_physics_step(self, step_size) -> None:
        """
        Physics call back, initialize robot (first frame) and call controller forward function to compute and apply joint torque

        """
        if self.first_step:
            self._anymal.initialize()
            self._poll_inspection_camera_commands()
            if self._intruder_scenario is not None:
                self._intruder_scenario.update(0.0)
                self._publish_intruder_states(force=True)
            self._update_lidar_follow_pose()
            self._update_inspection_camera_pose()
            self.first_step = False
        elif self.needs_reset:
            self._world.reset(True)
            self.needs_reset = False
            self.first_step = True
        else:
            self._poll_inspection_camera_commands()
            self._base_command = self._get_active_base_command()
            self._anymal.forward(step_size, self._base_command)
            if self._intruder_scenario is not None:
                self._intruder_scenario.update(step_size)
                self._sim_time += float(step_size)
                self._publish_intruder_states()
            self._update_lidar_follow_pose()
            self._update_inspection_camera_pose()

    def generate_replicator_dataset(
        self,
        output_dir: str,
        frame_count: int,
        image_width: int,
        image_height: int,
        seed: int,
        profile: str = "fence",
    ) -> None:
        if self._intruder_scenario is None:
            carb.log_warn("Replicator dataset requested, but intruder scenario is disabled.")
            return

        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        profile = profile if profile in ("fence", "clear", "hard", "mixed", "calibration") else "fence"

        camera = rep.create.camera()
        render_product = rep.create.render_product(camera, resolution=(int(image_width), int(image_height)))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir=str(output_path),
            rgb=True,
            bounding_box_2d_tight=True,
        )
        writer.attach([render_product])

        frame_count = max(1, int(frame_count))
        if profile == "calibration":
            clear_count = int(round(frame_count * 0.40))
            fence_count = int(round(frame_count * 0.40))
            hard_count = max(0, frame_count - clear_count - fence_count)
            frame_profiles = ["clear"] * clear_count + ["fence"] * fence_count + ["hard"] * hard_count
            rng.shuffle(frame_profiles)
        elif profile == "mixed":
            frame_profiles = list(rng.choice(("clear", "fence", "hard"), size=frame_count, p=(0.40, 0.40, 0.20)))
        else:
            frame_profiles = [profile] * frame_count

        profile_counts = {name: frame_profiles.count(name) for name in ("clear", "fence", "hard")}
        print(
            "Replicator person preview dataset: "
            f"frames={frame_count}, profile={profile}, "
            f"clear={profile_counts['clear']}, fence={profile_counts['fence']}, hard={profile_counts['hard']}, "
            f"output={output_path}, resolution={int(image_width)}x{int(image_height)}"
        )

        for frame_idx in range(frame_count):
            frame_profile = frame_profiles[frame_idx]
            intruder_positions = self._intruder_scenario.randomize_for_dataset()
            if intruder_positions:
                target_xy = intruder_positions[int(rng.integers(0, len(intruder_positions)))]
                target = (
                    float(target_xy[0]),
                    float(target_xy[1]),
                    float(_sample_height(self._terrain_x_values, self._terrain_y_values, self._terrain_heights, target_xy[0], target_xy[1]) + 1.0),
                )
            else:
                target = (0.0, self._fence_y + 2.0, 1.0)

            if frame_profile == "clear":
                camera_x = target[0] + rng.uniform(-6.0, 6.0)
                camera_y = self._fence_y + rng.uniform(3.0, 8.0)
                camera_z = rng.uniform(0.75, 1.35)
                look_at_jitter = (0.85, 0.55, 0.30)
            elif frame_profile == "hard":
                camera_x = target[0] + rng.uniform(-18.0, 18.0)
                camera_y = self._fence_y - rng.uniform(12.0, 24.0)
                camera_z = rng.uniform(0.52, 1.42)
                look_at_jitter = (2.2, 1.4, 0.55)
            else:
                camera_x = target[0] + rng.uniform(-10.0, 10.0)
                camera_y = self._fence_y - rng.uniform(5.0, 14.0)
                camera_z = rng.uniform(0.62, 1.18)
                look_at_jitter = (1.5, 0.8, 0.45)

            camera_x = float(np.clip(camera_x, -self._terrain_size * 0.38, self._terrain_size * 0.38))
            camera_y = float(np.clip(camera_y, -self._terrain_size * 0.46, self._terrain_size * 0.46))
            camera_z = float(camera_z)
            look_at = (
                target[0] + float(rng.uniform(-look_at_jitter[0], look_at_jitter[0])),
                target[1] + float(rng.uniform(-look_at_jitter[1], look_at_jitter[1])),
                target[2] + float(rng.uniform(-0.25, look_at_jitter[2])),
            )

            with camera:
                rep.modify.pose(position=(camera_x, camera_y, camera_z), look_at=look_at)

            simulation_app.update()
            try:
                rep.orchestrator.step(rt_subframes=4)
            except TypeError:
                rep.orchestrator.step()
            simulation_app.update()

            if (frame_idx + 1) % 10 == 0 or frame_idx == frame_count - 1:
                print(f"Captured Replicator frame {frame_idx + 1}/{frame_count} ({frame_profile})")

        try:
            rep.orchestrator.wait_until_complete()
        except Exception:
            pass
        print(f"Replicator preview dataset written to: {output_path}")

    def run(self) -> None:
        """
        Step simulation based on rendering downtime

        """
        # change to sim running
        while simulation_app.is_running():
            self._world.step(render=True)
            if self._world.is_stopped():
                self.needs_reset = True
        return

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        """
        Keyboard subscriber callback to when kit is updated.

        """

        # when a key is pressed for released  the command is adjusted w.r.t the key-mapping
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # on pressing, the command is incremented
            if event.input.name in self._input_keyboard_mapping:
                self._keyboard_command += np.array(self._input_keyboard_mapping[event.input.name])

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # on release, the command is decremented
            if event.input.name in self._input_keyboard_mapping:
                self._keyboard_command -= np.array(self._input_keyboard_mapping[event.input.name])
        return True


def main():
    """
    Parse arguments and instantiate the ANYmal runner

    """
    physics_dt = 1 / 200.0
    render_dt = 1 / 60.0

    print(
        "DMZ Sentry GP terrain: "
        f"amplitude={args.terrain_amplitude:.3f}m, "
        f"size={args.terrain_size:.1f}m, "
        f"resolution={args.terrain_resolution:.2f}m, "
        f"grid_step={args.terrain_grid_step:.2f}m, "
        f"trail_width={args.trail_width:.1f}m, "
        f"fence_y={args.fence_y:.1f}m, "
        f"river_width={args.river_width:.1f}m, "
        f"texture_scale={args.terrain_texture_scale:.2f}, "
        f"ground_detail={not args.no_ground_detail}, "
        f"gp_props={not args.no_gp_props}, "
        f"external_props={not args.no_external_props}, "
        f"ros2_sensors={not args.no_ros2_sensors}, "
        f"ros2_lidar={not args.no_ros2_lidar}, "
        f"ros2_cmd_vel={not args.no_ros2_cmd_vel}, "
        f"ros2_odom={not args.no_ros2_odom}, "
        f"cmd_vel_topic=/{args.cmd_vel_topic.lstrip('/')}, "
        f"lidar_mount={args.lidar_mount}, "
        f"intruder={not args.no_intruder}, "
        f"intruder_count={args.intruder_count}, "
        f"intruder_speed={args.intruder_speed:.2f}m/s, "
        f"intruder_visual={args.intruder_visual}, "
        f"intruder_yaw={args.intruder_yaw_deg:.1f}deg, "
        f"replicator_dataset={args.replicator_dataset}, "
        f"seed={args.terrain_seed}"
    )
    if args.terrain_texture:
        print(f"Terrain texture: {args.terrain_texture}")
    if args.terrain_normal_texture:
        print(f"Terrain normal texture: {args.terrain_normal_texture}")
    if args.terrain_roughness_texture:
        print(f"Terrain roughness texture: {args.terrain_roughness_texture}")
    if not args.no_external_props:
        print(f"Guard tower asset: {args.guard_tower_asset} scale={args.guard_tower_scale:.3f}")
        print(
            f"Fence asset: {args.fence_asset} "
            f"scale={args.fence_asset_scale:.3f}, count={args.fence_asset_count}"
        )
    print("Control: ROS 2 /cmd_vel has priority; keyboard arrows/N/M remain as fallback.")

    runner = Anymal_runner(
        physics_dt=physics_dt,
        render_dt=render_dt,
        terrain_amplitude=args.terrain_amplitude,
        terrain_size=args.terrain_size,
        terrain_resolution=args.terrain_resolution,
        terrain_grid_step=args.terrain_grid_step,
        trail_width=args.trail_width,
        fence_y=args.fence_y,
        river_width=args.river_width,
        terrain_texture=args.terrain_texture,
        terrain_normal_texture=args.terrain_normal_texture,
        terrain_roughness_texture=args.terrain_roughness_texture,
        terrain_texture_scale=args.terrain_texture_scale,
        add_ground_detail=not args.no_ground_detail,
        add_gp_props=not args.no_gp_props,
        add_external_props=not args.no_external_props,
        guard_tower_asset=args.guard_tower_asset,
        guard_tower_scale=args.guard_tower_scale,
        fence_asset=args.fence_asset,
        fence_asset_scale=args.fence_asset_scale,
        fence_asset_count=args.fence_asset_count,
        terrain_seed=args.terrain_seed,
        enable_ros2_sensors=not args.no_ros2_sensors,
        enable_ros2_lidar=not args.no_ros2_lidar,
        enable_ros2_cmd_vel=not args.no_ros2_cmd_vel,
        enable_ros2_odom=not args.no_ros2_odom,
        cmd_vel_topic=args.cmd_vel_topic,
        cmd_vel_timeout=args.cmd_vel_timeout,
        cmd_vel_linear_scale=args.cmd_vel_linear_scale,
        cmd_vel_yaw_scale=args.cmd_vel_yaw_scale,
        lidar_mount=args.lidar_mount,
        enable_intruder=not args.no_intruder,
        intruder_count=args.intruder_count,
        intruder_speed=args.intruder_speed,
        intruder_seed=args.intruder_seed,
        intruder_visual=args.intruder_visual,
        intruder_human_usd=args.intruder_human_usd,
        intruder_yaw_deg=args.intruder_yaw_deg,
    )
    simulation_app.update()
    runner._world.reset()
    simulation_app.update()
    try:
        if args.replicator_dataset:
            runner.generate_replicator_dataset(
                output_dir=args.dataset_output_dir,
                frame_count=args.dataset_frames,
                image_width=args.dataset_width,
                image_height=args.dataset_height,
                seed=args.dataset_seed,
                profile=args.dataset_profile,
            )
        else:
            runner.setup()
            simulation_app.update()
            runner.run()
    finally:
        runner.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
