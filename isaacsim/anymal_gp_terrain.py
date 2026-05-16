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
import carb
import numpy as np
import omni
import omni.appwindow  # Contains handle to keyboard
import omni.graph.core as og
import omni.kit.commands
import omni.replicator.core as rep
import usdrt.Sdf
from isaacsim.core.api import World
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.semantics import add_labels
from isaacsim.robot.policy.examples.robots import AnymalFlatTerrainPolicy
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


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
    default=4.0,
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
    help="Optional PNG/JPG satellite or orthophoto texture to map over the terrain.",
)
parser.add_argument("--no-ros2-sensors", action="store_true", help="Disable ROS 2 camera, LiDAR, TF, and clock publishers.")
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
parser.add_argument("--intruder-count", type=int, default=1, help="Number of moving intruder targets to spawn.")
parser.add_argument("--intruder-speed", type=float, default=0.45, help="Intruder approach speed in meters per second.")
parser.add_argument("--intruder-seed", type=int, default=31, help="Seed for deterministic intruder spawn positions.")
parser.add_argument("--no-gp-props", action="store_true", help="Disable lightweight GP visual props.")
parser.add_argument("--terrain-seed", type=int, default=7, help="Seed for deterministic terrain noise.")
args, unknown = parser.parse_known_args()

if not args.no_ros2_sensors or not args.no_ros2_cmd_vel or not args.no_ros2_odom:
    enable_extension("isaacsim.ros2.bridge")
    enable_extension("isaacsim.sensors.rtx")
    simulation_app.update()

FRONT_CAMERA_NAME = "SentryFrontCamera"
LIDAR_NAME = "SentryLidar"
FRONT_CAMERA_TRANSLATION = (0.95, 0.0, 0.64)
# USD camera local -Z looks forward. This quaternion points it along robot +X
# while keeping robot +Z as image up, so the viewport is not rolled sideways.
FRONT_CAMERA_ORIENTATION_IJKR = (0.5, -0.5, -0.5, 0.5)


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


def _add_terrain_uvs(mesh: UsdGeom.Mesh, x_values: np.ndarray, y_values: np.ndarray) -> None:
    x_min = float(x_values[0])
    x_range = float(x_values[-1] - x_values[0])
    y_min = float(y_values[0])
    y_range = float(y_values[-1] - y_values[0])
    texcoords = [
        Gf.Vec2f(float((x - x_min) / x_range), float((y - y_min) / y_range)) for y in y_values for x in x_values
    ]
    primvars_api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    st = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(texcoords)


def _bind_terrain_texture(stage, terrain_prim, texture_path: str) -> None:
    if not texture_path:
        return

    material = UsdShade.Material.Define(stage, Sdf.Path("/World/Looks/GP_TerrainSatelliteMaterial"))
    shader = UsdShade.Shader.Define(stage, Sdf.Path("/World/Looks/GP_TerrainSatelliteMaterial/PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.92)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    st_reader = UsdShade.Shader.Define(stage, Sdf.Path("/World/Looks/GP_TerrainSatelliteMaterial/StReader"))
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    texture = UsdShade.Shader.Define(stage, Sdf.Path("/World/Looks/GP_TerrainSatelliteMaterial/SatelliteTexture"))
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_path))
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
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


def _create_intruder_visual(stage, path: str) -> dict:
    root = UsdGeom.Xform.Define(stage, Sdf.Path(path))
    root_prim = root.GetPrim()
    _apply_class_label(root_prim, "person")

    clothing = (0.10, 0.17, 0.12)
    vest = (0.18, 0.21, 0.15)
    skin = (0.48, 0.35, 0.26)
    dark = (0.04, 0.045, 0.04)
    alert = (0.75, 0.10, 0.08)

    _add_visual_cube(stage, f"{path}/Torso", (0.0, 0.0, 0.95), (0.38, 0.22, 0.62), clothing, "person")
    _add_visual_cube(stage, f"{path}/Vest", (0.0, -0.015, 1.03), (0.42, 0.08, 0.42), vest, "person")
    _add_visual_sphere(stage, f"{path}/Head", (0.0, 0.0, 1.45), 0.16, skin, "person")
    _add_visual_sphere(stage, f"{path}/Helmet", (0.0, 0.0, 1.57), 0.17, dark, "person")
    _add_visual_cube(stage, f"{path}/LeftArm", (-0.31, 0.0, 0.96), (0.11, 0.12, 0.55), clothing, "person", rotate_xyz=(0.0, 0.0, -8.0))
    _add_visual_cube(stage, f"{path}/RightArm", (0.31, 0.0, 0.96), (0.11, 0.12, 0.55), clothing, "person", rotate_xyz=(0.0, 0.0, 8.0))
    _add_visual_cube(stage, f"{path}/LeftLeg", (-0.12, 0.0, 0.36), (0.13, 0.13, 0.68), dark, "person")
    _add_visual_cube(stage, f"{path}/RightLeg", (0.12, 0.0, 0.36), (0.13, 0.13, 0.68), dark, "person")
    _add_visual_cube(stage, f"{path}/DetectionMarker", (0.0, 0.0, 1.88), (0.55, 0.05, 0.10), alert, "person")

    xformable = UsdGeom.Xformable(root_prim)
    translate_op = xformable.AddTranslateOp()
    return {"prim": root_prim, "translate_op": translate_op}


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

        stage.DefinePrim("/World/Intruders", "Xform")
        for index in range(max(0, int(count))):
            path = f"/World/Intruders/Intruder_{index}"
            handles = _create_intruder_visual(stage, path)
            state = {
                "path": path,
                "translate_op": handles["translate_op"],
                "phase": float(self._rng.uniform(0.0, np.pi * 2.0)),
                "time": 0.0,
            }
            self._respawn_intruder(state)
            self._intruders.append(state)

        if self._intruders:
            print(
                f"Intruder scenario enabled: count={len(self._intruders)}, "
                f"speed={self._speed:.2f} m/s, label=person"
            )

    def _respawn_intruder(self, state: dict) -> None:
        x_limit = self._terrain_size * 0.30
        start_y_min = min(self._y_values[-1] - 2.0, self._fence_y + max(5.0, self._river_width * 0.38))
        start_y_max = min(self._y_values[-1] - 1.0, self._fence_y + max(8.0, self._river_width * 0.72))
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

    service_road_lines = []
    for road_offset in (-4.1, -5.7):
        line = []
        road_y = fence_y + road_offset
        for x in np.linspace(x_min, x_max, 90):
            z = _sample_height(x_values, y_values, heights, x, road_y)
            line.append((float(x), float(road_y), z + 0.025))
        service_road_lines.append(line)
    _add_curve_lines(stage, "/World/GP_Props/FenceServiceRoadEdges", service_road_lines, width=0.12, color=(0.22, 0.18, 0.11))

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

    for sign_idx, sign_x in enumerate(np.linspace(-terrain_size * 0.38, terrain_size * 0.38, 7)):
        sign_y = fence_y - 1.0
        sign_z = _sample_height(x_values, y_values, heights, sign_x, sign_y)
        _add_cylinder(stage, f"/World/GP_Props/WarningSign_{sign_idx}_Post", (sign_x, sign_y, sign_z + 0.65), 0.03, 1.3, (0.08, 0.08, 0.07))
        _add_cube(stage, f"/World/GP_Props/WarningSign_{sign_idx}_Plate", (sign_x, sign_y, sign_z + 1.23), (0.75, 0.05, 0.45), (0.85, 0.72, 0.18))


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
    add_gp_props: bool,
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
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.22, 0.29, 0.18)])
    _add_terrain_uvs(mesh, x_values, y_values)
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

    _bind_terrain_texture(stage, terrain_prim, texture_path)
    _add_terrain_grid(stage, x_values, y_values, heights, grid_step)
    _add_river(stage, x_values, y_values, heights, size, fence_y, river_width)
    if add_gp_props:
        _add_gp_props(stage, x_values, y_values, heights, size, fence_y, river_width)

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


def _setup_ros2_camera_graph(camera_path: str, graph_path: str = "/ROS2_Sentry_Camera"):
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
                ("createViewport.inputs:name", "ROS2_Sentry_CameraViewport"),
                ("createViewport.inputs:viewportId", 1),
                ("cameraHelperRgb.inputs:frameId", "SentryFrontCamera"),
                ("cameraHelperRgb.inputs:topicName", "camera/image_raw"),
                ("cameraHelperRgb.inputs:type", "rgb"),
                ("cameraHelperInfo.inputs:frameId", "SentryFrontCamera"),
                ("cameraHelperInfo.inputs:topicName", "camera/camera_info"),
                ("cameraHelperDepth.inputs:frameId", "SentryFrontCamera"),
                ("cameraHelperDepth.inputs:topicName", "camera/depth"),
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


def _setup_sentry_sensors(stage, robot_path: str = "/World/Anymal", lidar_mount: str = "robot") -> dict:
    mount_path = _find_anymal_sensor_mount(stage, robot_path)
    camera_path = _make_child_path(mount_path, FRONT_CAMERA_NAME)
    _add_sensor_visual_markers(stage, mount_path)
    camera_path = _create_front_camera(stage, camera_path)
    camera_graph = _setup_ros2_camera_graph(camera_path)

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

    lidar_handles = _create_lidar_and_ros2_writer(lidar_path, translation=lidar_translation)
    tf_clock_targets = [lidar_path] if lidar_mount == "world" else []
    tf_clock_graph = _setup_ros2_tf_clock_graph(tf_clock_targets)
    simulation_app.update()
    print(
        "ROS 2 sensors enabled: /camera/image_raw, /camera/depth, "
        "/camera/camera_info, /lidar/points, /tf, /clock"
    )
    print(f"Sentry sensors mounted on: {mount_path}")
    return {
        "mount_path": mount_path,
        "camera_path": camera_path,
        "camera_graph": camera_graph,
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
        add_gp_props,
        terrain_seed,
        enable_ros2_sensors,
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
            add_gp_props=add_gp_props,
        )
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
            )
            if enable_intruder
            else None
        )

        self._anymal = AnymalFlatTerrainPolicy(
            prim_path="/World/Anymal",
            name="Anymal",
            position=np.array([0.0, 0.0, terrain_center_height + 0.7]),
        )
        simulation_app.update()
        self._ros_sensor_handles = (
            _setup_sentry_sensors(self._stage, lidar_mount=lidar_mount) if enable_ros2_sensors else {}
        )
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
        return

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
            if self._intruder_scenario is not None:
                self._intruder_scenario.update(0.0)
            self._update_lidar_follow_pose()
            self.first_step = False
        elif self.needs_reset:
            self._world.reset(True)
            self.needs_reset = False
            self.first_step = True
        else:
            self._base_command = self._get_active_base_command()
            self._anymal.forward(step_size, self._base_command)
            if self._intruder_scenario is not None:
                self._intruder_scenario.update(step_size)
            self._update_lidar_follow_pose()

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
        f"gp_props={not args.no_gp_props}, "
        f"ros2_sensors={not args.no_ros2_sensors}, "
        f"ros2_cmd_vel={not args.no_ros2_cmd_vel}, "
        f"ros2_odom={not args.no_ros2_odom}, "
        f"cmd_vel_topic=/{args.cmd_vel_topic.lstrip('/')}, "
        f"lidar_mount={args.lidar_mount}, "
        f"intruder={not args.no_intruder}, "
        f"intruder_count={args.intruder_count}, "
        f"intruder_speed={args.intruder_speed:.2f}m/s, "
        f"seed={args.terrain_seed}"
    )
    if args.terrain_texture:
        print(f"Terrain texture: {args.terrain_texture}")
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
        add_gp_props=not args.no_gp_props,
        terrain_seed=args.terrain_seed,
        enable_ros2_sensors=not args.no_ros2_sensors,
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
    )
    simulation_app.update()
    runner._world.reset()
    simulation_app.update()
    runner.setup()
    simulation_app.update()
    try:
        runner.run()
    finally:
        runner.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
