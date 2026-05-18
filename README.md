# DMZ Sentry

Isaac Sim + ROS 2 Humble based autonomous reconnaissance quadruped simulation.

## Current Status

- Isaac Sim 5.1 standalone scene
- ANYmal C flat-terrain policy teleoperation
- GP-style terrain with fence, river, bunkers, watchtowers, and warning signs
- Denser border visuals: double fence, wire mesh, concertina wire, patrol road, lights, river markers, and reeds
- Moving intruder scenario for camera-based detection experiments
- RGB-D camera, RTX LiDAR, odometry, TF, and clock ROS 2 publishing

## Run

Open a terminal:

```bash
cd /home/rokey/dev_ws/dmz_sentry
./scripts/run_anymal_gp.sh
```

Useful options:

```bash
./scripts/run_anymal_gp.sh --no-ros2-sensors
./scripts/run_anymal_gp.sh --no-ros2-cmd-vel
./scripts/run_anymal_gp.sh --no-ros2-odom
./scripts/run_anymal_gp.sh --no-intruder
./scripts/run_anymal_gp.sh --intruder-count 3 --intruder-speed 0.65
./scripts/run_anymal_gp.sh --intruder-count 3 --intruder-speed 0.65 --intruder-visual isaac-human --intruder-yaw-deg 90
./scripts/run_anymal_gp.sh --terrain-amplitude 0.20
./scripts/run_anymal_gp.sh --terrain-seed 11
./scripts/run_anymal_gp.sh --terrain-texture /path/to/orthophoto.png --terrain-texture-scale 1 --no-ground-detail
./scripts/run_anymal_gp.sh --terrain-texture /path/to/ground_albedo.jpg --terrain-normal-texture /path/to/ground_normal.jpg --terrain-roughness-texture /path/to/ground_roughness.jpg --terrain-texture-scale 12 --no-ground-detail
```

Keyboard control:

- `UP` / `DOWN`: forward / backward
- `LEFT` / `RIGHT`: strafe
- `N` / `M`: yaw left / right

## Demo Quick Start

Use these wrappers for the current stable DMZ Sentry demo. They keep the long terrain, YOLO, GPU, and rqt-friendly image settings in one place.

Terminal 1: start Isaac Sim:

```bash
cd /home/rokey/dev_ws/dmz_sentry
./scripts/demo_dmz_sim.sh
```

Terminal 2: start YOLO person detection:

```bash
cd /home/rokey/dev_ws/dmz_sentry
./scripts/demo_yolo_detector.sh
```

Optional rqt view:

```bash
rqt_image_view /camera/annotated
```

The detector wrapper uses the calibrated model on GPU `0`, `confidence=0.25`, `image_size=320`, `publish_annotated=true`, `annotated_scale=0.5`, and `every_n=1`. The half-size annotated image keeps rqt responsive while the detector still runs on the resized YOLO input.

Useful overrides:

```bash
./scripts/demo_dmz_sim.sh --intruder-count 5 --intruder-speed 0.65
YOLO_PUBLISH_ANNOTATED=false ./scripts/demo_yolo_detector.sh
YOLO_CONFIDENCE=0.35 ./scripts/demo_yolo_detector.sh
YOLO_ANNOTATED_SCALE=1.0 ./scripts/demo_yolo_detector.sh
```

## Tactical Map and Patrol

The tactical map is a lightweight web view for the DMZ simulation coordinate frame. It draws the fence line, river band, safe patrol lane, towers, robot marker, current patrol waypoint, simulated intruder positions, and YOLO alert state. The web buttons publish high-level mission commands only; the ROS 2 patrol controller owns `/cmd_vel`.

Build the ROS 2 workspace after pulling or changing control nodes:

```bash
cd /home/rokey/dev_ws/dmz_sentry/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

Install rosbridge if it is not already available:

```bash
sudo apt install ros-humble-rosbridge-suite
```

Run the full tactical demo:

```bash
# Terminal 1: Isaac Sim
cd /home/rokey/dev_ws/dmz_sentry
./scripts/demo_dmz_sim.sh

# Terminal 2: YOLO detector
cd /home/rokey/dev_ws/dmz_sentry
./scripts/demo_yolo_detector.sh

# Terminal 3: waypoint patrol controller
cd /home/rokey/dev_ws/dmz_sentry
./scripts/demo_patrol_controller.sh

# Terminal 4: ROS bridge websocket
cd /home/rokey/dev_ws/dmz_sentry
./scripts/demo_rosbridge.sh

# Terminal 5: static tactical web map
cd /home/rokey/dev_ws/dmz_sentry
./scripts/demo_tactical_map.sh
```

Open:

```text
http://localhost:8080
```

The web controls publish:

- `start_patrol` on `/mission_command`: patrol on the clear straight lane between Tower W and Tower E
- `stop` on `/mission_command`: stop and hold position
- `resume` on `/mission_command`: resume the previous patrol mode

The patrol controller uses a conservative two-step motion style: turn in place toward the current waypoint first, then walk straight with yaw held still. This avoids the tip-over behavior seen when ANYmal receives forward and turning commands at the same time.

Patrol topics:

- `/mission_command` (`std_msgs/String`): high-level command from web or terminal
- `/patrol_state` (`std_msgs/String` JSON): controller state for the web UI
- `/cmd_vel` (`geometry_msgs/Twist`): velocity command sent to ANYmal
- `/alerts` (`std_msgs/String` JSON): YOLO person alert, used to stop patrol temporarily
- `/intruder_states` (`std_msgs/String` JSON): simulator ground-truth intruder positions for the tactical map

Manual command test without the web UI:

```bash
source /opt/ros/humble/setup.bash
source /home/rokey/dev_ws/dmz_sentry/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=129
ros2 topic pub --once /mission_command std_msgs/msg/String "{data: start_patrol}"
ros2 topic echo /patrol_state
```

## ROS 2 Teleoperation

Start Isaac Sim:

```bash
cd /home/rokey/dev_ws/dmz_sentry
./scripts/run_anymal_gp.sh
```

In another terminal:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=129
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p repeat_rate:=20.0
```

Nonzero `/cmd_vel` has priority over the in-window keyboard fallback. Keep `repeat_rate` enabled so teleop publishes zero commands when you release the keys.

## ROS 2 Check

In another terminal:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=129
ros2 topic list
ros2 topic hz /camera/image_raw
ros2 topic hz /lidar/points
ros2 topic echo /odom --once
```

Expected topics:

- `/camera/image_raw`
- `/camera/depth`
- `/camera/camera_info`
- `/lidar/points`
- `/odom`
- `/tf`
- `/clock`

For RViz2, set `Fixed Frame` to `world`, then add `Image`, `PointCloud2`, and `Odometry` displays. The LiDAR is currently in stable world-mounted debug mode; robot-following LiDAR was left off because it caused Isaac Sim RTX LiDAR crashes on this setup.

Current TF shape:

```text
world
  base
    SentryFrontCamera
  SentryLidar
```

`SentryLidar` stays under `world` while the stable debug LiDAR is used.

## Intruder Scenario

By default, one simple person-shaped `Intruder_0` target spawns near the river side of the fence and walks toward the fence line. It is visually primitive on purpose so the scenario stays stable before replacing it with an animated human USD.

Useful controls:

```bash
./scripts/run_anymal_gp.sh --intruder-count 1
./scripts/run_anymal_gp.sh --intruder-count 3 --intruder-speed 0.65
./scripts/run_anymal_gp.sh --intruder-visual isaac-human
./scripts/run_anymal_gp.sh --no-intruder
```

The default `--intruder-visual auto` mode tries to load an Isaac Sim human character from the configured Isaac asset root, then falls back to the primitive target if the asset is unavailable. The intruder prims are labeled as semantic class `person`, which leaves a path for later Replicator synthetic-data or bounding-box experiments.

If the default asset root is not configured, pass an explicit human USD:

```bash
./scripts/run_anymal_gp.sh --intruder-visual isaac-human --intruder-human-usd /path/or/url/to/human.usd
```

If the referenced human faces the wrong way, rotate it:

```bash
./scripts/run_anymal_gp.sh --intruder-visual isaac-human --intruder-yaw-deg 90
./scripts/run_anymal_gp.sh --intruder-visual isaac-human --intruder-yaw-deg -90
./scripts/run_anymal_gp.sh --intruder-visual isaac-human --intruder-yaw-deg 180
```

The Isaac human asset is currently a static visual target. Walking animation is a later step using either an animated human USD or Isaac Sim People/animation tooling.

## Synthetic Data Direction

Next dataset milestone:

```text
Isaac Sim GP scene
  -> randomize intruder pose, distance, count, lighting, weather, and camera view
  -> capture RGB + 2D bounding boxes with semantic label person
  -> convert annotations to YOLO format
  -> fine-tune YOLO
  -> compare stock YOLO vs DMZ Sentry custom detector
```

For sim-to-real credibility, prefer real or physically grounded environment inputs:

- Orthophoto/satellite image for `--terrain-texture` with `--terrain-texture-scale 1`
- DEM/heightmap in a later terrain import step
- PBR ground material maps for close camera realism:
  - albedo/basecolor
  - normal
  - roughness
- Sketchfab or other licensed USD/OBJ/FBX assets for props such as guard posts, fences, barriers, signs, boats, and human characters

Keep source/license notes for every downloaded asset in `docs/assets.md`.

## YOLO Person Detection

Install the detector dependency once:

```bash
python3 -m pip install --user ultralytics
```

Build the ROS 2 workspace:

```bash
cd /home/rokey/dev_ws/dmz_sentry/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

Run Isaac Sim first:

```bash
cd /home/rokey/dev_ws/dmz_sentry
./scripts/demo_dmz_sim.sh
```

In another terminal, start the detector:

```bash
cd /home/rokey/dev_ws/dmz_sentry
./scripts/demo_yolo_detector.sh
```

Outputs:

- `/detections_text`: JSON string detections
- `/alerts`: alert JSON string when a person crosses the confidence threshold
- `/camera/annotated`: optional camera image with YOLO boxes

By default, annotated image publishing is off to keep the system light. Use `/detections_text` and `/alerts` for robot logic:

```bash
YOLO_PUBLISH_ANNOTATED=false ./scripts/demo_yolo_detector.sh
```

To inspect bounding boxes in rqt:

```bash
./scripts/demo_yolo_detector.sh
```

Then view `/camera/annotated` in `rqt_image_view`. If the view lags, keep `YOLO_ANNOTATED_SCALE=0.5` or disable annotated publishing and rely on `/detections_text` and `/alerts`.

## Workspace Layout

```text
dmz_sentry/
  isaacsim/          Isaac Sim standalone scripts
  scripts/           Run/check helper scripts
  ros2_ws/src/       Future ROS 2 packages
  web/tactical_map/  Browser tactical map for odom, patrol state, and alerts
  assets/textures/   Satellite/orthophoto terrain textures
  docs/              Notes and diagrams
  media/             Screenshots and demo captures
```

From now on, edit `isaacsim/anymal_gp_terrain.py` here instead of editing the Isaac Sim `_build/release` example folder directly.
