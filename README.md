# DMZ Sentry

Isaac Sim + ROS 2 Humble based autonomous reconnaissance quadruped simulation.

## Current Status

- Isaac Sim 5.1 standalone scene
- ANYmal C flat-terrain policy teleoperation
- GP-style terrain with fence, river, bunkers, watchtowers, and warning signs
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
./scripts/run_anymal_gp.sh --terrain-amplitude 0.20
./scripts/run_anymal_gp.sh --terrain-seed 11
```

Keyboard control:

- `UP` / `DOWN`: forward / backward
- `LEFT` / `RIGHT`: strafe
- `N` / `M`: yaw left / right

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
./scripts/run_anymal_gp.sh --no-intruder
```

The intruder prims are labeled as semantic class `person`, which leaves a path for later Replicator synthetic-data or bounding-box experiments.

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
./scripts/run_anymal_gp.sh
```

In another terminal, start the detector:

```bash
cd /home/rokey/dev_ws/dmz_sentry
./scripts/run_yolo_person_detector.sh
```

Outputs:

- `/camera/annotated`: camera image with YOLO boxes
- `/detections_text`: JSON string detections

For RViz2 or rqt, view `/camera/annotated` as an image topic.

## Workspace Layout

```text
dmz_sentry/
  isaacsim/          Isaac Sim standalone scripts
  scripts/           Run/check helper scripts
  ros2_ws/src/       Future ROS 2 packages
  assets/textures/   Satellite/orthophoto terrain textures
  docs/              Notes and diagrams
  media/             Screenshots and demo captures
```

From now on, edit `isaacsim/anymal_gp_terrain.py` here instead of editing the Isaac Sim `_build/release` example folder directly.
