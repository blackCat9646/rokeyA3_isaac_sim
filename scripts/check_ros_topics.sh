#!/usr/bin/env bash
set -euo pipefail

set +u
source /opt/ros/humble/setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-129}"

ros2 topic list
echo
echo "Camera:"
timeout 5 ros2 topic hz /camera/image_raw || true
echo
echo "LiDAR:"
timeout 5 ros2 topic hz /lidar/points || true
echo
echo "Odometry sample:"
timeout 5 ros2 topic echo /odom --once || true
