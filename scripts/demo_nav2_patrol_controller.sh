#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set +u
source /opt/ros/humble/setup.bash
if [ -f "${PROJECT_ROOT}/ros2_ws/install/setup.bash" ]; then
  source "${PROJECT_ROOT}/ros2_ws/install/setup.bash"
fi
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-129}"

exec ros2 run dmz_sentry_control nav2_patrol_controller "$@"
