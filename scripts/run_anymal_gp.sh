#!/usr/bin/env bash
set -euo pipefail

ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/rokey/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set +u
source /opt/ros/humble/setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-129}"

cd "${ISAAC_SIM_ROOT}"
exec ./python.sh "${PROJECT_ROOT}/isaacsim/anymal_gp_terrain.py" "$@"
