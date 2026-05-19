#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATERIAL_DIR="${PROJECT_ROOT}/assets/materials/Ground081_2K-JPG"

TERRAIN_TEXTURE="${TERRAIN_TEXTURE:-${MATERIAL_DIR}/Ground081_2K-JPG_Color.jpg}"
TERRAIN_NORMAL_TEXTURE="${TERRAIN_NORMAL_TEXTURE:-${MATERIAL_DIR}/Ground081_2K-JPG_NormalGL.jpg}"
TERRAIN_ROUGHNESS_TEXTURE="${TERRAIN_ROUGHNESS_TEXTURE:-${MATERIAL_DIR}/Ground081_2K-JPG_Roughness.jpg}"
TERRAIN_TEXTURE_SCALE="${TERRAIN_TEXTURE_SCALE:-12}"
LIDAR_MOUNT="${LIDAR_MOUNT:-robot}"

for texture in "${TERRAIN_TEXTURE}" "${TERRAIN_NORMAL_TEXTURE}" "${TERRAIN_ROUGHNESS_TEXTURE}"; do
  if [ ! -f "${texture}" ]; then
    echo "Missing terrain texture: ${texture}" >&2
    exit 1
  fi
done

exec "${PROJECT_ROOT}/scripts/run_anymal_gp.sh" \
  --terrain-texture "${TERRAIN_TEXTURE}" \
  --terrain-normal-texture "${TERRAIN_NORMAL_TEXTURE}" \
  --terrain-roughness-texture "${TERRAIN_ROUGHNESS_TEXTURE}" \
  --terrain-texture-scale "${TERRAIN_TEXTURE_SCALE}" \
  --lidar-mount "${LIDAR_MOUNT}" \
  --no-ground-detail \
  "$@"
