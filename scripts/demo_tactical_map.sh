#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${TACTICAL_MAP_PORT:-8080}"

echo "DMZ Sentry tactical map: http://localhost:${PORT}"
exec python3 -m http.server "${PORT}" -d "${PROJECT_ROOT}/web/tactical_map"
