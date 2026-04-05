#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v pixi >/dev/null 2>&1; then
  echo "error: pixi is not installed or not on PATH" >&2
  exit 1
fi

cd "${repo_root}"

pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.WaveArm
