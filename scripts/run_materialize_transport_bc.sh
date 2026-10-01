#!/usr/bin/env bash
set -euo pipefail
repo=$(realpath "$(dirname "$0")/..")
cd "$repo"
set +u
source .pixi/envs/default/setup.bash
set -u
export ROS_DOMAIN_ID=94 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONPATH="$repo/aic_teacher_official:${PYTHONPATH:-}"
.pixi/envs/default/bin/python scripts/publish_moveit_joint_state.py \
 outputs/experiments/2026-09-29_moveit_gazebo_pilot/measured_start_joint_state.json \
 > outputs/experiments/2026-09-30_transport_continuation/fk_joint_source.log 2>&1 &
publisher_pid=$!
trap 'kill "$publisher_pid" 2>/dev/null || true' EXIT
.pixi/envs/default/bin/python scripts/materialize_moveit_bc_segments.py "$@"
