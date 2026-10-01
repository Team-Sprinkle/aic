#!/usr/bin/env bash
set -euo pipefail
scene=$(realpath "$1")
plan=$(realpath -m "$2")
orientation=${3:?comma-separated aligned TCP quaternion}
lane=${4:-0}
forward=${5:-0}
repo=$(realpath "$(dirname "$0")/..")
cd "$repo"
set +u
source .pixi/envs/default/setup.bash
set -u
export ROS_DOMAIN_ID=93 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONPATH="$repo/aic_teacher_official:${PYTHONPATH:-}"
.pixi/envs/default/bin/python scripts/publish_moveit_joint_state.py "$scene/measured_start_joint_state.json" > "$scene/joint_publisher.log" 2>&1 &
publisher_pid=$!
trap 'kill "$publisher_pid" 2>/dev/null || true' EXIT
.pixi/envs/default/bin/python scripts/plan_moveit_sc_fixed_route.py \
 --config "$scene/eval_config.yaml" --joint-state "$scene/measured_start_joint_state.json" \
 --route-plan "$scene/route_plan.json" --output-dir "$plan" \
 --lane-extra-m "$lane" --forward-extra-m "$forward" --strict-return \
 --low-cross-z-m .18 --diagonal-egress --final-standoff-z-m .2069 \
 --aligned-orientation-xyzw "$orientation" --prealign-from-goal-index 5 --tilt-before-return \
 > "$scene/planning.log" 2>&1
.pixi/envs/default/bin/python scripts/hold_moveit_route_segments.py "$plan" "${plan}_held" \
 --hold sweep_after_row:8 --hold tilt_clear_of_cards:5
.pixi/envs/default/bin/python scripts/make_moveit_insertion_handoff.py "${plan}_held" "${plan}_insert"
