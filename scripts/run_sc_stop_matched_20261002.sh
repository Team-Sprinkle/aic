#!/usr/bin/env bash
# Small same-controller development check of frozen stop signals.
set -euo pipefail
cd "$(dirname "$0")/.."
repo=$(pwd)
root=outputs/experiments/2026-10-01_transport_reliability
archive=/home/nas/chmin/aic_archives/2026-10-02_transport_stop_comparison
mkdir -p "$archive"
case_name=llb_0193
checkpoint=outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt
plan=outputs/experiments/2026-09-30_sc_llb_collection/continuation170/plans/$case_name/moveit_joint_trajectory.json
export DOCKER_HOST=unix:///run/user/1008/docker.sock
export AIC_TRANSPORT_EVAL_GPU=1 AIC_TRANSPORT_EVAL_POLICY_GPU=2
export AIC_TRANSPORT_EVAL_POLICY=aic_example_policies.ros.RunTemporalTransport
export AIC_TRANSPORT_EVAL_SCENE_DIR="$repo/outputs/experiments/2026-09-30_sc_llb_collection/continuation170/scenes/$case_name"
export AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA=.06
export AIC_TEMPORAL_TRANSPORT_CHECKPOINT="$repo/$root/frozen_actor_handoff/best.pt"
export AIC_TEMPORAL_TRANSPORT_IMAGENET=/var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth
export AIC_TEMPORAL_TRANSPORT_STOP_THRESHOLD=.95
export AIC_TEMPORAL_TRANSPORT_STOP_STREAK=2
export AIC_TEMPORAL_TRANSPORT_STOP_MIN_SIM_S=15
export AIC_SC_VISUAL_STANDOFF_CHECKPOINT="$repo/$root/actor_native_dual_port_locator/best.pt"
export AIC_SC_VISUAL_STANDOFF_CALIBRATION="$repo/configs/hierarchical_recovery/sc_visual_standoff_20261001.json"
export AIC_SC_VISUAL_STANDOFF_TRIGGER_M=.08
export AIC_SC_VISUAL_STANDOFF_STEP_M=.008
export AIC_SC_VISUAL_STANDOFF_STOP_M=.008
export AIC_SC_VISUAL_STANDOFF_STIFFNESS=1200
export AIC_SC_VISUAL_STANDOFF_DAMPING=80
export AIC_SC_VISUAL_STANDOFF_ORIENTATION=1
export AIC_SC_VISUAL_STANDOFF_ORIENTATION_PROBE_CHECKPOINT="$repo/$root/onpolicy_goal_adaptation/best.pt"
export AIC_SC_VISUAL_STANDOFF_ORIENTATION_YAW_BLEND=.5
for mode in transport_head visual; do
  audit="$archive/${case_name}_${mode}_v1"
  export AIC_SC_VISUAL_STOP_MODE="$mode"
  bash scripts/run_clean_transport_eval.sh "$case_name" "$checkpoint" "$audit" \
    > "$archive/${case_name}_${mode}_v1.launch.log" 2>&1
  .pixi/envs/default/bin/python scripts/measure_transport_teacher_path.py "$audit" \
    --plan "$plan" > "$archive/${case_name}_${mode}_v1.path.log"
done
