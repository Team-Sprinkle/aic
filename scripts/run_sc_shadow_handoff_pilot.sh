#!/usr/bin/env bash
# Development-only live pause/inspect/resume pilot; no insertion actor.
set -euo pipefail
cd "$(dirname "$0")/.."
repo=$(pwd)
audit=${1:-/var/tmp/chmin_aic_20261003_shadow_handoff/pilot_sc_llb0122}
case_name=${2:-llb_0122}
scene_root=${AIC_SC_SHADOW_SCENE_ROOT:-$repo/outputs/experiments/2026-09-30_sc_llb_collection/continuation170/scenes}
export AIC_TEMPORAL_HANDOFF_PRIOR_JSON=${AIC_TEMPORAL_HANDOFF_PRIOR_JSON:-$repo/configs/hierarchical_recovery/train_handoff_tcp_regions_20261005.json}
export DOCKER_HOST=unix:///run/user/1008/docker.sock
export AIC_TRANSPORT_EVAL_GPU=1 AIC_TRANSPORT_EVAL_POLICY_GPU=2
export AIC_TRANSPORT_EVAL_POLICY=aic_example_policies.ros.RunTemporalTransport
export AIC_TRANSPORT_EVAL_SCENE_DIR="$scene_root/$case_name"
export AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA=.06
export AIC_TEMPORAL_TRANSPORT_CHECKPOINT="$repo/outputs/experiments/2026-10-01_transport_reliability/frozen_actor_handoff/best.pt"
export AIC_TEMPORAL_TRANSPORT_IMAGENET=/var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth
export AIC_TEMPORAL_TRANSPORT_STOP_THRESHOLD=.95
export AIC_TEMPORAL_TRANSPORT_STOP_STREAK=2
export AIC_TEMPORAL_TRANSPORT_STOP_MIN_SIM_S=15
export AIC_SC_VISUAL_STANDOFF_CHECKPOINT="$repo/outputs/experiments/2026-10-01_transport_reliability/actor_native_dual_port_locator/best.pt"
export AIC_SC_VISUAL_STANDOFF_CALIBRATION="$repo/configs/hierarchical_recovery/sc_visual_standoff_20261001.json"
export AIC_SC_VISUAL_STANDOFF_TRIGGER_M=.08
export AIC_SC_VISUAL_STANDOFF_STEP_M=.008
export AIC_SC_VISUAL_STANDOFF_STOP_M=.008
export AIC_SC_VISUAL_STANDOFF_STIFFNESS=1200
export AIC_SC_VISUAL_STANDOFF_DAMPING=80
export AIC_SC_VISUAL_STANDOFF_ORIENTATION=1
export AIC_SC_VISUAL_STANDOFF_ORIENTATION_PROBE_CHECKPOINT="$repo/outputs/experiments/2026-10-01_transport_reliability/onpolicy_goal_adaptation/best.pt"
export AIC_SC_VISUAL_STANDOFF_ORIENTATION_YAW_BLEND=.5
export AIC_SC_VISUAL_STOP_MODE=visual
export AIC_SC_SHADOW_HANDOFF=1
export AIC_SC_SHADOW_LOCATOR="$repo/outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/best.pt"
export AIC_SC_SHADOW_IDENTITY=/var/tmp/chmin_aic_20260918_act/shadow_handoff_20261003/sc_spatial_identity.pt
export AIC_SC_SHADOW_ORIENTATION=/var/tmp/chmin_aic_20260918_act/shadow_handoff_20261003/sc_orientation_oneport.pt
export AIC_SC_SHADOW_REFINER="$repo/outputs/experiments/2026-10-02_two_family_perception/sc_native_refiner_v1/best.pt"
export AIC_SC_SHADOW_CALIBRATION="$repo/configs/hierarchical_recovery/sc_visual_standoff_20261001.json"
export AIC_SC_SHADOW_BOARD_MOUNT="$repo/configs/hierarchical_recovery/sc_board_to_port_training_fixed.json"
export AIC_SC_SHADOW_TCP_PRIOR="$repo/outputs/experiments/2026-10-02_two_family_perception/local_sc_orientation_prior_v1.json"
bash scripts/run_clean_transport_eval.sh "$case_name" \
  "$repo/outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt" "$audit"
