#!/usr/bin/env bash
# Test the frozen 39-scene insertion actor with a 60 mm teacher standoff.
# This changes only the diagnostic transport-to-local handoff, not actor weights.
set -euo pipefail
repo=$(realpath "$(dirname "$0")/..")
root="$repo/outputs/experiments/2026-10-02_two_family_perception"
checkpoint="$root/local_actor_sfp_native39_bodyconverted_v1/best.pt"
audit=/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_actor39_early_handoff_60mm_v1
export DOCKER_HOST=unix:///run/user/1008/docker.sock
export AIC_TRANSPORT_EVAL_GPU=3 AIC_TRANSPORT_EVAL_POLICY_GPU=3
export AIC_TRANSPORT_EVAL_POLICY=aic_example_policies.ros.CheatCodeThenLocal
export AIC_TRANSPORT_EVAL_TEACHER_PREFIX=true
export AIC_TRANSPORT_EVAL_SCENE_DIR="$root/sfp_native_scene_selection_v1/card2_port1_cards5_train"
export AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA=.02
export AIC_SFP_LOCAL_HANDOFF_Z_OFFSET_M=.06 AIC_SFP_LOCAL_HANDOFF_HOLD_SEC=.5
export AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_CHECKPOINT="$checkpoint"
export AIC_LOCAL_LOCATOR_CHECKPOINT="$root/joint_locator_native_sfp_v5_heatmap/best.pt"
export AIC_LOCAL_ORIENTATION_CHECKPOINT="$root/orientation_from_predicted_ports_v1/best.pt"
export AIC_LOCAL_SFP_COARSE_CHECKPOINT="$root/joint_locator_native_sfp_v5_heatmap/best.pt"
export AIC_LOCAL_SFP_BROAD_CHECKPOINT="$root/sfp_broad_crop480_v2/best.pt"
export AIC_LOCAL_SFP_FINE_CHECKPOINT="$root/sfp_fine_crop160_v2_inside_selection/best.pt"
export AIC_LOCAL_SFP_ORIENTATION_CHECKPOINT="$root/sfp_native_orientation_v2_highnative/best.pt"
export AIC_LOCAL_SFP_ORIENTATION_CHECKPOINT_2="$root/sfp_native_orientation_v3_runtime_matched/best.pt"
export AIC_LOCAL_CAMERA_CALIBRATION="$repo/configs/hierarchical_recovery/sc_visual_standoff_20261001.json"
export AIC_LOCAL_ACTOR_DEVICE=cpu
bash "$repo/scripts/run_clean_transport_eval.sh" card2_port1_cards5_train "$checkpoint" "$audit"
