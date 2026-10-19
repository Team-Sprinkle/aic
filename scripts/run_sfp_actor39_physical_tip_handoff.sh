#!/usr/bin/env bash
# Frozen local actor; privileged transport stops from measured tip standoff.
# Geometry is confined to this evaluation start generator, never local actor.
set -euo pipefail
repo=$(realpath "$(dirname "$0")/..")
root="$repo/outputs/experiments/2026-10-02_two_family_perception"
checkpoint="$root/local_actor_sfp_native39_bodyconverted_v1/best.pt"
scene=${AIC_SFP_PHYSICAL_SCENE:-card2_port1_cards5_train}
audit=${AIC_SFP_PHYSICAL_HANDOFF_AUDIT:-/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_actor39_physical_tip_handoff_35mm_v1}
export DOCKER_HOST=unix:///run/user/1008/docker.sock
export AIC_TRANSPORT_EVAL_GPU=${AIC_TRANSPORT_EVAL_GPU:-3}
export AIC_TRANSPORT_EVAL_POLICY_GPU=${AIC_TRANSPORT_EVAL_POLICY_GPU:-$AIC_TRANSPORT_EVAL_GPU}
export AIC_TRANSPORT_EVAL_POLICY=aic_example_policies.ros.CheatCodeThenLocal
export AIC_TRANSPORT_EVAL_TEACHER_PREFIX=true
export AIC_TRANSPORT_EVAL_SCENE_DIR="$root/sfp_native_scene_selection_v1/$scene"
export AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA=.02
export AIC_SFP_LOCAL_HANDOFF_Z_OFFSET_M=.03 AIC_SFP_LOCAL_HANDOFF_HOLD_SEC=.1
export AIC_SFP_LOCAL_HANDOFF_TIP_STANDOFF_M=${AIC_SFP_PHYSICAL_STANDOFF_M:-.035}
export AIC_SFP_LOCAL_HANDOFF_LATERAL_MAX_M=${AIC_SFP_PHYSICAL_LATERAL_MAX_M:-0}
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
bash "$repo/scripts/run_clean_transport_eval.sh" "$scene" "$checkpoint" "$audit"
