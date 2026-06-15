#!/usr/bin/env bash
set -euo pipefail

# Run inside the isaac-lab-base container from /workspace/isaaclab.
# This is the predefined expert policy path: a scripted guide/guard controller
# generates rollouts. It does not train from the multiplicative exponential-gated
# reward; that is handled by train_multiplicative_exp_gated_insertion_rl.sh.

cd /workspace/isaaclab

ROOT="aic/outputs/agentic_reward_curriculum_20260529"
RUN_ROOT="${AIC_EXPERT_ROLLOUT_ROOT:-/tmp/aic_expert_cheatcode_policy_true40x10}"
EPISODES="$ROOT/generated_episode_configs/v1254_v642lowtheta_postsettle_true40x10_precomp/episodes"
ACT_TS="aic/outputs/train/clean_sfp_sc/act/bc/20260510_clean_act_nact8_400k/act_policy_ts_175000_cuda0.pt"
BASE_CKPT="${AIC_EXPERT_BASE_CKPT:-/tmp/aic_v1367_clean_v1077_v1254_true40x10_4h_segments_clip1mm/2026-06-05_03-42-24_2026-06-04_train_v1367_clean_v1077_v1254_clip1mm_constant40x10_seg08/checkpoint_latest.pt}"
if [[ ! -f "$BASE_CKPT" ]]; then
  BASE_CKPT="$ROOT/policy_train_runs/2026-06-02_train_v1077_gentle_offline_true40x10/2026-06-02_23-44-40_isaac_online_serl/checkpoint_latest.pt"
fi

STEPS="${AIC_EXPERT_STEPS:-900}"
NUM_ENVS="${AIC_EXPERT_NUM_ENVS:-1}"
SEED="${AIC_EXPERT_SEED:-9401}"
SAVE_VIDEO="${AIC_EXPERT_SAVE_VIDEO:-1}"
mkdir -p "$RUN_ROOT"

VIDEO_FLAGS=(
  --camera_render_resolution 448
  --max_logged_image_steps "$STEPS"
  --image_log_every 1
  --video_fps 20
  --video_crf 16
  --video_final_hold_s 3.0
  --save_step_images
  --save_videos
)
if [[ "$SAVE_VIDEO" != "1" ]]; then
  VIDEO_FLAGS=(--max_logged_image_steps 0 --no-save_step_images --no-save_videos)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" /workspace/isaaclab/_isaac_sim/python.sh \
  aic/aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py \
  --task AIC-Task-v0 \
  --headless \
  --rendering_mode performance \
  --act_torchscript "$ACT_TS" \
  --policy_hz 20 \
  --enable_contact_sensor \
  --disable_ppo_resnet_observation_terms \
  --fix_isaac_ik_xy_sign \
  --absolute_ik_target_pose \
  --no-treat_time_limit_truncation_as_terminal \
  --gripper_joint_position 0.0035405 \
  --critic_image_encoder_override small_conv \
  --target_success_orientation_threshold 0.03 \
  --target_success_axial_threshold 0.0005 \
  --target_success_lateral_threshold 0.0005 \
  --target_reward_orientation_error_mode axis \
  --target_reward_orientation_axis_local 0 0 1 \
  --reward_preset multiplicative_exp_gated_insertion_v1 \
  --target_reward_exp_gated_action_axis_source action_manager \
  --target_reward_consistency_body none \
  --collision_contact_tune_prim_regex runtime_sdf_ \
  --collision_contact_tune_prim_regex cage_p0 \
  --replace_sfp_body_sdf_collision_with_shrunk_sdf_boxes \
  --near_gate_reset_max_iterations 8 \
  --near_gate_reset_position_tolerance 0.00005 \
  --near_gate_reset_orientation_tolerance 0.0005 \
  --target_action_guide_mode target_tip_stabilize \
  --target_action_guide_step_size 0.00006 \
  --target_action_guide_lateral_switch_m 0.0015 \
  --target_action_guide_preinsert_hover_depth -0.040 \
  --target_action_guide_preinsert_hover_step_m 0.00060 \
  --target_action_guide_preinsert_hover_ready_tolerance_m 0.0015 \
  --target_action_guide_preinsert_alignment_latch \
  --target_action_guide_final_axis_only_orientation \
  --target_action_guide_target_tip_body_name sfp_tip_link \
  --target_action_guide_target_tip_lateral_body_name sfp_tip_link \
  --target_action_guide_target_tip_orientation_body_name sfp_tip_link \
  --target_action_guide_target_tip_lateral_step_m 0.00045 \
  --target_action_guide_target_tip_goal_depth_m 0.0458 \
  --target_action_guide_target_tip_axial_step_m 8e-05 \
  --target_action_guide_target_tip_axial_lateral_gate_m 0.0010 \
  --target_action_guide_target_tip_clamp_positive_axial_when_gated \
  --target_action_guide_rotation_step_size 0.00030 \
  --target_action_guide_separate_rotation_compensation \
  --insertion_action_guard \
  --insertion_action_guard_lateral_threshold_m 0.0005 \
  --insertion_action_guard_lateral_step_m 0.00060 \
  --insertion_action_guard_reject_predicted_r_increase \
  --insertion_action_guard_reject_predicted_depth_when_offgate \
  --insertion_action_guard_zero_rotation_when_offcenter \
  --insertion_action_guard_adaptive_lateral_sign \
  --sfp_shrunk_box_margin_m 0.00030 0.0 0.00030 \
  --collision_contact_offset_m 0.00002 \
  --collision_rest_offset_m 0.0 \
  --replace_nic_cage_p0_with_aligned_cubes \
  --episode_config_dir "$EPISODES" \
  --episode_length_s 45.0 \
  --device cuda:0 \
  --updates 1000000 \
  --warmup_steps 1 \
  --actor_update_start_steps 1 \
  --actor_update_end_steps 0 \
  --update_every_steps 1 \
  --gradient_updates_per_step 1 \
  --batch_size 16 \
  --replay_capacity 12000 \
  --adapter_lr 0.0 \
  --critic_lr 1e-5 \
  --actor_q_weight 0.0 \
  --target_action_guide_weight 0.0 \
  --target_action_guide_collect_steps 1000000 \
  --target_action_guide_collect_blend 1.0 \
  --no-target_action_guide_use_episode_constant_action \
  --diagnostics_every 20 \
  --log_every 50 \
  --save_every_steps 0 \
  --save_latest_every_steps 0 \
  --save_replay_at_end \
  --save_replay_filter all \
  --no-save_final_checkpoint \
  "${VIDEO_FLAGS[@]}" \
  --output_dir "$RUN_ROOT" \
  --run_name "expert_cheatcode_policy_true40x10_native448" \
  --checkpoint "$BASE_CKPT" \
  --num_envs "$NUM_ENVS" \
  --seed "$SEED" \
  --steps "$STEPS"
