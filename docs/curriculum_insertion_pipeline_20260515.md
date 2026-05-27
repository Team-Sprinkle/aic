# One-Day Insertion Pipeline - 2026-05-15

Output root: `/data1/chmin/yj/ws_aic/src/aic/outputs/one_day_insertion_pipeline/curriculum_20260515_smart4_cd_from_b`
GPUs: `0,1,2,3`
ACT: `/data1/chmin/yj/ws_aic/src/aic/outputs/train/clean_sfp_sc/act/bc/20260510_clean_act_nact8_400k/act_policy_ts_175000_cuda0.pt`
Episodes: `/data1/chmin/yj/ws_aic/src/aic/outputs/analysis/isaac_near_gate_6mm_orientation_gate/episode_configs/episodes`
Curriculum mode: `staged`

## Summary

- Runs completed: 4
- Best alignment: `/data1/chmin/yj/ws_aic/src/aic/outputs/one_day_insertion_pipeline/curriculum_20260515_smart4_cd_from_b/runs/2026-05-15_19-42-08_stage_c_smart4_4_guarded_insert_20260515_194026_gpu3`
- Best insertion: `/data1/chmin/yj/ws_aic/src/aic/outputs/one_day_insertion_pipeline/curriculum_20260515_smart4_cd_from_b/runs/2026-05-15_19-42-08_stage_c_smart4_4_guarded_insert_20260515_194026_gpu3`

## Metric Table

| stage | name | score | final r mean mm | final r worst mm | best r mean mm | final theta mean | final theta worst | max s mm | bad inward r frac | force clip frac | run |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |

## Curriculum Episode Configs

- `stage_c` episodes: `/data1/chmin/yj/ws_aic/src/aic/outputs/analysis/curriculum_insertion_20260515/stage_c_6_8mm_entry/episode_configs/episodes` (20 eps, s -8.00..-6.00 mm, r 0.00..3.00 mm)
- `stage_d` episodes: `/data1/chmin/yj/ws_aic/src/aic/outputs/analysis/curriculum_insertion_20260515/stage_d_3_6mm_final/episode_configs/episodes` (20 eps, s -6.00..-3.00 mm, r 0.00..1.50 mm)

| stage_c | smart4_4_guarded_insert | -20.70 | 7.45 | 7.45 | 0.23 | 0.061 | 0.061 | -1.46 |  | 0.000 | `/data1/chmin/yj/ws_aic/src/aic/outputs/one_day_insertion_pipeline/curriculum_20260515_smart4_cd_from_b/runs/2026-05-15_19-42-08_stage_c_smart4_4_guarded_insert_20260515_194026_gpu3` |
| stage_c | smart4_3_medium_corridor | -22.27 | 8.01 | 8.11 | 0.23 | 0.061 | 0.062 | -1.61 |  | 0.000 | `/data1/chmin/yj/ws_aic/src/aic/outputs/one_day_insertion_pipeline/curriculum_20260515_smart4_cd_from_b/runs/2026-05-15_19-42-03_stage_c_smart4_3_medium_corridor_20260515_194026_gpu2` |
| stage_c | smart4_1_weak_insert_conservative | -28.34 | 10.24 | 10.45 | 0.23 | 0.061 | 0.061 | -0.97 |  | 0.000 | `/data1/chmin/yj/ws_aic/src/aic/outputs/one_day_insertion_pipeline/curriculum_20260515_smart4_cd_from_b/runs/2026-05-15_19-40-57_stage_c_smart4_1_weak_insert_conservative_20260515_194026_gpu0` |
| stage_c | smart4_2_weak_insert_balanced | -28.40 | 10.22 | 10.65 | 0.23 | 0.061 | 0.061 | -1.43 |  | 0.000 | `/data1/chmin/yj/ws_aic/src/aic/outputs/one_day_insertion_pipeline/curriculum_20260515_smart4_cd_from_b/runs/2026-05-15_19-40-54_stage_c_smart4_2_weak_insert_balanced_20260515_194026_gpu1` |

## Stage Conclusions

- `stage_c` best: `smart4_4_guarded_insert` at `/data1/chmin/yj/ws_aic/src/aic/outputs/one_day_insertion_pipeline/curriculum_20260515_smart4_cd_from_b/runs/2026-05-15_19-42-08_stage_c_smart4_4_guarded_insert_20260515_194026_gpu3`

## Recommendation

Use the best guide-only/alignment-imitation command as the safe fallback unless a weak insertion run satisfies strict alignment during positive depth.
Do not treat `sfp_tip_link` positive depth as success without lateral, orientation, and consistency-body checks.

## 2026-05-15 Stage Progression Check

The staged curriculum is not monotonically improving yet.

- Stage A to Stage B improved: the best Stage B alignment run reduced mean lateral error from about `2.01 mm` to `1.56 mm`, with worst-env lateral error about `1.62 mm`, while orientation stayed near `0.067 rad`. This is usable as an alignment checkpoint/reference.
- The selected Stage B checkpoint rerun later regressed (`r` grew to about `4.68 mm`, worst `7.85 mm`), so that later segment should not be treated as the best Stage B behavior.
- Stage C failed the intended transition: all four weak-insertion candidates moved axially toward the entrance (`s` from about `-6 mm` to `-1 to -1.6 mm`) while lateral error grew from about `0.23 mm` to `7.4-11.9 mm`. This is not real insertion progress and should not be promoted to Stage D.
- The separate single-stage 20 mm axial / 6 mm lateral radius sweep is currently more promising than Stage C. The radius-1 mm variants reduced `r` from about `5.96 mm` to about `2.0-2.3 mm` while also improving orientation from about `0.071 rad` to `0.062 rad`. Continue those runs before trying radius `0.5 mm`; radius `4 mm` and `2 mm` both failed by increasing `r`.

Decision: stop stale Stage B/C jobs, keep the best Stage A/B checkpoint as fallback, and do not advance staged C/D until C is changed to preserve lateral alignment before axial approach. The immediate candidate to continue is the single-stage radius-1 mm configuration.

## 2026-05-15 Single-Stage vs Four-Stage Decision

Current decision: the fixed single-stage radius-1 mm setup is better than the current four-stage curriculum, but it is not yet a complete insertion solution.

Four-stage status:

- Stage A/B can learn/use alignment behavior.
- Stage C is the blocker: weak insertion candidates start very close to centered (`r ~= 0.23 mm`) but lose alignment while approaching (`r` grows to `7.4-11.9 mm` as `s` moves from about `-6 mm` to `-1 mm`).
- Because Stage C breaks the intended invariant "maintain alignment while descending", Stage D should not be run from the current Stage C policy.

Single-stage radius status:

| run | s start mm | s final mm | r start mm | r final mm | best r mm | theta start | theta final | force final | checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| radius1_balanced | -19.97 | -38.77 | 5.95 | 2.18 | 1.68 | 0.0707 | 0.0684 | 16.74 | `outputs/one_day_insertion_pipeline/single20_20260515_axis_decay/runs/2026-05-15_19-40-51_single20_radius1_balanced_20260515_gpu5/checkpoint_latest.pt` |
| radius1_conservative | -19.97 | -7.64 | 5.95 | 1.84 | 1.51 | 0.0707 | 0.0754 | 5.23 | `outputs/one_day_insertion_pipeline/single20_20260515_axis_decay/runs/2026-05-15_19-40-53_single20_radius1_conservative_20260515_gpu4/checkpoint_latest.pt` |

Interpretation:

- Radius `4 mm` and `2 mm` were too permissive: the policy approached while lateral error grew.
- Radius `1 mm` is the first fixed-radius setup that produced early alignment. The conservative variant is better because it approaches from `s ~= -20 mm` to `s ~= -7.6 mm` while reducing `r` from `5.95 mm` to `1.84 mm`.
- The balanced variant aligns but moves away from the entrance (`s` becomes more negative), so it is not the selected candidate.
- A fixed-radius single-stage reward can plausibly align early and maintain alignment during descent, but only if the descent is gated tightly by lateral/orientation alignment. The current conservative radius-1 run nearly does this for lateral alignment, but theta drifted worse (`0.0707 -> 0.0754 rad`), so the next single-stage run should strengthen orientation guidance/penalty and keep axial progress slow until `r < 1.5 mm` and `theta < 0.06 rad`.

Recommended next run:

- Prefer single-stage fixed radius `1 mm` over the current four-stage pipeline.
- Start from the `radius1_conservative` checkpoint.
- Keep the tight lateral radius; do not reduce to `0.5 mm` yet because `1 mm` is improving.
- Add a stronger orientation-maintenance term/guide and stricter axial gating during descent.
- Only return to the four-stage curriculum after Stage C is changed so that axial motion is disabled or strongly penalized whenever `r/theta` leave the insertion tube.

## Commands

### stage_c / smart4_2_weak_insert_balanced

```bash
zsh -lc 'LC_USER_ID=yoonjung docker exec -e AIC_ISAAC_ALLOW_UNSUPPORTED_RTX_DRIVER=1 -e AIC_ISAAC_RANDOMIZATION_PROFILE=none -e CUDA_VISIBLE_DEVICES=1 -e DEVICE=cuda:0 -e NUM_ENVS=2 -e RENDERING_MODE=performance isaac-lab-base bash -lc '"'"'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py --headless --task AIC-Task-v0 --num_envs 2 --seed 802 --device cuda:0 --enable_cameras --rendering_mode performance --output_dir aic/outputs/one_day_insertion_pipeline/curriculum_20260515_smart4_cd_from_b/runs --run_name stage_c_smart4_2_weak_insert_balanced_20260515_194026_gpu1 --steps 1500 --updates 1500 --update_every_steps 4 --warmup_steps 50 --actor_update_start_steps 100 --batch_size 32 --replay_capacity 20000 --act_only --act_only_actor_mode act_direct --act_torchscript aic/outputs/train/clean_sfp_sc/act/bc/20260510_clean_act_nact8_400k/act_policy_ts_175000_cuda0.pt --n_action_steps 1 --tcp_action_frame root --no-fix_isaac_ik_xy_sign --no-isaac_ik_xy_sign_by_target_card --adapter_lr 1e-05 --adapter_delta_clip 0.003 --adapter_penalty_weight 0.05 --act_preservation_weight 0.5 --actor_q_weight 0.02 --tcp_translation_action_clip 0.0005 --tcp_rotation_action_clip 0.005 --reward_preset cheatcode_insertion_v1 --disable_command_pose_rewards --target_reward_body sfp_tip_link --target_reward_orientation_error_mode axis --target_reward_orientation_axis_local 0.0 0.0 1.0 --episode_config_dir aic/outputs/analysis/curriculum_insertion_20260515/stage_c_6_8mm_entry/episode_configs/episodes --near_gate_reset_max_iterations 8 --near_gate_reset_position_tolerance 0.002 --near_gate_reset_orientation_tolerance 0.05 --target_action_guide_weight 0.05 --target_action_guide_mode cheatcode_transform --target_action_guide_step_size 0.0003 --target_action_guide_rotation_step_size 0.005 --target_action_guide_rotation_sign 1.0 --target_action_guide_axial_step_size 0.0001 --target_action_guide_lateral_switch_m 0.0015 --target_action_guide_axial_blend_lateral_m 0.006 --target_action_guide_orientation_switch_rad 0.06 --target_action_guide_rotate_while_lateral --target_action_guide_preinsert_hover_depth -0.004 --target_action_guide_collect_blend 1.0 --target_action_guide_collect_steps 800 --target_action_guide_collect_decay --no-target_action_guide_prefix_decay --target_action_guide_train_executed --debug_diagnostics --diagnostics_every 10 --save_step_images --image_log_every 375 --max_logged_image_steps 1500 --debug_visual_overlays --log_every 25 --max_wall_time_minutes 15.0 --checkpoint aic/outputs/one_day_insertion_pipeline/curriculum_20260515_smart4_bcd_v2/runs/2026-05-15_19-36-53_stage_b_selected_balanced_align_ckpt_20260515_gpu0/checkpoint_latest.pt --save_latest_every_steps 375 --no-insertion_action_guard --target_reward_cheatcode_hover_weight 0.1 --target_reward_cheatcode_hover_depth -0.004 --target_reward_cheatcode_axial_progress_weight 0.1 --target_reward_cheatcode_corridor_weight 0.25 --target_reward_cheatcode_inside_alignment_weight 0.1 --target_reward_cheatcode_retreat_weight 0.1 --target_reward_consistency_body auto --target_reward_consistency_axial_std 0.004 --target_reward_consistency_lateral_sigma 0.003 --target_success_consistency_axial_threshold 0.001 --target_success_consistency_lateral_threshold 0.0015'"'"''
```

### stage_c / smart4_1_weak_insert_conservative

```bash
zsh -lc 'LC_USER_ID=yoonjung docker exec -e AIC_ISAAC_ALLOW_UNSUPPORTED_RTX_DRIVER=1 -e AIC_ISAAC_RANDOMIZATION_PROFILE=none -e CUDA_VISIBLE_DEVICES=0 -e DEVICE=cuda:0 -e NUM_ENVS=2 -e RENDERING_MODE=performance isaac-lab-base bash -lc '"'"'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py --headless --task AIC-Task-v0 --num_envs 2 --seed 801 --device cuda:0 --enable_cameras --rendering_mode performance --output_dir aic/outputs/one_day_insertion_pipeline/curriculum_20260515_smart4_cd_from_b/runs --run_name stage_c_smart4_1_weak_insert_conservative_20260515_194026_gpu0 --steps 1500 --updates 1500 --update_every_steps 4 --warmup_steps 50 --actor_update_start_steps 100 --batch_size 32 --replay_capacity 20000 --act_only --act_only_actor_mode act_direct --act_torchscript aic/outputs/train/clean_sfp_sc/act/bc/20260510_clean_act_nact8_400k/act_policy_ts_175000_cuda0.pt --n_action_steps 1 --tcp_action_frame root --no-fix_isaac_ik_xy_sign --no-isaac_ik_xy_sign_by_target_card --adapter_lr 3e-06 --adapter_delta_clip 0.002 --adapter_penalty_weight 0.05 --act_preservation_weight 0.5 --actor_q_weight 0.02 --tcp_translation_action_clip 0.0003 --tcp_rotation_action_clip 0.005 --reward_preset cheatcode_insertion_v1 --disable_command_pose_rewards --target_reward_body sfp_tip_link --target_reward_orientation_error_mode axis --target_reward_orientation_axis_local 0.0 0.0 1.0 --episode_config_dir aic/outputs/analysis/curriculum_insertion_20260515/stage_c_6_8mm_entry/episode_configs/episodes --near_gate_reset_max_iterations 8 --near_gate_reset_position_tolerance 0.002 --near_gate_reset_orientation_tolerance 0.05 --target_action_guide_weight 0.08 --target_action_guide_mode cheatcode_transform --target_action_guide_step_size 0.0002 --target_action_guide_rotation_step_size 0.004 --target_action_guide_rotation_sign 1.0 --target_action_guide_axial_step_size 5e-05 --target_action_guide_lateral_switch_m 0.001 --target_action_guide_axial_blend_lateral_m 0.006 --target_action_guide_orientation_switch_rad 0.06 --target_action_guide_rotate_while_lateral --target_action_guide_preinsert_hover_depth -0.004 --target_action_guide_collect_blend 1.0 --target_action_guide_collect_steps 1000 --target_action_guide_collect_decay --no-target_action_guide_prefix_decay --target_action_guide_train_executed --debug_diagnostics --diagnostics_every 10 --save_step_images --image_log_every 375 --max_logged_image_steps 1500 --debug_visual_overlays --log_every 25 --max_wall_time_minutes 15.0 --checkpoint aic/outputs/one_day_insertion_pipeline/curriculum_20260515_smart4_bcd_v2/runs/2026-05-15_19-36-53_stage_b_selected_balanced_align_ckpt_20260515_gpu0/checkpoint_latest.pt --save_latest_every_steps 375 --no-insertion_action_guard --target_reward_cheatcode_hover_weight 0.1 --target_reward_cheatcode_hover_depth -0.004 --target_reward_cheatcode_axial_progress_weight 0.1 --target_reward_cheatcode_corridor_weight 0.25 --target_reward_cheatcode_inside_alignment_weight 0.1 --target_reward_cheatcode_retreat_weight 0.1 --target_reward_consistency_body auto --target_reward_consistency_axial_std 0.004 --target_reward_consistency_lateral_sigma 0.003 --target_success_consistency_axial_threshold 0.001 --target_success_consistency_lateral_threshold 0.0015'"'"''
```

### stage_c / smart4_3_medium_corridor

```bash
zsh -lc 'LC_USER_ID=yoonjung docker exec -e AIC_ISAAC_ALLOW_UNSUPPORTED_RTX_DRIVER=1 -e AIC_ISAAC_RANDOMIZATION_PROFILE=none -e CUDA_VISIBLE_DEVICES=2 -e DEVICE=cuda:0 -e NUM_ENVS=2 -e RENDERING_MODE=performance isaac-lab-base bash -lc '"'"'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py --headless --task AIC-Task-v0 --num_envs 2 --seed 803 --device cuda:0 --enable_cameras --rendering_mode performance --output_dir aic/outputs/one_day_insertion_pipeline/curriculum_20260515_smart4_cd_from_b/runs --run_name stage_c_smart4_3_medium_corridor_20260515_194026_gpu2 --steps 1500 --updates 1500 --update_every_steps 4 --warmup_steps 50 --actor_update_start_steps 100 --batch_size 32 --replay_capacity 20000 --act_only --act_only_actor_mode act_direct --act_torchscript aic/outputs/train/clean_sfp_sc/act/bc/20260510_clean_act_nact8_400k/act_policy_ts_175000_cuda0.pt --n_action_steps 1 --tcp_action_frame root --no-fix_isaac_ik_xy_sign --no-isaac_ik_xy_sign_by_target_card --adapter_lr 3e-06 --adapter_delta_clip 0.002 --adapter_penalty_weight 0.05 --act_preservation_weight 0.5 --actor_q_weight 0.05 --tcp_translation_action_clip 0.0003 --tcp_rotation_action_clip 0.005 --reward_preset cheatcode_insertion_v1 --disable_command_pose_rewards --target_reward_body sfp_tip_link --target_reward_orientation_error_mode axis --target_reward_orientation_axis_local 0.0 0.0 1.0 --episode_config_dir aic/outputs/analysis/curriculum_insertion_20260515/stage_c_6_8mm_entry/episode_configs/episodes --near_gate_reset_max_iterations 8 --near_gate_reset_position_tolerance 0.002 --near_gate_reset_orientation_tolerance 0.05 --target_action_guide_weight 0.05 --target_action_guide_mode cheatcode_transform --target_action_guide_step_size 0.0003 --target_action_guide_rotation_step_size 0.005 --target_action_guide_rotation_sign 1.0 --target_action_guide_axial_step_size 0.0001 --target_action_guide_lateral_switch_m 0.001 --target_action_guide_axial_blend_lateral_m 0.006 --target_action_guide_orientation_switch_rad 0.05 --target_action_guide_rotate_while_lateral --target_action_guide_preinsert_hover_depth -0.004 --target_action_guide_collect_blend 1.0 --target_action_guide_collect_steps 800 --target_action_guide_collect_decay --no-target_action_guide_prefix_decay --target_action_guide_train_executed --debug_diagnostics --diagnostics_every 10 --save_step_images --image_log_every 375 --max_logged_image_steps 1500 --debug_visual_overlays --log_every 25 --max_wall_time_minutes 15.0 --checkpoint aic/outputs/one_day_insertion_pipeline/curriculum_20260515_smart4_bcd_v2/runs/2026-05-15_19-36-53_stage_b_selected_balanced_align_ckpt_20260515_gpu0/checkpoint_latest.pt --save_latest_every_steps 375 --no-insertion_action_guard --target_reward_cheatcode_hover_weight 0.1 --target_reward_cheatcode_hover_depth -0.004 --target_reward_cheatcode_axial_progress_weight 0.15 --target_reward_cheatcode_corridor_weight 0.5 --target_reward_cheatcode_inside_alignment_weight 0.1 --target_reward_cheatcode_retreat_weight 0.1 --target_reward_consistency_body auto --target_reward_consistency_axial_std 0.004 --target_reward_consistency_lateral_sigma 0.003 --target_success_consistency_axial_threshold 0.001 --target_success_consistency_lateral_threshold 0.0015'"'"''
```

### stage_c / smart4_4_guarded_insert

```bash
zsh -lc 'LC_USER_ID=yoonjung docker exec -e AIC_ISAAC_ALLOW_UNSUPPORTED_RTX_DRIVER=1 -e AIC_ISAAC_RANDOMIZATION_PROFILE=none -e CUDA_VISIBLE_DEVICES=3 -e DEVICE=cuda:0 -e NUM_ENVS=2 -e RENDERING_MODE=performance isaac-lab-base bash -lc '"'"'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py --headless --task AIC-Task-v0 --num_envs 2 --seed 804 --device cuda:0 --enable_cameras --rendering_mode performance --output_dir aic/outputs/one_day_insertion_pipeline/curriculum_20260515_smart4_cd_from_b/runs --run_name stage_c_smart4_4_guarded_insert_20260515_194026_gpu3 --steps 1500 --updates 1500 --update_every_steps 4 --warmup_steps 50 --actor_update_start_steps 100 --batch_size 32 --replay_capacity 20000 --act_only --act_only_actor_mode act_direct --act_torchscript aic/outputs/train/clean_sfp_sc/act/bc/20260510_clean_act_nact8_400k/act_policy_ts_175000_cuda0.pt --n_action_steps 1 --tcp_action_frame root --no-fix_isaac_ik_xy_sign --no-isaac_ik_xy_sign_by_target_card --adapter_lr 3e-06 --adapter_delta_clip 0.002 --adapter_penalty_weight 0.05 --act_preservation_weight 0.5 --actor_q_weight 0.02 --tcp_translation_action_clip 0.0003 --tcp_rotation_action_clip 0.005 --reward_preset cheatcode_insertion_v1 --disable_command_pose_rewards --target_reward_body sfp_tip_link --target_reward_orientation_error_mode axis --target_reward_orientation_axis_local 0.0 0.0 1.0 --episode_config_dir aic/outputs/analysis/curriculum_insertion_20260515/stage_c_6_8mm_entry/episode_configs/episodes --near_gate_reset_max_iterations 8 --near_gate_reset_position_tolerance 0.002 --near_gate_reset_orientation_tolerance 0.05 --target_action_guide_weight 0.08 --target_action_guide_mode cheatcode_transform --target_action_guide_step_size 0.0002 --target_action_guide_rotation_step_size 0.005 --target_action_guide_rotation_sign 1.0 --target_action_guide_axial_step_size 5e-05 --target_action_guide_lateral_switch_m 0.001 --target_action_guide_axial_blend_lateral_m 0.006 --target_action_guide_orientation_switch_rad 0.05 --target_action_guide_rotate_while_lateral --target_action_guide_preinsert_hover_depth -0.004 --target_action_guide_collect_blend 1.0 --target_action_guide_collect_steps 1000 --target_action_guide_collect_decay --no-target_action_guide_prefix_decay --target_action_guide_train_executed --debug_diagnostics --diagnostics_every 10 --save_step_images --image_log_every 375 --max_logged_image_steps 1500 --debug_visual_overlays --log_every 25 --max_wall_time_minutes 15.0 --checkpoint aic/outputs/one_day_insertion_pipeline/curriculum_20260515_smart4_bcd_v2/runs/2026-05-15_19-36-53_stage_b_selected_balanced_align_ckpt_20260515_gpu0/checkpoint_latest.pt --save_latest_every_steps 375 --insertion_action_guard --target_reward_cheatcode_hover_weight 0.1 --target_reward_cheatcode_hover_depth -0.004 --target_reward_cheatcode_axial_progress_weight 0.1 --target_reward_cheatcode_corridor_weight 0.25 --target_reward_cheatcode_inside_alignment_weight 0.1 --target_reward_cheatcode_retreat_weight 0.1 --target_reward_consistency_body auto --target_reward_consistency_axial_std 0.004 --target_reward_consistency_lateral_sigma 0.003 --target_success_consistency_axial_threshold 0.001 --target_success_consistency_lateral_threshold 0.0015'"'"''
```

## 2026-05-15 Mixed Long Scheduled Radius/Orientation Run

User requested stopping the short orientation-schedule jobs and restarting with tighter requirements.

Changes made before launch:

- Stopped the stale `single20_sched_lat4to*ori*` runs; no old orientation run remained active before relaunch.
- Fixed `aic_utils/aic_isaac/scripts/isaac_episode_configs.py` so randomized board/card position variation is materialized into each episode YAML before computing target, entrance, insertion axis, and near-gate reset metadata. This avoids stale reward geometry from reset-time object randomization.
- Generated a mixed 150-episode dataset:
  - 50 episodes at `20 mm` axial outside / `6 mm` lateral.
  - 50 episodes at `30 mm` axial outside / `8 mm` lateral.
  - 50 episodes at `40 mm` axial outside / `10 mm` lateral.
  - Port index randomized: 70 port 0, 80 port 1.
  - Target card index randomized: card counts observed `{0: 70, 1: 38, 2: 24, 3: 13, 4: 5}`.
  - Board position and NIC-card offset variation are materialized in the YAMLs.

Dataset paths:

- Requests: `outputs/analysis/mixed_20_30_40mm_tight_20260515/episode_configs/requests/`
- Episodes: `outputs/analysis/mixed_20_30_40mm_tight_20260515/episode_configs/episodes/`
- Summary: `outputs/analysis/mixed_20_30_40mm_tight_20260515/episode_configs/summary.json`

Training launch:

- Six independent replicas on GPUs `0-5`.
- `num_envs=2` per GPU.
- `batch_size=32`.
- Wall time: `90 min`.
- Based on the prior GPU5 hyperparameters from `single20_sched_lat4to075_ori10to04_balanced_gpu5`.
- Scheduled lateral tolerance:
  - far depth `s=-40 mm`: `sigma_lat_pre=4 mm`, `sigma_lat_insert=4 mm`
  - entrance `s=0`: `sigma_lat_pre=0.25 mm`, `sigma_lat_insert=0.25 mm`
- Scheduled orientation tolerance:
  - far depth `s=-40 mm`: `sigma_theta_pre=0.12 rad`, `sigma_theta_insert=0.10 rad`
  - entrance `s=0`: `sigma_theta_pre=0.07 rad`, `sigma_theta_insert=0.04 rad`
- Insertion reward strengthened relative to the old GPU5 run:
  - axial progress `0.15`
  - corridor `1.00`
  - inside alignment `0.25`
  - retreat `0.20`

Run root:

`outputs/one_day_insertion_pipeline/mixed_20_30_40mm_tight_20260515/`

Run directories:

- `outputs/one_day_insertion_pipeline/mixed_20_30_40mm_tight_20260515/runs/2026-05-15_20-40-28_mixed_long_tight_lat4to025_ori_gpu5hp_rep0`
- `outputs/one_day_insertion_pipeline/mixed_20_30_40mm_tight_20260515/runs/2026-05-15_20-40-29_mixed_long_tight_lat4to025_ori_gpu5hp_rep1`
- `outputs/one_day_insertion_pipeline/mixed_20_30_40mm_tight_20260515/runs/2026-05-15_20-40-30_mixed_long_tight_lat4to025_ori_gpu5hp_rep2`
- `outputs/one_day_insertion_pipeline/mixed_20_30_40mm_tight_20260515/runs/2026-05-15_20-40-32_mixed_long_tight_lat4to025_ori_gpu5hp_rep3`
- `outputs/one_day_insertion_pipeline/mixed_20_30_40mm_tight_20260515/runs/2026-05-15_20-40-32_mixed_long_tight_lat4to025_ori_gpu5hp_rep4`
- `outputs/one_day_insertion_pipeline/mixed_20_30_40mm_tight_20260515/runs/2026-05-15_20-40-35_mixed_long_tight_lat4to025_ori_gpu5hp_rep5`

Immediate launch check:

- All six runs reached at least `step=1`.
- Initial reward was `-0.527025` for each replica, consistent with the same first two mixed episodes.
- No argparse/config crash occurred.
