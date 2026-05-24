# Agent Reward Funnel Audit - 2026-05-23

## Scope

Repository: `Team-Sprinkle/aic`
Branch: `feat/hybrid-train`

Read first:

- `docs/isaac_insertion_experiment_summary_20260523.md`
- `docs/training_debug_findings_20260513.md`
- `docs/isaac_near_gate_experiments_20260515.md`
- `docs/cheatcode_insertion_reward_validation_20260515.md`
- `docs/one_day_insertion_pipeline_20260515.md`
- `docs/curriculum_insertion_pipeline_20260515.md`
- `docs/isaac_tip_orientation_reward_findings_20260515.md`
- `docs/action_axis_reward_iteration_20260516.md`

## Current Reward Terms

Main implementation:

- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/insertion_geometry.py`
- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py`
- SERL CLI wiring in `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`

Current insertion reward support includes:

- Semantic insertion geometry: signed axial depth `s`, lateral error `r`, target depth, lateral gate, depth fraction.
- `signed_axial_progress_reward`: positive only when gated; forward motion off-center becomes negative.
- `insertion_corridor_reward`: centered depth reward minus bypass penalty.
- `cheatcode_insertion_phase_reward`: phase-conditioned lateral progress, orientation progress, near-misalignment penalty, preinsert hover, gated axial progress, corridor, inside alignment, retreat, semantic/trailing-body progress, action-axis gate.
- Scheduled lateral and orientation widths through `schedule_lateral_radius` and `schedule_orientation_tolerance`.
- Semantic consistency gate through `sfp_module_link` when reward body is `sfp_tip_link`.
- Action-axis gating from action manager or realized body delta.

Presets:

- `near_gate_corridor_v1`
- `cheatcode_insertion_v1`
- `cheatcode_alignment_v1`

## Current Success Metrics

Strict success is not tip depth. The current strict checks require:

- axial threshold near the target depth,
- lateral threshold,
- orientation threshold using the semantic tip axis,
- optional consistency body axial/lateral thresholds,
- visual/video sanity when running Isaac/Gazebo.

Relevant CLI knobs:

- `--target_success_axial_threshold`
- `--target_success_lateral_threshold`
- `--target_success_orientation_threshold`
- `--target_success_consistency_axial_threshold`
- `--target_success_consistency_lateral_threshold`
- `--target_reward_consistency_body auto`

## Known Failure Modes

- Tip-depth false positives: `sfp_tip_link` can cross the entrance plane while `sfp_module_link` remains outside.
- Reset/contact instability: some near-gate starts are physically stressed and eject under zero action.
- Lateral bypass: policies can move axially while the plug is off-center unless action and reward are tightly gated.
- Orientation residual: best prior samples reached good `s/r` but stayed around `theta ~= 0.04-0.06 rad`, above strict `0.03 rad`.
- Rotation-induced lateral sweep: wrist rotations move the offset SFP tip sideways unless compensated.
- Controller realization mismatch: intended TCP/root-frame action can differ from realized semantic tip motion.
- Reward-only success risk: high reward or positive `s` is insufficient without module consistency and video.

## Hook Points

- Reward funnel formulas: `insertion_geometry.py::cheatcode_insertion_phase_reward`.
- Reward term plumbing: `rewards.py::body_to_object_cheatcode_phase_reward`, `body_to_object_axial_progress`, `body_to_object_insertion_corridor`.
- Reward CLI/config: `serl/train.py` around `--target_reward_cheatcode_*`.
- Guide logic: `serl/train.py::_cheatcode_transform_guided_policy_action` and `_target_guided_policy_action`.
- Servo guard: `serl/train.py::_apply_insertion_action_guard`.
- Diagnostics and run artifacts: `serl/train.py` writes `train_config.json`, `metrics.jsonl`, `audit_log.jsonl`, `diagnostics_summary.json`, images/videos when enabled.
- Offline formula audits: `aic_utils/aic_isaac/scripts/audit_insertion_reward_geometry.py`.
- New geometry-level hooks added in this pass:
  - `aic_utils/aic_isaac/scripts/privileged_insertion_servo_sweep.py`
  - `aic_utils/aic_isaac/scripts/audit_phase_reward_funnel.py`

## Smoke Commands

Pure tests:

```bash
.pixi/envs/default/bin/python -m pytest \
  aic_utils/aic_isaac/test/test_insertion_reward_geometry.py \
  aic_utils/aic_isaac/test/test_agent_reward_funnel_scripts.py -q
```

Privileged servo sweep:

```bash
.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/privileged_insertion_servo_sweep.py \
  --run-name 20260523_phase2_privileged_servo_sweep
```

Reward funnel audit:

```bash
.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/audit_phase_reward_funnel.py \
  --run-name 20260523_phase3_hand_tuned \
  --version hand_tuned
```

Auto-tuned reward funnel audit:

```bash
.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/audit_phase_reward_funnel.py \
  --run-name 20260523_phase3_auto_from_servo \
  --version auto_from_servo \
  --servo-metrics-json outputs/agent_reward_funnel/servo_sweeps/20260523_phase2_privileged_servo_sweep/metrics.json
```

Isaac guarded servo smoke template:

```bash
LC_USER_ID=yoonjung docker exec \
  -e AIC_ISAAC_ALLOW_UNSUPPORTED_RTX_DRIVER=1 \
  -e AIC_ISAAC_RANDOMIZATION_PROFILE=none \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e DEVICE=cuda:0 \
  -e NUM_ENVS=1 \
  -e RENDERING_MODE=performance \
  isaac-lab-base bash -lc '
cd /workspace/isaaclab &&
./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py \
  --headless --task AIC-Task-v0 --num_envs 1 --seed 236 --device cuda:0 \
  --enable_cameras --rendering_mode performance \
  --output_dir aic/outputs/agent_reward_funnel/isaac_smoke \
  --run_name guarded_servo_smoke \
  --steps 120 --updates 120 --update_every_steps 100000 --warmup_steps 120 \
  --actor_update_start_steps 100000 --batch_size 32 \
  --act_only --act_only_actor_mode act_direct \
  --act_torchscript aic/outputs/train/clean_sfp_sc/act/bc/20260510_clean_act_nact8_400k/act_policy_ts_175000_cuda0.pt \
  --n_action_steps 1 --tcp_action_frame root \
  --reward_preset cheatcode_insertion_v1 \
  --disable_command_pose_rewards \
  --target_reward_body sfp_tip_link \
  --target_reward_orientation_error_mode axis \
  --target_reward_orientation_axis_local 0.0 0.0 1.0 \
  --target_reward_consistency_body auto \
  --target_reward_consistency_axial_std 0.004 \
  --target_reward_consistency_lateral_sigma 0.0015 \
  --target_success_consistency_axial_threshold 0.001 \
  --target_success_consistency_lateral_threshold 0.0015 \
  --target_action_guide_mode cheatcode_transform \
  --target_action_guide_collect_blend 1.0 \
  --target_action_guide_collect_steps 120 \
  --target_action_guide_step_size 0.00025 \
  --target_action_guide_rotation_step_size 0.0015 \
  --target_action_guide_orientation_switch_rad 0.03 \
  --target_action_guide_final_orientation_depth_m 0.006 \
  --target_action_guide_final_orientation_lateral_m 0.0006 \
  --target_action_guide_final_orientation_threshold_rad 0.03 \
  --target_action_guide_final_orientation_axial_step_size 0.0 \
  --insertion_action_guard \
  --insertion_action_guard_lateral_threshold_m 0.0006 \
  --insertion_action_guard_lateral_step_m 0.0002 \
  --insertion_action_guard_centered_axial_step_m 0.00004 \
  --insertion_action_guard_blocked_axial_step_m -0.00005 \
  --insertion_action_guard_zero_rotation_when_offcenter \
  --insertion_action_guard_retention \
  --debug_diagnostics --diagnostics_every 1 \
  --save_step_images --image_log_every 20 --max_logged_image_steps 120 \
  --debug_visual_overlays --max_wall_time_minutes 20
'
```
