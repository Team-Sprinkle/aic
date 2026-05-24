# Agentic Reward/Curriculum Audit 2026-05-23

Status: strict insertion has not been demonstrated. This audit records the current reward, curriculum, guide, evaluator, and run commands used for the 2026-05-23 agentic tuning loop.

## Current Controls

Reward geometry lives in `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/insertion_geometry.py`.
- `compute_insertion_geometry` computes signed axial depth `s`, lateral error `r`, depth fraction, and seated-depth consistency.
- `rewards.py` wires this into `body_to_object_cheatcode_phase_reward`, `body_to_object_success`, and `body_to_object_success_once_bonus`.
- Strict success can include `target_success_consistency_axial_threshold` and `target_success_consistency_lateral_threshold`; current runs use `sfp_tip_link` as the rewarded body and `sfp_module_link` through `--target_reward_consistency_body auto`.

Reward flags are exposed through `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`.
- Cheatcode funnel: `--reward_preset cheatcode_insertion_v1`, `--target_reward_cheatcode_*`, `--target_reward_cheatcode_action_axis_gate`, and `--target_reward_cheatcode_action_axis_source body_delta`.
- Semantic/module consistency: `--target_reward_consistency_body auto`, `--target_reward_consistency_axial_std`, `--target_reward_consistency_lateral_sigma`, `--target_success_consistency_axial_threshold`, `--target_success_consistency_lateral_threshold`.
- Orientation: `--target_reward_orientation_error_mode axis --target_reward_orientation_axis_local 0.0 0.0 1.0`. This uses semantic tip/body orientation rather than gripper quaternion alignment.

Curriculum episode generation is in `aic_utils/aic_isaac/scripts/isaac_episode_configs.py`.
- `materialize_episode_configs` and `materialize_many_episode_configs` consume YAML request files.
- `_apply_start_near_gate` materializes near-gate starts from `scene.start_near_gate`.
- Existing useful episode dirs:
  - `outputs/analysis/isaac_near_gate_handoff_r105/episode_configs/episodes`: final handoff, about 0.5 mm axial and 0.2 mm lateral.
  - `outputs/analysis/curriculum_insertion_20260515/stage_d_3_6mm_final/episode_configs/episodes`: final-stage starts, about 3-6 mm axial and 0-1.5 mm lateral.
  - `outputs/analysis/mixed_10_20_30_40_x_4_6_8_10_multgate_20260516/episode_configs_interleaved`: prior mixed long-start curriculum.

Guide/action guard/evaluator logic is in `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`.
- Guide flags: `--target_action_guide_mode cheatcode_transform`, `--target_action_guide_step_size`, `--target_action_guide_rotation_step_size`, `--target_action_guide_final_orientation_*`, `--target_action_guide_final_axis_only_orientation`, `--target_action_guide_separate_rotation_compensation`.
- Action guard flags: `--insertion_action_guard`, `--insertion_action_guard_zero_rotation_when_offcenter`, `--insertion_action_guard_retention`, `--insertion_action_guard_centered_axial_step_m`, `--insertion_action_guard_reject_predicted_r_increase`.
- New guard flag added here: `--insertion_action_guard_retention_require_orientation_gate`, which prevents retention-driven positive axial motion while semantic tip orientation is outside the orientation gate.
- New controller/orientation diagnostics added here: `--target_action_guide_adaptive_orientation_sign`, `--target_action_guide_orientation_probe_basis`, and `--debug_audit_rotation_axes`.
- New module recovery flags added here: `--insertion_action_guard_module_recovery` and `--insertion_action_guard_module_recovery_zero_rotation`.
- Diagnostics include `post_step_insertion_geometry`, `post_step_all_body_insertion_geometry`, `axis_alignment_realization`, `insertion_action_guard_*`, force/contact metrics, camera images, and `cheatcode_phase_summary.json`.

## Best Prior Checkpoints

- `outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517/best_semantic_progress10_poststep_finalrot_latest_checkpoint.pt`
- `outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517/best_semantic_progress7_final_hold_checkpoint.pt`
- `outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517/best_semantic_progress3_seat_step100_checkpoint.pt`
- Guided fallback run: `outputs/one_day_insertion_pipeline/orientation_calibration_20260519/runs/2026-05-19_23-22-46_guard_final_orientation_hold_full_2026-05-19_23-22-00`

Current best prior/fallback classification: `tip_depth_false_positive`. Best row had `s=0.0957 mm`, `r=0.950 mm`, `theta=0.0297 rad`, depth fraction `0.012`, module consistency `0.0187`, max force `33.83 N`, strict success false.

## Smoke/Short Commands

Compile and tests:

```bash
.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py aic_utils/aic_isaac/scripts/agentic_insertion_reward_curriculum_loop.py
.pixi/envs/default/bin/python -m pytest aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_agent_reward_funnel_scripts.py -q
```

Generate candidate commands without running Isaac:

```bash
.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/agentic_insertion_reward_curriculum_loop.py --max-iterations 4
```

Run bounded near-gate loop:

```bash
.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/agentic_insertion_reward_curriculum_loop.py --execute --max-iterations 1 --output-root outputs/agentic_reward_curriculum_20260523_next
```

Summarize an existing run:

```bash
.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/agentic_insertion_reward_curriculum_loop.py --summarize-run outputs/one_day_insertion_pipeline/orientation_calibration_20260519/runs/2026-05-19_23-22-46_guard_final_orientation_hold_full_2026-05-19_23-22-00
```

## Current Failure Modes

- Tip-depth-only false positives remain possible if module/body consistency is ignored.
- Retention without an orientation gate can push inward with tight `r` but bad `theta`; the 2026-05-23 near-gate smoke reached depth but stayed at `theta ~= 0.062 rad`.
- Enforcing orientation-gated retention correctly prevents that false positive, but then axial progress stalls because bounded final orientation refinement did not reduce `theta` below 0.030 rad.
- Axis-only final-window refinement slightly reduced best theta to `0.05897 rad` but worsened lateral/force behavior and still failed strict success.
- Controller-aware basis probing can reach full tip depth and tight `r`, but theta remains around `0.064 rad` and module consistency remains around `0.398`, so it is a strict false positive rather than success.
- Module recovery can now trigger and back out from false full-depth states when rotation is zeroed, but it has not improved theta or module consistency enough for strict success.
