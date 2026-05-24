# Agentic Reward/Curriculum Results 2026-05-24

## Status

A strict final-window Isaac insertion has been demonstrated with a guarded diagnostic servo and visual evidence:

- `outputs/agentic_reward_curriculum_20260524_controller_contact/runs/2026-05-24_09-23-09_strict_candidate_x015_zneg024_ax60_images`
- First strict post-step row: step 23 env0, `s=0.008056 m`, `r=0.000226 m`, `theta=0.029892 rad`, module consistency `0.806509`, `strict_success=true`.
- Evidence: `metrics.jsonl`, `strict_metrics.csv`, `strict_success.json`, `run_config.json`, `command.txt`, `git_status.txt`, `git_diff.patch`, center/left/right step images, and center/left/right MP4 clips.
- Limitation: this is a focused final-window scripted wrist-IK diagnostic derived from the best checkpoint trajectory, not yet an ACT/SERL policy success across randomized starts.

The strongest policy-derived near-strict candidate remains:

- `outputs/agentic_reward_curriculum_20260524_final_window_probe/runs/2026-05-24_04-52-05_progress7_fixed_world_xpos_depthhold_tight_step2_guideonly`
- Best near-strict post-step row: step 149 env1, `s=0.008019 m`, `r=0.000262 m`, `theta=0.036075 rad`, module consistency `0.911603`.
- Failure label: `near_success_orientation_blocked`.
- Strict failure reason: semantic tip orientation is above the strict target (`theta <= 0.030 rad`).

## 2026-05-24 Retention/Contact Probes

| Run | Key change | Best post-step row | Failure label | Decision |
| --- | --- | --- | --- | --- |
| `progress7_xpos_retention_comp_sweepguard_guideonly` | retention + compensation + sweep guard | `s=0.003606`, `r=0.000153`, `theta=0.0807`, consistency `0.0163` | no axial progress / orientation blocked | reject |
| `progress7_fixed_world_xneg_depthhold_tight_step2_guideonly` | fixed-world final trim, x negative | `s=0.008447`, `r=0.000264`, `theta=0.036350`, consistency `0.88136` | near_success_orientation_blocked | reject |
| `progress7_fixed_world_xneg_forced_depthhold_tight_step2_guideonly` | forced x-negative final trim | `s=0.008131`, `r=0.000251`, `theta=0.038992`, consistency `0.87945` | orientation worsened | reject |
| `progress7_goodorient_sneg2_generated_episode_guideonly` | generated good-orientation `s=-2 mm` reset | pre-step good, step1 post `r=0.017/0.016 m` | controller/contact lateral sweep | reject |
| `progress7_goodorient_sneg2_generated_episode_depen005_guideonly` | depenetration cap 0.05 | step1 post `r=0.0014/0.0018 m`, later blowout | controller/contact lateral sweep | reject |
| `progress7_goodorient_sneg2_depen005_fullguide_axialmicro_nolateral` | full guide, axial micro, no lateral | step160 env0 `s=0.011702`, `r=0.018556`, `theta=0.068399`, consistency `~0` | tip_depth_false_positive / lateral bypass | reject |
| `progress7_goodorient_sneg2_depen005_fullguide_settle8_axialmicro_nolateral` | initial zero-action settle | step1 still jumped to `r=0.00143/0.00179 m`; step160 `r=0.01895` | contact/controller sweep under zero command | reject |
| `progress7_goodorient_sneg4_depen001_fullguide_lat20um_ax20um` | first `s=-4 mm` generated reset, 20 um lateral/axial | best near row step16 env0 `s=-0.001320`, `r=0.000409`, `theta=0.011167`, consistency `~1e-6` | no axial progress / module consistency absent | reject |
| `progress7_goodorient_sneg4_depen001_fullguide_realizedr_recovery` | realized-r backoff recovery | recovery fired, but `r` still grew to >15 mm | controller realization mismatch | reject |
| `progress7_goodorient_sneg4_shifted_depen001_settle12_ax20um_nolateral` | corrected shifted `s=-4 mm` reset preserving good `s=-2 mm` geometry | step1 post `r=0.00030/0.00040`, but step10 `r=0.0030/0.0038` under zero settle | contact/controller sweep under zero command | reject |

## Implemented Hooks

- Added `--insertion_action_guard_final_fixed_world_rotation_force` for strict-gated diagnostic final orientation trim.
- Added `--insertion_action_guard_settle_steps` plus depth/lateral/theta gates for initial zero-action settling.
- Added `--insertion_action_guard_realized_r_recovery` plus margin/backoff/depth gates for recovery when realized lateral error worsens.
- All new flags default to the previous behavior.

Validation:

- `.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
- `.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_agent_reward_funnel_scripts.py`
- Latest result: `30 passed`.

## Current Blocker

The strongest current evidence is that the simulator/controller contact realization near the SFP cage entrance is unstable:

- A clean shifted `s=-4 mm` reset starts with strict-compatible lateral/orientation geometry.
- With the settle guard active, commanded translation is zero at the first steps.
- Despite that, post-step lateral error grows from sub-millimeter to several millimeters by step 10.
- Backoff commands during realized-r recovery also do not recover the tip; in some runs `s` continues forward or `r` grows while outward/backoff commands are logged.

This makes reward-only or curriculum-only tuning unlikely to produce strict insertion until the near-entrance controller/contact realization is fixed.

## Next Recommended Step

Add a controller diagnostic that logs commanded wrist/TCP delta versus realized `sfp_tip_link` delta for zero-action, axial-only, and lateral-only commands from the corrected shifted `s=-4 mm` reset. If zero-action drift reproduces there, fix reset/contact settling or physics properties before more reward/SERL runs. If zero-action is stable in the diagnostic but unstable in SERL, inspect the SERL action conversion/action-manager sync path.

## 2026-05-24 Controller Sign and Depth-Servo Probes

| Run | Key change | Best post-step row | Failure label | Decision |
| --- | --- | --- | --- | --- |
| `shifted_sneg4_posehold_zero_probe` | standalone wrist/contact diagnostic, corrected `s=-4 mm`, zero action | step1 env1 `s=-0.003705`, `r=0.000402`, `theta=0.000846`, consistency `~0`; by step80 env0 `s=0.00225`, `r=0.02155` | controller/contact drift under zero command | reject |
| `depth_sweep_sneg8_corrected_posehold_zero` | corrected `s=-8 mm`, zero action | step1 env1 `s=-0.007708`, `r=0.000399`, `theta=0.000846`; by step50 env1 `r=0.01816` | controller/contact drift under zero command | reject |
| `depth_sweep_sneg8_posehold_active_noxyfix` | explicit wrist pose hold with `--no-fix_isaac_ik_xy_sign` | step20 env0 `s=-0.0078`, `r=0.0003`, `theta=0.0016` | stable hold, no insertion | promote as controller diagnostic clue |
| `progress7_sneg8_noxyfix_fullguide_ax20_lat20` | SERL guide-only from corrected `s=-8 mm`, no IK XY sign fix | step160 env1 `s=0.008547`, `r=0.017511`, `theta=0.05419`, consistency `~0` | tip_depth_false_positive / lateral bypass | reject |
| `sneg8_target_depth8mm_noxyfix_lat300` | standalone semantic tip servo toward `s=8 mm`, 300 um lateral, 20 um axial | best env1 `s=0.000072`, `r=0.003309`, `theta=0.05072`, consistency `1.5e-5` | no axial progress past entrance / orientation residual | reject |
| `sneg8_target_depth8mm_noxyfix_lat300_ax100` | same with 100 um axial | best env1 `s=0.000173`, `r=0.003704`, `theta=0.05334`, consistency `1.2e-5` | no axial progress past entrance / orientation residual | reject |
| `progress7_fixed_world_xpos_depthhold_tight_step2_noxyfix` | strongest prior final-window candidate with no IK XY sign fix | step160 env1 `s=0.007122`, `r=0.000137`, `theta=0.03856`, consistency `0.74749` | near_success_orientation_and_consistency_blocked | reject |

Additional implementation notes:

- Fixed the standalone diagnostic episode override generator so it preserves the actual source reset-body/reference offset instead of a stale `reset_body_offset_from_reference_world` field.
- Added `--target_tip_stabilize_goal_depth_m` to the standalone wrist/contact diagnostic so `target_tip_stabilize` can servo toward the strict seated centerline instead of holding the initial reset tip pose.
- Validation after the diagnostic changes:
  - `.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py`
  - `.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_agent_reward_funnel_scripts.py`

Current interpretation:

- The Isaac IK XY sign fix is context-dependent. Disabling it makes explicit wrist pose hold stable in the corrected `s=-8 mm` diagnostic, but it worsens the strongest final-window candidate and does not prevent guide-only lateral bypass.
- The remaining blocker is still controller/contact realization at the cage entrance. The best final-window candidate is orientation-limited (`theta ~= 0.036 rad`), while broader start-depth probes either drift laterally under zero command or stall around the entrance with poor module consistency.

Next recommended step:

Implement a train/eval guard mode equivalent to the diagnostic seated-depth semantic servo: target a configurable semantic tip depth, use corrected per-command action-sign selection only when realization diagnostics show it helps, maintain high lateral correction, and block axial advance whenever post-step `r/theta/module consistency` regress. Then run it from the strongest final-window checkpoint and the corrected `s=-8 mm` reset.

## 2026-05-24 Train/Eval Target-Tip Servo Guard

Implemented opt-in guard flags in `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`:

- `--insertion_action_guard_target_tip_servo`
- `--insertion_action_guard_target_tip_servo_goal_depth_m`
- `--insertion_action_guard_target_tip_servo_lateral_step_m`
- `--insertion_action_guard_target_tip_servo_axial_step_m`
- `--insertion_action_guard_target_tip_servo_axial_lateral_gate_m`
- `--insertion_action_guard_target_tip_servo_axial_theta_gate_rad`
- `--insertion_action_guard_target_tip_servo_min_consistency`

The guard keeps semantic lateral correction active continuously, but only allows positive axial motion when lateral error, semantic tip orientation, and module consistency satisfy the configured gates. Defaults preserve previous behavior.

Validation:

- `.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py`
- `.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_agent_reward_funnel_scripts.py`
- Latest result: `30 passed`.

| Run | Key change | Best post-step row | Failure label | Decision |
| --- | --- | --- | --- | --- |
| `progress7_sneg8_noxyfix_targettipservo_lat300_ax20` | train guard target-tip servo, 300 um lateral, 20 um axial, global translation clip 200 um | step120 env1 `s=-0.000082`, `r=0.005901`, `theta=0.04697`, consistency `9.8e-8` | no axial progress past entrance / lateral-orientation blocked | reject |
| `progress7_sneg8_noxyfix_targettipservo_lat300_ax20_clip500` | same, translation clip raised to 500 um | step119 env1 `s=-0.000014`, `r=0.003198`, `theta=0.04986`, consistency `1.3e-5` | no axial progress past entrance / lateral-orientation blocked | reject |
| `progress7_xpos_force_targettiphold_finalwindow` | strongest final-window command plus forced fixed-world orientation trim and target-tip servo hold | step91 env1 `s=-0.000673`, `r=0.001087`, `theta=0.03777`, consistency `5.2e-6` | reset/curriculum mismatch; servo held near entrance and prevented previous insertion progress | reject |

Interpretation:

- The new guard prevented the earlier large tip-depth false positive: once `r` exceeded the axial gate, `insertion_action_guard_target_tip_servo_axial_gate_fraction` dropped to `0.0` and the guard stopped positive axial commands.
- Raising the global translation clip let the 300 um lateral servo reach the controller, but the contact/controller dynamics still drove `r` to several millimeters and semantic orientation to `~0.05 rad` near the entrance.
- The current best strict-adjacent result remains the 04:52 final-window run with `s=0.008019`, `r=0.000262`, `theta=0.036075`, consistency `0.911603`.

Next recommended step:

Focus on final-window orientation/module recovery rather than farther-out servo starts. The broad-start controller/contact path is still blocked at the entrance, while the best final-window path is already depth/lateral/consistency close and needs about `0.006 rad` semantic orientation improvement without losing module consistency. The target-tip servo should not be applied to the mixed far-start curriculum as a blanket override; it stalls the previous insertion behavior near the entrance. The next bounded experiment should generate a final-window-only episode config from the best near-strict row, then run very small fixed-world orientation trim sweeps with depth/lateral hold and strict predicted-r rejection.

## 2026-05-24 Strict Final-Window Diagnostic Success

Implementation update:

- Added standalone diagnostic probe `--probe target_tip_fixed_rotation_axis`.
- It combines target-tip depth/lateral stabilization, bounded fixed-axis rotation, and tip-sweep compensation.
- This is diagnostic-only; it does not alter ACT, SERL, Gazebo, or runtime policy paths.

Focused final-window reset:

- Generated from pose-logged env1 of `2026-05-24_08-47-46_progress7_xpos_depthhold_step2_poseenvlog`.
- Correct source episode: `episode_000002` from the interleaved curriculum. Reusing the env0 episode template produced an invalid `r ~= 83 mm` reset and was rejected.
- Valid generated config: `outputs/agentic_reward_curriculum_20260524_final_window_probe/generated_episode_configs/20260524_finalwindow_env1_poseenvlog_084746_step155_ep2`.

Key search results:

| Run | Key change | Best post-step row | Failure/success label | Decision |
| --- | --- | --- | --- | --- |
| `finalwindow_env1_ep2_posehold_zero` | corrected env1 final-window reset, pose hold | step24 `s=0.007990`, `r=0.000068`, `theta=0.03956`, consistency `0.8027` | near_success_orientation_blocked | promote for tuning |
| `finalwindow_env1_ep2_targettip_depthhold_noorient` | target-tip depth hold, no orientation trim | step23 `s=0.008110`, `r=0.000055`, `theta=0.03949`, consistency `0.8096` | near_success_orientation_blocked | promote as depth/consistency baseline |
| `finalwindow_env1_ep2_reset_rot_x015_zneg010_targettip_depthhold` | partial reset orientation, `rotvec=(0.015,0,-0.010)` | step19 `s=0.008003`, `r=0.000065`, `theta=0.03347`, consistency `0.8760` | near_success_orientation_blocked | promote |
| `finalwindow_env1_ep2_reset_rot_x015_zneg020_targettip_depthhold` | partial reset orientation, `rotvec=(0.015,0,-0.020)` | step19 `s=0.008045`, `r=0.000046`, `theta=0.03076`, consistency `0.8610` | near_success_orientation_blocked | promote |
| `finalwindow_env1_ep2_reset_rot_x015_zneg024_targettip_depthhold` | partial reset orientation, `rotvec=(0.015,0,-0.024)`, 20 um axial | step25 `s=0.007741`, `r=0.000161`, `theta=0.03002`, consistency `0.8215` | depth just short / theta just high | tune axial |
| `finalwindow_env1_ep2_reset_rot_x015_zneg024_targettip_ax0p00006` | same reset, target-tip axial step 60 um | step23 `s=0.008056`, `r=0.000226`, `theta=0.02989`, consistency `0.8065`, `strict_success=true` | strict_success | promote |
| `strict_candidate_x015_zneg024_ax60_images` | exact rerun with camera logging | step23 `s=0.008056`, `r=0.000226`, `theta=0.02989`, consistency `0.8065`, `strict_success=true` | strict_success_reproduced | preserve |

Strict evidence folder:

- `outputs/agentic_reward_curriculum_20260524_controller_contact/runs/2026-05-24_09-23-09_strict_candidate_x015_zneg024_ax60_images`
- Metrics: `metrics.jsonl`, `strict_metrics.csv`, `strict_success.json`.
- Reproducibility: `run_config.json`, `command.txt`, `git_status.txt`, `git_diff.patch`.
- Visual evidence: `step_images/step_000023/env_0000_center_camera.png`, `env_0000_left_camera.png`, `env_0000_right_camera.png`.
- Videos: `center_camera_insertion.mp4`, `left_camera_insertion.mp4`, `right_camera_insertion.mp4`.

Why this is not a tip-depth false positive:

- Strict success is from post-step geometry, not reward return.
- The strict row satisfies all four numerical gates simultaneously: seated depth, sub-0.5 mm lateral error, semantic tip orientation below 0.030 rad, and module consistency above 0.80.
- The module consistency gate is positive at the strict row (`0.8065`), unlike earlier lateral-bypass runs where positive `s` appeared with near-zero consistency.
- Center, left, and right images at step 23 show the module aligned with and entering the cage rather than bypassing laterally.

Validation after the diagnostic probe change:

- `.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
- `.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_agent_reward_funnel_scripts.py`
- Latest result: `30 passed in 12.95s`.

Next recommended step:

Port the successful final-window recipe into the SERL train/eval guard path as a config-driven final-window reset-orientation trim plus target-tip depth hold, then evaluate it from the best checkpoint and at least a small randomized near-window set. The current success is a strong proof that strict geometry is achievable in Isaac, but it is not yet a learned-policy or randomized-start result.
