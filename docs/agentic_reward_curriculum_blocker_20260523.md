# Agentic Reward/Curriculum Blocker 2026-05-23

Strict SFP-to-NIC insertion has not been demonstrated. The closest runs reach the configured seated semantic tip depth (`target_depth_m ~= 0.008`) with tight lateral error, but fail strict success on semantic tip orientation and `sfp_module_link`/module-body consistency. This is not a reward-return failure; it is a final-seat controller/contact/module-consistency blocker.

## Current Best Candidate

The strongest candidate remains the small-step basis orientation probe:

- Run: `outputs/agentic_reward_curriculum_20260523_probe_basis_small/runs/2026-05-23_16-18-19_r105_probe_basis_rot001_ax5um`
- Best post-step `s`: `8.190 mm` (`depth_fraction=1.0`)
- Best post-step `r`: `0.130 mm`
- Best post-step `theta`: `0.0641 rad`
- Best module consistency: `0.3978`
- Strict success: `false`
- Failure: near-full semantic tip depth with bad orientation and bad module consistency.

This is a near-seated false positive. It satisfies tip-depth and lateral-error criteria, but it misses the strict orientation target (`<0.030 rad`) by about `0.034 rad` and misses the module-consistency target (`>=0.80`) by about `0.40`.

## Recovery Experiments

| run | best s mm | best r mm | best theta rad | best module consistency | final state | strict |
|---|---:|---:|---:|---:|---|---|
| one-step module recovery, zero rotation | 8.022 | 0.080 | 0.0659 | 0.3713 | recovery override | false |
| state machine, 2mm backout target | 8.022 | 0.080 | 0.0659 | 0.3713 | backout | false |
| state machine, 7mm fast backout | 7.573 | 0.110 | 0.0639 | 0.4011 | trim | false |
| state machine, 7mm short hold | 7.573 | 0.110 | 0.0639 | 0.4011 | reinsert/hold | false |

The persistent recovery state machine verified that backout/trim/reinsert states can be commanded and logged. It did not improve the strict blocker. In the short-hold run, recovery reached reinsert/hold, but final metrics were still `s=6.493 mm`, `r=0.115 mm`, `theta=0.0721 rad`, and module consistency `0.254`.

## Classification

Primary label: `near_success_module_consistency_blocked`

Secondary labels:
- `near_success_orientation_blocked`
- `tip_depth_false_positive`
- `controller_realization_mismatch`

The repeated pattern is: semantic tip depth and lateral centering are achievable, but the final contact/servo behavior does not rotate/seat the semantic module body into the strict target. Backout and reinsertion do not recover the module consistency gap.

## Why Reward/Curriculum Is Not The Next Lever

The reward already penalizes the known false positive through orientation and semantic/module consistency gates. The guide/evaluator can produce full tip depth, but the controller cannot realize the remaining orientation/module seating. Longer SERL from this state is likely to reinforce a false-positive strategy unless the low-level final-seat behavior is fixed first.

## Recommended Next Code Change

Stop reward-only tuning and add a controller diagnostic that measures realized semantic tip and module motion from pure bounded wrist-frame rotations/translations while in shallow contact. The diagnostic should identify whether any admissible wrist command can reduce `theta` below `0.030 rad` without increasing `r` or lowering module consistency.

If no such local command exists, the likely blocker is asset/contact/evaluator alignment rather than reward/curriculum. If such a command exists, convert it into a guarded final-seat servo primitive and only then resume reward/curriculum tuning.

## Contact Command Diagnostic Update

Two staged contact diagnostics were added after this blocker was first written:

- Rotation probe: `outputs/agentic_reward_curriculum_20260523_contact_diagnostic/runs/2026-05-23_18-28-09_r105_staged_rotation_axis_probe`
- Translation probe: `outputs/agentic_reward_curriculum_20260523_contact_diagnostic/runs/2026-05-23_18-31-13_r105_staged_translation_axis_probe`

Both runs used the guide/guard until step 154, then switched to pure bounded audit actions at step 155. The handoff state was already near seated: `s=7.637 mm`, `r=0.038 mm`, `theta=0.0638 rad`, module consistency `0.380`.

Rotation audit result: no +/- root rotation axis improved `theta`; the best audit theta was `0.0643 rad`, already worse than the handoff value. By step 180, `theta` worsened to `0.0687 rad` and module consistency fell to `0.355`.

Translation audit result: tiny root translations preserved/increased semantic depth but still worsened `theta`; final `theta=0.0678 rad`, module consistency `0.390`, `r=0.220 mm`.

The diagnostic suggests the remaining module-consistency gap is not solved by naive wrist rotations/translations in contact. The next bounded code change is a semantic `sfp_module_link` lateral-alignment guard: when the tip is centered and near seated, command a tiny module-body lateral correction while rejecting commands that increase tip lateral error or semantic orientation error.

## Module-Lateral Guard Update

A semantic `sfp_module_link` lateral-alignment guard was added and tested after the contact diagnostic. It is config-driven and only activates when the semantic tip is centered near the final window. It logs activation, module lateral error, predicted tip-r rejection, and post-step strict geometry.

Runs:

| run | best s mm | best r mm | best theta rad | best module consistency | note |
|---|---:|---:|---:|---:|---|
| `2026-05-23_18-42-58_r105_module_lateral_align_priority_smoke` | 8.386 | 0.229 | 0.0636 | 0.4436 | best module consistency so far, still strict false |
| `2026-05-23_18-47-15_r105_module_lateral_hold72_smoke` | 8.747 | 0.227 | 0.0634 | 0.4365 | hold depth did not prevent inward drift |
| `2026-05-23_18-49-57_r105_module_lateral_hold68_backoff200_smoke` | 7.552 | 0.232 | 0.0660 | 0.3893 | stronger backoff held shallower depth but did not improve consistency |
| `2026-05-23_18-53-01_r105_module_lateral_hold68_backoff200_signneg_smoke` | 7.405 | 0.235 | 0.0670 | 0.3688 | opposite lateral sign was worse |

The guard-order bug is fixed: module-lateral commands are no longer overwritten by retention/final-orientation overrides. The remaining failure is not a simple sign issue. With the nominal sign, module consistency improves only to `0.4436`; with shallower hold/backoff, the realized controller motion trades away depth and worsens orientation without moving module consistency toward the strict `>=0.80` gate.

Updated classification:
- Primary: `controller_realization_mismatch`
- Secondary: `near_success_module_consistency_blocked`, `near_success_orientation_blocked`, `contact_spike`

Updated recommendation: add a module-motion realization diagnostic before more reward or curriculum tuning. It should log commanded versus realized `sfp_tip_link` and `sfp_module_link` deltas and the projection of realized module motion onto the desired module-lateral correction. If the module body cannot be moved in the desired direction while the tip is in shallow contact, the next blocker is likely IK/contact/asset geometry rather than reward design.

## 2026-05-23 Late Iteration Update

The module-motion realization diagnostic was added to the guard metrics and used in the 19:16 UTC `rot006_long` run. It showed a controller/contact mismatch: the guard predicted about `142 um` of relative module correction per step, but after contact the realized `sfp_module_link - sfp_tip_link` projection collapsed to approximately zero. Despite this, the run reached full configured semantic tip depth:

- Run: `outputs/agentic_reward_curriculum_20260523_module_lateral/runs/2026-05-23_19-16-02_r105_module_lateral_rotation_align_rot006_long`
- Final/best `s=8.214 mm`
- Final `r=0.222 mm`
- Final `theta=0.06096 rad`
- `success_geometry_by_env=true`, `strict_partial_insertion_by_env=true`, explicit consistency thresholds pass
- Strict success remains false because `theta > 0.030 rad`

Follow-up bounded probes:

| probe | result |
|---|---|
| earlier module-lateral activation at 5.5 mm | stalled at `s=6.742 mm`, `theta=0.0610` |
| disabled predicted-r rejection | compensation executed but stalled at `s=7.197 mm`, `theta=0.0613` |
| delayed full-quat final trim at 7.6 mm | never reached activation; best `s=6.832 mm` |
| delayed full-quat final trim at 6.5 mm | best `s=6.561 mm`, `theta=0.0606` |
| semantic final-axis rotation guard, broad lateral gate | briefly `theta=0.0589`, but lateral sweep to about `0.95 mm` |
| semantic final-axis rotation guard, `r<=0.3 mm` | avoided broad sweep but stalled at `s=5.688 mm`, `theta=0.0607` |
| original adapter-policy eval path | exercised checkpoint adapter but stalled at `s≈4.6-4.8 mm`, `theta≈0.0607` |

Updated classification:
- Primary: `near_success_orientation_blocked`
- Secondary: `controller_realization_mismatch`, `rotation_induced_lateral_sweep`, `no_axial_progress`

The closest current artifact is no longer a tip-depth-only false positive: it satisfies full semantic depth and lateral geometry, but it is still not strict insertion because the semantic tip axis remains about `0.031 rad` above the strict threshold. The next bounded path is to reproduce the original longer May 17 adapter-training run from `best_semantic_progress7_final_hold_checkpoint.pt`, because short no-learning adapter eval did not reproduce the previously reported `theta≈0.036-0.037` near-success. If that does not recover a lower-theta candidate, the evidence points to controller/contact realization of semantic-axis rotation as the blocker.

## 2026-05-23 Separate Video and Historical Reproduction

A separate video artifact was generated without disturbing the tuning runs:

- Run: `outputs/agentic_reward_curriculum_20260523_video/runs/2026-05-23_19-35-07_r105_best_checkpoint_rot006_video`
- Videos: `videos/env0_center_camera_h264.mp4`, `videos/env0_left_camera_h264.mp4`, `videos/env0_right_camera_h264.mp4`
- Metrics: `s=8.214 mm`, `r=0.222 mm`, `theta=0.06096 rad`, smooth module consistency `0.5049`

This is useful visual evidence of the best current full-depth attempt, but it is not strict success because the semantic orientation and module consistency remain outside strict thresholds.

The bounded historical reproduction from `best_semantic_progress7_final_hold_checkpoint.pt` was also run with the May 17 mixed curriculum and loose reward schedule:

- Run: `outputs/agentic_reward_curriculum_20260523_reproduce_may17/runs/2026-05-23_20-38-14_progress10_repro220_historical_curriculum_reward`
- Best env0: `s=4.579 mm`, `r=0.230 mm`, `theta=0.05607 rad`, consistency `0.108`
- Best env1: `s=5.226 mm`, `r=0.275 mm`, `theta=0.03633 rad`, consistency `0.257`
- No row reached `s>=7 mm`; strict success count was zero.

This failed reproduction keeps the blocker classification unchanged. The strongest evidence remains that full semantic tip depth and lateral centering are achievable, but orientation/module-body realization under contact is not solved by the current guide, reward, short adapter eval, or bounded reproduction.

## 2026-05-23 Historical No-Compensation and Depth-Hold Update

The historical full-depth candidate was recovered only after disabling the current final-orientation compensation path:

- `outputs/agentic_reward_curriculum_20260523_reproduce_may17/runs/2026-05-23_21-13-39_progress10_repro220_no_final_comp`
- Best env1: `s=10.286 mm`, `r=0.279 mm`, `theta=0.03638 rad`, smooth module consistency `0.409`.
- Seated-window rows had good depth/lateral/module consistency, for example `s=8.445 mm`, `r=0.276 mm`, `theta=0.03662 rad`, consistency `0.887`.
- Strict success remained false because theta stayed above `0.030 rad`.

A longer 16 s no-compensation run confirmed that extra horizon does not solve theta and instead over-inserts:

- `outputs/agentic_reward_curriculum_20260523_reproduce_may17/runs/2026-05-23_21-55-42_progress10_no_comp_ep16s`
- Best seated row: `s=8.071 mm`, `r=0.276 mm`, `theta=0.03666 rad`, consistency `0.909`.
- Later best-depth row: `s=13.887 mm`, `r=0.296 mm`, `theta=0.03633 rad`, consistency `0.0039`.

A new config-driven final seated-depth hold was implemented and validated. With final orientation delayed to 7.5 mm, the hold preserves the best low-theta/full-depth state:

- `outputs/agentic_reward_curriculum_20260523_depth_hold/runs/2026-05-23_22-34-03_progress10_no_comp_depth_hold8mm_latefinal`
- Best env1: `s=8.034 mm`, `r=0.284 mm`, `theta=0.03643 rad`, consistency `0.917`.
- `success_geometry_by_env=true`, depth hold active, strict false only because `theta > 0.030 rad`.

Updated blocker classification:
- Primary: `near_success_orientation_blocked`
- Secondary: `controller_realization_mismatch`
- Resolved for this trajectory: module consistency collapse from over-insertion, when the final-depth hold is active.

Next single code change: add or tune a very small final-window orientation trim that only activates with `s>=7.5 mm`, `r<=0.5 mm`, and module consistency passing, while the final-depth hold is active. Reject any candidate trim that increases lateral error or reduces module consistency. Do not loosen the strict theta threshold.

## 2026-05-24 Strict Final-Axis Gate Update

A strict-gated final-axis trim was implemented in `train.py`, disabled by default:

- `--insertion_action_guard_final_axis_alignment_strict_gate`
- `--insertion_action_guard_final_axis_alignment_strict_min_consistency`
- `--insertion_action_guard_final_axis_alignment_strict_max_lateral_m`
- `--insertion_action_guard_final_axis_alignment_strict_lateral_margin_m`
- `--insertion_action_guard_final_axis_alignment_strict_max_depth_m`
- `--insertion_action_guard_final_axis_alignment_strict_require_depth_hold`

Validation passed:

- `.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
- `.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_agent_reward_funnel_scripts.py`
- Result: `30 passed`

The latest controlled continuation run did not improve strict success:

- Run: `outputs/agentic_reward_curriculum_20260523_orientation_trim/runs/2026-05-23_23-27-52_progress10_hold8_latefinal_continue_best`
- Best seated row: `s=8.076 mm`, `r=0.283 mm`, `theta=0.03639 rad`, consistency `0.918`
- Strict success count: `0`
- Classification: `near_success_orientation_blocked`

Two strict-axis probes were then run:

| probe | result |
|---|---|
| positive sign, `0.001 rad` | trim activated, but seated theta worsened to `0.04077 rad`; later over-inserted to `s=14.935 mm` and consistency collapsed to `0.00046` |
| negative sign, `0.0005 rad` | trim never activated because predicted axis error did not improve; best seated row `s=8.907 mm`, `r=0.271 mm`, `theta=0.03685 rad`, consistency `0.802` |

Updated classification:
- Primary: `near_success_orientation_blocked`
- Secondary: `controller_realization_mismatch`, `rotation_realization_overinsert`
- Rejected path: more final-axis rotation authority or simple sign flipping.

Updated next step: implement a bounded seated-window recovery/settling policy instead of more rotation. When `s` is seated but `theta > 0.030`, back out to shallow positive depth with zero rotation, settle lateral/module consistency, then reinsert with zero rotation and the final-depth hold active. If theta remains at `~0.036 rad` after this recovery, the blocker is likely asset/contact/IK realization rather than reward/curriculum.

## 2026-05-24 Seated Recovery Probe

The first seated-window recovery probe used the existing module-recovery state machine from the low-theta depth-hold checkpoint:

- Run: `outputs/agentic_reward_curriculum_20260523_seated_recovery/runs/2026-05-24_00-25-35_progress10_hold8_seated_recovery_target65`
- Trigger: `s>=7.9 mm`, `theta>0.030`, or consistency `<0.90`
- Target backout depth: `6.5 mm`
- Backoff step: `80 um`
- Rotation zeroed during recovery

Result:

- Strict success count: `0`
- Recovery activated and entered backout (`max_recovery=1.0`, `max_backout=1.0`)
- It never reached trim or reinsert (`max_trim=0`, `max_reinsert=0`)
- Best seated row: `s=8.066 mm`, `r=0.180 mm`, `theta=0.04142 rad`, consistency `0.823`
- Final rows remained non-strict; env1 ended over-inserted at `s=11.623 mm`, consistency `0.089`

Updated next step: run one stronger recovery realization probe with a much larger outward backoff and a shallower target. If that still cannot reduce `s` reliably while preserving `r`, stop guide/reward tuning and classify the remaining blocker as controller/contact realization under seated contact.

## 2026-05-24 Strong Recovery Realization Probe

The stronger recovery probe used the same low-theta depth-hold checkpoint but targeted a much shallower positive depth:

- Run: `outputs/agentic_reward_curriculum_20260523_seated_recovery/runs/2026-05-24_00-41-42_progress10_hold8_seated_recovery_target40_backoff300`
- Trigger: `s>=7.9 mm`, `theta>0.030`, or consistency `<0.90`
- Target backout depth: `4.0 mm`
- Backoff step: `300 um`
- Rotation zeroed during recovery

Result:

- Strict success count: `0`
- Recovery activated and entered backout (`max_recovery=1.0`, `max_backout=1.0`)
- It still never reached trim or reinsert (`max_trim=0`, `max_reinsert=0`)
- Best seated row: `s=8.027 mm`, `r=0.218 mm`, `theta=0.04143 rad`, consistency `0.823`
- Best seated consistency row: `s=8.117 mm`, `r=0.163 mm`, `theta=0.04163 rad`, consistency `0.835`
- Best depth row over-inserted to `s=10.556 mm`, `r=0.105 mm`, `theta=0.05340 rad`, consistency `0.248`

Updated blocker classification:
- Primary: `near_success_orientation_blocked`
- Secondary: `controller_realization_mismatch`
- Additional: `recovery_backout_realization_mismatch`

Conclusion: the current bounded reward/curriculum/final-guard search has produced a credible near-seated candidate but not strict insertion. The remaining gap is not reward return or axial depth. It is semantic orientation under seated contact, plus inconsistent realization of recovery/backout commands once the module is in contact. The next single code change should be a controller/contact realization diagnostic that logs commanded versus realized `sfp_tip_link` and `sfp_module_link` deltas during seated backout and semantic-axis correction. If that diagnostic confirms commanded outward or axis-correction motion is not realized under contact, the next work item should move to controller/contact/asset handling rather than additional reward tuning.

## 2026-05-24 Controller/Contact Realization Diagnostic

New diagnostics were added to `train.py`:

- `insertion_action_guard_guarded_world_delta_env0`
- `insertion_action_guard_guarded_world_delta_by_env`
- `diagnostics.guarded_command_realization`
- top-level realized projection metrics for tip and module motion along the guarded command

Validation passed:

- `.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
- `.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_agent_reward_funnel_scripts.py`
- Result: `30 passed`

Diagnostic runs:

| run | result |
|---|---|
| `2026-05-24_01-03-03_hold8_recovery_target40_command_realization_160` | Recovery commanded negative axial backout, averaging `-0.233 mm` during active recovery and reaching `-0.300 mm` per step, but signed depth still increased. Env1 went from seated `s=8.027 mm` to `s=10.346 mm`; consistency fell to `0.304`. |
| `2026-05-24_01-08-11_hold8_recovery_target40_pure_axial_backout` | Backout lateral correction disabled. Near-zero lateral plus `-0.300 mm` axial still did not retract; env1 ended at `s=8.404 mm`, `theta=0.04570 rad`, consistency `0.688`. |
| `2026-05-24_01-12-24_hold8_recovery_target40_pure_axial_backout_signpos` | Opposite axial sign drove deeper insertion, reaching `s=13.529 mm` and consistency `0.006`. |
| `2026-05-24_01-15-39_hold8_recovery_target40_rootframe_pure_axial_backout` | Root-frame pure axial backout also failed to reduce `s`; env1 ended at `s=8.270 mm`, `theta=0.04444 rad`, consistency `0.723`. |

Updated blocker classification:

- Primary: `controller_realization_mismatch`
- Secondary: `near_success_orientation_blocked`
- Additional: `recovery_backout_realization_mismatch`

This is now stronger than a reward/curriculum blocker. In seated or shallow-contact states, the guard can request outward axial motion, and the sign is not ambiguous: positive axial moves deeper, negative axial does not retract under contact. The next recommended code path is not another reward sweep. It is a controller/contact/asset diagnostic outside the learning loop: hold a shallow-contact state, send direct wrist pose/IK outward deltas in both frames, and measure semantic `sfp_tip_link` and `sfp_module_link` depth changes. If direct wrist pose targets cannot retract either, inspect port/module collision geometry, contact solver settings, module articulation/cable constraints, and whether the configured insertion axis/entrance frame is physically compatible with the collision model.

## 2026-05-24 Final Orientation Exhaustion Update

Additional bounded final-orientation and depth-hold experiments were run after the controller/contact diagnostic:

| run | best strict-relevant row | result |
|---|---|---|
| `2026-05-24_01-25-34_guarded_approach_direct_axis_backout_start105` | `s=7.988 mm`, `r=0.265 mm`, `theta=0.04115 rad`, consistency `0.819` | Direct semantic-axis backout after guarded approach did not retract from contact; final theta and consistency worsened. |
| `2026-05-24_01-30-40_strict_final_axis_trim_rot0015` | `s=8.463 mm`, `r=0.203 mm`, `theta=0.04171 rad`, consistency `0.817` | Positive strict final-axis trim did not improve theta and later over-inserted. |
| `2026-05-24_01-34-27_strict_final_axis_trim_rot0015_signneg` | `s=8.088 mm`, `r=0.163 mm`, `theta=0.04177 rad`, consistency `0.833` | Opposite sign did not activate useful trim. |
| `2026-05-24_01-37-42_progress7_orientation_basis_depth_hold8mm` | `s=8.018 mm`, `r=0.205 mm`, `theta=0.04143 rad`, consistency `0.837` | Kinematic basis prediction was not reliable under contact. |
| `2026-05-24_01-52-59_guarded_approach_rotation_axis_audit_start103` | `s=8.095 mm`, `r=0.350 mm`, `theta=0.03880 rad`, consistency `0.777` | Empirical pure rotations reduced theta modestly but did not satisfy orientation or consistency together. |
| `2026-05-24_01-56-56_progress7_no_basis_depth_hold8mm_extended320` | `s=8.683 mm`, `r=0.267 mm`, `theta=0.03743 rad`, consistency `0.802` | Longer hold kept valid geometry briefly but theta plateaued above strict; later over-insertion destroyed consistency. |
| `2026-05-24_02-17-14_progress7_early_orientation_hold76mm` | `s=8.911 mm`, `r=0.277 mm`, `theta=0.03682 rad`, consistency `0.805` | Earlier orientation hold did not reduce theta below `0.030`; later consistency collapsed. |

Updated blocker classification:

- Primary: `near_success_orientation_blocked`
- Secondary: `controller_realization_mismatch`
- Additional: `recovery_backout_realization_mismatch`
- Additional: `orientation_plateau_env_or_card_dependent`

The best candidate is still the 8 mm depth-hold run from 2026-05-23: `s=8.034 mm`, `r=0.284 mm`, `theta=0.03643 rad`, consistency `0.917`. The remaining gap to strict success is about `0.0064 rad` of semantic tip orientation. Further reward/curriculum/guide sweeps in this branch have not moved that gap below `0.030 rad` without losing module consistency.

Recommended next single code change: add a standalone controller/contact realization diagnostic that starts from shallow positive insertion, bypasses SERL, sends direct wrist pose deltas/rotations through the same wrist IK controller, and logs realized semantic tip/module motion plus contacts. If direct wrist pose targets cannot reduce theta or retract under contact, inspect asset/contact geometry and solver/controller settings before running more reward experiments.

## 2026-05-24 Reusable Controller/Contact Probe and Pre-seat Gate Test

Implemented `aic_utils/aic_isaac/scripts/controller_contact_realization_diagnostic.py` to make the controller/contact probe reproducible. The script reuses a known near-seated `train_config.json`, launches staged debug-audit cases, writes `agent_decision.json`, parses all-env post-step semantic metrics, and writes `controller_contact_summary.json` plus a markdown summary.

Validation:

- `.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/scripts/controller_contact_realization_diagnostic.py aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
- `.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_agent_reward_funnel_scripts.py`
- Result: `30 passed`

Diagnostic runs in `outputs/agentic_reward_curriculum_20260524_controller_contact/`:

| run | best row | failure |
|---|---|---|
| `semantic_axis_backout_300um_start105` | `s=8.028 mm`, `r=0.236 mm`, `theta=0.04154 rad`, consistency `0.847` | Backout produced only `-0.007 mm` final depth change and worsened theta/consistency. |
| `pure_rotation_axes_4mrad_start103` | `s=8.106 mm`, `r=0.303 mm`, `theta=0.04018 rad`, consistency `0.778` | No empirical wrist rotation axis reached strict theta; the best theta row failed consistency. |
| `semantic_axis_forward_100um_start105` | `s=8.388 mm`, `r=0.243 mm`, `theta=0.04127 rad`, consistency `0.816` | Inward command increased depth but worsened theta and consistency, confirming tip-depth-only false-positive risk. |
| `shallow_rotation_axes_4mrad_start90` | `s=8.010 mm`, `r=0.257 mm`, `theta=0.04521 rad`, consistency `0.826` | Starting from shallow positive insertion did not make orientation correction tractable; rotations worsened theta. |
| `shallow_semantic_axis_backout_200um_start90` | `s=7.768 mm`, `r=0.082 mm`, `theta=0.04374 rad`, consistency `0.777` | Shallow backout still realized as additional insertion; final `ds=+2.858 mm`. |

Also tested a config-only tighter pre-seat orientation gate:

- Run: `outputs/agentic_reward_curriculum_20260524_preseat_orientation_gate/runs/2026-05-24_02-46-07_preseat_orientgate035_finaldepth65_ax10um`
- Main changes: `--target_action_guide_orientation_switch_rad 0.035`, final orientation from `6.5 mm`, final threshold `0.030`, centered axial step `10 um`.
- Final env1: `s=-0.336 mm`, `r=1.499 mm`, `theta=0.03704 rad`, consistency approximately `0`.
- Strict success count: `0`.

Updated blocker classification:

- Primary: `controller_realization_mismatch`
- Secondary: `near_success_orientation_blocked`
- Additional: `recovery_backout_realization_mismatch`
- Additional: `preseat_orientation_gate_stalls_approach`

The current evidence supports this interpretation: if the guide/guard seats the module, semantic theta plateaus above the strict threshold and final-contact commands cannot fix it; if axial insertion is gated tightly on pre-seat orientation, the approach stalls before module consistency is established. Shallow positive insertion probes now show the same command-realization problem before full seating. The next single code path should be a wrist-pose/controller diagnostic outside SERL, not another reward sweep. It should initialize or drive to `s≈4-7 mm`, send direct wrist pose targets/deltas through the same IK/controller path, and log realized `sfp_tip_link` and `sfp_module_link` depth/lateral/theta plus contacts. If that diagnostic can rotate/retract at shallow contact, implement a pre-contact orientation servo phase. If it cannot, inspect collision geometry, solver/contact settings, and module/cable constraints.

## 2026-05-24 Fixed-axis final trim follow-up

A final bounded sweep tested whether the last `theta≈0.036 rad` gap could be closed by a strict-gated world-axis wrist trim with tip-motion compensation. A new config-driven guard path was added:

- `--insertion_action_guard_final_fixed_world_rotation`
- `--insertion_action_guard_final_fixed_world_rotation_axis {x,y,z,best}`
- `--insertion_action_guard_final_fixed_world_rotation_sign`
- `--insertion_action_guard_final_fixed_world_rotation_step_rad`
- `--insertion_action_guard_final_fixed_world_rotation_compensation_clip_m`

Validation passed with `30 passed` across the insertion reward geometry and agent reward funnel script tests.

Evidence:

| run | best row | failure label |
|---|---|---|
| `progress7_guard_finalaxis_depthhold_tight_step2_signneg_guideonly` | `s=8.298 mm`, `r=0.277 mm`, `theta=0.03631 rad`, consistency `0.906` | `near_success_orientation_blocked` |
| `progress7_fixed_world_ypos_depthhold_tight_step2_guideonly` | `s=8.160 mm`, `r=0.276 mm`, `theta=0.03848 rad`, consistency `0.886` | `orientation_plateau_env_or_card_dependent` |
| `progress7_fixed_world_yneg_depthhold_tight_step2_guideonly` | `s=8.298 mm`, `r=0.277 mm`, `theta=0.03631 rad`, consistency `0.906` | `near_success_orientation_blocked` |
| `progress7_fixed_world_best_depthhold_tight_step2_guideonly` | `s=8.249 mm`, `r=0.267 mm`, `theta=0.03754 rad`, consistency `0.888` | `orientation_plateau_env_or_card_dependent` |
| `progress7_fixed_world_best_depthhold_tight_step05_guideonly` | `s=8.245 mm`, `r=0.242 mm`, `theta=0.03802 rad`, consistency `0.808` | `near_success_orientation_blocked` |

The sweep did not produce strict success. It also did not improve on the best known near-seat row. The remaining best valid candidate is still:

- Run: `outputs/agentic_reward_curriculum_20260523_depth_hold/runs/2026-05-23_22-34-03_progress10_no_comp_depth_hold8mm_latefinal`
- Metrics: `s=8.034 mm`, `r=0.284 mm`, `theta=0.03643 rad`, consistency `0.917`
- Strict gap: `theta` remains about `0.0064 rad` above the `0.030 rad` threshold.

Visual artifacts were generated separately for the best env1 near-seat sequence:

- `outputs/agentic_reward_curriculum_20260523_depth_hold/runs/2026-05-23_22-34-03_progress10_no_comp_depth_hold8mm_latefinal/videos/env1_center_camera_h264.mp4`
- `outputs/agentic_reward_curriculum_20260523_depth_hold/runs/2026-05-23_22-34-03_progress10_no_comp_depth_hold8mm_latefinal/videos/env1_left_camera_h264.mp4`
- `outputs/agentic_reward_curriculum_20260523_depth_hold/runs/2026-05-23_22-34-03_progress10_no_comp_depth_hold8mm_latefinal/videos/env1_right_camera_h264.mp4`

Updated blocker classification:

- Primary: `near_success_orientation_blocked`
- Secondary: `controller_realization_mismatch`
- Additional: `orientation_plateau_env_or_card_dependent`
- Additional: `tip_depth_false_positive` risk remains controlled by consistency/theta checks.

Recommended next code path remains outside reward tuning: build or extend the standalone wrist-pose/controller diagnostic so it sends direct wrist pose targets at shallow positive insertion and logs semantic tip/module geometry plus contact state. If direct wrist pose targets cannot reduce theta below `0.030 rad` without module consistency collapse, the likely blocker is asset/contact/controller realization rather than reward/curriculum.

## 2026-05-24 Standalone wrist reset/contact diagnostic

The standalone wrist/contact diagnostic was extended with a closed-loop direct wrist orientation servo and generated episode reset overrides:

- `--probe orientation_servo_best`
- `--override_start_signed_depth_m`
- `--override_start_lateral_m`
- `--override_start_orientation_wxyz`

Validation passed: `30 passed`.

Evidence:

| run | best row | failure label |
|---|---|---|
| `wrist_orientation_servo_best_shallow5_step2` | `s=3.219 mm`, `r=0.331 mm`, `theta=0.06556 rad`, consistency `0.012` | `controller_realization_mismatch` |
| `wrist_orientation_servo_best_shallow6_step2_longapproach` | `s=-1.326 mm`, `r=0.472 mm`, `theta=0.07222 rad`, consistency `0.000` | `reset_regression` / `controller_realization_mismatch` |
| `wrist_orientation_servo_best_reset_s4mm_step2` | `s=3.116 mm`, `r=0.551 mm`, `theta=0.06883 rad`, consistency `0.010` | `near_success_module_consistency_blocked` at shallow reset |
| `wrist_reset_s8mm_zero_probe` | `s=7.990 mm`, `r=0.196 mm`, `theta=0.07079 rad`, consistency `0.561` | `tip_depth_false_positive` risk: depth/lateral pass but theta/consistency fail |
| `wrist_reset_s8mm_targetquat_zero_probe` | `s=6.648 mm`, `r=14.735 mm`, `theta=0.29766 rad`, consistency `0.000` | invalid gripper reset orientation |
| `wrist_orientation_servo_best_reset_s8mm_step4` | `s=8.051 mm`, `r=0.284 mm`, `theta=0.07100 rad`, consistency `0.613` | `controller_realization_mismatch` / `near_success_orientation_blocked` |

This strengthens the blocker classification. Direct reset can place the semantic tip near strict seated depth and lateral error, but not with valid semantic orientation or module/body consistency. A direct wrist orientation servo under that seated reset only marginally improves theta and remains far from `0.030 rad`. The best guided-policy near-seat row is still better than the direct reset rows, which means the issue is not simply that reward tuning has not found an obvious controller action.

Updated blocker classification:

- Primary: `controller_realization_mismatch`
- Secondary: `near_success_orientation_blocked`
- Additional: `module_consistency_failure`
- Additional: `tip_depth_false_positive`

Next recommended code change: derive or solve a reset/controller target for the gripper from the desired semantic `sfp_tip_link` pose, rather than copying the outside-gate gripper orientation into shallow/seated generated resets. Then rerun the direct wrist reset diagnostic. If the derived gripper reset still cannot place `sfp_tip_link` and `sfp_module_link` into a consistent seated pose, inspect collision geometry, cable/module constraints, and solver/contact settings before further reward/curriculum experiments.

## 2026-05-24 Updated blocker after reset diagnostics and fixed-x sweep

New instrumentation:

- `wrist_contact_realization.py` now writes `reset_diagnostic.json` with episode reset targets, realized `gripper_tcp` / `sfp_tip_link` / `sfp_module_link` poses, reset insertion geometry, module geometry, and contact summary.

Most important new evidence:

- Tight seated reset (`wrist_reset_s8mm_tightik_resetdiag_zero_probe`) places `gripper_tcp` accurately and places `sfp_tip_link` at `s≈8.00 mm`, `r≈0.30 mm`, but reset semantic tip orientation is already `theta≈0.0706 rad` and module consistency is only `0.60-0.62`.
- Small reset orientation offsets do not solve this. `+x=0.005 rad` reduces reset theta only to `≈0.0672 rad` and lowers consistency to `≈0.55`; larger `+x` offsets reduce theta more but destroy seated depth/module consistency.
- The best guided final-window improvement is `progress7_fixed_world_xpos_depthhold_tight_step2_guideonly`: `s=8.019 mm`, `r=0.262 mm`, `theta=0.03608 rad`, consistency `0.912`, strict false.
- Larger `+x`, earlier final hold, applying the same trim to the online-adapted progress10 run, and a solver-tuned run all failed to improve strict metrics. The solver-tuned run with `AIC_ISAAC_SOLVER_POSITION_ITERATIONS=32`, `AIC_ISAAC_SOLVER_VELOCITY_ITERATIONS=16`, and external-force updates did not insert.

Current closest candidate:

- Run: `outputs/agentic_reward_curriculum_20260524_final_window_probe/runs/2026-05-24_04-52-05_progress7_fixed_world_xpos_depthhold_tight_step2_guideonly`
- Gap: `theta=0.03608 rad`, about `0.00608 rad` above the strict `0.030 rad` threshold.
- Other strict gates pass on the best row: `s=8.019 mm`, `r=0.262 mm`, consistency `0.912`.

Updated blocker classification:

- Primary: `near_success_orientation_blocked`
- Secondary: `controller_realization_mismatch`
- Additional: `module_consistency_failure` for direct seated resets
- Additional: `solver_contact_regression` for the high-iteration/external-force run

Updated next code change: add a local-transform/contact diagnostic that compares the actual transform chain `gripper_tcp -> sfp_module_link -> sfp_tip_link` before contact, at shallow insertion, and at the seated plateau. The reset diagnostic shows that the gripper can be accurately commanded while the semantic tip/module orientation remains wrong, so the next useful evidence is whether the module/tip link relationship changes under contact/cable constraints or whether the episode reset/reference transform itself is inconsistent with the asset.

## 2026-05-24 Updated blocker after compensated strict reset and pose-hold probes

New evidence changes the blocker slightly:

- The actual local transform chain `gripper_tcp -> sfp_module_link -> sfp_tip_link` is stable between the best guided plateau and direct reset diagnostics.
- A derived gripper reset with orientation-aware position compensation can create a reset-only strict pose for both envs: `s>=8 mm`, `r<0.5 mm`, `theta=0`, and consistency `>0.96`.
- That reset-only pose is not success. On the first physics step, with no meaningful corrective action yet, the tip moves roughly `8.5 mm` out of the seated state and module consistency collapses.
- A stronger direct wrist pose-hold controller can recover depth/lateral/module consistency in env1 (`s=8.036 mm`, `r=0.380 mm`, consistency `0.922`) but semantic theta plateaus at `0.04006 rad`, still above the strict `0.030 rad` threshold.
- Gated semantic final trim and `+/-x` reset-orientation pre-biases did not produce a post-step strict row; they either degraded module consistency/lateral error or worsened theta.
- A target-tip stabilizer that commands wrist deltas from the reset `sfp_tip_link` target was worse than gripper pose-hold: it drove lateral error into the millimeter range and module consistency collapsed. This suggests direct semantic tip correction through the current wrist IK/action path is not a reliable final-seat controller.
- Pose-hold plus gated fixed `+x` rotation was stable but insufficient. `+x=0.0005 rad` reached `s=8.038 mm`, `r=0.455 mm`, `theta=0.03960 rad`, consistency `0.727`; repeated `+x=0.001 rad` lowered theta transiently to `0.03882` only at `s=7.35 mm`, below strict seated depth.

Closest current candidates:

| candidate | metrics | strict gap |
|---|---|---|
| Guided final-window `progress7_fixed_world_xpos_depthhold_tight_step2_guideonly` | `s=8.019 mm`, `r=0.262 mm`, `theta=0.03608 rad`, consistency `0.912` | theta high by `0.00608 rad` |
| Direct compensated reset + 2 mm pose hold | `s=8.036 mm`, `r=0.380 mm`, `theta=0.04006 rad`, consistency `0.922` | theta high by `0.01006 rad` |
| Direct compensated reset + pose hold + fixed `+x` trim | `s=8.038 mm`, `r=0.455 mm`, `theta=0.03960 rad`, consistency `0.727` | theta high by `0.00960 rad`, consistency below strict |

Updated blocker classification:

- Primary: `controller_realization_mismatch`
- Secondary: `near_success_orientation_blocked`
- Additional: `contact_stability_failure` for the reset-only strict pose ejecting on the first post-reset physics step
- Additional: `tip_depth_false_positive` remains controlled by strict theta/consistency checks

Recommended next code change: stop reward-only tuning and inspect final-seat contact/controller feasibility. The highest-value test is to change only contact/solver/final-seat stabilization mechanics and ask whether the compensated strict reset can remain strict for one post-step frame. Candidate bounded changes: increase final-seat damping/solver stability for the cable/module contact, temporarily disable or soften the problematic collision pair for a diagnostic only, or add a pre-step low-level Cartesian hold that can apply the gripper correction before the contact impulse ejects the module. If none of those can hold strict post-step geometry, classify the blocker as asset/contact/controller rather than reward/curriculum.
