# Agentic Reward/Curriculum Results 2026-05-23

Strict success was not achieved in the bounded probes run so far. The closest candidate is a near-gate retention run that reached large semantic tip depth and tight lateral error, but it failed strict success on tip orientation, module/body consistency, and force/contact sanity.

## Insertion Depth Coordinate

The strict evaluator's axial coordinate `s` is the semantic signed depth of `sfp_tip_link` from the configured port entrance plane to the configured seated target pose. In the current near-gate episode this target is `seated_depth_m: 0.008`, so full semantic insertion is 8 mm. This is not the full 48.72 mm physical cage depth. The geometry implementation computes `target_depth_m = dot(target_pose_world - entrance_pose_world, insertion_axis_world)` and rejects depths outside 3-30 mm, so 48.72 mm cannot be the current `s` success target.

## Runs

| run | command/config | checkpoint | best s mm | best r mm | best theta rad | depth frac | module consistency | max force N | strict | label | visual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| prior guard_final_orientation_hold_full | `outputs/one_day_insertion_pipeline/orientation_calibration_20260519/...guard_final_orientation_hold_full...` | run checkpoint | 0.096 | 0.950 | 0.0297 | 0.012 | 0.0187 | 33.83 | false | tip_depth_false_positive | center/left/right images |
| mixed smoke, tight retention | `outputs/agentic_reward_curriculum_20260523_smoke/iterations/iter_001_smoke_retention_r05_ax5um_rejectr/agent_decision.json` | best_semantic_progress10 | -7.394 | 3.970 | 0.0766 | 0.000 | 0.0000 | 34.81 | false | contact_spike | center/left/right images |
| final handoff retention, predicted-r reject | `outputs/agentic_reward_curriculum_20260523_r105_smoke/iterations/iter_001_r105_retention_r05_ax10um_rejectr/agent_decision.json` | best_semantic_progress10 | 6.892 | 0.229 | 0.0621 | 0.862 | 0.3891 | 34.13 | false | contact_spike | center/left/right images |
| orientation-gated full-quat final trim | `outputs/agentic_reward_curriculum_20260523_r105_orientgate/iterations/iter_001_r105_orientgate_fullquat_rot004_ax5um/agent_decision.json` | best_semantic_progress10 | 0.228 | 0.206 | 0.0611 | 0.028 | 0.0108 | 33.53 | false | tip_depth_false_positive | center/left/right images |
| orientation-gated axis-only final trim | `outputs/agentic_reward_curriculum_20260523_r105_axisfinal/iterations/iter_001_r105_orientgate_axisonly_rot006_ax5um/agent_decision.json` | best_semantic_progress10 | 0.200 | 0.368 | 0.0590 | 0.025 | 0.0078 | 35.00 | false | contact_spike | center/left/right images |
| adaptive orientation-sign final trim | `outputs/agentic_reward_curriculum_20260523_adaptive_orient/iterations/iter_001_r105_adaptive_orientsign_fullquat_rot006_ax5um/agent_decision.json` | best_semantic_progress10 | 0.173 | 0.487 | 0.0705 | 0.022 | 0.0067 | 35.00 | false | contact_spike | center/left/right images |
| six-direction basis orientation probe | `outputs/agentic_reward_curriculum_20260523_probe_basis/iterations/iter_001_r105_probe_basis_rot003_ax5um/agent_decision.json` | best_semantic_progress10 | 0.902 | 0.277 | 0.0837 | 0.113 | 0.0063 | 34.43 | false | contact_spike | center/left/right images |
| small-step basis orientation probe | `outputs/agentic_reward_curriculum_20260523_probe_basis_small/iterations/iter_001_r105_probe_basis_rot001_ax5um/agent_decision.json` | best_semantic_progress10 | 8.190 | 0.130 | 0.0641 | 1.000 | 0.3978 | 34.15 | false | contact_spike | center/left/right images |
| pure rotation-axis audit | `outputs/agentic_reward_curriculum_20260523_rotation_audit/runs/2026-05-23_16-17-11_r105_rotation_axis_audit` | n/a | -0.105 | 0.083 | 0.0718 | 0.000 | n/a | 31.65 | false | diagnostic | center/left/right images |
| module recovery, rotation allowed | `outputs/agentic_reward_curriculum_20260523_module_recovery/iterations/iter_001_r105_module_recovery_rot001_backoff50um/agent_decision.json` | best_semantic_progress10 | 9.215 | 0.222 | 0.0623 | 1.000 | 0.4247 | 34.15 | false | contact_spike | center/left/right images |
| module recovery, zero rotation | `outputs/agentic_reward_curriculum_20260523_module_recovery_zero/iterations/iter_001_r105_module_recovery_zero_rot_backoff100um/agent_decision.json` | best_semantic_progress10 | 8.022 | 0.080 | 0.0659 | 1.000 | 0.3713 | 34.15 | false | contact_spike | center/left/right images |
| module recovery state machine, 2mm target | `outputs/agentic_reward_curriculum_20260523_module_recovery_sm/runs/2026-05-23_18-04-00_r105_module_recovery_state_machine` | best_semantic_progress10 | 8.022 | 0.080 | 0.0659 | 1.000 | 0.3713 | 34.15 | false | contact_spike | center/left/right images |
| module recovery state machine, 7mm fast backout | `outputs/agentic_reward_curriculum_20260523_module_recovery_sm/runs/2026-05-23_18-07-15_r105_module_recovery_state_machine_fast_backout` | best_semantic_progress10 | 7.573 | 0.110 | 0.0639 | 0.947 | 0.4011 | 34.15 | false | contact_spike | center/left/right images |
| module recovery state machine, 7mm short hold | `outputs/agentic_reward_curriculum_20260523_module_recovery_sm/runs/2026-05-23_18-10-24_r105_module_recovery_state_machine_short_hold` | best_semantic_progress10 | 7.573 | 0.110 | 0.0639 | 0.947 | 0.4011 | 34.15 | false | contact_spike | center/left/right images |
| staged contact rotation-axis diagnostic | `outputs/agentic_reward_curriculum_20260523_contact_diagnostic/runs/2026-05-23_18-28-09_r105_staged_rotation_axis_probe` | best_semantic_progress10 | 8.053 | 0.057 | 0.0655 | 1.000 | 0.3687 | 9.96 | false | near_success_orientation_blocked/module_consistency_blocked | center/left/right images |
| staged contact translation-axis diagnostic | `outputs/agentic_reward_curriculum_20260523_contact_diagnostic/runs/2026-05-23_18-31-13_r105_staged_translation_axis_probe` | best_semantic_progress10 | 8.059 | 0.090 | 0.0646 | 1.000 | 0.3893 | 8.64 | false | near_success_orientation_blocked/module_consistency_blocked | center/left/right images |
| module-lateral priority smoke | `outputs/agentic_reward_curriculum_20260523_module_lateral/runs/2026-05-23_18-42-58_r105_module_lateral_align_priority_smoke` | best_semantic_progress10 | 8.386 | 0.229 | 0.0636 | 1.000 | 0.4436 | 34.15 | false | contact_spike | center/left/right images |
| module-lateral hold 7.2mm | `outputs/agentic_reward_curriculum_20260523_module_lateral/runs/2026-05-23_18-47-15_r105_module_lateral_hold72_smoke` | best_semantic_progress10 | 8.747 | 0.227 | 0.0634 | 1.000 | 0.4365 | 34.15 | false | contact_spike | center/left/right images |
| module-lateral hold 6.8mm, backoff 200um | `outputs/agentic_reward_curriculum_20260523_module_lateral/runs/2026-05-23_18-49-57_r105_module_lateral_hold68_backoff200_smoke` | best_semantic_progress10 | 7.552 | 0.232 | 0.0660 | 0.944 | 0.3893 | 34.15 | false | contact_spike | center/left/right images |
| module-lateral hold 6.8mm, opposite sign | `outputs/agentic_reward_curriculum_20260523_module_lateral/runs/2026-05-23_18-53-01_r105_module_lateral_hold68_backoff200_signneg_smoke` | best_semantic_progress10 | 7.405 | 0.235 | 0.0670 | 0.926 | 0.3688 | 34.15 | false | contact_spike | center/left/right images |
| module-lateral rotation 0.006 long | `outputs/agentic_reward_curriculum_20260523_module_lateral/runs/2026-05-23_19-16-02_r105_module_lateral_rotation_align_rot006_long` | best_semantic_progress10 | 8.214 | 0.222 | 0.0610 | 1.000 | 0.5049 | 34.15 | false | near_success_orientation_blocked | center/left/right images + separate video |
| earlier module-lateral activation | `outputs/agentic_reward_curriculum_20260523_module_lateral/runs/2026-05-23_19-40-56_r105_module_lateral_rotation_align_rot006_act55` | best_semantic_progress10 | 6.742 | 0.222 | 0.0610 | 0.843 | 0.4565 | 34.15 | false | no_axial_progress | center/left/right images |
| module-lateral no predicted-r reject | `outputs/agentic_reward_curriculum_20260523_module_lateral/runs/2026-05-23_19-44-54_r105_module_lateral_rot006_no_predr_reject` | best_semantic_progress10 | 7.197 | 0.285 | 0.0613 | 0.900 | 0.4100 | 34.05 | false | no_axial_progress | center/left/right images |
| delayed full-quat final trim, 7.6mm | `outputs/agentic_reward_curriculum_20260523_orientation_final/runs/2026-05-23_19-54-30_r105_rot006_delayed_fullquat_finalrot006` | best_semantic_progress10 | 6.832 | 0.241 | 0.0611 | 0.854 | 0.3937 | 34.14 | false | no_axial_progress | center/left/right images |
| delayed full-quat final trim, 6.5mm | `outputs/agentic_reward_curriculum_20260523_orientation_final/runs/2026-05-23_19-58-06_r105_rot006_delayed65_fullquat_finalrot006` | best_semantic_progress10 | 6.561 | 0.211 | 0.0606 | 0.820 | 0.4294 | 34.14 | false | no_axial_progress | center/left/right images |
| semantic final-axis guard, broad lateral | `outputs/agentic_reward_curriculum_20260523_axis_guard/runs/2026-05-23_20-08-27_r105_final_axis_guard_rot003` | best_semantic_progress10 | 0.424 | 0.949 | 0.0589 | 0.053 | 0.0107 | 35.00 | false | rotation_induced_lateral_sweep | center/left/right images |
| semantic final-axis guard, 0.3mm lateral | `outputs/agentic_reward_curriculum_20260523_axis_guard/runs/2026-05-23_20-12-18_r105_final_axis_guard_rot003_lat03` | best_semantic_progress10 | 5.688 | 0.200 | 0.0607 | 0.711 | 0.3481 | 35.00 | false | no_axial_progress | center/left/right images |
| original adapter-policy eval path | `outputs/agentic_reward_curriculum_20260523_checkpoint_eval/runs/2026-05-23_20-16-37_r105_progress10_adapter_eval_original_path` | best_semantic_progress10 | 4.759 | 0.208 | 0.0607 | 0.595 | 0.2478 | 18.42 | false | no_axial_progress | center/left/right images |
| historical May17 curriculum/reward reproduction, 220 steps | `outputs/agentic_reward_curriculum_20260523_reproduce_may17/runs/2026-05-23_20-38-14_progress10_repro220_historical_curriculum_reward` | progress7 -> online adapter | 5.226 | 0.275 | 0.0363 | 0.653 | 0.257 | n/a | false | no_axial_progress / module_consistency_blocked | center/left/right images |
| depth hold 8mm, late final orientation | `outputs/agentic_reward_curriculum_20260523_depth_hold/runs/2026-05-23_22-34-03_progress10_no_comp_depth_hold8mm_latefinal` | progress7 -> online adapter | 8.034 | 0.284 | 0.0364 | 1.004 | 0.917 | n/a | false | near_success_orientation_blocked | center/left/right images |
| continue best depth-hold checkpoint | `outputs/agentic_reward_curriculum_20260523_orientation_trim/runs/2026-05-23_23-27-52_progress10_hold8_latefinal_continue_best` | depth-hold latest | 8.076 | 0.283 | 0.0364 | 1.010 | 0.918 | n/a | false | near_success_orientation_blocked | center/left/right images |
| strict final-axis gate, +sign 0.001rad | `outputs/agentic_reward_curriculum_20260523_strict_axis/runs/2026-05-23_23-55-22_progress10_hold8_strict_axis_rot001_pos` | depth-hold latest | 8.344 | 0.265 | 0.0408 | 1.043 | 0.809 | n/a | false | rotation_realization_overinsert / orientation_regression | center/left/right images |
| strict final-axis gate, -sign 0.0005rad | `outputs/agentic_reward_curriculum_20260523_strict_axis/runs/2026-05-24_00-12-45_progress10_hold8_strict_axis_rot0005_neg` | depth-hold latest | 8.907 | 0.271 | 0.0368 | 1.113 | 0.802 | n/a | false | near_success_orientation_blocked | center/left/right images |
| seated recovery to 6.5mm | `outputs/agentic_reward_curriculum_20260523_seated_recovery/runs/2026-05-24_00-25-35_progress10_hold8_seated_recovery_target65` | depth-hold latest | 8.066 | 0.180 | 0.0414 | 1.008 | 0.823 | n/a | false | recovery_backout_realization_mismatch | center/left/right images |

## Interpretation

The final handoff retention run is the closest numerical candidate:
- It reached `s=6.892 mm` out of an 8 mm seated depth and maintained `r=0.229 mm`, which satisfies the lateral target.
- It failed strict orientation: `theta=0.0621 rad`, above the strict `0.030 rad` target.
- It failed module/body consistency: `0.389`, below the `0.80` strict gate.
- It had a force/contact spike near the configured 35 N clip.
- Its phase summary reported `fraction_delta_s_positive_and_theta_gt_0p06 = 0.9496`, so this is not strict insertion and should not be counted as success.

The orientation-gated retention patch is working as intended:
- Full-quat gated run: `retention_active=1.0` after entry but `retention_axial_active=0.0` while `misoriented=1.0`; axial progress stopped instead of creating a false positive.
- Axis-only gated run slightly improved best theta to `0.05897 rad` but did not approach the strict `0.030 rad` threshold and worsened lateral/force metrics.

The controller-aware orientation probes did not solve the final theta:
- Adaptive orientation sign flipped 24 times, but the sign-reversed command made semantic theta worse; this is not a simple scalar sign error.
- A pure rotation-axis audit showed `-rx` and `-ry` reduce semantic axis error locally by about `0.0011-0.0012 rad` per `0.003 rad` command, while `+rx/+ry` worsen it.
- The small-step six-direction basis servo reached full tip depth and very tight lateral error, but theta stayed at `0.064 rad` and module consistency stayed at `0.398`. This is a near-seated false positive, not strict insertion.

The module recovery guard now functions mechanically but has not solved seating:
- With rotation allowed during recovery, the guard activated but realized `s` kept increasing, indicating rotation/contact realization overpowered the 50 um backoff.
- With recovery zeroing rotation and using 100 um backoff, realized `s` decreased after recovery activated (`s=8.148 mm` at step 176 to `s=7.882 mm` at step 201). This validates the backoff mechanism.
- Module consistency still stayed low (`~0.34-0.37`) and theta stayed high (`~0.066 rad`), so recovery did not produce strict insertion.
- The persistent state-machine recovery also did not recover strict insertion. The 2 mm-target run remained in backout through the end of the 260-step probe; best strict metrics were unchanged from the one-step zero-rotation run. A 7 mm fast-backout diagnostic reached trim at the final step, but `theta` worsened to `0.0816 rad` and module consistency fell to `0.203`. A 7 mm short-hold diagnostic reached reinsert/hold state; by step 340, `theta` improved only to `0.0721 rad` and module consistency to `0.254`, still far outside strict success.

The staged contact command diagnostics ruled out naive local wrist commands as the missing final-seat primitive:
- At the audit handoff, post-step metrics were `s=7.637 mm`, `r=0.038 mm`, `theta=0.0638 rad`, module consistency `0.380`.
- Pure bounded rotations from that state did not reduce `theta`; final `theta=0.0687 rad` and module consistency `0.355`.
- Pure bounded translations increased depth but did not solve orientation or module consistency; final `theta=0.0678 rad`, module consistency `0.390`.
- Since tip `r` is tight while module consistency remains low, the next targeted guard is module-body lateral alignment rather than more tip-depth or reward tuning.

The module-body lateral alignment guard was implemented and tested:
- A guard-order fix was required because the new module-lateral command was initially overwritten by later retention/final-orientation overrides. After the fix, the guard became active in the final window.
- The best priority smoke improved module consistency to `0.4436`, but still missed the strict `>=0.80` consistency gate and strict orientation target by a wide margin (`theta=0.0636 rad` versus `<0.030 rad`).
- Adding a module-lateral hold depth showed controller/contact coupling: a 7.2 mm hold still drifted inward to `s=8.747 mm`; a stronger 6.8 mm hold/backoff held shallower depth but did not improve module consistency.
- Flipping the module-lateral correction sign made module consistency worse (`0.3688`) and increased `theta` to `0.0670 rad`.
- This rules out a simple lateral sign bug and suggests the remaining blocker is controller/contact realization of the module body under contact, not just reward shaping.

Additional 19:40-20:20 UTC updates:
- The best current full-depth artifact is `r105_module_lateral_rotation_align_rot006_long`: it reaches configured seated depth (`s=8.214 mm`) with tight lateral error (`r=0.222 mm`) and `success_geometry_by_env=true`, but strict success remains false because semantic tip orientation plateaus at `theta=0.06096 rad`, above the strict `0.030 rad` threshold. Its explicit consistency axial/lateral thresholds pass, but the smooth consistency gate remains only `0.505`.
- Earlier module-lateral activation and disabling predicted-r rejection did not improve strict metrics. Earlier activation stalled at `s=6.742 mm`; disabling predicted-r rejection allowed the compensation command to execute but traded away depth and consistency.
- Delayed full-quat final trims did not solve orientation. A 7.6 mm activation never reached the activation depth; a 6.5 mm activation held `r` tight but stalled at `s=6.56 mm` with `theta=0.0606 rad`.
- A new config-gated semantic final-axis rotation guard was added. With broad lateral activation it briefly reduced theta to `0.0589 rad` but induced lateral sweep (`r≈0.95 mm`) and stalled near the entrance. Tightening activation to `r<=0.3 mm` avoided the sweep but still stalled at `s=5.688 mm`, `theta=0.0607 rad`.
- A no-learning eval of the original adapter-policy path was run after noticing that earlier checkpoint evals used `--act_only --act_only_actor_mode act_direct`. The adapter-path eval did exercise the checkpoint policy, but under the short deterministic near-gate eval it stalled around `s=4.6-4.8 mm` and did not recover the older `theta≈0.036-0.037` near-success reported in the May 17 training run.

Additional 20:38 UTC update:
- A separate video artifact was generated from the current best full-depth `rot006` checkpoint/guard path: `outputs/agentic_reward_curriculum_20260523_video/runs/2026-05-23_19-35-07_r105_best_checkpoint_rot006_video`. It includes center/left/right MP4s and a markdown summary. It reaches `s=8.214 mm` and `r=0.222 mm`, but remains strict false because `theta=0.06096 rad` and smooth module consistency is only `0.5049`.
- A closer historical reproduction was run from `best_semantic_progress7_final_hold_checkpoint.pt` using the May 17 mixed curriculum and loose reward schedule. It did not reproduce the old env1 full-depth near-success. Best env1 post-step metrics were `s=5.226 mm`, `r=0.275 mm`, `theta=0.03633 rad`, module consistency `0.257`; no row reached `s>=7 mm`, and final metrics backed out to negative depth. The run preserved `git_status.txt`, `git_diff.patch`, config, metrics, phase summary, and images.

Updated recommendation: keep the separate video artifact as visual evidence of the best current full-depth attempt, but do not claim strict success. The historical reproduction suggests the older low-theta/full-depth candidate is not recovered by simply restoring the old curriculum/reward for 220 steps. The next bounded experiment should either run a longer exact May 17 reproduction from `best_semantic_progress7_final_hold_checkpoint.pt` or move to controller-level realization diagnostics for semantic-axis rotation under contact; short guarded axis commands either sweep laterally or stall before seated depth.

Additional 20:54-22:51 UTC updates:
- Restoring the historical May 17 action-gate schedule did not by itself recover the old full-depth trajectory. `2026-05-23_20-54-58_progress10_repro260_historical_action_gate_fixed` reached only `s=5.226 mm`, `r=0.275 mm`, `theta=0.03633 rad`, smooth consistency `0.2565`.
- Disabling the current final-orientation compensation reproduced the historical full-depth/low-theta path. `2026-05-23_21-13-39_progress10_repro220_no_final_comp` reached `s=10.286 mm`, `r=0.279 mm`, `theta=0.03638 rad`, but consistency fell to `0.409`; around the seated window it had `s=7.224-8.445 mm`, `r≈0.276 mm`, `theta≈0.0366-0.0368`, and consistency `0.815-0.887`. Strict success remained false because theta is above `0.030 rad`.
- A longer no-compensation 16 s run confirmed the failure mode rather than solving it. `2026-05-23_21-55-42_progress10_no_comp_ep16s` had zero strict rows. The best seated row was env1 step 146: `s=8.071 mm`, `r=0.276 mm`, `theta=0.03666 rad`, consistency `0.909`; later over-insertion reached `s=13.887 mm` and collapsed consistency to `0.0039`.
- A config-gated final seated-depth hold was added to the insertion action guard, disabled by default: `--insertion_action_guard_final_orientation_hold_depth_m`, `--insertion_action_guard_final_orientation_hold_margin_m`, and `--insertion_action_guard_final_orientation_hold_backoff_step_m`. Validation passed with `py_compile` and `30` focused tests.
- The first 8 mm hold probe activated too late because the run stalled before seated depth: `2026-05-23_22-18-33_progress10_no_comp_depth_hold8mm` reached only `s=5.043 mm`.
- Delaying final orientation activation to 7.5 mm while enabling the 8 mm hold produced the closest current strict-metric candidate: `2026-05-23_22-34-03_progress10_no_comp_depth_hold8mm_latefinal` reached `s=8.034 mm`, `r=0.284 mm`, `theta=0.03643 rad`, consistency `0.917`, `success_geometry_by_env=true`, and the hold activated (`hold_fraction_max=0.5`). Strict success remained false solely on theta.

Additional 23:27-00:25 UTC updates:
- Continuing from the best depth-hold checkpoint reproduced the near-success but did not improve theta: `s=8.076 mm`, `r=0.283 mm`, `theta=0.03639 rad`, consistency `0.918`. Strict success count was zero.
- A strict-gated final-axis trim was added, disabled by default. Validation passed with `py_compile` and the focused reward/script tests (`30 passed`).
- The positive-sign strict-axis probe activated the trim (`max_axis_active=0.5`) but worsened the seated orientation to `theta=0.04077 rad` and later over-inserted to `s=14.935 mm` with consistency collapse. This is rejected.
- The negative-sign smaller-step strict-axis probe did not activate because the one-step semantic-axis predictor rejected it (`max_axis_active=0.0`). It behaved like the depth-hold baseline and remained strict false with `theta=0.03685 rad`.
- A seated-window recovery run activated module recovery and backout (`max_recovery=1.0`, `max_backout=1.0`) but never reached trim/reinsert. Best seated row regressed to `theta=0.04142 rad`, and the run ended over-inserted with low consistency. This suggests the 80 um backoff was too weak once contact/guide pressure had already over-inserted.

Current best controlled candidate remains `outputs/agentic_reward_curriculum_20260523_depth_hold/runs/2026-05-23_22-34-03_progress10_no_comp_depth_hold8mm_latefinal`.

Current recommendation: do not claim success. The final-depth hold fixes the over-insertion/module-consistency collapse in the low-theta historical trajectory, but semantic tip orientation plateaus about `0.0064 rad` above the strict threshold. The final-axis trim family either fails the predictor or worsens realized seating. The first recovery attempt confirmed that recovery triggers but does not realize enough outward motion, so the next bounded test should increase recovery backoff substantially and target a shallower positive depth before declaring the blocker controller/contact realization.

Additional 00:41 UTC update:
- A stronger seated recovery realization probe was run with a 4.0 mm target depth and 300 um backoff: `outputs/agentic_reward_curriculum_20260523_seated_recovery/runs/2026-05-24_00-41-42_progress10_hold8_seated_recovery_target40_backoff300`.
- Strict success count remained `0`.
- Recovery and backout activated (`max_recovery=1.0`, `max_backout=1.0`), but the state machine still never reached trim or reinsert (`max_trim=0`, `max_reinsert=0`).
- Best seated row: `s=8.027 mm`, `r=0.218 mm`, `theta=0.04143 rad`, consistency `0.823`.
- Best consistency seated row: `s=8.117 mm`, `r=0.163 mm`, `theta=0.04163 rad`, consistency `0.835`.
- Best depth row over-inserted to `s=10.556 mm` with consistency collapse to `0.248`.
- Final rows remained non-strict; env0 was seated by depth/lateral only but had `theta=0.06620 rad` and consistency `0.597`, while env1 was over-inserted with consistency `0.241`.

Updated recommendation: stop treating reward/curriculum as the main bottleneck for the current best near-seated state. The bounded final-axis and recovery probes did not reduce semantic theta below the strict gate or restore module consistency after contact. The next useful experiment is a controller/contact realization diagnostic that directly measures whether requested outward and semantic-axis correction commands produce the intended `sfp_tip_link` and `sfp_module_link` motion under seated contact. If not, the blocker should be classified as controller/contact or asset/contact realization rather than a reward-funnel issue.

Additional 01:00-01:18 UTC update:
- Added command-realization diagnostics to `train.py` that log the guard's requested world-space delta and the realized `sfp_tip_link` / `sfp_module_link` / gripper / wrist projection onto that command.
- Validation passed after each diagnostic/control edit:
  - `.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
  - `.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_agent_reward_funnel_scripts.py`
  - Result: `30 passed`
- `outputs/agentic_reward_curriculum_20260524_contact_realization/runs/2026-05-24_01-03-03_hold8_recovery_target40_command_realization_160` confirmed the controller/contact mismatch. During recovery/backout the guard commanded negative axial motion averaging `-0.233 mm` and later `-0.300 mm` per step, but realized signed-depth still increased. Env1 increased from `s=8.027 mm` at recovery entry to `s=10.346 mm` by step 160 while consistency fell to `0.304`; strict success count was `0`.
- Added a config-driven recovery backout lateral scale:
  - `--insertion_action_guard_module_recovery_backout_lateral_scale`
- `outputs/agentic_reward_curriculum_20260524_contact_realization/runs/2026-05-24_01-08-11_hold8_recovery_target40_pure_axial_backout` disabled lateral correction during backout. This reduced the inward drift but still did not realize outward motion. With near-zero lateral command and `-0.300 mm` axial backoff, env1 still increased to `s=8.404 mm`; strict success count was `0`.
- Added a config-driven recovery backoff direction sign:
  - `--insertion_action_guard_module_recovery_backoff_direction_sign`
- `outputs/agentic_reward_curriculum_20260524_contact_realization/runs/2026-05-24_01-12-24_hold8_recovery_target40_pure_axial_backout_signpos` tested the opposite axial sign. It drove deeper insertion, reaching `s=13.529 mm` with consistency collapse to `0.006`; strict success count was `0`.
- `outputs/agentic_reward_curriculum_20260524_contact_realization/runs/2026-05-24_01-15-39_hold8_recovery_target40_rootframe_pure_axial_backout` tested pure axial recovery in `root` action frame. Negative axial commands still failed to reduce `s`; env1 ended at `s=8.270 mm`, `theta=0.04444 rad`, consistency `0.723`; strict success count was `0`.

Updated recommendation: stop reward/curriculum/final-guard tuning for this branch until the low-level controller/contact issue is fixed. The immediate next code path should be a controller or asset/contact diagnostic outside the policy loop: command a held shallow-contact state with direct scripted wrist IK/pose targets, sweep outward axial deltas in both action frames, and verify whether the simulated module can physically retract along the port axis without direct `sfp_tip_link` IK. If direct wrist pose targets also cannot reduce semantic tip depth under contact, inspect collision/contact geometry, solver settings, and the SFP/module articulation/cable constraints.

Additional 01:25-02:23 UTC update:
- `outputs/agentic_reward_curriculum_20260524_direct_axis_audit/runs/2026-05-24_01-25-34_guarded_approach_direct_axis_backout_start105` used the normal guide/guard through the seated window, then bypassed the guard and sent direct semantic-axis backout actions. Env1 reached a near-seated state at step 105 (`s=7.988 mm`, `r=0.265 mm`, `theta=0.04115 rad`, consistency `0.819`), but direct backout did not retract under contact; final env1 remained around `s=7.976 mm`, with `theta=0.04516 rad` and consistency `0.711`. Strict count: `0`.
- `outputs/agentic_reward_curriculum_20260524_final_axis/runs/2026-05-24_01-30-40_strict_final_axis_trim_rot0015` tested strict-gated final semantic-axis rotation from the low-theta depth-hold checkpoint. Positive-sign trim activated only briefly, rejected most strict candidates, and over-inserted to `s=21.226 mm` with consistency collapse. Best valid seated row was `s=8.463 mm`, `r=0.203 mm`, `theta=0.04171 rad`, consistency `0.817`. Strict count: `0`.
- `outputs/agentic_reward_curriculum_20260524_final_axis/runs/2026-05-24_01-34-27_strict_final_axis_trim_rot0015_signneg` tested the opposite final-axis sign. It never activated the trim and regressed to the same `theta≈0.0418 rad` seated plateau. Strict count: `0`.
- `outputs/agentic_reward_curriculum_20260524_final_axis/runs/2026-05-24_01-37-42_progress7_orientation_basis_depth_hold8mm` added the kinematic orientation-basis probe to the original progress7 online adaptation path. The predicted-basis choice did not transfer under contact. Best seated row was `s=8.018 mm`, `r=0.205 mm`, `theta=0.04143 rad`, consistency `0.837`; later rows drifted to `theta≈0.051` with lower consistency. Strict count: `0`.
- `outputs/agentic_reward_curriculum_20260524_final_axis/runs/2026-05-24_01-52-59_guarded_approach_rotation_axis_audit_start103` empirically cycled pure `+/-x/y/z` wrist rotations after a guarded approach. It reduced env1 theta from about `0.0411` to `0.0380` over the audit, but no row satisfied the full strict gate; the best depth row had `s=8.095 mm`, `r=0.350 mm`, `theta=0.03880 rad`, consistency `0.777`.
- `outputs/agentic_reward_curriculum_20260524_final_axis/runs/2026-05-24_01-56-56_progress7_no_basis_depth_hold8mm_extended320` extended the original no-basis depth-hold training to 320 steps. While consistency was valid, the best seated row was `s=8.683 mm`, `r=0.267 mm`, `theta=0.03743 rad`, consistency `0.802`. Later, theta improved only to `0.03638 rad` while module consistency collapsed to near zero due to over-insertion. Strict count: `0`.
- `outputs/agentic_reward_curriculum_20260524_final_axis/runs/2026-05-24_02-17-14_progress7_early_orientation_hold76mm` started final orientation hold earlier at `7.6 mm`. This did not cross the orientation gate. Best valid seated row was `s=8.911 mm`, `r=0.277 mm`, `theta=0.03682 rad`, consistency `0.805`; later theta plateaued around `0.0364` while consistency collapsed. Strict count: `0`.

Updated recommendation: the current family is exhausted for this branch. Empirical rotation can shave only about `0.003 rad` from the near-seated theta, and neither extended hold nor earlier hold reaches the strict `<0.030 rad` semantic orientation threshold before consistency collapses. The next code path should leave reward/curriculum tuning and implement a standalone wrist-pose/controller realization diagnostic outside the SERL loop, starting from a shallow-contact state and sweeping direct wrist pose deltas/rotations while logging `sfp_tip_link`, `sfp_module_link`, and contact/solver state.

Additional 02:37-02:54 UTC update:
- Added `aic_utils/aic_isaac/scripts/controller_contact_realization_diagnostic.py`, a reusable launcher/analyzer for staged controller/contact probes. It loads a known near-seated trainer command from `train_config.json`, runs bounded debug-audit cases, writes `agent_decision.json`, saves launcher logs, parses post-step semantic geometry for all envs, and writes `controller_contact_summary.json` plus `summary.md`.
- Validation passed:
  - `.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/scripts/controller_contact_realization_diagnostic.py aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
  - `.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_agent_reward_funnel_scripts.py`
  - Result: `30 passed`
- The diagnostic triad was run under `outputs/agentic_reward_curriculum_20260524_controller_contact/`:

| run | best strict-relevant row | final delta from pre-audit | result |
|---|---|---|---|
| `semantic_axis_backout_300um_start105` | `s=8.028 mm`, `r=0.236 mm`, `theta=0.04154 rad`, consistency `0.847` | `ds=-0.007 mm`, `dtheta=+0.00444 rad`, `dcons=-0.084` | Backout commands barely reduced semantic depth under contact and worsened theta/consistency. |
| `pure_rotation_axes_4mrad_start103` | `s=8.106 mm`, `r=0.303 mm`, `theta=0.04018 rad`, consistency `0.778` | `ds=+1.505 mm`, `dtheta=+0.00113 rad`, `dcons=-0.217` | No bounded wrist rotation axis closed the theta gap; the best theta row failed module consistency. |
| `semantic_axis_forward_100um_start105` | `s=8.388 mm`, `r=0.243 mm`, `theta=0.04127 rad`, consistency `0.816` | `ds=+0.299 mm`, `dtheta=+0.01951 rad`, `dcons=-0.131` | Tiny inward commands can increase depth but worsen theta and module consistency, so tip-depth-only progress remains a false-positive risk. |
| `shallow_rotation_axes_4mrad_start90` | `s=8.010 mm`, `r=0.257 mm`, `theta=0.04521 rad`, consistency `0.826` | `ds=+3.210 mm`, `dtheta=+0.00240 rad`, `dcons=+0.568` | Starting near shallow positive insertion did not make orientation correction easier; rotations still worsened theta. |
| `shallow_semantic_axis_backout_200um_start90` | `s=7.768 mm`, `r=0.082 mm`, `theta=0.04374 rad`, consistency `0.777` | `ds=+2.858 mm`, `dtheta=+0.00238 rad`, `dcons=+0.603` | Even shallow backout commands did not reduce semantic depth; the system continued inserting while theta worsened. |

- A config-only pre-seat orientation-gate probe was run under `outputs/agentic_reward_curriculum_20260524_preseat_orientation_gate/runs/2026-05-24_02-46-07_preseat_orientgate035_finaldepth65_ax10um`. It tightened `--target_action_guide_orientation_switch_rad` to `0.035`, started final orientation at `6.5 mm`, lowered centered axial steps to `10 um`, and kept the existing guard otherwise unchanged.
- That probe failed in the opposite direction: it prevented over-seating but stalled outside the port. Final env1 was about `s=-0.336 mm`, `r=1.499 mm`, `theta=0.03704 rad`, with near-zero module consistency. Strict success count remained `0`.

Updated recommendation: the diagnostic evidence now separates two failure modes. Loose orientation gating seats the tip/module but enters contact with an unrecoverable `theta≈0.036-0.041 rad`; tight orientation gating avoids that contact state but cannot complete the approach/insertion. The shallow probes show this is not only a fully seated-contact problem: direct debug-audit backout and pure rotations at about `s=4.9-5.1 mm` also failed to reduce depth/theta. The next iteration should not be reward-only. It should add a wrist-pose/controller diagnostic outside the SERL loop, preferably with direct scripted wrist targets and contact/solver logging, to verify whether the low-level controller can actually realize semantic-axis backout or semantic orientation correction without relying on the policy/action adapter path.

## 2026-05-24 Final-window fixed-axis trim update

Implemented a config-driven final fixed world-axis trim in `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`:

- `--insertion_action_guard_final_fixed_world_rotation`
- `--insertion_action_guard_final_fixed_world_rotation_axis {x,y,z,best}`
- `--insertion_action_guard_final_fixed_world_rotation_sign`
- `--insertion_action_guard_final_fixed_world_rotation_step_rad`
- `--insertion_action_guard_final_fixed_world_rotation_compensation_clip_m`

The `best` mode evaluates all six signed world-axis micro-rotations and only applies a command if the one-step semantic tip-axis prediction improves theta and the existing strict final-axis guard rails pass.

Validation:

- `.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py aic_utils/aic_isaac/scripts/controller_contact_realization_diagnostic.py`
- `.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_agent_reward_funnel_scripts.py`
- Result: `30 passed`

Runs:

| run | best strict-relevant row | result |
|---|---|---|
| `2026-05-24_03-59-27_progress7_guard_finalaxis_depthhold_tight_step2_signneg_guideonly` | `s=8.298 mm`, `r=0.277 mm`, `theta=0.03631 rad`, consistency `0.906` | Opposite final-axis command sign recovered the same near-seat plateau; no strict success. |
| `2026-05-24_04-05-42_progress7_fixed_world_ypos_depthhold_tight_step2_guideonly` | `s=8.160 mm`, `r=0.276 mm`, `theta=0.03848 rad`, consistency `0.886` | Fixed world `+y` activated but worsened theta relative to the best plateau. |
| `2026-05-24_04-08-57_progress7_fixed_world_yneg_depthhold_tight_step2_guideonly` | `s=8.298 mm`, `r=0.277 mm`, `theta=0.03631 rad`, consistency `0.906` | Fixed world `-y` did not activate useful trim and matched the plateau. |
| `2026-05-24_04-12-47_progress7_fixed_world_best_depthhold_tight_step2_guideonly` | `s=8.249 mm`, `r=0.267 mm`, `theta=0.03754 rad`, consistency `0.888` | All-axis chooser activated but did not improve theta below the plateau. |
| `2026-05-24_04-16-15_progress7_fixed_world_best_depthhold_tight_step05_guideonly` | `s=8.245 mm`, `r=0.242 mm`, `theta=0.03802 rad`, consistency `0.808` | Smaller all-axis micro-step also failed; deeper rows reached `theta≈0.0364` only with consistency below `0.8`. |

No row in this family satisfied strict success. The closest valid row remains `theta≈0.0363 rad`, about `0.0063 rad` above the strict threshold.

Separate visual artifact generated from the current best near-seat run:

- Source run: `outputs/agentic_reward_curriculum_20260523_depth_hold/runs/2026-05-23_22-34-03_progress10_no_comp_depth_hold8mm_latefinal`
- Videos:
  - `videos/env1_center_camera_h264.mp4`
  - `videos/env1_left_camera_h264.mp4`
  - `videos/env1_right_camera_h264.mp4`

Updated recommendation: the fixed-axis patch is useful diagnostic instrumentation but did not solve the final orientation gap. Continue with the controller/contact diagnostic direction: bypass SERL, command direct wrist pose targets at shallow positive insertion, and verify whether semantic `sfp_tip_link` theta can physically be reduced below `0.030 rad` without destroying `sfp_module_link` consistency.

## 2026-05-24 Direct wrist reset/contact diagnostic update

Extended `aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py` with:

- `--probe orientation_servo_best`, a closed-loop direct wrist IK probe that evaluates signed world-axis wrist rotations from current semantic tip geometry and applies only predicted theta-improving rotations.
- `--override_start_signed_depth_m`, `--override_start_lateral_m`, and `--override_start_orientation_wxyz`, which generate a reproducible temporary episode-config copy before Isaac starts so the diagnostic can reset directly into shallow or seated contact.

Validation:

- `.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py aic_utils/aic_isaac/scripts/controller_contact_realization_diagnostic.py`
- `.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_agent_reward_funnel_scripts.py`
- Result: `30 passed`

Runs:

| run | best row | result |
|---|---|---|
| `2026-05-24_04-23-21_wrist_orientation_servo_best_shallow5_step2` | `s=3.219 mm`, `r=0.331 mm`, `theta=0.06556 rad`, consistency `0.012` | Direct wrist approach did not reach the intended `5 mm` shallow contact and did not improve theta enough. |
| `2026-05-24_04-25-03_wrist_orientation_servo_best_shallow6_step2_longapproach` | `s=-1.326 mm`, `r=0.472 mm`, `theta=0.07222 rad`, consistency `0.000` | Longer/more aggressive direct approach regressed and never reached contact; rejected as approach-controller failure evidence. |
| `2026-05-24_04-31-21_wrist_orientation_servo_best_reset_s4mm_step2` | `s=3.116 mm`, `r=0.551 mm`, `theta=0.06883 rad`, consistency `0.010` | Generated `s=4 mm` shallow reset relaxed to lower depth and poor module consistency; orientation servo found little usable correction. |
| `2026-05-24_04-32-25_wrist_reset_s8mm_zero_probe` | `s=7.990 mm`, `r=0.196 mm`, `theta=0.07079 rad`, consistency `0.561` | Generated seated-depth reset can place tip depth/lateral but not semantic orientation or module/body consistency. |
| `2026-05-24_04-33-40_wrist_reset_s8mm_targetquat_zero_probe` | `s=6.648 mm`, `r=14.735 mm`, `theta=0.29766 rad`, consistency `0.000` | Using the port target quaternion as the reset-body orientation is invalid for the gripper reset body. |
| `2026-05-24_04-34-28_wrist_orientation_servo_best_reset_s8mm_step4` | `s=8.051 mm`, `r=0.284 mm`, `theta=0.07100 rad`, consistency `0.613` | Direct wrist orientation servo under seated reset made only marginal theta improvement (`~0.071` to `~0.069`) and did not approach strict theta. |

Interpretation: direct wrist IK can be reset to depth/lateral near the seated criterion, but the resulting semantic tip orientation and module/body consistency are much worse than the best guided policy near-seat state. Under direct wrist commands, the semantic orientation servo cannot close the strict `theta <= 0.030 rad` gap. This supports a controller/contact/reset-realization blocker: the system can report correct tip depth while the module/body pose and semantic tip axis remain physically inconsistent.

Next recommended code path: inspect and instrument the reset/controller geometry itself. In particular, compare the gripper reset pose, `sfp_tip_link` pose, and `sfp_module_link` pose immediately after reset against the target episode entrance/axis, then derive a gripper reset orientation from the desired semantic tip orientation instead of reusing the outside-gate gripper orientation.

## 2026-05-24 Reset diagnostics, fixed-x trim, and solver probe

Added reset-target versus realized semantic body logging to `aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py`. Each wrist/contact run now writes `reset_diagnostic.json` and embeds the same data in `wrist_contact_summary.json`: episode reset target, actual `gripper_tcp`, `sfp_tip_link`, and `sfp_module_link` poses, reset insertion geometry, module geometry, and contact summary.

Key reset finding:

- `2026-05-24_04-49-33_wrist_reset_s8mm_tightik_resetdiag_zero_probe` placed the gripper TCP at the requested pose and placed the semantic tip at `s≈8.00 mm`, `r≈0.30 mm`, but the actual reset semantic tip orientation was already `theta≈0.0706 rad` and module consistency was only `0.60-0.62`. A correct gripper reset target does not imply a correct semantic tip/module pose under the current cable/module articulation and contact state.

Additional reset-orientation probes:

| run | best row | result |
|---|---|---|
| `wrist_reset_s8mm_tightik_zero_probe` | `s=7.997 mm`, `r=0.208 mm`, `theta=0.07073 rad`, consistency `0.584` | Tight reset IK did not fix semantic tip orientation or module consistency. |
| `wrist_reset_s8mm_derived_targettip_tightik` | `s=-144.457 mm`, `r=56.816 mm`, `theta=3.09583 rad`, consistency `0.000` | Rejected: the supplied desired tip quaternion was flipped relative to the port axis, so this was an invalid derivation target. |
| `wrist_reset_s8mm_rot_zpos_004_tightik` | `s=8.258 mm`, `r=0.347 mm`, `theta=0.07088 rad`, consistency `0.625` | Preserved depth but did not improve theta. |
| `wrist_reset_s8mm_rot_xpos_004_tightik` | `s=3.146 mm`, `r=0.217 mm`, `theta=0.05811 rad`, consistency `0.016` | Reduced theta, but lost seated depth and module consistency. |
| `wrist_reset_s8mm_rot_xpos_002_tightik` | `s=-0.761 mm`, `r=1.341 mm`, `theta=0.05488 rad`, consistency `0.000` | More disruptive than `x=0.004`; rejected. |
| `wrist_reset_s8mm_rot_xpos002_zpos004_tightik` | `s=-0.715 mm`, `r=1.442 mm`, `theta=0.05953 rad`, consistency `0.000` | Combining `+x` with `+z` destroyed insertion. |
| `wrist_reset_s8mm_rot_xpos0005_zpos004_tightik` | `s=8.400 mm`, `r=0.319 mm`, `theta=0.06790 rad`, consistency `0.621` | Preserved depth but remained far from strict theta/consistency. |
| `wrist_reset_s8mm_rot_xpos0005_tightik_resetdiag` | `s=8.039 mm`, `r=0.050 mm`, `theta=0.06721 rad`, consistency `0.552` | Small `+x` improved theta slightly but reduced module consistency. |

Guided final-window follow-up:

| run | best strict-relevant row | result |
|---|---|---|
| `progress7_fixed_world_xpos_depthhold_tight_step2_guideonly` | `s=8.019 mm`, `r=0.262 mm`, `theta=0.03608 rad`, consistency `0.912` | Best strict-relevant row so far, but still misses theta by `0.00608 rad`. |
| `progress7_fixed_world_xpos_depthhold_tight_step4_guideonly` | `s=8.012 mm`, `r=0.433 mm`, `theta=0.03712 rad`, consistency `0.833` | Larger `+x` step worsened theta/lateral/consistency; rejected. |
| `progress10_fixed_world_xpos_depthhold8mm_step2` | `s=8.243 mm`, `r=0.399 mm`, `theta=0.03628 rad`, consistency `0.814` | Applying `+x` to the online-adapted progress10 setup did not beat the progress7 guide-only candidate. |
| `progress7_fixed_world_xpos_depthhold_tight_step2_guideonly_240` | `s=8.019 mm`, `r=0.262 mm`, `theta=0.03608 rad`, consistency `0.912` | Extending to 240 steps did not improve the seated plateau. Lowest theta was `0.03602` at `s=7.789 mm`, below seated depth. |
| `progress7_fixed_world_xpos_earlyhold775_step2_guideonly` | `s=8.011 mm`, `r=0.430 mm`, `theta=0.03661 rad`, consistency `0.825` | Earlier final hold/trim regressed; rejected. |
| `progress7_xpos_step2_solver32_16_extforces_guideonly` | no valid inserted row; best score row had `s=-8.565 mm`, `theta=0.05545`, consistency `0` | Higher solver iterations plus external-force updates broke approach/insertion in this setup; rejected. |

Current best strict-relevant candidate:

- Run: `outputs/agentic_reward_curriculum_20260524_final_window_probe/runs/2026-05-24_04-52-05_progress7_fixed_world_xpos_depthhold_tight_step2_guideonly`
- Metrics: `s=8.019 mm`, `r=0.262 mm`, `theta=0.03608 rad`, consistency `0.912`
- Strict success: `false`
- Remaining gap: semantic tip orientation is still about `0.00608 rad` above the strict `0.030 rad` threshold.

Updated recommendation: keep the `+x, 0.002 rad` final trim as the best guarded final-window variant, but do not keep increasing rotation authority or moving the hold window earlier. The remaining work should target why the semantic tip/module orientation plateaus under contact: inspect the cable/module articulation and collision geometry, and add a diagnostic that compares the actual local transform between `gripper_tcp`, `sfp_module_link`, and `sfp_tip_link` before contact, at shallow insertion, and at the seated plateau. The evidence so far suggests that reward/curriculum and final-window guide changes can choose a near-seat state but cannot realize the last `~0.006 rad` of semantic tip orientation without violating module consistency.

## Artifacts

Reproducible run folders:
- `outputs/agentic_reward_curriculum_20260523_smoke/`
- `outputs/agentic_reward_curriculum_20260523_r105_smoke/`
- `outputs/agentic_reward_curriculum_20260523_r105_orientgate/`
- `outputs/agentic_reward_curriculum_20260523_r105_axisfinal/`
- `outputs/agentic_reward_curriculum_20260523_adaptive_orient/`
- `outputs/agentic_reward_curriculum_20260523_probe_basis/`
- `outputs/agentic_reward_curriculum_20260523_probe_basis_small/`
- `outputs/agentic_reward_curriculum_20260523_rotation_audit/`
- `outputs/agentic_reward_curriculum_20260523_module_recovery/`
- `outputs/agentic_reward_curriculum_20260523_module_recovery_zero/`
- `outputs/agentic_reward_curriculum_20260523_module_recovery_sm/`
- `outputs/agentic_reward_curriculum_20260523_contact_diagnostic/`
- `outputs/agentic_reward_curriculum_20260523_module_lateral/`

Each executed folder includes config, git status/diff, command logs, iteration decisions, metrics, phase summaries, and center/left/right camera images. No video was produced by these short probes.

## Recommendation

Do not move to reward-only or longer SERL training from this state. The remaining blocker is guide/controller orientation realization and module/body seating in the final millimeter: bounded full-quat, final-window axis-only, adaptive sign, six-direction basis refinement, staged contact command probes, and module-lateral correction could not bring semantic tip theta below about `0.059-0.064 rad` in the inserted regime, and module consistency remains far below the strict gate.

The next single code change should be a controller realization diagnostic for module-body motion: log commanded versus realized `sfp_tip_link` and `sfp_module_link` deltas under the module-lateral guard, including the projection of realized module motion onto the desired correction vector. If the module body cannot be moved laterally while the tip is in shallow contact, stop reward tuning and inspect asset/contact/IK constraints. Keep direct `sfp_tip_link` IK disabled.

## 2026-05-24 Compensated strict reset and pose-hold recovery

Added relative transform logging for `gripper_tcp -> sfp_module_link -> sfp_tip_link` in both guided SERL diagnostics and the standalone wrist/contact diagnostic. The local transforms were effectively stable between the guided plateau and direct reset cases, so the remaining error is not explained by a changing local module/tip transform alone.

Derived a gripper reset orientation from the stable actual `gripper_tcp -> sfp_tip_link` transform and a desired semantic tip orientation. With `--derive_reset_position_from_orientation`, the reset diagnostic can place both envs in a pre-step strict pose:

- Env0 reset: `s=8.007 mm`, `r=0.298 mm`, `theta=0.000 rad`, consistency `0.978`, strict true.
- Env1 reset: `s=8.077 mm`, `r=0.385 mm`, `theta=0.000 rad`, consistency `0.962`, strict true.

This is not counted as success because it is reset-only and visually unverified. The first physics step ejects the plug: zero/forward/pose-hold probes all had zero post-step strict rows.

Controller/contact recovery probes:

| run | best post-step row | result |
|---|---|---|
| `wrist_reset_s8mm_actualrel_targettip_poscomp_forward500` | `s=0.127 mm`, `r=0.477 mm`, `theta=0.04169 rad`, consistency `0.00004` | Forward command did not preserve insertion after reset ejection. |
| `wrist_reset_s8mm_actualrel_targettip_poscomp_posehold20` | `s=-0.401 mm`, `r=0.221 mm`, `theta=0.04075 rad`, consistency `0.00001` | Small pose-hold correction was too weak to recover after the first-step ejection. |
| `wrist_reset_s8mm_actualrel_targettip_poscomp_posehold2mm50` | `s=8.036 mm`, `r=0.380 mm`, `theta=0.04006 rad`, consistency `0.922` | Larger wrist pose-hold recovers depth/lateral/consistency but not strict semantic orientation. |
| `wrist_reset_s8mm_actualrel_targettip_poscomp_posehold_orient_holdrot02_trim0005` | `s=8.238 mm`, `r=0.505 mm`, `theta=0.03958 rad`, consistency `0.778` | Gated semantic orientation trim failed to improve theta and degraded lateral/consistency. |
| `wrist_reset_s8mm_actualrel_targettip_poscomp_posehold2mm50_bias_xpos02` | `s=-1.024 mm`, `r=1.672 mm`, `theta=0.04268 rad`, consistency `0.000002` | `+x` pre-bias destabilized insertion. |
| `wrist_reset_s8mm_actualrel_targettip_poscomp_posehold2mm50_bias_xneg02` | `s=8.600 mm`, `r=0.086 mm`, `theta=0.05780 rad`, consistency `0.558` | `-x` pre-bias preserved depth/lateral in one env but worsened theta/consistency. |
| `wrist_reset_s8mm_actualrel_targettip_poscomp_tipstab_1mm70` | `s=-0.780 mm`, `r=2.006 mm`, `theta=0.04218 rad`, consistency `0.000001` | Targeting the reset `sfp_tip_link` position through wrist deltas destabilized lateral/module consistency. |
| `wrist_reset_s8mm_actualrel_targettip_poscomp_tipstab_posonly2mm70` | `s=-0.439 mm`, `r=2.071 mm`, `theta=0.04026 rad`, consistency `0.000001` | Disabling orientation trim did not fix target-tip stabilization; direct tip-position correction is worse than gripper pose-hold. |
| `wrist_reset_s8mm_actualrel_targettip_poscomp_posehold_xpos0005_holdrot02_start24` | `s=8.038 mm`, `r=0.455 mm`, `theta=0.03960 rad`, consistency `0.727` | Stable but still misses theta and consistency. |
| `wrist_reset_s8mm_actualrel_targettip_poscomp_posehold_xpos001_holdrot02_gate06` | `s=7.581 mm`, `r=0.338 mm`, `theta=0.04113 rad`, consistency `0.656` | Repeated `+x=0.001 rad` trims gave a lower transient theta (`0.03882`) only below seated depth; rejected. |

Current best overall strict-relevant row remains the guided final-window run `progress7_fixed_world_xpos_depthhold_tight_step2_guideonly`: `s=8.019 mm`, `r=0.262 mm`, `theta=0.03608 rad`, consistency `0.912`, strict false.

Updated recommendation: the reset/controller can mathematically create a strict pose, but the contact step ejects it and active wrist pose-hold only recovers to a non-strict theta plateau. Direct semantic tip-position stabilization through wrist IK is worse than gripper pose-hold, which points to controller/contact coupling rather than a missing reward term. The next useful code path is a contact/controller feasibility fix: either modify final-seat contact/solver/asset constraints so the compensated strict pose remains stable for one post-step frame, or add a lower-level Cartesian hold that can apply the required correction before the contact impulse grows. Only after that should the guide/reward path be retried for visual evidence.
