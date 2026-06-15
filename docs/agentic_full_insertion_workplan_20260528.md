# Agentic Full-Insertion Workplan - 2026-05-28

Goal: achieve strict full SFP-to-NIC insertion from a non-contact near-gate start, initially targeting 40 mm axial and 10 mm lateral semantic tip offset. Strict success must remain the existing full-depth checker: full seated depth, lateral error, tip orientation, module/body consistency, and visual sanity.

## Current State

- Historical 8-10 mm insertion checkpoints are useful warm starts, but they were evaluated under the old shallow-depth interpretation and are not strict full insertion.
- Current 40/10 reset geometry is approximately correct at step 1, but guide behavior is sensitive to lateral-sign adaptation and reset/controller realization.
- Recent stable historical guide runs (`v247`, `v255`) did not reproduce under the current runtime when adaptive lateral sign was enabled.
- Fixed lateral sign with adaptive sign disabled (`v265`) restored lateral convergence from 40/10: step 180 reached `s=-27.30 mm`, `r=0.54 mm`, `theta=0.0907 rad`. This is not success, but it is a viable training base.
- Current blocker before full insertion is not depth alone: orientation remains far outside the strict `theta <= 0.03 rad` target while lateral centering improves.

## Time Allocation

Each block is capped at 7 hours. Stop a block early if it produces a strict success, clear blocker, or inferior result.

1. Stabilize reset/guide/controller baseline, up to 7 hours.
   - Reproduce a non-contact 40/10 guide rollout.
   - Audit gripper-to-tip and wrist-to-tip initialization randomness.
   - Disable or repair adaptive lateral sign when it flips against measured post-step improvement.
   - Promote only configs that improve post-step `s/r/theta/module` without false-positive depth.

2. Reward and curriculum ablations, up to 7 hours.
   - Compare fixed-sign guide with existing reward, stricter orientation-gated reward, and depth-gated semantic/module consistency.
   - Bias curriculum toward 40/10, then expand to 40/6, 30/10, 20/10 only after the 40/10 near-gate path improves.
   - Reject reward variants that pay positive axial progress while `r/theta/module` are off gate.

3. Literature-review-based experiments, up to 7 hours.
   - Review SERL/HIL-SERL, Eureka-style reward synthesis, VLM reward learning, and contact-rich insertion work.
   - Convert only testable ideas into config or small code patches.
   - Prioritize phase-conditioned reward and residual/guarded policy control over broad reward-only tuning.

4. Model architecture experiments, up to 7 hours.
   - Audit whether the online actor/critic uses only current observations.
   - Test low-risk history stacking or recurrent/state-history adapter before replacing the full policy.
   - Evaluate stronger visual encoders only if the current code path can load/train them without breaking ACT/SERL compatibility.

5. Training and rollout cadence, continuing across the above blocks.
   - Run conservative SERL from the best 8-10 mm/full-depth warm checkpoint only after guide geometry is sane.
   - Roll out every roughly 30 minutes or every major checkpoint.
   - During rollouts, save center/left/right videos or equidistant snapshots from episode start.
   - Stop training early if post-step lateral error diverges or if positive `s` appears with poor module consistency.

## Immediate Experiment Queue

1. `v266`: conservative SERL from fixed-sign 40/10 guide (`v265` base), delayed actor updates. Rejected: lateral stayed near gate but theta monotonically worsened while depth increased.
2. `v267`: fixed-sign guide with `--target_action_guide_target_tip_clamp_positive_axial_when_gated`. Rejected: explicit positive axial command stayed zero, but lateral correction/controller realization still produced inward semantic depth while theta was bad.
3. `v268`: early probe-basis orientation trim with strict predicted lateral gates. Rejected: theta and lateral both worsened; rotation prediction did not transfer to realized controller behavior.
4. Next: add measured orientation-trim acceptance/cooldown or collect near-centered rollouts for imitation-heavy training with axial progress disabled until theta is controlled.
5. If full-depth progress remains blocked after lateral/theta are good: increase axial step modestly and add module-consistency-gated depth reward.

## Runs Since Workplan Start

| Run | Type | Result | Decision |
| --- | --- | --- | --- |
| `eval_v265_fixed_lateral_sign_pos_noadaptive_step180` | guide-only ablation | Step 180: `s=-27.30 mm`, `r=0.54 mm`, `theta=0.0907 rad`. Fixed lateral sign restores centering, but orientation degrades. | Promote only as lateral-centering base. |
| `train_v266_fixedsign40x10_theta_reward_lr5e8_q1e4_step3200` | short SERL training | Stopped around step 540. Step 500: `s=-13.52 mm`, `r=0.60 mm`, `theta=0.1579 rad`. This is depth progress with bad theta. | Reject as false-progress risk. |
| `eval_v267_fixedsign_clamp_positive_axial_when_theta_bad_step260` | guide-only ablation | Same geometry as v265/v266 through step 260 despite positive-axial clamp; realized depth still increased from lateral/control coupling. | Reject; clamp is insufficient alone. |
| `eval_v268_fixedsign_early_probe_orientation_step260` | guide-only orientation ablation | Step 260: `s=-33.40 mm`, `r=21.13 mm`, `theta=0.1468 rad`. | Reject; early rotation causes lateral loss. |
| `eval_v269_fixedsign_late_tight_probe_orientation_step320` | guide-only orientation ablation | Late tight activation preserved lateral (`r=0.12 mm` at step 320) but theta still worsened to `0.1160 rad` while depth advanced to `s=-14.48 mm`. | Reject; current orientation trim direction/realization is not reducing semantic theta. |
| `eval_v270_late_orientation_realized_reject_step320` | guide-only measured-rejection ablation | Same `s/r/theta` as v269. New metrics showed `final_orientation_active=1.0` but `final_orientation_command_active=0.0`, so probe-basis strict gating selected no rotation. | Reject; measured rejection works as instrumentation, but no command was issued to reject. |
| `eval_v271_late_fullquat_orientation_reject_step260` | guide-only full-quat orientation ablation | Final orientation command became active near step 172, immediately increased theta, and the measured rejection cooldown fired. Step 260: `s=-20.03 mm`, `r=0.10 mm`, `theta=0.1030 rad`. | Reject; positive-sign full-quat rotation is harmful, but the realized rejection guard is working. |
| `eval_v272_late_fullquat_orientation_reject_rotsignneg_step260` | guide-only full-quat sign ablation | Negative rotation sign also triggered rejection after one command. Step 260: `s=-24.34 mm`, `r=0.20 mm`, `theta=0.1049 rad`; `final_orientation_command_active=0.0` after cooldown. | Reject; the current full-quat orientation trim cannot reduce semantic theta from the 40/10 approach state. |
| `train_v273_fixedsign40x10_realized_reject_theta_gate_step3200` | conservative online SERL training | Stopped early after step 550. Step 500: `s=-27.50 mm`, `r=0.15 mm`, `theta=0.0837 rad`; step 550 regressed to `s=-28.48 mm`, `r=1.51 mm`, `theta=0.0887 rad`. | Reject; training did not fix the semantic orientation blocker and began losing lateral alignment. |
| `v274_40x10_preinsert_rotation_axes` | wrist/contact rotation diagnostic | Failed before stepping because cameras were configured but `--enable_cameras` was omitted. | Preserve as failed-run evidence; rerun serially with cameras enabled. |
| `v275_40x10_preinsert_rotation_axes` | wrist/contact rotation diagnostic | Diagnostic approach used the wrong lateral sign and drove `r` from about 10 mm to 41 mm. | Reject as invalid orientation evidence. |
| `v276_40x10_preinsert_rotation_axes_latsignneg` | wrist/contact rotation diagnostic | Flipped lateral sign centered near the entrance but overshot to `s=-8.70 mm` by step 180. `+rx` reduced theta only about `0.0003 rad` per 0.001 rad command; `-rx`, `+/-y`, and `+/-z` worsened theta/lateral. | Weak directional evidence for bounded `+x`, but not sufficient alone. |
| `eval_v277_guard_fixed_x_orientation_step260` | guide-only guard-axis transfer test | Fixed `+x` final orientation did not transfer: step 260 `s=-36.08 mm`, `r=8.67 mm`, `theta=0.1035 rad`. | Reject; do not train on this guard setting. |
| `smoke_v278_critic_history4_step2` | architecture smoke | New critic-only history path ran for steps 1-2. Config recorded `critic_state_history_steps=4`, `critic_state_dim=328`; no Isaac process remained afterward. Step 2 reset/settle geometry: `s=-39.90 mm`, `r=11.35 mm`, `theta=0.0726 rad`. | Plumbing accepted; use only after guide improves. |
| `v279_40x10_reset_settle_seed56800` | reset-settle ablation | No-learning zero-action settle via `validate_serl_reset_settle.py`: step 1 `s=-38.80 mm`, `r=9.69 mm`, `theta=0.0678`; step 2 `s=-38.09 mm`, `r=11.57 mm`, `theta=0.0731`. | Reset reaches roughly 40/10, but first-step settle can add ~0.7 mm inward and ~1.9 mm lateral drift; controller-settle variance must be accounted for before reward/training claims. |
| `smoke_v280_convnext_critic_step1` | architecture smoke | `--critic_image_encoder_override convnext_tiny` initialized and ran one Isaac step. Config recorded `critic_image_encoder=convnext_tiny`; no Isaac process remained afterward. | Stronger critic vision path is config-selectable; do not train it until guide probes stop failing theta/lateral. |
| `v281_40x10_v279calib_reset_settle_seed56802` | reset-settle calibration validation | Patched `calibrate_near_gate_reset_from_settle_metrics.py` so SERL metrics can serve as the calibration source. The single-seed calibrated config worsened reset geometry: step 2 `s=-43.72 mm`, `r=18.35 mm`, `theta=0.0719`. | Reject generated config; single-seed settle/orientation calibration is too aggressive and should not seed training. |
| `eval_v282_broad_initial_settle_step260` | no-learning guide | Broadened initial settle guard for 40/10 starts. Step 260: `s=-19.30 mm`, `r=0.14 mm`, `theta=0.0755`, module consistency gate `0`. | Promote as better approach guide than v272, but still orientation/module blocked. |
| `train_v283_broadsettle_history4_full_depth_step3200` | online SERL training attempt | Failed at the first critic update because the new critic-history path exposed a projection dimension mismatch. | Preserve as architecture-smoke failure; patch history handling before using it again. |
| `train_v284_broadsettle_history4_full_depth_step3200` | online SERL training attempt | Failed at replay sampling because sampled `critic_state` history vectors had variable lengths. | Preserve as history-path failure; replay stacking now pads variable critic-state lengths and the encoder truncates/pads to configured size. |
| `train_v285_broadsettle_full_depth_step3200` | online SERL training, no critic history | Stopped at checkpoint 800. Step 800: `s=-12.11 mm`, `r=0.08 mm`, `theta=0.1042`, module consistency gate `0`. | Reject as tip-depth false-positive risk; training learned inward motion while semantic orientation and module consistency worsened. |
| `eval_v286_broadsettle_theta_depth_recovery_step300` | guide-only recovery probe | Step 300: `s=-22.17 mm`, `r=0.14 mm`, `theta=0.0755`, module consistency `0`. | Reject; slowed false depth but did not improve theta/module consistency. |
| `eval_v287_guard_targettipservo_module_gate_step300` | guide-only target-tip servo/module-gated guard | Step 300: `s=-27.94 mm`, `r=0.89 mm`, `theta=0.0728`, module consistency `0`. | Reject; axial gate worked but theta remained blocked and lateral drifted. |
| `v288-v293_reset_theta_correction` | reset/theta correction ablation | Scaled forward theta correction `0.75` plus settle calibration (`v293`) gave reset-settle step 3 `s=-40.82 mm`, `r=10.63 mm`, `theta=0.0486`. | Promote as best 40/10 reset orientation seed so far. |
| `eval_v294_theta075_settlecal_guardservo_step300` | guide-only on v293 reset | Step 300: `s=-31.88 mm`, `r=0.13 mm`, `theta=0.0488`, module consistency `0`. | Promote to short training candidate, with strict false-positive checks. |
| `train_v295_theta075_settlecal_guardservo_step1600` | short SERL training attempt | Stopped after guide failure was reproduced: step 320 `s=-30.67 mm`, `r=0.12 mm`, `theta=0.0464`; step 360 `s=-24.39 mm`, `r=41.70 mm`, `theta=0.0923`. | Reject; the teacher/guard failed before actor learning could be credited. |
| `eval_v296_theta075_settlecal_guardservo_step500` | extended guide-only reproduction | Same transition failure: step 320 `s=-30.67 mm`, `r=0.12 mm`, `theta=0.0464`; step 330 `s=-25.12 mm`, `r=40.97 mm`, `theta=0.0848`. | Reject; concrete bypass transition near step 330. |
| `eval_v297_theta075_recovery_keep_lateral_step500` | recovery without zeroing lateral | Step 500: `s=-22.43 mm`, `r=1.90 mm`, `theta=0.0819`. | Reject; catastrophic bypass avoided but lateral/theta worsened. |
| `eval_v298_theta075_recovery_stronger_lateral_step500` | stronger lateral correction | Held controlled approach through step 480: `s=-19.58 mm`, `r=0.08 mm`, `theta=0.0477`; failed by step 500: `s=-12.89 mm`, `r=40.01 mm`, `theta=0.0889`. | Best pre-contact guide state, but still not trainable past the contact transition. |
| `eval_v299_theta075_early_module_recovery_step600` | early module/shallow bypass recovery | Recovery activated near `s=-20 mm` but still failed: step 490 `s=-19.42 mm`, `r=0.03 mm`, `theta=0.0485`; step 500 `s=-13.46 mm`, `r=39.98 mm`, `theta=0.0888`. | Reject; recovery branch commanded backoff but realized depth still moved inward. |
| `eval_v300_module_recovery_signflip_step540` | module recovery backoff sign flip | Failed earlier before module recovery activation: step 350 `s=-26.54 mm`, `r=37.71 mm`, `theta=0.0811`. | Reject; sign flip alone is not a stable fix. |
| `eval_v301_outer_module_recovery_signflip_step460` | earlier module recovery activation | Module recovery activated but did not stop bypass: step 340 `s=-27.87 mm`, `r=0.47 mm`, `theta=0.0492`; step 350 `s=-21.41 mm`, `r=41.94 mm`, `theta=0.0894`. | Reject; recovery axial/backoff remains unsafe under contact realization. |
| `train_v302_theta075_terminated_bypass_step2400` | short SERL with lateral/off-gate failure terminations | Stopped at checkpoint 800. Terminations reset after bad off-gate/bypass segments, but reward became unstable after updates. | Preserve checkpoint as failed training evidence; evaluate before any continuation. |
| `eval_v303_train_v302_ckpt800_policy_guard_step600` | held-out rollout of v302 checkpoint 800 | Best depth was step 600 `s=-9.1 mm`, but `r=31.02 mm`, `theta=0.1971`, module consistency `0`, strict success `false`. | Reject; training worsened orientation and produced lateral bypass/false progress. |
| `eval_v304_recovery_zero_axial_step540` | guide/guard ablation | `--insertion_action_guard_recovery_zero_axial` made recovery axial zeroing active, but realized motion still moved inward/laterally; step 540 `s=17.62 mm`, `r=15.87 mm`, `theta=0.1621`, module consistency `0`. | Reject; axial-zero recovery alone is not a safe training target. |
| `eval_v305_prelip_offgate_hold_zero_lateral_step420` | zero-command/hold diagnostic | Guarded command was zero, body-relative plug distances stayed constant, but the controlled chain drifted from step 1 `s=-43.68 mm`, `r=11.57 mm` to step 420 `s=-8.97 mm`, `r=12.08 mm`. | Evidence of controller/contact drift under hold; not a policy success/failure signal. |
| `eval_v307_debug_zero_action_settle80` / `eval_v308_debug_zero_action_gravityoff_settle80` | true zero-action diagnostics | True zero action was much more stable than the earlier reset wrapper; gravity-off reduced force after settling but worsened lateral drift (`r=15.42 mm` at step 80). | Keep gravity on; reset wrapper zero-action mode was not a pure zero-action test. |
| `eval_v310_initial_settle100_then_guide_step520` | guide-only staged settle | Reproduced useful approach segment: step 180 `s=-37.16 mm`, `r=0.13 mm`, `theta=0.0787`; later failed by step 460 `s=-21.47 mm`, `r=45.07 mm`, `theta=0.1101`. | Promote only the approach-centering segment; do not train beyond the failure window. |
| `train_v311_settle100_terminated_bypass_step900` | short SERL training attempt | Stopped after checkpoint 300. It cycled in settle/pre-contact with step 359 `s=-40.77 mm`, `r=9.26 mm`, `theta=0.0524`; no useful insertion data. | Reject and stop; terminations/seed prevented collecting the good v310 centered segment. |
| `train_v312_v310seed_centering_segment_step320` | staged SERL/guide training | Trained only the v310 approach segment. Checkpoints at 160/320; teacher reached step 160 `s=-37.59 mm`, `r=0.06 mm`, `theta=0.0775`. | Promote checkpoint 160 as approach-centering base only. |
| `eval_v313_train_v312_ckpt160_policy_guard_step520` | guide-off rollout | Policy+guard preserved centering and avoided the previous 40 mm bypass: step 520 `s=-24.17 mm`, `r=0.08 mm`, but `theta=0.1298`, module consistency `0`. | Promote as better policy base for approach; reject for insertion because orientation/module remain false-positive risks. |
| `eval_v315_ckpt160_allowfinal_orientprobe_step360` | guide-on orientation probe | Enabling final orientation during realized-r recovery made orientation commands engage. Step 360 `s=-23.57 mm`, `r=0.16 mm`, `theta=0.0600`, module consistency `0`. | Promote as orientation-teacher candidate; still not strict. |
| `train_v316_orientprobe_teacher_step420` | staged orientation training | Teacher improved theta to step 420 `s=-19.16 mm`, `r=1.66 mm`, `theta=0.0439`, module consistency `0`, with checkpoints at 140/280/420. | Preserve checkpoint, but train/eval must verify guide-off learning. |
| `eval_v317_train_v316_ckpt420_policy_guard_step520` | guide-off rollout | Policy did not learn the orientation improvement; step 280 `theta=0.1099`, and later bypassed at step 500 `s=-19.79 mm`, `r=42.62 mm`. | Reject as strict insertion policy; orientation must be explicit guard/residual or trained differently. |
| `eval_v319_ckpt420_policy_guard_finalrot_always_step420` | policy+guard final-rotation residual | Enabling the existing guarded `frame_quat_best` final rotation with `0.0005 rad` step avoided bypass and held `r=0.10 mm` at step 420, but theta plateaued at `0.0760`. | Promote as safer guard setting, but not sufficient for strict orientation. |
| `eval_v320_ckpt420_policy_guard_finalrot_step001_step420` | rotation-step sweep | `0.001 rad` reduced theta to `0.0500` by step 380 but lateral grew to `2.67 mm` and bypass occurred by step 400 (`r=42.77 mm`). | Reject; larger rotation improves theta but breaks lateral retention. |
| `eval_v321_ckpt420_finalrot001_holddepth33_step420` | rotation plus depth hold | Depth hold engaged near `s=-33 mm`, but once lateral exceeded the final-orientation gate, rotation/hold shut off and the same bypass occurred by step 400. | Reject; next change must keep recovery active after lateral grows, not just hold while centered. |
| `eval_v322_ckpt420_finalrot001_recoverylat3mm_step420` | patched guard recovery lateral window | Added `--insertion_action_guard_final_orientation_recovery_lateral_m`. With `3 mm`, behavior improved through step 300 (`s=-30.51 mm`, `r=0.09 mm`, `theta=0.0458`) but still jumped by step 320 (`r=40.55 mm`). | Patch helps pre-jump stability but 3 mm recovery is too small for the realized contact jump. |
| `eval_v323_ckpt420_finalrot001_recoverylat50mm_step460` | large recovery-window diagnostic | With `50 mm`, hold/rotation/recovery stayed active after the jump and slowly reduced `r` from `40.55 mm` to `34.49 mm` by step 460, but theta stayed about `0.082` and module consistency stayed `0`. | Confirms guard can stay active after bypass, but correction is too slow; must prevent the jump rather than recover from it. |
| `eval_v324_ckpt420_finalrot001_holddepth36_step700` | shallower final-orientation hold | Holding around `s=-36 mm` kept lateral tight much longer: step 500 `s=-26.52 mm`, `r=0.21 mm`, `theta=0.0458`; still jumped by step 540 (`r=40.57 mm`). | Best current stable pre-jump trajectory; strict failure remains orientation residual plus contact/controller lateral bypass. |
| `eval_v325_ckpt420_finalrot002_holddepth36_step560` | stronger rotation sweep | `0.002 rad` rotation reduced theta faster but caused lateral growth and early bypass: step 280 `r=2.54 mm`, step 300 `r=43.41 mm`. | Reject; stronger rotation authority is unsafe. |
| `train_v326_stable_orienthold36_step500` | short SERL from v324-style guarded behavior | Completed 500 steps with `updates_done=76`, checkpoints at 250/500. Step 500: `s=-26.52 mm`, `r=0.21 mm`, `theta=0.0458`, module consistency `0`. | Preserve checkpoint; training did not solve strict orientation/module consistency but produced a reproducible stable pre-jump checkpoint. |
| `eval_v327_train_v326_ckpt500_policy_guard_step700` | held-out rollout of v326 checkpoint 500 | Reproduced v324 almost exactly. Step 500 `s=-26.52 mm`, `r=0.21 mm`, `theta=0.0458`; step 520 jumped to `s=-19.13 mm`, `r=40.98 mm`, `theta=0.0863`, module consistency `0`. | Reject as strict insertion; failure label is controller/contact-induced lateral bypass with orientation residual. |
| `eval_v328_train_v326_ckpt500_failure_images_step560` | held-out rollout with images through failure window | Same metrics as v327 and saved center/left/right frames through `step_000560`, including `step_000500` before the jump and `step_000520` after it. | Preserve as visual evidence for blocker analysis. |

## Literature Notes

- SERL emphasizes sample-efficient off-policy robotic RL with strong reset/control infrastructure; for this task, that argues for fixing reset/servo/controller realization before expecting reward-only training to succeed. Source: https://arxiv.org/abs/2401.16013
- HIL-SERL reinforces the same practical pattern: precise manipulation benefits from intervention/guide data and conservative online updates rather than letting the actor chase early untrained Q estimates. Source: https://hil-serl.github.io/
- Eureka-style reward synthesis is relevant, but its reward-code loop must be constrained by strict geometry/video checks here because positive reward or positive tip depth can be false progress. Source: https://arxiv.org/abs/2310.12931
- RL-VLM-F suggests preference/visual feedback can help reward learning, but for this setup VLM feedback should be used as an audit signal on saved camera frames, not as the primary success metric. Source: https://arxiv.org/abs/2402.03681
- Contact-rich insertion literature repeatedly favors residual policies, compliance/force signals, phase decomposition, and demonstrations for peg/connector insertion. The immediate actionable idea is residual learning around a guarded controller with force/contact and module-consistency gates, not a free actor. Examples: https://www.sciencedirect.com/science/article/pii/S0278612523002352 and https://arxiv.org/abs/2106.04306
- Partial observability is a credible architecture issue because the online SERL replay currently stores current observation/action transitions, while controller coupling/contact state depends on recent realized motion. A bounded architecture experiment should test state/action history stacking before a larger recurrent policy rewrite.

## Architecture Notes

- The online SERL critic path already supports `small_conv`, `resnet18`, `resnet18_imagenet`, `convnext_tiny`, and `convnext_tiny_imagenet`.
- Added `--critic_image_encoder_override` to `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py` so critic-encoder ablations can be run config-only while preserving existing checkpoint configs and ACT actor behavior.
- Added `--critic_state_history_steps` to the same script. This is critic-only: `obs["state"]` remains the current 82D policy state for ACT, while the online critics may consume a flattened current-plus-recent state history through `obs["critic_state"]`. This tests partial observability without changing exported/runtime ACT policy interfaces.
- These architecture switches do not solve the current guide/controller issue by themselves. They are for the architecture ablation block after a non-false-positive guide/training target exists.

## Literature-Driven Experiment Rules

- SERL and HIL-SERL both argue that robot RL success depends on resets, controller quality, and useful guide/intervention data, not reward tuning alone. For this task, guide-only probes must improve strict `s/r/theta/module` before long online updates.
- Eureka-style reward iteration is useful only under strict geometric/video gates here. Candidate rewards must be audited offline over `s/r/theta/action-axis/module` and must assign low value to forward motion when orientation or lateral gates fail.
- RL-VLM-F motivates using visual feedback as an auxiliary audit of saved frames, but the success label remains the strict post-step geometry checker.
- Contact-rich insertion work repeatedly favors phase decomposition, residual control around a stable controller, force/contact recovery, and history/tactile/force state. The next controller experiment should therefore be a state-aware retention/orientation guard or residual imitation target, not global rotation authority.
- Architecture experiments are capped to critic-only changes first: `--critic_state_history_steps 4` and `--critic_image_encoder_override convnext_tiny`/`convnext_tiny_imagenet` once a guide configuration stops creating false depth progress.

## Logging Requirements

Every promoted run must preserve:
- `train_config.json`
- `metrics.jsonl`
- checkpoint path if training
- post-step insertion geometry table
- image/video evidence for rollouts
- markdown summary with strict success verdict and false-positive checks

Strict success remains unachieved until the strict checker reports success and visual evidence agrees.
