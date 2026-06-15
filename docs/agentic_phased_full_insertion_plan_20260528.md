# Agentic Phased Full-Insertion Plan - 2026-05-28

## Current Ground Truth

- All training and rollout jobs were stopped before this plan was written.
- Strict full insertion is still unachieved. The best stable current candidate is still the v326/v327/v328 family: around step 500 it reaches `s=-26.52 mm`, `r=0.21 mm`, `theta=0.0458`, and module consistency `0`; it then suffers a contact/controller lateral bypass.
- The old 8-10 mm insertion behavior must be treated as a shallow-depth warm start only. Under the corrected full-depth metric it is not success.
- The reset generator starts semantic tip references outside the entrance plane by `start_near_gate.axial_distance_m` and lateralizes by a sampled perpendicular direction. Existing repair scripts compensate measured `wrist_3_link`/`sfp_tip_link` offsets, but single-seed settle calibration has already produced regressions and must be validated per generated episode.

## Time Allocation

Each track is capped at 7 hours of wall-clock experiment time before it must either promote a candidate, reject the track with evidence, or move to the next track.

| Track | Cap | Main question | Promotion gate |
| --- | ---: | --- | --- |
| A. Reset and controller audit | 7 h | Is the semantic tip reset and realized wrist-to-tip motion stable enough for learning? | 40/10 start validates within tolerance after settle; no contact-induced jump before controlled insertion begins. |
| B. Reward/curriculum ablations | 7 h | Can phased gates prevent tip-depth false positives while preserving approach? | Offline reward audit plus guide/training rollout improves `s/r/theta/module` jointly. |
| C. Literature-derived controller ideas | 7 h | Do residual/guarded/contact-aware ideas from SERL/HIL-SERL/Eureka/RL-VLM-F/contact insertion help? | A config/code patch with better strict metrics than v326 without visual bypass. |
| D. Architecture ablations | 7 h | Is partial observability or weak vision blocking learning? | History or stronger vision improves held-out rollout metrics without breaking ACT/SERL paths. |
| E. Training from warm starts | 7 h per promoted base | Can SERL improve a promoted guide or shallow-insertion checkpoint? | Held-out 40/10 rollout improves strict metrics; bad false-depth replay is stopped early. |

## Immediate Commands

Stop any accidental processes:

```bash
docker exec isaac-lab-base bash -lc 'pkill -9 -f "aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py|isaaclab.sh -p" || true'
```

Rank shallow/full-depth warm starts:

```bash
.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/select_insertion_candidates.py \
  outputs/agentic_reward_curriculum_20260523_orientation_trim \
  outputs/agentic_reward_curriculum_20260523_module_lateral \
  outputs/agentic_reward_curriculum_20260524_retention_guard \
  outputs/agentic_reward_curriculum_20260527 \
  --limit 40 --max-rows-per-file 2500
```

Initial bounded scan result:

- Final-window runs can report `s ~= 45-49 mm` at step 1, but the scan shows `strict_success=false`; these are not 40/10 non-contact insertion policies and must not be used as success evidence.
- The closest scanned shallow-depth 40/10-style run was `eval_guide_teacher_rotclip_thetagate04_40x10_v222_novideo_step800`: best scanned depth `s=10.04 mm` at step 674, but `r=24.25 mm`, `theta=0.1035`, module consistency `0`, strict success `false`. This is a warm-start/debug candidate only, not insertion.
- The safer current policy base remains `train_v326_stable_orienthold36_step500`, because it preserves centering before contact (`s=-26.52 mm`, `r=0.21 mm`, `theta=0.0458`, consistency `0`) even though it later fails by controller/contact lateral bypass.

Validate reset/settle before training a candidate:

```bash
.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/validate_serl_reset_settle.py \
  --episode-config-dir outputs/agentic_reward_curriculum_20260527/generated_episode_configs/full_depth_start40x10_theta0005_theta_corr_scale075_settlecal_v293 \
  --run-name reset_settle_40x10_current --steps 5 --num-envs 1
```

## Ablation Matrix

1. Reset/gripper-tip ablations:
   - Current `wrist_3_link` reset with measured `sfp_tip_link` repair.
   - Same episode without single-seed settle correction.
   - Two or more seeds for lateral direction sampling.
   - Promotion requires step-1/step-3 semantic `s/r/theta` close to requested 40 mm / 10 mm and no large first-step drift.

2. Reward/curriculum ablations:
   - Existing cheatcode insertion reward.
   - Tight phase-conditioned funnel: no axial credit unless `r`, `theta`, action-axis, and module consistency gates pass.
   - Depth-gated semantic consistency: module consistency penalty only after shallow stable insertion.
   - Bypass penalty: positive tip `s` with low module consistency is negative evidence.

3. Guard/controller ablations:
   - Current v326/v340 contact/module recovery.
   - Backout-only recovery with lateral correction preserved and no reinsert until theta recovers.
   - Force/contact spike preemption before the large lateral jump.
   - Reject any setting that commands or realizes inward motion while `r/theta/module` are off gate.

4. Architecture ablations:
   - Current ACT/SERL actor, current observation only.
   - Online critic history via `--critic_state_history_steps 4` after history replay validation.
   - Stronger critic vision via `--critic_image_encoder_override convnext_tiny_imagenet`.
   - New ACT BC option: `configs/train/act_resnet50.yaml`, leaving default `configs/train/act.yaml` unchanged.
   - Offline SERL stronger critic option: `configs/train/vision_offline_serl_convnext_history.yaml`.

## Literature-Derived Rules

- SERL emphasizes that sample-efficient robot RL depends on off-policy learning plus high-quality controllers, resets, and reward implementations; here that means reset/servo evidence must precede long training. Source: https://arxiv.org/abs/2401.16013
- HIL-SERL supports intervention/guide data for precise manipulation; for this task, failed guide segments should be treated as intervention labels to avoid, not replay to imitate. Source: https://hil-serl.github.io/
- Eureka motivates automated reward-code search, but every generated reward must be audited against strict geometry/video checks because tip-depth-only positive `s` is a known false positive. Source: https://arxiv.org/abs/2310.12931
- RL-VLM-F motivates visual preference checks on saved center/left/right frames, but VLM or visual judgment remains auxiliary; strict success is still post-step geometry plus visual sanity. Source: https://arxiv.org/abs/2402.03681
- Contact-rich insertion literature favors phase decomposition, residual control, force/contact monitoring, and recovery. The next useful experiment is a guarded residual/servo target with history/contact features, not free actor updates.

## Stop/Promote Rules

- Stop a run early if lateral error exceeds 3 mm after the centered phase, if theta grows while `s` increases, or if module consistency remains low during positive depth progress.
- Promote only if post-step `s`, `r`, `theta`, and module consistency improve together and saved frames do not show lateral bypass.
- Strict success still requires the repository strict checker plus center/left/right visual evidence. Reward return is not evidence of success.

## 2026-05-28 Continuation Notes

- All active Isaac training/rollout jobs were stopped after the v344 interruption request. A follow-up `pgrep` only matched the verification command itself.
- Reset-settle validation on `full_depth_start40x10_theta0005_theta_corr_scale075_settlecal_v293` showed substantial zero-action inward settle drift: step 1 `s=-42.03 mm`, `r=10.89 mm`, `theta=0.0499`; step 5 `s=-29.01 mm`, `r=9.49 mm`, `theta=0.0472`. This means nominal 40/10 episodes can effectively become much closer after settle.
- A settle-step-5 calibrated episode was generated at `outputs/agentic_reward_curriculum_20260528/generated_episode_configs/full_depth_start40x10_v293_settle5_calibrated`. Validation improved the post-settle effective start: step 5 `s=-38.42 mm`, `r=9.38 mm`, `theta=0.0474`.
- `eval_v343_settle5cal_ckpt500_guard_step420` used the calibrated episode but was rejected for lateral blow-up: step 40 `s=-46.41 mm`, `r=47.33 mm`, `theta=0.0708`; step 420 `s=-19.49 mm`, `r=21.95 mm`, `theta=0.0711`; module consistency `0`, strict success `false`.
- `train_v344_40x10_orientrefine_from_v342_step1200` was intentionally stopped at step 183. It warm-started from v342 and preserved centering while improving depth, but remained orientation/module blocked: best/latest step 183 `s=-34.86 mm`, `r=0.075 mm`, `theta=0.0575`, module consistency `0`, strict success `false`.
- Important caveat: v344 inherited the older `full_depth_start40x10_theta0005_theta_corr_scale075_settlecal_v293` episode config rather than the new settle-5 calibrated episode. Any promoted 40/10 training should use the calibrated episode explicitly and persist the exact generated command.
- Added `aic_utils/aic_isaac/scripts/build_serl_command_variant.py` to create reproducible command variants by replacing flags instead of appending duplicates.
- `eval_v345_calibrated_adaptive_lateral_ckpt_v342_step260` used a clean command with the settle-5 calibrated episode and adaptive lateral sign enabled. It was rejected: step 1 `s=-54.69 mm`, `r=12.33 mm`, `theta=0.0511`; step 20 `s=-47.51 mm`, `r=51.56 mm`, `theta=0.0722`; step 260 `s=-44.78 mm`, `r=49.98 mm`, `theta=0.0715`; module consistency `0`, strict success `false`.
- `eval_v346_calibrated_pureguide_step220` removed actor contribution with `target_action_guide_collect_blend=1.0`. It did not blow laterally: step 1 `s=-54.69 mm`, `r=12.25 mm`, `theta=0.0507`; step 120 `s=-48.39 mm`, `r=0.07 mm`, `theta=0.0614`; step 220 `s=-43.37 mm`, `r=0.08 mm`, `theta=0.0591`; module consistency `0`, strict success `false`.
- `eval_v347_pureguide_early_finalorient_step360` activated the final-orientation window earlier at `s=-48 mm`. It was rejected: best tight-lateral point was step 183 `s=-45.18 mm`, `r=0.20 mm`, `theta=0.0567`; by step 200 `r=1.39 mm`, and later rows show reset/fallout. Earlier final orientation did not solve theta and reduced corridor robustness.
- `train_v348_calibrated_pureguide_history_step600` tested conservative SERL with `critic_state_history_steps=4`, decayed pure-guide collection, and strict bypass/off-gate terminations. It was stopped at step 279 because it was slow and unpromising: step 220 `s=-43.36 mm`, `r=0.10 mm`, `theta=0.0595`; step 260 `s=-41.34 mm`, `r=0.12 mm`, `theta=0.0626`; step 279 `s=-40.30 mm`, `r=0.09 mm`, `theta=0.0657`; module consistency `0`, strict success `false`.
- `eval_v349_guard_prefinal_trim_step320` tested a guard-side pre-final orientation trim using tiny pulsed `frame_quat_best` rotations, strict lateral/depth prediction, and no module-consistency requirement before contact. This is the best current controller direction: best tight-lateral theta was step 295 `s=-44.31 mm`, `r=0.09 mm`, `theta=0.0412`; final step 320 `s=-43.95 mm`, `r=0.12 mm`, `theta=0.0413`; module consistency `0`, strict success `false`.
- `eval_v350_guard_prefinal_trim_stronger_step650` increased trim step/pulse rate and was rejected: lateral error exceeded 2 mm by step 200 and later reached about 10 mm. Stronger rotation authority recreates the known lateral-sweep failure mode.
- `eval_v351_guard_prefinal_trim_long_step700` extended the safer v349 settings. It reproduced v349 through step 320 and then reset/fell out around step 360. Best tight-lateral theta remained step 295 `s=-44.31 mm`, `r=0.09 mm`, `theta=0.0412`; best depth was step 354 `s=-43.36 mm`, `r=0.07 mm`, `theta=0.0420`; module consistency `0`, strict success `false`.
- Current best trainable evidence is not strict insertion. Calibrated-reset plus ACT/checkpoint rollout can still produce immediate lateral blow-up, while pure guide is laterally safe but too slow and orientation blocked. History-critic training preserves centering but does not solve the orientation gap and is too slow for broad sweeps. The v349/v351 guard-side pre-final trim narrows theta from about `0.059` to about `0.041` while preserving `r < 0.5 mm`, but it plateaus above the strict `0.030` rad gate and does not produce module consistency. The next bounded change should add plateau-aware micro-orientation dithering or a learned/servo residual only inside this v349-style strict lateral/depth envelope.
- `train_v362_earlyguard_guided_policy_30min` was the first actual training attempt after revalidating a stable early guard. It used guide-heavy actor updates from the v349-style command. It reached best tight-lateral geometry at step 391: `s=-34.77 mm`, `r=0.15 mm`, `theta=0.0420`, module consistency `0`, strict success `false`, then reset at step 392 and again later. Training was stopped at step 600 because later episodes degraded into large lateral errors while consistency stayed zero.
- `eval_v365_no_failure_terms_diagnostic_step430` disabled failure terminations only to expose the hidden post-step failure. It showed the controller failure directly: step 391 was centered (`s=-34.77 mm`, `r=0.15 mm`, `theta=0.0420`), but step 392 jumped to `s=-32.06 mm`, `r=17.13 mm`, `theta=0.0327`, and step 393 reached `r=35.91 mm`. This is a rotation/controller-induced lateral sweep, not success.
- Added config flag `--insertion_action_guard_prelip_offgate_axial_lock_bidirectional` in `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`. It clamps pre-lip off-gate axial command components in both signed directions because the current wrist/port sign convention can realize inward semantic tip motion from the opposite axial sign.
- `eval_v366_bidirectional_offgate_lock_step430` proved the new bidirectional lock prevents the centimeter-scale sweep: final step 430 was `s=-33.36 mm`, `r=0.108 mm`, but `theta=0.0966`, module consistency `0`, strict success `false`. This is safer laterally but worse for orientation and not yet a training base by itself.
- Next bounded probe: combine the safer v349/v351 pre-final orientation-trim envelope with bidirectional off-gate lock only when off gate, then train only if `r` stays below 0.5 mm and theta trends down rather than up. If theta remains above 0.03 with good `r`, move to a final-window orientation ablation rather than more generic SERL updates.
- `eval_v367_v349_earlyguard_bidirectional_step430` used the v349 trim envelope with early guard and bidirectional off-gate lock. It avoided the sweep but was rejected as a training base with final step `s=-33.36 mm`, `r=0.108 mm`, `theta=0.0966`, module consistency `0`, strict success `false`.
- `eval_v368_bidirectional_flip_orient_sign_step260` flipped both orientation signs and produced the same trajectory as v367 through step 260. The `frame_quat_best` candidate selection already covers the effective sign, so the failure is not a simple fixed-sign error.
- `eval_v369_bidirectional_no_finalrot_step260` disabled final fixed rotation under the same bidirectional lock. It improved centered depth relative to v367 at step 260 (`s=-38.66 mm`, `r=0.066 mm`) but theta was still bad (`0.0791`). This became the least-bad short training base because it avoided lateral sweep without adding worsening rotation commands.
- `train_v370_bidir_no_finalrot_orientq_20min` trained briefly with orientation reward weights increased and conservative Q/guide losses. It was stopped early: it made centered axial progress but theta worsened while `s` increased. Best tight-lateral point before reset was step 360: `s=-30.03 mm`, `r=0.053 mm`, `theta=0.0860`, module consistency `0`, strict success `false`; it reset at step 361.
- `eval_v372_policy_v370_noguide_step180` evaluated the learned checkpoint with guide blend `0.0`. It did not reproduce or improve the guide: step 180 was `s=-45.66 mm`, `r=0.131 mm`, `theta=0.0830`, module consistency `0`, strict success `false`. This rejects the current reward-only SERL update as a path to strict insertion.
- `train_v373_convnext_history_smoke` tested the architecture track with `--critic_image_encoder_override convnext_tiny_imagenet` and `--critic_state_history_steps 4`. The smoke ran successfully, but through step 260 it matched v370 geometry exactly: step 260 `s=-38.66 mm`, `r=0.066 mm`, `theta=0.0791`, module consistency `0`, strict success `false`. This is expected for a short guide-blend rollout and shows critic-side history/vision alone is not an immediate controller fix.
- Fixed a critic-history training bug in `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`: history buffers were mixing raw Isaac policy observations with the LeRobot/ACT state used by the online critics, causing a reset-time shape mismatch (`400` vs `616`) during history runs. The buffer now uses `act_obs["state"]` and `next_act_obs["state"]` consistently. `python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py` passed.
- `train_v374_convnext_history_policy30` warm-started from v373 with `target_action_guide_collect_blend=0.80`. It initially looked trainable: step 260 `s=-39.99 mm`, `r=0.135 mm`, `theta=0.0564`, consistency `0`, strict success `false`. It then crashed at the first reset because of the critic-history bug above.
- `train_v375_convnext_history_policy30_fixedhistory` relaunched after the fix from the v374 checkpoint. It no longer crashed, but the 0.8 guide blend was rejected because later episodes destabilized laterally. Best centered depth before reset: step 294 `s=-36.17 mm`, `r=0.120 mm`, `theta=0.0558`, consistency `0`, strict success `false`; later rows returned to `r=12-15 mm`.
- `train_v376_convnext_history_guidedistill_fixedhistory` used full guide execution (`target_action_guide_collect_blend=1.0`), no actor Q pressure, and adapter distillation of the guarded executed action. It improved the first centered episode's orientation relative to v375 but remained far from strict insertion: step 280 `s=-38.40 mm`, `r=0.138 mm`, `theta=0.0455`, consistency `0`, strict success `false`, then reset before module consistency appeared.
- `eval_v378_policy_v376_noguide_step300` evaluated the v376 checkpoint with guide blend `0.0` and no updates. The learned policy did not reproduce the guided behavior safely: best centered depth was step 238 `s=-42.92 mm`, `r=0.054 mm`, but `theta=0.1472` and consistency `0`; strict success remained `false`. This rejects guide distillation alone as sufficient.
- Current decision: policy training is being run, but the present guide/reward setup trains lateral centering more readily than semantic tip orientation and module consistency. The next bounded track should patch controller-side realized-orientation handling or add actor/adapter-side temporal state/action history before further long training; more generic Q updates or guide distillation on the same signal are unlikely to achieve strict insertion.

## 2026-05-28 Continuation: v380-v383 Training Triage

- All active Isaac training/rollout jobs were stopped again before this continuation. Host and `isaac-lab-base` process scans only matched the verification commands.
- `eval_v380_bidir_fixedrot_realized_reject_step360` added guard-side final fixed-world rotation with realized-theta rejection. It improved semantic orientation while centered, but still failed at the same contact/controller boundary. Best centered step 259: `s=-39.94 mm`, `r=0.145 mm`, `theta=0.0452`, module signed depth `-63.57 mm`, module lateral `1.16 mm`, consistency axial error `85.74 mm`, strict success `false`. Step 260 terminated after a realized lateral/contact jump to `r=12.34 mm`.
- `eval_v381_bidir_fixedrot_contact_recovery_step340` enabled earlier contact-force recovery. It reproduced the exact same best centered state and step-260 termination. The force spike is visible post-step, but the guard force signal at the decision step was still about `9.54 N`, so the configured force recovery was one control tick too late.
- `eval_v382_bidir_fixedrot_no_guide_rrecovery_step340` disabled the guide realized-r recovery branch. It reproduced the same step-260 failure, ruling that branch out as the direct cause.
- `train_v383_bidir_fixedrot_guided_policy_45min` was an actual policy-training run, not just rollout. It warm-started from `train_v376_convnext_history_guidedistill_fixedhistory/checkpoint_latest.pt`, used `--critic_image_encoder_override convnext_tiny_imagenet`, `--critic_state_history_steps 4`, `target_action_guide_collect_blend=0.90`, adapter distillation weight `3.0`, and a small Q term `0.01`. It was stopped at about 1267 rows after checkpoints `400`, `800`, and `1200` for checkpoint rollouts. Best centered state was step 959: `s=-30.05 mm`, `r=0.186 mm`, `theta=0.0933`, module signed depth `-53.61 mm`, module lateral `2.03 mm`, consistency axial error `75.78 mm`, strict success `false`. This is deeper but clearly worse orientation and module consistency.
- Current failure label: controller/contact-induced lateral bypass plus semantic orientation/module-consistency blocker. Training on the current guide can increase centered depth, but it also teaches or tolerates a high-theta state and repeated reset jumps (`v383` terminations at steps 274, 362, 539, 960, and 1052).

### Next Bounded Schedule

Each block remains capped at 7 hours before promotion, rejection, or a documented blocker:

1. **Checkpoint rollouts and architecture ablation.** Roll out `v383` checkpoints `400`, `800`, and `1200` with guide blend `0.0`, strict post-step metrics, and saved images. Compare against v378/v372 to decide whether ConvNeXt/history helped the policy or only the critic during guided collection.
2. **Reset/randomization audit.** Re-run repaired 40/10 reset-settle validation across at least two lateral seeds and compare current wrist reset, repaired semantic-tip reset, and no-settle-correction variants. Do not introduce broader randomization until deterministic 40/10 is stable.
3. **Controller preemption ablation.** Add or expose a pre-step realized-motion predictor based on commanded wrist delta plus recent tip realization history, because force/contact recovery triggers too late. Promotion requires avoiding the step-260/274 jump without loosening strict success thresholds.
4. **Reward/curriculum ablation.** Use the depth-gated bypass tolerance audit from `docs/agentic_40x10_ablation_literature_20260528.md` and test only variants that still assign low value to forward motion when `r/theta/module` are off gate.
5. **Literature-driven experiment.** Convert residual-control/contact-rich insertion ideas into one runnable guarded-residual experiment; do not spend this block on generic reward tuning.

### Checkpoint Rollout Ablation Result

| Run | Checkpoint | Best centered post-step state | Termination | Decision |
| --- | --- | --- | --- | --- |
| `eval_v384_policy_v383_ckpt400_noguide_step300` | v383 step 400 | step 213: `s=-45.40 mm`, `r=0.067 mm`, `theta=0.1722`, module `s=-69.02 mm`, module `r=1.39 mm`, consistency axial error `91.19 mm` | step 214 | Reject. Policy centers laterally but drives semantic orientation far away from strict. |
| `eval_v385_policy_v383_ckpt800_noguide_step300` | v383 step 800 | step 213: `s=-45.32 mm`, `r=0.061 mm`, `theta=0.1729`, module `s=-68.94 mm`, module `r=1.37 mm`, consistency axial error `91.11 mm` | step 214 | Reject. No material improvement over checkpoint 400. |
| `eval_v386_policy_v383_ckpt1200_noguide_step300` | v383 step 1200 | step 215: `s=-45.03 mm`, `r=0.055 mm`, `theta=0.1684`, module `s=-68.65 mm`, module `r=1.29 mm`, consistency axial error `90.82 mm` | step 216 | Reject. Slightly lower theta but still far outside `0.030 rad`; no module consistency. |

Conclusion: ConvNeXt/history critic training plus guide-heavy adapter updates did not produce a usable guide-off policy. The learned policy reaches excellent lateral centering but with severe semantic tip orientation error and no module/body consistency. The next work should not be more generic training on this signal; it should target reset/controller realization and orientation-aware residual control.

### Reset/Transform Validation Result

Two 5-step settle probes were run with deterministic randomization off and relaxed acceptance thresholds only to measure realized reset geometry:

| Run | Episode config | Step 1 post-step | Step 5 post-step | Decision |
| --- | --- | --- | --- | --- |
| `reset_v387_current_settle5_steps5` | `outputs/agentic_reward_curriculum_20260528/generated_episode_configs/full_depth_start40x10_v293_settle5_calibrated` | `s=-53.62 mm`, `r=11.58 mm`, `theta=0.0497`, module `s=-77.25 mm`, module `r=12.51 mm` | `s=-47.17 mm`, `r=7.84 mm`, `theta=0.0485`, module `s=-70.80 mm`, module `r=8.90 mm` | Usable as an outside-start config, but the effective start is still farther axially than nominal 40 mm after reset and has large lateral error. |
| `reset_v388_repaired_tip_transform_settle_steps5` | `outputs/agentic_reward_curriculum_20260527/generated_episode_configs/full_depth_start40x10_theta0005_repaired_tip_transform_settle` | `s=-39.11 mm`, `r=10.28 mm`, `theta=0.0683`, module `s=-62.72 mm`, module `r=8.82 mm` | `s=-35.57 mm`, `r=11.25 mm`, `theta=0.0753`, module `s=-59.16 mm`, module `r=9.68 mm` | Better axial distance for 40 mm, but orientation is significantly worse. Do not promote without an orientation correction. |

Interpretation: the gripper/wrist-to-tip reset transform matters. The current settle-5 config gives lower theta but starts effectively farther out; the repaired semantic-tip transform gives the requested 40 mm axial start more closely but worsens theta. This supports a reset block before additional long training: generate a repaired-tip transform with measured theta correction and validate it before policy updates.

### Literature Review Update

Recent/public work reinforces the current technical direction rather than reward-only tuning:

- SERL stresses that sample-efficient robot RL depends on reward, reset, and high-quality controller infrastructure in addition to off-policy learning. That maps directly to the current blocker: the guide/controller reset geometry must be fixed before long SERL updates. Source: https://arxiv.org/abs/2401.16013
- HIL-SERL uses human/intervention guidance for precise manipulation. Here the analogous intervention should be privileged servo/guard trajectories and negative labels for the contact-jump segment, not replaying failed insertion transitions as if they were demonstrations. Source: https://arxiv.org/abs/2410.21845
- Eureka supports automated reward-code iteration, but the task-specific lesson is to keep generated rewards under strict geometry/video gates because tip-depth progress is a known false positive. Source: https://arxiv.org/abs/2310.12931
- RL-VLM-F uses VLM preferences to learn rewards from visual observations. For this task, saved center/left/right frames can provide visual sanity or preference labels, but VLM judgment should remain auxiliary to semantic `s/r/theta/module` success. Source: https://arxiv.org/abs/2402.03681
- IndustReal specifically highlights signed-distance-style rewards, sampling curricula, and a policy-level action integrator for contact-rich assembly transfer. The actionable idea is a controller-aware action integrator or preemptive realized-motion filter around the wrist IK path. Source: https://arxiv.org/abs/2305.17110
- Residual feedback learning for contact-rich insertion shows that residual policies can improve a controller by modifying both command output and feedback signals under position/orientation uncertainty. The next learning experiment should therefore be a residual/guarded policy inside a stable controller envelope, not a free actor. Source: https://arxiv.org/abs/2106.04306
- Curriculum studies for contact-rich insertion support staged curricula with domain randomization introduced only after the nominal task is solved. Source: https://arxiv.org/abs/2204.12844

### Controller Preemption and Training Continuation Result

- Added `--insertion_action_guard_module_recovery_command_clip_m` in `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py` to clip module-recovery translation command norms before they override the guarded delta. `python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py` passed.
- `eval_v389_early_module_recovery_prejump_step360` activated module recovery before the late jump. It delayed the failure from about step 260 to step 268, but did not prevent it. Best centered state was step 267: `s=-44.40 mm`, `r=0.535 mm`, `theta=0.0453`, module `s=-68.03 mm`, module `r=1.51 mm`, strict success `false`.
- `eval_v390_early_module_microbackoff_step360` reduced recovery backoff and trim steps. It failed earlier at step 259. Best centered state was step 258: `s=-41.64 mm`, `r=0.135 mm`, `theta=0.0469`, module `s=-65.27 mm`, module `r=1.19 mm`, strict success `false`.
- `eval_v391_early_module_recovery_clip80um_step360` clipped module-recovery commands to `80 um`. It was rejected because lateral error drifted to `2.87 mm` by step 260 before the same reset/fallout around step 263. Best centered state was only step 211: `s=-44.38 mm`, `r=0.687 mm`, `theta=0.0543`, module `s=-68.00 mm`, module `r=1.90 mm`, strict success `false`.
- `train_v392_bidir_fixedrot_guided_policy_continue60` was an actual policy-training continuation from `train_v383_bidir_fixedrot_guided_policy_45min/checkpoint_latest.pt`. It kept the stable v380-style guard base, used the 40 mm / 10 mm calibrated episode config, increased guide distillation (`target_action_guide_weight=5.0`), used stronger guide collection (`target_action_guide_collect_blend=0.95`), and reduced actor Q pressure (`actor_q_weight=0.005`). It was stopped after checkpoint `001800` for an intermediate rollout instead of running the full 60 minutes without feedback.
- `eval_v393_policy_v392_ckpt1800_noguide_step320` rolled out checkpoint `001800` with guide blend `0.0`. It did not improve over v383 guide-off rollouts: best centered state was step 215: `s=-44.99 mm`, `r=0.076 mm`, `theta=0.1686`, module `s=-68.61 mm`, module `r=1.32 mm`, strict success `false`; termination occurred at step 216.

Conclusion: yes, policy training is required and was run, but continuing the same guide-heavy SERL signal is not producing a guide-off insertion policy. The actor learns lateral centering but not semantic tip orientation or module consistency. The next training attempt should change the learning target to a guarded residual/controller-aware formulation or add explicit orientation/module-consistency supervision from successful privileged servo segments; simply extending v392 is unlikely to solve full insertion.

### Actor-History and Rotation-Distillation Architecture Probe

- Added `--actor_state_history_steps` in `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`. This gives the trainable ACT adapter a flattened history of recent low-dimensional states while leaving the frozen ACT TorchScript base on the current observation and images. Older checkpoints remain loadable; incompatible adapter input weights are reinitialized through the existing compatible-state loading path. Validation:
  - `python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
  - `./isaaclab.sh -p -m pytest -q aic/aic_utils/aic_isaac/test/test_insertion_reward_geometry.py` -> `30 passed`
  - `eval_v394_actor_history_smoke_step30` ran 30 Isaac steps with `--actor_state_history_steps 4` and `--critic_state_history_steps 4`.
- `train_v395_actor_history_guided_policy25` trained a history-aware adapter from the v383 checkpoint using the stable 40/10 guard base. It was stopped at checkpoint `001000` for guide-off evaluation.
- `eval_v396_policy_v395_ckpt1000_noguide_step320` showed actor history alone did not solve the policy failure. Best centered state was step 218: `s=-44.69 mm`, `r=0.060 mm`, `theta=0.1680`, module `s=-68.30 mm`, module `r=1.32 mm`, strict success `false`. This is only a `0.30 mm` axial improvement over v393 and still far outside the `0.030 rad` orientation gate.
- Added `--target_action_guide_rotation_loss_weight` so rotation imitation inside target-action guide distillation is configurable. The old hard-coded behavior was equivalent to `0.1`, which likely underweighted the semantic orientation part of the guide relative to translation.
- `train_v397_actor_history_rotguide_policy25` resumed from v395 checkpoint `001000`, disabled actor Q (`actor_q_weight=0.0`), set `target_action_guide_weight=6.0`, and used `target_action_guide_rotation_loss_weight=2.0`. It was stopped at checkpoint `000500` because guided-collection behavior looked less stable.
- `eval_v398_policy_v397_ckpt500_noguide_step320` rejected the high-rotation-distillation setting. Best centered state was step 213: `s=-45.50 mm`, `r=0.063 mm`, `theta=0.1722`, module `s=-69.12 mm`, module `r=1.37 mm`, strict success `false`. The final state had lower theta (`0.0432`) but only after lateral error blew up to `14.26 mm`, so it is a lateral-bypass artifact, not insertion.

Conclusion: stronger actor architecture/history and heavier rotation distillation did not produce a usable guide-off policy. The failure label remains `near_success_orientation_blocked` while centered, followed by `lateral_bypass` after the policy/guard leaves the tight corridor. The next bounded change should not be more guide-heavy supervised adapter training; it should produce a privileged residual dataset or controller-aware target that labels the centered high-theta state as unsafe and separates orientation correction from lateral sweep.

### Privileged Residual Target Extraction and Contact Preemption

- Added `aic_utils/aic_isaac/scripts/extract_privileged_residual_targets.py`. This is intentionally not a fake imitation dataset: recent `metrics.jsonl` files do not persist full policy observations, so they cannot be converted into a complete LeRobot BC dataset. The script instead extracts privileged controller-aware labels and target command rows from post-step geometry, guarded command vectors, force/contact, and next-step failures.
- Validation: `python -m py_compile aic_utils/aic_isaac/scripts/extract_privileged_residual_targets.py` passed.
- Extraction over `v380`, `v393`, `v396`, and `v398` wrote `outputs/agentic_reward_curriculum_20260528/residual_targets/v399_privileged_targets/privileged_residual_targets.csv` and summary JSON. It found `1320` rows, `481` centered-high-theta rows, `4` pre-jump realization mismatches, `0` safe-centered rows, and `521` lateral-bypass rows. Best centered row was the known v380 pre-jump state at step 259: `s=-39.94 mm`, `r=0.145 mm`, `theta=0.0452`, module `s=-63.57 mm`, module `r=1.16 mm`; the next row jumped by `12.19 mm` lateral. This confirms there is no successful/near-success residual target in the current logs; the data is mostly negative supervision.
- Added `--insertion_action_guard_final_post_override_r_reject*` flags. This runs a final predicted-lateral-error rejection after all recovery, module, prelip, and orientation overrides, so late override branches cannot bypass the earlier predicted-r check.
- `eval_v399_v380_postoverride_rreject_step360` reproduced v380 exactly through the failure. The final post-override predicted-r check never triggered because the one-step geometric prediction remained near zero lateral error at step 259, while the simulator jumped to `r=12.34 mm` at step 260. This rules out pure kinematic predicted-r rejection for this failure.
- `eval_v400_v380_force8_preempt_step360` lowered contact-force recovery from `12 N` to `8 N`, activated it earlier at `s=-55 mm`, and used an `80 um` backoff. It delayed the jump from step 260 to step 275. Best centered row improved slightly in depth to step 274: `s=-39.48 mm`, `r=0.301 mm`, `theta=0.0486`, module `s=-63.10 mm`, module `r=1.27 mm`, strict success `false`. This proves the pre-jump force signal is causally useful but does not solve insertion.
- `eval_v401_v380_force8_backoff300um_step360` increased backoff to `300 um` and reduced lateral scale to `0.2`. It was rejected: the jump moved earlier to step 265 and lateral centering degraded before failure (`r` around `1.0 mm` by steps 240-260). Best centered row under `0.7 mm` was much worse, step 148: `s=-49.72 mm`, `r=0.696 mm`, `theta=0.0671`.
- Follow-up extraction over `v380`, `v399`, `v400`, and `v401` wrote `outputs/agentic_reward_curriculum_20260528/residual_targets/v401_guard_followup_targets/`. It found `1440` rows, `517` centered-high-theta rows, `4` pre-jump realization mismatches, `0` safe-centered rows, `286` contact-spike rows, and `486` lateral-bypass rows.

Conclusion: the current blocker is not an actor architecture issue alone and not a reward-only issue. The simulator/controller can remain tightly centered until about `s=-39.5 mm`, then contact produces a non-kinematic lateral jump despite predicted-r safety. Lower force threshold delays the jump but sustained/strong backoff harms centering. The next bounded code change should be a stateful contact-retreat controller that exits the contact manifold completely when force exceeds about `8-10 N`, resets the module/tip lateral relationship, then re-approaches with orientation held. Training should only resume after that controller produces at least one safe-centered row past the current `s=-39.5 mm` barrier; otherwise SERL has no positive residual target to imitate.

### Stateful Contact-Retreat and Policy-Training Probe

- Added opt-in contact-retreat state-machine flags to `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`:
  - `--insertion_action_guard_contact_force_retreat_state_machine`
  - exit depth, backout/hold steps, clear-force threshold, recenter lateral gate, and max retries.
- Validation:
  - `python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
  - `./isaaclab.sh -p -m pytest -q aic/aic_utils/aic_isaac/test/test_insertion_reward_geometry.py` -> `30 passed`
- `eval_v402_v380_contact_retreat_state_step360` enabled stateful contact retreat on the v400/v380 guard. It improved the best centered post-step row to step 290: `s=-37.664 mm`, `r=0.302 mm`, `theta=0.04370`, module `s=-61.294 mm`, module `r=1.218 mm`, strict success `false`. This is better than the prior `s=-39.5 mm` barrier, but the run then reset; strict insertion is still not achieved.
- `train_v403_contact_retreat_guided_policy45` was a real policy-training run from `train_v392_bidir_fixedrot_guided_policy_continue60/checkpoint_latest.pt`, with conservative actor-Q weight `0.003` and the v402 contact-retreat guard. It was stopped at step 843 after checkpoint `000800` because actor drift worsened geometry. Best centered row was step 282: `s=-38.878 mm`, `r=0.302 mm`, `theta=0.04616`, module `s=-62.506 mm`, module `r=1.229 mm`, strict success `false`.
- `train_v404_contact_retreat_imitation_policy30` disabled actor-Q pressure (`actor_q_weight=0.0`), increased target-action guide loss, and reduced adapter LR. It was stopped after checkpoint `000400` because it also failed earlier than v402. Best centered row was step 277: `s=-39.247 mm`, `r=0.306 mm`, `theta=0.04679`, module `s=-62.874 mm`, module `r=1.246 mm`, strict success `false`.

Conclusion: yes, policy training was resumed after the guarded evals, but both training variants degraded the current best guided behavior. The best evidence-backed candidate remains the guided/controller path, not a learned checkpoint. The next bounded step should be another small controller change before more training: make contact-retreat leave hold state through a bounded re-approach/abort transition instead of saturating at high force, and add an explicit training label or loss that penalizes the centered high-theta/module-inconsistent contact state.
