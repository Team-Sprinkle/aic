# Agentic Full-Insertion Iteration - 2026-05-29

This note records the post-step evidence from the reset/guard/training iteration after calibrating the 40 mm axial / 10 mm lateral start to the v413 episode config.

Strict success was not achieved. All rows below are post-step metrics and use the strict full-depth checker, not reward return or tip-depth-only progress.

| Run | Kind | Best centered s mm | r mm | theta rad | module r mm | Strict | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| v414 | eval | -37.490 | 0.041 | 0.06901 | 1.663 | False | reject: centered but orientation/module blocked |
| v415 | eval | -33.421 | 0.082 | 0.06619 | 1.644 | False | reject: deeper, still theta/module blocked |
| v417 | eval | -29.960 | 0.124 | 0.06411 | 1.556 | False | useful sign-flip evidence, not strict |
| v418 | eval | -27.820 | 0.344 | 0.06612 | 1.577 | False | reject: larger final rotation worsened theta |
| v420 | eval | -27.924 | 0.124 | 0.06695 | 1.568 | False | reject: module-priority flags without module alignment were insufficient |
| v421 | eval | -25.883 | 0.339 | 0.07718 | 1.730 | False | reject: module lateral alignment activated but worsened deeper insertion |
| v422 | eval | -29.936 | 0.121 | 0.06606 | 1.590 | False | reject: stronger hold/trim worsened module consistency |
| v423 | eval | -29.974 | 0.124 | 0.06319 | 1.534 | False | near module threshold; still theta blocked |
| v424 | eval | -27.934 | 0.353 | 0.06169 | 1.466 | False | promoted as guide: module strict, better theta, deeper |
| v426 | eval | -30.082 | 0.283 | 0.05572 | 1.577 | False | promoted as orientation/module guide; depth regression |
| v427 | eval | -27.826 | 0.343 | 0.06148 | 1.458 | False | reject: recovered depth but lost theta |
| v416train | train | -26.222 | 0.285 | 0.07569 | 1.776 | False | no strict success; learned deeper centered but bad theta/module |
| v419train | train | -26.966 | 0.362 | 0.07562 | 1.725 | False | no strict success; sign-flip training did not preserve guide gains |
| v425train | train | -27.662 | 0.362 | 0.07106 | 1.608 | False | no strict success; module-rotation guide not learned cleanly |
| v428train | train | -31.009 | 0.482 | 0.05214 | 1.697 | False | no strict success; better theta point lost module consistency/depth |

## Findings

- The v413 reset repair is the current best 40/10 start calibration: it preserves the low-theta reset family and starts near `s=-41.5 mm`, `r=10.0 mm`.
- The old 8-10 mm insertion candidates remain shallow warm starts/debug evidence only; they are not full insertion under the corrected full-depth target.
- Final orientation sign was wrong or at least non-optimal in this regime. `target_action_guide_rotation_sign=-1` and `insertion_action_guard_final_fixed_world_rotation_sign=-1` improved realized theta deltas.
- The module lateral alignment hook existed but must be enabled and must use a negative activation depth for the current pre-seat regime. With bounded module rotation plus compensation, v424 crossed the strict module-lateral threshold (`1.466 mm`) while keeping tip lateral tight.
- The best orientation-only tradeoff is v426 (`theta=0.05572 rad`) but it gives up axial progress and module consistency at the selected best-theta point.
- Short SERL updates from v409 did not improve strict metrics. Training tended to learn the guide neighborhood but not the coupled final correction; longer training on the same guide is unlikely to solve the strict checker without a better servo target.

## Current Blocker

The closest candidates now satisfy tight tip lateral and can satisfy module lateral, but not simultaneously with strict orientation and full depth:

- v424: better depth/module, theta still about `0.062 rad`.
- v426: better theta/module early, but axial depth stays near `s=-30 mm` and final module consistency can drift.
- v428 training: theta improved to `0.052 rad` at one centered point, but module lateral rose to `1.697 mm` and depth regressed.

Failure label: `near_success_orientation_blocked` plus `controller_realization_mismatch` for the coupled module/orientation/depth correction.

## Next Code Direction

Add a config-driven two-stage final servo schedule instead of a single static final-orientation step:

1. Trim stage: hold depth near `s=-30 mm`, require `r <= 0.5 mm`, enable module-rotation alignment, and continue final orientation until `theta` decreases below a configurable intermediate gate.
2. Reinsert stage: only after module lateral and theta gates pass, resume tiny positive axial steps, keeping module rotation active and rejecting commands that increase theta or module lateral.
3. Log per-stage realized `theta_delta`, module lateral delta, and axial delta so the failure classifier can separate wrong rotation authority from contact/controller realization.

Do not start another long SERL run until a guide-only probe beats v424/v426 on the combined metric.

## 2026-05-29 Two-Stage Servo / Training Update

Implemented a disabled-by-default final two-stage servo in `serl/train.py`:

- `--insertion_action_guard_final_two_stage_servo`
- final-window activation depth/lateral gate
- separate trim theta/module gates
- separate reinsert theta/module gates
- bounded final reinsert axial step
- metrics for two-stage window/trim/reinsert-ready fractions

Validation:

- `python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
- `docker exec ... ./isaaclab.sh -p -m pytest -q aic/aic_utils/aic_isaac/test/test_insertion_reward_geometry.py`
- Result: `30 passed`

Runs:

| Run | Kind | Best post-step s mm | r mm | theta rad | module r mm | Strict | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| v429 | eval | -29.136 | 0.091 | 0.05452 | 1.341 | False | reject: two-stage never activated because module recovery masked the window |
| v430 | eval | -27.271 | 0.060 | 0.05398 | 1.255 | False | useful: module recovery disabled, two-stage active, but axial gate remained blocked |
| v431 | train | -27.009 | 0.220 | 0.04662 | 1.269 | False | stopped: Q-weighted actor drift; later rows degraded to theta about 0.11 and module r > 2 mm |
| v432 | train | -27.271 | 0.060 | 0.05398 | 1.255 | False | stopped: imitation-heavy training reproduced guide but did not improve final consistency/depth |
| v433 | eval | -27.271 | 0.060 | 0.05398 | 1.255 | False | stable-count bug fixed; axial gate can open, but depth still plateaued |
| v434 | eval | -27.271 | 0.060 | 0.05398 | 1.255 | False | reject: trim/reinsert separation and 20 um step did not improve post-step depth |

New findings:

- The first two-stage implementation was masked by `module_recovery_active`; disabling module recovery in the final window allowed the new window/ready metrics to activate.
- `target_tip_servo_stable_count` stayed at zero even when `final_two_stage_reinsert_ready=1.0` because the older orientation-recovery gate reset stability whenever theta exceeded the strict 0.03 rad gate. This was patched so the two-stage reinsert-ready state can accumulate stability.
- After the patch, the axial gate opens in some final-window rows, but the realized post-step geometry still plateaus around `s=-27.3 mm`, `theta=0.054 rad`, module lateral `1.25 mm`, with module axial consistency still about `73 mm` from the strict target.
- Training was performed, but both Q-weighted and imitation-heavy updates failed to improve strict metrics. The Q-weighted run drifted; the imitation-heavy run mainly reproduced the guide.

Current failure label:

- `near_success_module_consistency_blocked`
- `near_success_orientation_blocked`
- `controller_realization_mismatch`

Next recommended change:

Stop tuning reward/training alone until the controller/guard can produce a guide trajectory that improves module axial consistency. The immediate code change should add realized command diagnostics for the target-tip servo final window: commanded axial step, realized tip axial delta, realized module axial delta, and whether another guard branch overrides target-tip servo axial motion. The current evidence says the servo commands are not translating into module/body insertion even after the axial gate opens.

## Fixed-Lateral Reset Ablation - 2026-05-29

Added a deterministic near-gate reset hook for controlled 40/10 ablations:

- `scene.start_near_gate.lateral_direction_world` in `aic_utils/aic_isaac/scripts/isaac_episode_configs.py`
- `aic_utils/aic_isaac/scripts/set_near_gate_lateral_direction.py`

This keeps randomized lateral starts available for broad curricula, but lets fixed 40 mm axial / 10 mm lateral experiments pin the semantic lateral start direction. Validation:

- `python -m py_compile aic_utils/aic_isaac/scripts/isaac_episode_configs.py`
- `python -m py_compile aic_utils/aic_isaac/scripts/set_near_gate_lateral_direction.py`
- `.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_isaac_online_serl.py -k 'near_gate'`
- Result: `5 passed, 9 deselected`

Generated fixed-direction episode folders from the repaired v413 config:

| Config | Direction | Reset-settle step 5 s mm | r mm | theta rad | Accepted for loose 40/10 reset? |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_depth_start40x10_v413_fixed_lat_ypos` | projected +Y | -45.947 | 8.907 | 0.04948 | yes |
| `full_depth_start40x10_v413_fixed_lat_yneg` | projected -Y | -46.549 | 12.514 | 0.04829 | no: lateral over 12 mm probe cutoff |

The +Y folder was promoted to a short guide-only two-stage eval:

| Run | Kind | Best post-step s mm | r mm | theta rad | module r mm | final axial consistency error mm | Strict | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v435 `eval_v435_fixed_lat_ypos_twostage_step520` | guide-only | -40.051 | 3.442 | 0.04125 | 2.923 | 85.858 | false | reject: fixed +Y improves theta but fails lateral/module and does not reach the final two-stage window |

Additional v435 notes:

- Best theta occurred at step 359 (`theta=0.02090 rad`) but with poor lateral/module consistency (`r=4.315 mm`, module r `4.677 mm`), so it is not near success.
- The two-stage final window and axial gate never activated in the best records because the approach never recentered tightly enough.
- This ablation supports the reset-randomization hypothesis: lateral start direction materially changes the guide trajectory. Fixed-direction starts are useful for ablations, but +Y is worse than the original v413 direction for insertion progress.

Next action:

Use the deterministic reset hook for controlled curriculum sweeps, but return to the original v413 lateral direction or a measured good direction for training. Do not train on the +Y fixed direction unless the guide is first repaired to center below 1 mm.

## Guarded vs No-Guard Branch Update - 2026-05-29

Latest diagnosis: the guarded action/servo layer improves lateral alignment but can prevent axial progress and confound credit assignment because executed actions differ from actor actions. The next branch keeps the guarded code path available for diagnostics, but disables hard action overrides for training.

Completed guarded diagnostic:

| Run | Start | Videos | Best post-step s mm | r mm | theta rad | module s mm | module r mm | final axial error mm | Strict | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v456 `eval_v456_guarded_diagnostic_v455_ckpt600_40x10_fullvideo_step1200` | 40/10 | center/left/right full episode, snapshots | -9.323 | 0.073 | 0.08021 | -32.898 | 1.838 | 55.073 | false | preserve as guarded diagnostic baseline |

Guard behavior note: at the best-depth frame, the guard was applied and orientation recovery / predicted-r rejection / module lateral alignment were active, but the chosen guarded axial command was outward/backoff. This is useful diagnostic evidence but not a success.

No-guard reward-only branch settings:

- Disable `insertion_action_guard` and hard target-tip servo / final two-stage servo / module lateral override / orientation recovery / contact retreat / predicted-r rejection / prelip clamps / offgate locks / hard final orientation holds by config.
- Keep normal policy action clipping, strict success checker, strict failure metrics, post-step geometry logging, and module consistency penalties.
- Preserve multiplicative insertion credit: `G_insert = G_lateral * G_orientation * G_action_axis`.
- Increase axial-progress reward only inside the gates; do not reward forward/tip-depth progress when lateral/orientation/module gates fail.

No-guard evidence so far:

| Run | Start | Best post-step s mm | r mm | theta rad | module s mm | module r mm | final axial error mm | Strict | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v466 `eval_v466_noguard_shallow2x0_v463_ckpt400_fullvideo_step320` | 2/0 shallow | 8.285 | 0.206 | 0.06030 | -15.361 | n/a | 37.536 | false | useful axial motion, but tip-depth false positive / module consistency blocked |
| v467 `train_v467_noguard_shallow2x0_thetafail_from_v463_ckpt400_policy1200_bs8` | 2/0 shallow | 0.279 | 0.298 | 0.05199 | -23.349 | 1.297 | 45.524 | false | reject: theta-failure gate too conservative; axial progress suppressed |

Next training action:

Run a short no-guard reward-only smoke from `v463/checkpoints/checkpoint_000400.pt` on the shallow centered start. Relax the v467 theta failure gate enough to allow axial progress, keep no hard overrides, strengthen semantic/module consistency and bypass penalties, and monitor whether axial `s` improves without widening `r`/`theta`. If axial improves but theta/module worsen, tune reward gates/curriculum first rather than reintroducing hard corrective servo. If it immediately bypasses or destabilizes, fall back only to minimal safety: clipping plus termination/logging.

## No-Guard Smoke Results - 2026-05-29

Ran the no-hard-override branch with reward-only learning and preserved strict metrics. No strict success occurred.

| Run | Start / Eval | Best post-step s mm | r mm | theta rad | module s mm | module r mm | final axial error mm | Strict | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v468 `train_v468_noguard_shallow2x0_relaxedtheta_modulepen_from_v463_ckpt400_policy1200_bs8` | train shallow 2/0 | 6.315 | 0.198 | 0.05788 | -17.312 | 0.926 | 39.486 | false | useful local axial progress; reject as tip-depth false positive |
| v469 `eval_v469_noguard_v468_ckpt1200_policyonly_40x10_fullvideo_step1200` | eval 40/10 | -40.005 | 12.144 | 0.10792 | -63.555 | 10.220 | 85.730 | false | reject: shallow no-guard training did not transfer to 40/10 |

Comparison anchors:

| Run | Mode | Best post-step s mm | r mm | theta rad | module s mm | module r mm | Strict | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v456 | guarded diagnostic 40/10 | -9.323 | 0.073 | 0.08021 | -32.898 | 1.838 | false | guard centers but blocks or backs off axial progress |
| v451 | prior policy-only 40/10 | -20.836 | 0.209 | 0.04709 | -44.460 | 1.309 | false | better approach/generalization than v469, still not full depth |
| v466 | prior no-guard shallow rollout | 8.285 | 0.206 | 0.06026 | -15.361 | 0.479 | false | local tip insertion without module/body consistency |
| v469 | no-guard v468 policy-only 40/10 | -40.005 | 12.144 | 0.10792 | -63.555 | 10.220 | false | shallow-only reward branch regressed far-start approach |

Current diagnosis:

- Guarded servo remains useful as a diagnostic and centering fallback, but it is not the right training execution layer for credit assignment.
- No-guard shallow training can create local axial tip progress, but the current reward/curriculum allows tip-depth false positives unless module consistency and theta become stricter at shallow positive `s`.
- Training only on shallow 2/0 starts loses the 40/10 approach skill. The next no-guard training should mix near-gate/final starts with earlier 40/10 or 20/4 starts, and should use phase probabilities rather than a shallow-only branch.
- The next change should tune curriculum and reward gates first: preserve alignment phases, gradually introduce shallow insertion, and increase module consistency pressure only after tight `r` and acceptable `theta`. Do not reintroduce hard action replacement unless the minimal no-guard branch bypasses or destabilizes.

Artifacts:

- v456 guarded videos: `outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-29_09-46-02_eval_v456_guarded_diagnostic_v455_ckpt600_40x10_fullvideo_step1200/env0000_{center,left,right}_full_episode_20fps_quality.mp4`
- v469 no-guard videos: `outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-29_11-46-48_eval_v469_noguard_v468_ckpt1200_policyonly_40x10_fullvideo_step1200/env0000_{center,left,right}_full_episode_20fps_quality.mp4`

## Reset-Settle and Mixed-Curriculum Ablation - 2026-05-29

Reset-settle probes found that the nominal shallow centered starts are not actually stable after five runtime settle/action steps:

| Probe | Config | post-step s mm | r mm | theta rad | Accepted? | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v470 | 40/10 v413 | -43.378 | 8.773 | 0.04877 | yes | loose 40/10 bounds; usable far-start anchor |
| v470 | 20/4 v460 | -23.085 | 5.138 | 0.04988 | yes | loose 20/4 bounds; usable near-gate anchor |
| v470 | 10/2 v462 centered | -8.922 | 3.808 | 0.04914 | no | lateral drift after settle |
| v470 | 2/0 v464 shallow | -2.074 | 6.326 | 0.04888 | no | severe lateral drift after settle |
| v472 | 10/2 after one recenter pass | -7.202 | 2.036 | 0.04826 | no | improved but still off-center |
| v472 | 2/0 after one recenter pass | -5.034 | 2.535 | 0.04827 | no | improved lateral, worsened axial |
| v474 | 10/2 after second recenter pass | -10.505 | 3.508 | 0.04858 | no | non-linear/unstable correction |
| v474 | 2/0 after second recenter pass | -5.049 | 3.506 | 0.05021 | no | non-linear/unstable correction |

Interpretation: the current shallow reset family is a real confound. It can show local tip-depth progress, but the reset/settle itself can place the tip several millimeters off center before learning acts. Do not use shallow-only training as evidence of full insertion, and do not keep blindly applying settle correction without a better model of the cable relaxation.

Generated mixed curriculum:

- `outputs/agentic_reward_curriculum_20260529/generated_episode_configs/full_depth_mixed_v475_40x10_20x4_10x2_2x0_weighted`
- weights: 40/10 x2, 20/4 x3, 10/2 x3, 2/0 x2
- reset-settle loose validation accepted all 10 episodes under broad curriculum-start bounds, but the 10/2 and 2/0 phases remain known noisy starts.

Mixed no-guard training from the stronger 40/10 partial-alignment checkpoint:

| Run | Start / Eval | Best post-step s mm | r mm | theta rad | module s mm | module r mm | final axial error mm | Strict | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v476 `train_v476_noguard_mixedcurr_from_v450_policy1600_bs8` | mixed train | 0.954 | 0.160 | 0.05911 | -22.693 | 0.378 | 44.868 | false | retains local shallow progress but still theta/module blocked |
| v477 `eval_v477_noguard_v476_ckpt1600_policyonly_40x10_fullvideo_step1200` | 40/10 eval | -40.001 | 11.235 | 0.09952 | -63.558 | 9.526 | 85.733 | false | reject: mixed training still did not transfer to 40/10 |

Comparison to prior anchors:

- v476 is better than v469 only inside the mixed/shallow training distribution. It did not improve held-out 40/10 rollout.
- v451 remains the best policy-only 40/10 approach anchor from this set (`s=-20.836 mm`, `r=0.209 mm`, `theta=0.04709`), but it is still not full insertion.
- v456 remains the best guarded 40/10 centering/depth diagnostic (`s=-9.323 mm`, `r=0.073 mm`) but theta and guard-induced axial blocking remain unresolved.

Next decision:

Do not continue shallow or mixed no-guard SERL with the same reset family. The next engineering step should fix or bypass the shallow reset confound and add an explicit phase-aware sampler that can avoid unstable shallow starts until reset-settle validation passes. If continuing learning before that, train from v450/v451 on 40/10 and 20/4 only, then add shallow phases only after a validated shallow reset exists.

## Latest No-Guard Axial/Action-Axis Smoke - 2026-05-29

User-directed update: preserve the guarded diagnostic rollout, then test a no-hard-override reward-only branch because the guard appears to help alignment while blocking axial insertion and confusing policy credit assignment.

Guarded diagnostic status:

- Preserved run: `outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-29_09-46-02_eval_v456_guarded_diagnostic_v455_ckpt600_40x10_fullvideo_step1200`
- Evidence bundle includes `command.txt`, `train_config.json`, `git_status.txt`, `git_diff.patch`, `metrics.jsonl`, `metrics_selected.csv`, `metrics_summary.json`, `cheatcode_phase_summary.json`, snapshots, and separate center/left/right full-episode videos.
- Guard behavior at best depth: strong centering (`r=0.073 mm`) but non-success orientation (`theta=0.08021 rad`) and module lag (`module_s=-32.898 mm`). Guard was applied continuously and the chosen axial command was outward/backoff at the best-depth row, so this is diagnostic evidence, not success.

Fresh no-guard branch:

- Command: `outputs/agentic_reward_curriculum_20260529/commands/train_v478_noguard_rewardonly_axialaxis_from_v455_ckpt600_40x10_smoke800_bs8.txt`
- Checkpoint source: `v455/checkpoints/checkpoint_000600.pt`, selected because it was the strongest guarded partial-alignment checkpoint among v453/v454/v455 for `r/theta` balance.
- Disabled by config: `insertion_action_guard`, `target_action_guide_train_executed`, target-tip servo, final two-stage servo, module lateral alignment override, module-alignment rotation, target-tip orientation recovery, contact force recovery/retreat, predicted-r rejection, prelip clamp, offgate axial lock, final orientation hard-hold compensation/rejection, final fixed-world rotation, and final-axis-alignment rotation.
- Kept: normal policy action clipping, strict success checker, strict failure metrics, post-step semantic tip/module geometry, module consistency and bypass penalty.
- Reward change: axial progress weight increased to `0.85`, action-axis gate kept from body delta, bypass penalty increased to `120`, semantic progress/loss increased to `1.7/2.4`.

Results:

| Run | Mode | Best post-step s mm | r mm | theta rad | module s mm | module r mm | final axial error mm | Strict | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v456 | guarded diagnostic 40/10 | -9.323 | 0.073 | 0.08021 | -32.898 | 1.838 | 55.073 | false | preserve diagnostic; guard centers but blocks/backs off axial progress |
| v451 | prior policy-only 40/10 | -20.836 | 0.209 | 0.04709 | -44.460 | 1.309 | 66.635 | false | best policy-only 40/10 anchor so far |
| v478 | no-guard reward-only train 40/10 | -40.013 | 20.475 | 0.08074 | -63.621 | 21.249 | 85.796 | false | reject: stronger axial/action reward lost alignment |
| v479 | no-guard v478 policy-only 40/10 video | -40.025 | 15.765 | 0.06465 | -63.646 | 16.048 | 85.821 | false | reject: no axial approach transfer; visual evidence saved |
| v477 | prior no-guard mixed policy-only 40/10 | -40.001 | 11.235 | 0.09952 | -63.558 | 9.526 | 85.733 | false | reject: mixed/shallow training did not transfer |

v479 artifacts:

- Run folder: `outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-29_13-06-17_eval_v479_noguard_v478_ckpt800_policyonly_40x10_fullvideo_step1200`
- Videos: `env0000_center_full_episode_20fps_quality.mp4`, `env0000_left_full_episode_20fps_quality.mp4`, `env0000_right_full_episode_20fps_quality.mp4`
- Metrics/config: `metrics_summary.json`, `metrics_selected.csv`, `metrics.jsonl`, `command.txt`, `train_config.json`, `git_status.txt`, `git_diff.patch`

Diagnosis:

- The main hypothesis is still plausible, but this particular no-guard reward-only setting is too aggressive from 40/10. It rewards axial/action-axis pressure strongly enough that the actor loses lateral alignment before entering the gate.
- Do not re-enable hard servo overrides for training yet. The guard remains a diagnostic baseline.
- Next training branch should keep no hard overrides, but back off axial pressure and recover the v451-style approach skill first: train from v450/v451 on 40/10 + 20/4 only, with stronger lateral/corridor/orientation retention and a lower axial reward until `r < 1 mm` is repeatedly achieved. Add shallow/final phases only after reset-settle validation is fixed or after the phase-aware sampler can reject unstable shallow starts.

## Alignment-First No-Guard Branch - 2026-05-29

Implemented the next branch implied by the v478/v479 diagnosis:

- Generated curriculum: `outputs/agentic_reward_curriculum_20260529/generated_episode_configs/full_depth_mixed_v480_40x10_20x4_only_weighted`
- Weights: 40/10 x3, 20/4 x5. No 10/2 or 2/0 shallow episodes, because those reset-settle probes are unstable.
- Command: `outputs/agentic_reward_curriculum_20260529/commands/train_v480_noguard_40x10_20x4_alignmentfirst_from_v450_policy1600_bs8.txt`
- Checkpoint source: v450 `checkpoint_latest.pt`, not the degraded v478 actor.
- Guard/servo hard overrides remained disabled by config.
- Reward changes relative to v478: axial progress reduced to `0.28`; corridor, inside-alignment, lateral-progress, and orientation-progress increased; bypass penalty kept high at `100`.

Run management:

- v480 reached clean checkpoints at 400, 800, and 1200.
- After checkpoint 1200 the policy drifted again, so the run was stopped at step 1274 to preserve the checkpoint and avoid wasting runtime.

Results:

| Run | Mode | Best post-step s mm | r mm | theta rad | module s mm | module r mm | final axial error mm | Strict | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v480 | no-guard 40/10+20/4 train, best s | -27.364 | 6.794 | 0.06869 | -50.986 | 5.899 | 73.161 | false | reject for insertion; axial approach improved but lateral bad |
| v480 | no-guard 40/10+20/4 train, best balanced | -27.947 | 1.028 | 0.04475 | -51.573 | 1.300 | 73.748 | false | useful near-gate signal; still outside strict lateral and far from full insertion |
| v480 | stopped row | -40.571 | 16.802 | 0.06634 | -64.194 | 17.538 | 86.369 | false | stopped after drift |

Interpretation:

- Lower axial pressure plus 40/10+20/4 curriculum recovers more approach progress than v478/v479, and briefly gets close to the lateral gate (`r=1.028 mm`) while keeping theta moderate (`0.04475 rad`).
- It still does not get below the strict lateral gate, does not reach the port entrance, and module depth remains about 51.6 mm behind at the best balanced row.
- The failure label is `lateral_bypass/controller_realization_mismatch risk`, not strict insertion: final force was high and the policy later drifted laterally.

Next action:

- Do not rollout v480 yet as a candidate success video. It is not close enough and the latest policy drifted.
- Next branch should use the v480 best-balanced checkpoint only as a diagnostic if needed, but likely train from v450/v451 again with even stronger lateral retention and lower actor Q weight. Consider adding a soft failure termination/log-only curriculum reset when `r > 3 mm` after approach rather than hard corrective servo.
- The main open blocker is still getting a policy-only actor from 40/10 to consistently reach `r < 0.5-1.0 mm` before rewarding any inward axial motion.

## No-Guard Shallow-to-Full Bridge - 2026-05-29

Updated diagnosis after v481-v485:

- v481 tested actor state history on the 40/10+20/4 no-guard branch. It did not improve transfer: best `s=-27.292 mm`, `r=6.974 mm`, `theta=0.06608`; strict success remained false.
- v482/v483 resumed from the best shallow-positive no-guard policy and increased module-consistency pressure. This produced the best no-guard full-depth direction so far on shallow/final starts, but still not strict insertion.
- v484 evaluated the v483 checkpoint on the required 40 mm axial / 10 mm lateral start with full videos. It failed to enter: best balanced row was `s=-41.751 mm`, `r=4.098 mm`, `theta=0.01973`; strict success false.
- v485 mixed 40/10, 20/4, 10/2, and 2/0 starts from v483 checkpoint 300. It preserved some shallow/final insertion progress and found a shallow strict-theta row, but did not combine strict theta with depth.

Results:

| Run | Mode | Best post-step s mm | r mm | theta rad | module s mm | module r mm | final axial error mm | Strict | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v481 | no-guard actor-history 40/10+20/4 train | -27.292 | 6.974 | 0.06608 | -50.913 | 6.243 | 73.088 | false | reject: no 40/10 transfer improvement |
| v482 | no-guard shallow module-depth train | 11.624 | 0.418 | 0.05317 | -11.992 | 1.016 | 34.167 | false | partial: useful depth/module progress, theta blocked |
| v483 | no-guard v482 resume, dense checkpoints | 17.372 | 0.445 | 0.05138 | -6.250 | 0.842 | 28.425 | false | promote for eval: deepest no-guard shallow/final candidate |
| v484 | v483 ckpt300 policy-only 40/10 video | -40.010 | 32.931 | 0.11553 | -63.556 | 30.800 | 85.731 | false | reject for 40/10; full videos saved |
| v484 | best balanced row | -41.751 | 4.098 | 0.01973 | -65.396 | 4.552 | 87.571 | false | good theta only; no axial approach |
| v485 | no-guard mixed bridge, best depth | 13.702 | 0.132 | 0.05277 | -9.942 | 0.664 | 32.117 | false | partial: depth/lateral good, theta blocked |
| v485 | no-guard mixed bridge, best theta | 1.652 | 0.660 | 0.02746 | -21.990 | 0.840 | 44.165 | false | partial: strict theta only at shallow depth |

Key artifacts:

- v483 training: `outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-29_14-02-54_train_v483_noguard_shallow2x0_deep_module_resume_v482_ckpt200_policy900_bs8`
- v484 40/10 rollout: `outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-29_14-14-54_eval_v484_noguard_v483_ckpt300_policyonly_40x10_fullvideo_step1200`
- v484 videos: `env0000_center_full_episode_20fps_quality.mp4`, `env0000_left_full_episode_20fps_quality.mp4`, `env0000_right_full_episode_20fps_quality.mp4`
- v485 mixed bridge: `outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-29_14-31-08_train_v485_noguard_mixed_bridge_from_v483_ckpt300_policy1600_bs8`

Current failure label:

- `orientation_blocked_at_depth`: no-guard policy can reach `s=13-17 mm` with sub-mm lateral/module lateral error, but tip orientation remains near `0.05 rad`, above strict `0.03 rad`.
- `no_axial_progress_from_40x10`: the shallow/final skill does not transfer back to the 40/10 pre-contact start; policy-only 40/10 evaluation remains outside the gate.

Next branch:

- Keep hard action overrides disabled.
- Train an orientation-at-depth branch from v483 or v485, biased toward 2/0 and 10/2 starts, with higher orientation progress and lower axial pressure until theta is below strict threshold at positive depth.
- Then reintroduce 20/4 and 40/10 curriculum only after the final-window policy can combine `s > 10 mm`, `r < 0.5 mm`, `theta < 0.03 rad`, and improving module depth.

## No-Guard Reproducibility and Orientation-at-Depth Probes - 2026-05-29

Updated diagnosis after v486-v490:

- v486 tested the planned orientation-at-depth branch from v483 checkpoint 300 with high orientation pressure and reduced axial pressure. It regressed: best `s=0.676 mm`, `r=0.224 mm`, `theta=0.05631`; strict success false.
- v487 resumed v483 checkpoint 400 with a milder theta-tightening setup. It also regressed: best `s=2.756 mm`, `r=0.798 mm`, `theta=0.05410`; strict success false.
- v488 continued the exact v483 reward recipe from v483 checkpoint 400. It did not preserve the prior depth behavior: best `s=3.237 mm`, `r=0.211 mm`, `theta=0.05420`; strict success false.
- v489 reran the original v483 start from the v482 checkpoint with dense 20-step checkpoints to capture the transient depth policy. The transient did not recur: best `s=3.163 mm`, `r=0.746 mm`, `theta=0.05799`; strict success false.
- v490 reran the original v483 start with original 100-step checkpoint cadence. The transient again did not recur: best `s=2.238 mm`, `r=0.366 mm`, `theta=0.05798`; strict success false.

Results:

| Run | Mode | Best post-step s mm | r mm | theta rad | module s mm | module r mm | final axial error mm | Strict | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v486 | high-orientation shallow/final from v483 ckpt300 | 0.676 | 0.224 | 0.05631 | -22.955 | 1.155 | 45.130 | false | reject: orientation pressure suppressed insertion |
| v487 | modest-theta recovery from v483 ckpt400 | 2.756 | 0.798 | 0.05410 | -20.869 | 0.886 | 43.044 | false | reject: depth not recovered |
| v488 | exact v483 continuation from ckpt400 | 3.237 | 0.211 | 0.05420 | -20.392 | 0.869 | 42.567 | false | reject: checkpoint resume does not reproduce depth |
| v489 | exact v483 rerun, dense checkpoints | 3.163 | 0.746 | 0.05799 | -20.449 | 1.163 | 42.624 | false | reject: v483 transient not reproduced |
| v490 | exact v483 rerun, original cadence | 2.238 | 0.366 | 0.05798 | -21.389 | 0.789 | 43.564 | false | reject: v483 transient not reproduced |

Important clarification:

- The original v483 `s=17.372 mm` row was a real post-step geometry row, not a parser artifact. It remained strict-false because theta was about `0.05138 rad`, module_s was about `-6.250 mm`, and final axial consistency error was about `28.425 mm`.
- The current blocker is not simply "more training from v483". The v483 depth behavior is not reproducibly captured by its saved checkpoints or by rerunning the same recipe from v482.

Current failure label:

- `unstable_learning_or_actor_drift`: fresh online SERL runs from the same actor checkpoint can lose the shallow insertion behavior quickly.
- `orientation_blocked_at_depth`: the one deep transient still had theta around `0.05 rad`, above the strict `0.03 rad` threshold.
- `module_consistency_blocked`: even the best v483 transient had module depth about `6.25 mm` behind the tip and final axial consistency error about `28.4 mm`.

Next branch:

- Stop repeating v483-style online Q updates as-is.
- Try a conservative no-guard branch with delayed or tiny actor updates and stronger imitation/action-preservation pressure, so the actor does not drift while the critic learns final-window values.
- If that still fails, treat the blocker as online SERL update instability for final insertion and switch to generating supervised/residual data from the best guarded or v483-like trajectories before returning to online RL.

## No-Guard Loose-Gate Probe and Conservative Training Branch - 2026-05-29

Updated diagnosis after v491:

- v491 loosened the no-guard insertion orientation gate on the shallow/final 2 mm / 0 mm curriculum to test whether the earlier branches were suppressing axial progress by over-penalizing theta around 0.05 rad.
- Strict success criteria were unchanged. The looser gate was a reward/curriculum probe only; it did not change the evaluator.
- The run stayed numerically stable, but did not recover the v483 transient deep insertion. Best post-step depth was only `s=2.459 mm`, with `r=0.368 mm`, `theta=0.05789 rad`, module depth `-21.169 mm`, and final axial consistency error `43.344 mm`.

Results:

| Run | Mode | Best post-step s mm | r mm | theta rad | module s mm | module r mm | final axial error mm | Strict | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v491 | no-guard loose-orientation shallow/final from v482 | 2.459 | 0.368 | 0.05789 | -21.169 | 0.770 | 43.344 | false | reject: no meaningful axial insertion |

Current training branch:

- v492 is running from the v483 checkpoint 300 actor with all hard guard/servo overrides disabled.
- Compared with v491, v492 uses a smaller actor-Q weight, delayed actor updates, lower adapter learning rate, and nonzero action-preservation pressure. The purpose is to train the policy rather than evaluate only, while reducing the online SERL actor drift observed in v486-v491.
- Rollout/eval should happen only after a meaningful checkpoint or about 30 minutes, unless metrics clearly show immediate instability.

Updated outcome after v492-v493:

- v492 stopped after a clean checkpoint because it plateaued. Best post-step depth was `s=3.141 mm`, with `r=0.507 mm`, `theta=0.05789 rad`, module depth `-20.478 mm`, and final axial consistency error `42.652 mm`; strict success remained false.
- v493 widened the training-only insertion gates and increased gated axial reward while keeping strict success unchanged. It did not improve: best post-step depth was `s=2.027 mm`, with `r=0.785 mm`, `theta=0.05570 rad`, module depth `-21.597 mm`, and final axial consistency error `43.772 mm`; strict success remained false.
- The current no-guard reward-only branches are not learning enough inward action from reward alone on the shallow/final curriculum. The next bounded branch should use a soft target-action guide loss without executing guide actions, so the policy still controls the robot but receives a supervised action target for the final-window behavior.

Results:

| Run | Mode | Best post-step s mm | r mm | theta rad | module s mm | module r mm | final axial error mm | Strict | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v491 | no-guard loose-orientation shallow/final | 2.459 | 0.368 | 0.05789 | -21.169 | 0.770 | 43.344 | false | reject: no meaningful axial insertion |
| v492 | no-guard conservative actor updates | 3.141 | 0.507 | 0.05789 | -20.478 | 0.705 | 42.652 | false | reject: plateaued before 600 |
| v493 | no-guard wide-gate axial discovery | 2.027 | 0.785 | 0.05570 | -21.597 | 0.829 | 43.772 | false | reject: shallow tip motion only |
| v494 | soft guide loss, guide not executed | 2.672 | 0.791 | 0.05065 | -20.955 | 0.993 | 43.130 | false | reject: shallow tip motion only |

v494 note:

- `target_action_guide_collect_blend=0.0`, so the privileged guide was not executed as an action override.
- The guide was used only as a supervised action-loss target. This did not solve axial insertion; the best row stayed shallow and module/body depth lag remained about `43 mm`.
- The evidence now points away from another shallow 2/0 no-guard reward tweak. The next structural issue is that online SERL resumes preserve only actor weights while critic/replay/optimizer state are fresh, so rare good transients like v483 are not reproducibly captured. A full-state checkpoint/resume path or an explicit offline imitation dataset from guarded/privileged trajectories is the next code-level change to test.
