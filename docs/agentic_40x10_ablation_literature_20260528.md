# Agentic 40x10 Insertion Ablation and Literature Plan

Date: 2026-05-28

Scope: deterministic SFP-to-NIC insertion from the 40 mm axial / 10 mm lateral near-gate start. Strict success remains the existing post-step full-depth checker: target depth about 45.8-46.9 mm, lateral error about <= 0.5 mm, tip orientation <= 0.03 rad, module/body consistency, and visual sanity.

## Current Status

The latest 40x10 training run inspected here was:

`outputs/agentic_reward_curriculum_20260527/policy_train_runs/2026-05-28_09-06-39_train_from_v197_v220teacher_sparse_full_depth_v225_step1200`

It produced 551 metric rows and saved `checkpoint_000300.pt` plus `checkpoint_latest.pt`, but the tail metrics show it did not reach the insertion regime:

| metric | best/last observed |
|---|---:|
| axial depth mean | last 2.840 mm, max 2.840 mm |
| lateral error mean | last 28.114 mm, best 4.738 mm |
| orientation error mean | last 0.1313 rad, best 0.0548 rad |
| module consistency gate mean | last 0.0, max 0.421 |
| centered axial fraction | last 0.0 |
| batch reward mean | last -93.85 |

Failure label: `lateral_bypass / approach_alignment_failed`. This is not yet a final-window seating problem. The model/guide combination is failing to hold the 40x10 approach corridor tightly enough before it pushes forward.

## Reward Audit Finding

I added an Isaac-free scalar audit:

`aic_utils/aic_isaac/scripts/audit_40x10_reward_cases.py`

The audit evaluates the current cheatcode phase reward at hand-picked `(s, r, theta, semantic_gate)` states that match the observed 40x10 failure modes.

Validation:

```bash
.pixi/envs/default/bin/python -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/insertion_geometry.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py \
  aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py \
  aic_utils/aic_isaac/scripts/audit_40x10_reward_cases.py

.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py
```

Result: `30 passed`.

Legacy reward issue:

`outputs/agentic_reward_curriculum_20260527/reward_audits/20260528_40x10_reward_cases_v1/summary.md`

The old aggressive bypass penalty correctly punished the 10 mm lateral-bypass state, but it also made a strict-like final state unattractive:

| case | s | r | theta | semantic | total | corridor |
|---|---:|---:|---:|---:|---:|---:|
| v222_10mm_lateral_bypass | 10.037 mm | 24.247 mm | 0.10346 | 0.0 | -408.7902 | -6.5745 |
| strict_candidate | 45.800 mm | 0.350 mm | 0.02500 | 1.0 | -47.7857 | -19.2359 |

Patch:

I added config-driven `bypass_gate_tolerance`, default `0.0`, to the insertion corridor and cheatcode phase reward. The tolerance is depth-gated, so it only softens the bypass penalty near full insertion and preserves strong penalties for shallow tip-only false positives.

New audit:

`outputs/agentic_reward_curriculum_20260527/reward_audits/20260528_40x10_reward_cases_bypass_tol080_depthgated/summary.md`

| case | s | r | theta | semantic | total | corridor |
|---|---:|---:|---:|---:|---:|---:|
| shallow_centered_module_bad | 10.000 mm | 0.350 mm | 0.02800 | 0.0 | -12.4660 | -4.8717 |
| v222_10mm_lateral_bypass | 10.037 mm | 24.247 mm | 0.10346 | 0.0 | -408.7902 | -6.5745 |
| strict_candidate | 45.800 mm | 0.350 mm | 0.02500 | 1.0 | 1.1721 | 0.3472 |

Interpretation: the reward no longer crushes legitimate near-strict full-depth states, while it still rejects tip-depth-only and lateral-bypass false positives.

## Recommended Next Training Change

Run the next 40x10 training iteration from the same best checkpoint, but add:

```bash
--target_reward_insertion_bypass_gate_tolerance 0.80
```

Keep:

```bash
--target_reward_insertion_bypass_penalty_scale 30.0
AIC_ISAAC_RANDOMIZATION_PROFILE=none
```

Do not enable gripper-tip randomization until deterministic 40x10 reaches the strict checker or gets very close. Randomization should be phased back in after the controller can solve the nominal semantic geometry.

## Phased 40x10 Approach

Phase 1: approach lock.

Start at 40 mm axial / 10 mm lateral. Reward only lateral and orientation correction until the tip is near the port corridor. Axial credit remains zero or negative while `r` and `theta` are outside gates.

Phase 2: entrance lock.

Near `s=-6 mm` to `s=-2 mm`, hold/hover until `r < 0.75 mm`, then tighten toward `0.5 mm`; hold orientation below `0.05 rad`, then tighten toward `0.04/0.03 rad`. Do not allow inward axial motion while off-gate.

Phase 3: shallow insertion.

Allow tiny axial steps only if tip `r/theta` and module consistency do not contradict each other. Penalize positive tip `s` with poor module consistency as bypass, not success.

Phase 4: final seating.

Use the depth-gated bypass tolerance only near full depth. This avoids punishing real near-seated states while keeping strict success unchanged.

Phase 5: randomization.

After deterministic success, introduce randomization in stages: no randomization, then tip/gripper offset only, then full visual/contact randomization. Each stage must keep semantic `sfp_tip_link` and `sfp_module_link` metrics in the evaluator.

## Ablation Matrix

| ablation | variants | purpose |
|---|---|---|
| reward | legacy, depth-gated bypass tolerance 0.70/0.80/0.90, tighter entrance gates | confirm reward does not prefer false positives |
| curriculum | 40x10 only, reverse curriculum from entrance, phased mix | decide whether far start is too hard before final seating is solved |
| guide/servo | lateral-only, lateral+theta, predicted-r reject, target-tip servo, backoff/retry | isolate controller/servo limitations |
| orientation | none, final full-quat, final axis-only, hybrid final-window | prevent global axis-only from hurting approach |
| randomization | none, gripper-tip only, full | quantify randomization cost after nominal solve |
| model | current ACT adapter, stronger critic image encoder, privileged-state critic/control | determine whether vision/model expressivity is the bottleneck |
| observations | vision+proprio, add semantic privileged metrics during training, force/contact stats | test whether contact/geometry signals are necessary |

## Model Architecture Notes

The online SERL trainer is not limited to a tiny image stack. It already supports these critic image encoders in `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`:

- `small_conv`
- `resnet18`
- `resnet18_imagenet`
- `convnext_tiny`
- `convnext_tiny_imagenet`

The current default config uses `small_conv`. The actor path is still the ACT TorchScript plus trainable adapter/residual. My recommendation is not to change the actor architecture first. Run a privileged/guide-heavy solve first. If the privileged-state or guide-controlled version reaches strict success but vision-based SERL does not, then run the critic backbone ablation with `resnet18_imagenet` and `convnext_tiny_imagenet`, and consider retraining the ACT actor only after proving that the controller/reward can solve the nominal geometry.

## Literature Notes

SERL is directly relevant because it emphasizes sample-efficient off-policy robotic RL from pixels, demonstrations, controllers, and real robot reset/reward engineering. The repository's current ACT-adapter SERL setup is aligned with this style, but the SFP task needs stricter geometry and false-positive checks than ordinary sparse success. Source: SERL, arXiv 2401.16013, https://arxiv.org/abs/2401.16013

HIL-SERL shows that precise manipulation can benefit from interventions and corrections rather than pure autonomous exploration. For this task, the analogue is privileged guide/servo trajectories and residual learning, not arbitrary reward-only tuning. Source: Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning, https://huggingface.co/papers/2410.21845

Eureka is relevant as an LLM reward-code search baseline, but this task cannot accept reward return as success because tip-depth false positives are known. Any agent-generated reward must be filtered through strict geometry and videos. Source: Eureka, arXiv 2310.12931, https://arxiv.org/abs/2310.12931

RL-VLM-F is relevant for visual reward learning, but VLM preference feedback is likely too coarse for sub-millimeter insertion unless paired with semantic geometry or high-resolution close-up views. It may be useful for visual sanity labels, not the primary success checker. Source: RL-VLM-F, arXiv 2402.03681, https://arxiv.org/abs/2402.03681

IndustReal is relevant because it targets contact-rich assembly transfer with specialized rewards and action integration. The practical lesson is that insertion tasks often need controller-aware action integration, not just policy outputs with a generic reward. Source: IndustReal, arXiv 2305.17110, https://huggingface.co/papers/2305.17110

Contact-rich insertion literature repeatedly treats insertion as staged search/alignment plus force-aware insertion, often with compliance or residual control. This supports the phased plan above: solve near-gate alignment, then shallow insertion, then final seating with recovery. Sources: contact-rich RL review, https://www.sciencedirect.com/science/article/pii/S0736584522001995; residual learning from demonstration, https://arxiv.org/abs/2008.07682; residual feedback learning, https://arxiv.org/abs/2106.04306; curriculum/contact-rich insertion study, https://arxiv.org/abs/2204.12844

## Concrete Next Experiments

1. No-learning guide probe from 40x10 with depth-gated bypass tolerance enabled, videos on, one env, deterministic randomization off.
2. If the probe still ends with large `r`, tune the guide/servo, not the reward: smaller axial allowance, larger lateral correction until `r < 1 mm`, predicted-r rejection, and no rotation while off-center.
3. If `r` becomes tight but `theta` blocks insertion, run final-window-only orientation ablation: full-quat, axis-only, hybrid, and no final refinement.
4. If `r/theta` are tight but no axial progress occurs, modestly increase tiny axial step or axial reward while preserving gates.
5. Only then run SERL from the best checkpoint with `--target_reward_insertion_bypass_gate_tolerance 0.80`, conservative actor updates, and checkpoint/video rollouts every 30 minutes.

## Current Best Hypothesis

The gap from 40x10 is currently controller/servo/curriculum more than raw model expressivity. The evidence is that the latest run never reached low lateral error at the entrance and the reward audit now shows a reward bug only near final depth. A stronger vision backbone may help later, but first the guide/servo must reliably convert 40x10 into a near-gate state with `r <= 0.5-1.0 mm` and `theta <= 0.04-0.05 rad`.

## 2026-05-28 Continuation: Reset/Settle And Training Triage

User request: stop all ongoing training/rollout jobs, then continue autonomously with bounded task families: reset/randomization fixes, ablations, literature-review-driven ideas, and model architecture experiments. No task family should consume more than 7 hours before either promotion, rejection, or blocker documentation.

Process status after stop check:

- Host process scan: no active `serl/train.py`, `isaaclab.sh`, or rollout jobs.
- Container process scan in `isaac-lab-base`: no active `serl/train.py`, `isaaclab.sh`, or rollout jobs.
- Active strict success count remains 0. No reward-only or tip-depth-only result is treated as success.

### Time Allocation

| task family | time cap | promotion criterion | rejection/blocker criterion |
|---|---:|---|---|
| Reset/settle and gripper-tip transform validation | 7 h | post-step reset starts near requested 40/10 geometry without large theta/lateral drift | semantic tip cannot be placed repeatably from wrist IK metadata |
| Reward/curriculum ablations | 7 h | improves post-step `s/r/theta/module` without increasing bypass false positives | reward changes improve return but not strict geometry |
| Guide/servo/controller ablations | 7 h | guide-only probe reaches entrance with `r <= 0.5-1.0 mm`, `theta <= 0.04-0.05 rad`, no large sweep | small commands produce large realized lateral jumps |
| Model architecture ablations | 7 h | a stronger critic/backbone improves held-out geometry after guide is sane | no improvement over current ACT adapter under identical guide/curriculum |
| Literature-driven experiments | 7 h | yields a concrete runnable experiment that beats current best strict metrics | ideas remain speculative or require unavailable infrastructure |
| Reporting and paper framing | 7 h | complete table of runs, evidence, and recommendation | missing metrics/videos/configs |

### Reset/Transform Finding

I added:

`aic_utils/aic_isaac/scripts/repair_near_gate_reset_tip_transform.py`

Purpose: repair near-gate reset-body poses from the calibrated local reset-body-to-`sfp_tip_link` transform instead of trusting stale world offsets. The script rewrites:

- `body_start_position_world`
- `tcp_start_position_world`
- `reset_body_orientation_wxyz`
- `reset_body_offset_from_reference_world`
- metadata under `semantic_tip_transform_repair`

Validation commands:

```bash
python -m py_compile aic_utils/aic_isaac/scripts/repair_near_gate_reset_tip_transform.py

.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/repair_near_gate_reset_tip_transform.py \
  --episodes-dir outputs/agentic_reward_curriculum_20260527/generated_episode_configs/full_depth_start40x10_theta0005_measured_quat_corrected/episodes \
  --output-dir outputs/agentic_reward_curriculum_20260527/generated_episode_configs/full_depth_start40x10_theta0005_repaired_tip_transform_settle \
  --apply-settle-compensation \
  --no-apply-strict-theta-correction
```

Repair summary for the active 40x10 episode:

| field | old | repaired with settle |
|---|---:|---:|
| reset offset x | -8.248 mm | -20.993 mm |
| reset offset y | -54.237 mm | -43.908 mm |
| reset offset z | 247.399 mm | 250.273 mm |

This confirms a nontrivial reset-body / semantic-tip inconsistency in the current generated 40x10 YAML.

### Reset/Settle Ablation Results

| run | purpose | post-step result | decision |
|---|---|---|---|
| `v234_sminus26_small_backout_lateral_controller_only` | controller-only override at `s=-26 mm`, `r=0.073 mm`, injected orientation error | started at `r=23.3 mm`, then `r>40 mm` | reject; override path did not preserve semantic tip |
| `v235_sminus26_small_backout_no_rot_controller_only` | same override without injected rotation | started at `r=23.5 mm` | reject; stale reset offset reproduced semantic-tip mismatch |
| `v236_repaired40x10_reset_validation` | repaired transform, no settle compensation | step 0 correct: `s=-39.96 mm`, `r=10.00 mm`, `theta=0`; step 1 zero-command settled to `r=33.4 mm` | reject as direct training config; needs settle compensation |
| `v237_repaired40x10_settle_reset_validation` | repaired transform plus existing settle compensation | step 1: `s=-40.77 mm`, `r=9.69 mm`, `theta=0.0616` | partial promote for reset geometry; theta remains bad |
| `v238_repaired40x10_settle_theta_reset_validation` | additionally apply stored strict-theta correction | step 1: `s=-43.31 mm`, `r=14.27 mm`, `theta=0.0613` | reject; theta correction did not improve post-step theta/lateral |
| `v239_eval_guide_repairedsettle_defer_finalrot_40x10` | no-learning guide on repaired+settle episode | step 1 `r=9.80 mm`, step 260 `r=23.63 mm`, `theta=0.0833` | reject for training; drift worse than v230/v233 |

Interpretation:

- The reset metadata was indeed inconsistent, and the new repair script can make step-0 semantic geometry correct.
- The first physics/control settle is still a separate problem; applying the existing settle shift improves lateral but not orientation.
- The repaired+settle guide is not yet a training candidate because it drifts laterally during approach.
- The next reset/fix iteration should calibrate the post-step semantic orientation and lateral settle together, not independently. A valid reset config must satisfy post-step, not only pre-step, `s/r/theta`.

### Architecture Ablation Path

The current online SERL stack already supports stronger critic image encoders through config:

- `--critic_image_encoder resnet18_imagenet`
- `--critic_image_encoder convnext_tiny_imagenet`

These should be tried only after the guide/servo produces non-bypass trajectories, because the current failure appears in no-learning guide/controller probes. The first model ablation should therefore be:

1. Same fixed guide/curriculum as the best non-bypass candidate.
2. Current `small_conv` critic baseline.
3. `resnet18_imagenet`.
4. `convnext_tiny_imagenet`.
5. Same checkpoint/eval cadence and strict post-step metrics.

Actor/history issue:

- The online actor is a frozen ACT TorchScript with residual adapter; ACT itself emits an action chunk, but the adapter correction is current-state plus base-action, not an explicit recurrent history module.
- Do not replace the actor first. If controller/guide can solve the nominal geometry but policy learning cannot reproduce it, then add a history-conditioned adapter or retrain ACT with temporal observations as a separate architecture ablation.

### Literature-Driven Experiment Notes

- SERL argues that robotic RL needs sample-efficient off-policy learning, engineered rewards, resets, and controllers, matching this repo's current direction. Source: https://arxiv.org/abs/2401.16013
- HIL-SERL supports intervention/correction-heavy learning for precision manipulation. For this repo, privileged guide/servo rollouts are the practical analogue. Source: https://arxiv.org/abs/2410.21845
- Eureka supports LLM-generated reward search, but this task must filter any reward proposal through strict geometry and videos due known tip-depth false positives. Source: https://arxiv.org/abs/2310.12931
- RL-VLM-F suggests VLM feedback can shape visual rewards, but sub-millimeter insertion likely needs semantic geometry as the checker; VLMs may help label visual sanity, not define success. Source: https://arxiv.org/abs/2402.03681
- IndustReal/contact-rich assembly work supports staged curricula and controller-aware residual action integration rather than reward-only tuning. Source: https://arxiv.org/abs/2305.17110

### Next Experiments

1. Reset calibration v240: derive a post-step correction that jointly targets 40 mm axial, 10 mm lateral, and low semantic theta after one zero-action settle step.
2. Controller-only validation v241: use the repaired post-step reset and apply tiny lateral/backoff commands. Promote only if realized `r` does not jump.
3. Guide-only validation v242: no-learning guide with final rotation disabled until near lip; stop at the first large lateral drift and inspect command vs realized motion.
4. Training v243: only if v242 reaches near-gate without bypass, train from the best shallow-insertion checkpoint with conservative actor updates and checkpoint/video rollouts every 30 minutes.
5. Architecture ablation v244+: only after a guide-safe dataset exists; compare `small_conv`, `resnet18_imagenet`, and `convnext_tiny_imagenet` critics under identical configs.

## 2026-05-28 Update: 40x10 Training Gate And Lateral-Sign Finding

The 40 mm axial / 10 mm lateral path has now been trained briefly, but not successfully.
Short online SERL was attempted only after guide/eval probes; training was stopped whenever
post-step geometry showed lateral bypass.

| run | type | key post-step result | decision |
|---|---|---|---|
| `v240_poststep_calibrated40x10_reset_validation` | reset validation | step 1 `s=-40.00 mm`, `r=10.01 mm`, `theta=0.0616` | promote for position reset only; theta still bad |
| `v242_poststep_position_theta_calibrated40x10_reset_validation` | reset+theta calibration | step 1 `s=-43.31 mm`, `r=14.31 mm`, `theta=0.0613` | reject; orientation calibration worsened position and did not reduce theta |
| `v243_eval_guide_poststepcalib_defer_finalrot_40x10` | no-learning guide | step 260 `s=-23.40 mm`, `r=0.06 mm`, `theta=0.0990`; strict false | lateral success but orientation worsened; training candidate only with caution |
| `v244_train_poststepcalib40x10_short` | online SERL | stopped after lateral drift; step 190 `r=40.60 mm` | reject; training/warmup path bypassed |
| `v245_eval_poststepcalib_repeat_seed56764` | no-learning repeat | seed repeat also drifted; step 59 `r=19.02 mm` | reject; failure not caused by gradient updates |
| `v246_train_poststepcalib_seed56763` | online SERL | step 150 `r=33.59 mm` before useful learning | reject |
| `v247_eval_latsignneg` | no-learning guide, negative lateral sign, smaller lateral steps | step 180 `s=-27.53 mm`, `r=0.47 mm`, `theta=0.0717` | promote; first current non-bypass 40x10 guide |
| `v248_train_latsignneg_seed56766` | online SERL | step 150 `r=28.36 mm` | reject; seed-sensitive |
| `v249_train_latsignneg_seed56765` | online SERL | stable to step 190, then after updates `r=3.38 mm` at step 210 | reject; actor updates destabilized lateral retention |
| `v250_train_latsignneg_imitationonly_lr2e8` | online SERL, actor-Q disabled, tiny LR | no updates before drift; step 220 `r=5.24 mm` | reject; underlying guide loses retention after shallow approach |
| `v251_eval_targettipservo_theta_gate` | no-learning, target-tip servo theta gate, early final orientation | stopped axial progress but final rotation induced lateral sweep; step 240 `r=8.48 mm` | reject; rotation-induced lateral sweep |
| `v252_eval_targettipservo_no_finalrot` | no-learning, target-tip servo, final rotation disabled | step 240 `s=-24.23 mm`, `r=0.16 mm`, `theta=0.0737` | promote for lateral retention; still orientation-blocked/no full insertion |
| `v253_eval_targettipservo_no_finalrot_depthrecovery` | v252 plus realized-depth recovery | step 260 `s=-25.53 mm`, `r=0.06 mm`, `theta=0.0727` | partial promote; less lateral drift but still unintended inward creep |
| `v254_eval_targettipservo_hold_theta_bad` | stronger backoff with smaller lateral step | step 260 `s=-35.28 mm`, `r=14.72 mm`, `theta=0.0717` | reject; over-constrained lateral correction and drifted sideways |
| `v255_eval_axisonly_finalrot_small` | tiny axis-only final orientation, fixed-world guard rotation off | step 260 `s=-28.06 mm`, `r=0.07 mm`, `theta=0.0721` | current best centered hold; no orientation improvement |
| `v256_eval_axisonly_strong_backoff` | v255 with stronger backoff | step 260 `s=-36.47 mm`, `r=16.68 mm`, `theta=0.0746` | reject; strong backoff destabilized lateral retention |
| `v257_eval_axisonly_depthrecovery_zerolat` | patched depth-recovery lateral scale set to 0 | step 260 `s=-36.37 mm`, `r=25.68 mm`, `theta=0.0807` | reject; zeroing lateral during recovery loses centering badly |

Current interpretation:

- The original lateral correction sign is wrong or unreliable for the repaired 40x10 starts. `--target_action_guide_lateral_direction_sign -1.0` plus smaller lateral steps is required for the current non-bypass path.
- Training has been attempted, but actor updates are not yet the limiting factor. The guide itself reaches a shallow, well-centered state and then either:
  - keeps advancing while `theta` is bad, or
  - rotates to reduce theta and sweeps the semantic tip laterally.
- The best current controlled state is v252: centered (`r <= 0.2 mm`) but orientation-blocked (`theta ~= 0.074 rad`) and far from full insertion (`s=-24.23 mm` vs full target about `45.8 mm`).
- The current best after the follow-up guard probes is v255: centered (`r ~= 0.07 mm`) and less shallow than v252 (`s=-28.06 mm`), but still orientation-blocked (`theta ~= 0.072 rad`).
- Diagnostics from v255 show controller realization mismatch: the guard commanded a negative axial/backoff component around `-59 um`, but the semantic tip still realized about `+46 um` inward motion on that step. Pure config backoff is therefore not enough; too much backoff destabilizes lateral retention, as v256 showed.
- I added a backward-compatible guard flag, `--insertion_action_guard_target_tip_servo_realized_depth_recovery_lateral_scale`, to scale lateral correction during realized-depth recovery. Validation passed with `python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py` and `.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py` (`30 passed`). Setting the scale to `0.0` in v257 was too aggressive and caused lateral loss, so any future use should sweep intermediate scales such as `0.25` and `0.5`.
- Full strict success remains 0. This is not a reward-success problem; it is a guide/servo/controller realization problem.

Next concrete change:

1. Sweep intermediate realized-depth-recovery lateral scales (`0.25`, `0.5`) around the v255 configuration. The endpoints are known: scale `1.0` creeps inward but holds `r`; scale `0.0` holds back but loses `r`.
2. Add an orientation trim mode that is accepted only if the next measured step reduces theta without increasing `r`; otherwise automatically revert to no-rotation hold. This should explicitly classify `rotation_induced_lateral_sweep` instead of continuing.
3. Only resume online SERL after a guide-only run holds `r <= 0.5 mm` and reduces theta toward `<= 0.03 rad` without losing module consistency.

### 2026-05-28 Continuation: Architecture And Reset-Settle Results

Validation run before further experiments:

```bash
python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py aic_utils/aic_isaac/scripts/train_isaac_online_serl.py
.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_isaac_online_serl.py
```

Result: `43 passed`.

Implemented online SERL architecture switches:

- `--critic_image_encoder_override {small_conv,resnet18,resnet18_imagenet,convnext_tiny,convnext_tiny_imagenet}` for config-only critic backbone ablations.
- `--critic_state_history_steps N` for critic-only low-dimensional history. The ACT actor still receives the current 82D state; critics receive `obs["critic_state"]` with current-first flattened history.

Smoke results:

| run | result |
|---|---|
| `smoke_v278_critic_history4_step2` | ran steps 1-2, config recorded `critic_state_history_steps=4`, `critic_state_dim=328`; no lingering process |
| `smoke_v280_convnext_critic_step1` | ran step 1, config recorded `critic_image_encoder=convnext_tiny`; no lingering process |

Reset-settle ablation:

| run | post-step result | decision |
|---|---|---|
| `v279_40x10_reset_settle_seed56800` | step 1 `s=-38.80 mm`, `r=9.69 mm`, `theta=0.0678`; step 2 `s=-38.09 mm`, `r=11.57 mm`, `theta=0.0731` | confirms first-step settle can add inward and lateral drift before meaningful policy action |
| `v281_40x10_v279calib_reset_settle_seed56802` | single-seed calibrated config worsened to step 2 `s=-43.72 mm`, `r=18.35 mm`, `theta=0.0719` | reject generated config; do not train from it |

Utility patch:

- `calibrate_near_gate_reset_from_settle_metrics.py` now falls back to SERL `post_step_selected_body_poses` when a train run has no `reset_diagnostic.json`.
- This makes the utility reusable for future calibration, but v281 shows single-seed calibration is not robust enough by itself.

Current recommendation: do not run long SERL yet. Next guide/controller experiment should explicitly compensate first-step settle and keep lateral correction active while rejecting orientation commands that increase realized `r/theta`.

### 2026-05-28 Continuation: Training Was Started, Then Stopped On False-Progress Risk

The broad initial-settle guard was evaluated before resuming training.

| run | post-step result | decision |
|---|---|---|
| `eval_v282_broad_initial_settle_step260` | step 260 `s=-19.30 mm`, `r=0.14 mm`, `theta=0.0755`, module consistency gate `0` | promote for short training; improves axial approach and centering relative to v272 but is still not strict insertion |
| `train_v283_broadsettle_history4_full_depth_step3200` | failed at first critic update due critic-history projection mismatch | reject as failed architecture path; fixed encoder-side critic-state normalization |
| `train_v284_broadsettle_history4_full_depth_step3200` | failed at replay sampling due variable-length `critic_state` tensors | reject as failed architecture path; fixed replay padding for variable critic-state length |
| `train_v285_broadsettle_full_depth_step3200` | stopped at checkpoint 800: `s=-12.11 mm`, `r=0.08 mm`, `theta=0.1042`, module consistency gate `0` | reject; policy learned inward motion while semantic orientation and module consistency degraded |

Validation after the critic-history fixes:

```bash
python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py
.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_isaac_online_serl.py
```

Result: `43 passed`.

Conclusion: training has started, but the current reward/guide still permits depth progress with bad semantic orientation and zero module consistency. The next training recipe should not simply run longer. It should gate axial progress more strongly on `theta` and module consistency, or use a guide-following/imitation-heavy phase with axial progress disabled until orientation is under control.
