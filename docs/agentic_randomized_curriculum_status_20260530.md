# Agentic Randomized Curriculum Status - 2026-05-30

## Why the single-reset branch stopped

Recent shallow no-guard runs were overfitting to one repeated entrance-hover reset:

| run | episode config | envs | step-1 post-step reset |
|---|---|---:|---|
| v529 | `full_depth_start2x0_v464_settle_centered_from_v462` | 1 | `s=-2.053 mm`, `r=0.082 mm`, `theta=0.0507 rad`, module `s=-25.678 mm`, module `r=1.010 mm` |
| v530 | `full_depth_start2x0_v464_settle_centered_from_v462` | 1 | identical to v529 |

That config has one YAML episode. With `num_envs=1`, the policy repeatedly saw the same shallow entrance state, which explains entrance hover/alignment behavior without robust full axial/module insertion.

## Randomized Curriculum Configs

Existing accepted mixed generator:

- `randomized_curriculum_v543_interleaved_from_v464/train_mixed_50_30_20`
- 24 YAML episodes, first 8 envs include shallow/final, near-gate, and bridge samples.
- Validation `validate_v543_reset_train_mixed_50_30_20_interleaved` step 1: `s=-20.07..+0.339 mm`, `r=0.959..2.539 mm`, `theta=0.0509..0.0574 rad`.

New configs built in this pass:

- `randomized_curriculum_v563_robotfinal_v543_bridge`
  - rejected for training order: first 8 envs were all final-window robot-joint starts, so it did not expose near/bridge states in the first reset batch.
- `randomized_curriculum_v565_interleaved_robotfinal_v543_bridge`
  - accepted for short smoke: 24 episodes, first 8 envs = 4 final, 2 near-gate, 2 bridge; full list = 12 final/shallow, 7 near-gate, 5 bridge.
  - strict success target unchanged: full depth around `46.864 mm` plus lateral, theta, module consistency, and visual sanity.
- `randomized_curriculum_v569_lowtheta_from_v552`
  - rejected. It was generated from a low-theta metric-derived pose, but reset/settle validation placed the wrist/tip far outside the port geometry.
  - Validation `validate_v569_lowtheta_mixed_reset20`: step 20 `s=-265.45..-94.10 mm`, `r=48.28..436.93 mm`, `theta=1.592..3.139 rad`, strict success false for all envs.
- `full_depth_start2x0_v570_theta_scale100_from_v464`
  - rejected. Increasing the stored strict-theta correction scale to `1.0` on the physically valid v464 family broke lateral/axial reset quality.
  - Validation `validate_v570_theta_scale100_reset20`: step 1 `s=-12.10 mm`, `r=24.97 mm`, `theta=0.0670 rad`; step 20 `s=+35.71 mm`, `r=29.91 mm`, `theta=0.0685 rad`, strict success false.

Validation commands:

- `outputs/agentic_reward_curriculum_20260529/commands/validate_v564_reset_mixed_v563_constant0_20.txt`
- `outputs/agentic_reward_curriculum_20260529/commands/validate_v566_reset_mixed_v565_constant0_20.txt`

v565 validation (`validate_v566_reset_mixed_v565_constant0_20`) showed the intended reset spread:

| row | s | r | theta | module_s | module_r | note |
|---|---:|---:|---:|---:|---:|---|
| step 1 min/mean/max | `-20.08 / -6.31 / -1.80 mm` | `0.043 / 1.200 / 4.054 mm` | `0.0505 / 0.0563 / 0.0849 rad` | `-43.70 / -29.94 / -25.42 mm` | `0.880 / 1.485 / 3.850 mm` | no immediate terminations |
| step 20 min/mean/max | `-2.03 / -1.42 / -0.85 mm` | `0.043 / 0.360 / 0.792 mm` | `0.0495 / 0.0502 / 0.0507 rad` | `-25.66 / -25.04 / -24.48 mm` | `1.026 / 1.229 / 1.571 mm` | near/bridge envs terminated and recycled |

## Training Results

| run | config/checkpoint | decision | best post-step metrics |
|---|---|---|---|
| v562 | final-window robot-joint replay imitation from v561 | reject as orientation plateau | best theta `0.0397 rad` at `s=-0.660 mm`, `r=0.759 mm`; no positive `s`, no low-theta samples |
| v567 | v565 randomized, from v544 checkpoint 400 | partial axial exploration, not success | best `s=+2.102 mm`, `r=0.120 mm`, `theta=0.0540 rad`, module `s=-21.529 mm`, module `r=1.086 mm`; no theta <= `0.030 rad` |
| v568 | v565 randomized, tighter orientation gate / lower axial reward | reject: lateral bypass and no theta fix | best theta `0.0464 rad` at negative `s`; best `s=+3.466 mm` had `r=40.49 mm`, module `r=38.88 mm` |
| v573 | v565 randomized, from v567, stronger orientation pressure / reduced axial reward / 8 envs | reject as infrastructure failure | Isaac/Vulkan `ERROR_DEVICE_LOST` during scene startup before metrics/checkpoint; rerun at lower load |
| v574 | v565 randomized, from v567, same reward idea, 4 envs, no image logging | reject after checkpoint 150 as orientation-residual drift | best `s=+1.319 mm`, `r=0.333 mm`, `theta=0.0572 rad`, module `r=0.846 mm`, strict false; best theta `0.0395 rad` occurred at `s=-0.708 mm`, not insertion |

Strict success remains `false` for all runs. Positive tip depth in v567/v568 is not success because theta remains above threshold and module depth lags by roughly `23-25 mm`.
v574 confirms the same pattern under a randomized no-guard continuation: the policy can recover small positive tip depth while lateral error is low, but theta moves away from the strict threshold. This is still a tip-depth false-positive risk, not insertion.

## Orientation Realization Diagnostics

The strict theta floor is not just a reward-weight issue. Two controller probes were run from the v574/v567 near-gate state:

| probe | command family | result | decision |
|---|---|---|---|
| v575 | `pure_rotation_axes_4mrad_start103` | best theta after audit `0.04960 rad`, only `0.00038 rad` below baseline; final `r` worsened to `0.588 mm` | reject: too little realized orientation correction |
| v577 | `pure_rotation_axes_20mrad_start103` with unclipped diagnostic rotation | `+ry` reduced theta to `0.04281 rad`, but backed out to `s=-9.30 mm` and `r=1.005 mm`; final swept to `r=43.76 mm`, theta `0.07676 rad` | reject: rotation-induced lateral/axial sweep |

This supports the controller-realization diagnosis: bounded wrist rotation can affect semantic tip theta, but useful theta reduction is coupled to unacceptable tip sweep/backout unless tip-motion compensation is solved.

## Current Diagnosis

The randomized curriculum fixed the data-distribution problem, but it did not fix the deeper orientation/module consistency bottleneck. The policy can create shallow positive tip `s` under randomized starts, but:

- strict theta never drops below `0.030 rad`,
- module `s` remains far behind tip `s`,
- stricter orientation gating can trigger lateral bypass instead of solving orientation,
- the reset distribution itself still has a theta floor around `0.049-0.056 rad` even in robot-joint final-window starts.

## Next Recommendation

Do not return to single-reset shallow training. The next bounded change should target the reset/controller orientation floor directly:

1. Keep `randomized_curriculum_v565_interleaved_robotfinal_v543_bridge` as the best validated randomized curriculum for reward-only training.
2. Do not promote the low-theta metric reset (`v569`) or direct theta-scale repair (`v570`) families; both failed reset/settle validation.
3. Stop reward-only axial tuning until the semantic tip theta floor is addressed. The quaternion audit showed the strict checker is using the semantic tip orientation offset as intended; the remaining floor is a reset/controller realization problem, not just a mislabeled raw quaternion.
4. Next bounded code change should be a minimal compensated-orientation diagnostic: apply wrist rotation with explicit predicted semantic-tip translation compensation and reject any command that increases post-step `r` or backs out `s` beyond a tiny tolerance. Promote it only if post-step semantic-tip theta drops below `0.030 rad` while preserving `r <= 0.5 mm` and module consistency.

Training more on v565 without fixing the orientation floor is likely to produce more shallow tip-depth false positives, not strict full insertion.

## Update - Collision Replacement Does Not Rescue Randomized Starts

The latest contact-side diagnostic tested whether Isaac's converted SFP collision mesh was the reason randomized
shallow/final starts could not settle into valid insertion states.

| run | collision mode | reset/contact result | decision |
|---|---|---|---|
| v682 | four authored SFP body boxes | randomized shallow/final validation accepted `0/12`; step-20 `s=-1.750/+0.063/+2.742 mm`, `r=0.075/1.155/2.729 mm`, theta `0.0429/0.0443/0.0477 rad` | reject |
| v683 | all authored SFP module boxes, near-full replay | best transient `s=44.417 mm`, `r=0.591 mm`, theta `0.0432 rad`, module consistency `0.0258`; strict false | reject |
| v684 | all authored SFP module boxes | randomized shallow/final validation accepted `0/12`; step-20 theta worsened to `0.0524/0.2357/0.4190 rad`, module consistency zero | reject |

This keeps the randomized curriculum decision unchanged: do not train on the rejected shallow/final reset families, and
do not use collision disable/replacement to claim insertion. The next viable training data source must first produce
valid, non-false-positive module-following trajectories under the strict post-step checker.

## Update - Orientation Source Is Not The Randomized Curriculum Fix

I added a train-path diagnostic flag, `--episode_target_orientation_source`, because the strict wrist diagnostic uses
`reference_reward_body_start_orientation_wxyz` while historical `train.py` diagnostics used
`target_pose_world.orientation_wxyz`.

The comparison did not produce valid starts:

| run | source | strict accepted | key result |
|---|---|---:|---|
| v685 | `target_pose` | `0/44` | step-4 theta mean `0.0610 rad`; r mean `5.43 mm` |
| v687 | `reference_reward_body_start`, body-offset skipped | `0/44` | step-1 theta mean improved to `0.0234 rad`, but r mean `5.23 mm` and module consistency error mean `7.64 mm` |
| v688 | same source plus target-tip guide | `0/240` | low-theta rows over-inserted tip and lost lateral/module consistency |

So the orientation convention explains part of the diagnostic mismatch, but it does not rescue the randomized or
final-window train distributions. The current training gate remains: no long SERL run until reset/contact realization
can produce low-r, low-theta, module-consistent post-step states without tip-depth false positives.

## Update - Module-Depth Reward Branch v586-v595

After the orientation-floor rejection, I tested whether a better module-following final-window reset plus explicit
module-depth reward could preserve the best partial insertion state and advance it toward full seating.

Accepted reset source:

- `outputs/agentic_reward_curriculum_20260529/reset_validation/2026-05-30_02-12-18_validate_v587_robotstate_v483_module_following_zeroaction_reset20/accepted_episodes`
- Validation accepted 4/4 robot-joint starts from v483 module-following rows.
- Step-20 starts were physically plausible partial insertions: tip `s≈11.1 mm`, `r≈0.277-0.689 mm`,
  `theta≈0.047-0.053 rad`, module `s≈-12.5 mm`, strict false.

Reward/code change:

- Added optional `module_depth_progress` to the cheatcode insertion phase reward.
- Added train flags:
  - `--target_reward_cheatcode_module_depth_progress_scale`
  - `--target_reward_cheatcode_module_depth_progress_weight`
  - `--target_reward_cheatcode_module_depth_loss_weight`
- Positive module forward progress is gated by the existing alignment/action gates. Module retreat is penalized
  directly. Defaults are zero, so old configs remain backward-compatible.

Runs:

| run | branch | result | decision |
|---|---|---|---|
| v591 | v587 starts, low penalty, axial 200 | best module-following row `s=17.235 mm`, module `s=-6.386 mm`, `theta=0.0513 rad`; best theta `0.0395 rad` only after retreat to `s=10.606 mm`, module `s=-13.037 mm` | reject |
| v592 | current-depth consistency | improved reward scale, but best reward after retreat to `s≈9.5-10.0 mm`, module `s≈-14 mm` | reject |
| v593 | axial-dominant reward | still backed out; final `s≈9.9-11.7 mm`, module `s≈-13.7..-11.9 mm` | reject |
| v594 | module-depth progress/loss | best module-following row `s=17.348 mm`, module `s=-6.274 mm`, `theta=0.0514 rad`; only tight-theta row `theta=0.0281 rad` was a retreat state at `s=9.824 mm`, module `s=-13.818 mm` | reject as tip-depth/module false-positive risk |
| v595 | strong module loss, standalone orientation reward off | best module-following depth remained the initial reset; final retreated to `s≈11.3-11.9 mm`, module `s≈-12.3..-11.7 mm` | reject |

Current strict-success status:

- No run has `strict_success=true`.
- Closest module-following depth remains around tip `s≈17.3 mm`, still roughly `29.5 mm` short of the
  `46.864 mm` strict seated-depth target.
- The best strict-theta row is not success because it retreats to shallow tip depth and poor module depth.

Decision:

The current no-guard reward-only branch is blocked. The actor repeatedly trades module-following depth for improved
lateral/orientation scalar reward. Reward tuning alone, including current-depth consistency and explicit module-depth
retreat loss, did not produce deeper module-consistent insertion from the validated v587 starts.

Next recommended code path:

Add a teacher/demonstration or residual data source that can produce deeper module-following trajectories, then use
imitation or conservative residual learning. Pure reward-only updates from the current partial reset distribution are
not providing a positive example of advancing from approximately `17 mm` tip depth toward the full `46.864 mm` seated
depth.

## Update - Randomized Curriculum Rebuild v612-v617

The single-reset shallow branch remains stopped. I audited v519/v520/v524/v526/v529/v530 and confirmed they all used
`full_depth_start2x0_v464_settle_centered_from_v462` with `num_envs=1` and one YAML episode. Their first post-step
reset was effectively identical: tip `s≈-2.05 mm`, `r≈0.08-0.10 mm`, `theta≈0.0507 rad`, module
`s≈-25.68 mm`; no run produced strict success.

Reset/curriculum changes:

| config | validation | post-step result | decision |
|---|---|---|---|
| `randomized_curriculum_v612_phase_curriculum_from_v464` | separate shallow/near/bridge/heldout/mixed zero-action probes | axial targets were offset by about `-8 mm`; requested shallow `s=-3..+2 mm` settled around `s=-11.5..-5.2 mm` | reject; generator base-settle offset wrong for semantic-tip-preserving reconstruction |
| `randomized_curriculum_v613_phase_curriculum_axialcal_from_v464` | `validate_v613_*_zeroaction_reset20` | axial mapping fixed; mixed step 20 `s=-19.95..-0.15 mm`, `r=1.39..5.97 mm`, `theta=0.052..0.069` | accepted only as an intermediate; lateral still too loose |
| `randomized_curriculum_v615_train_mixed_poststepcal_v613` | `validate_v615_train_mixed_poststepcal_zeroaction_reset20` | rejected with meter-scale resets because calibration mixed local targets with Isaac global env offsets | reject; bug in calibration method |
| `randomized_curriculum_v616_train_mixed_poststepcal_envtarget_v613` | `validate_v616_train_mixed_poststepcal_zeroaction_reset20` | step 20 `s=-20.01..-0.36 mm`, `r=0.129..2.463 mm`, `theta=0.054..0.067`, 8/8 accepted under curriculum thresholds | promote for short no-guard smoke |

Code changes:

- `validate_serl_reset_settle.py` now writes `post_step_reset_metrics.csv` and
  `post_step_reset_metrics_summary.json` with step-1/final min/mean/max for `s`, `r`, `theta`, module `s/r`, strict
  count, and terminations.
- `calibrate_randomized_curriculum_from_reset_metrics.py` calibrates randomized starts from zero-action settle rows.
  The first version exposed the env-offset bug; the fixed version uses `target_world_by_env` and `target_depth_m_by_env`
  from the metrics row, so each env calibration is done in the correct Isaac world frame.

Training smokes:

| run | curriculum | result | decision |
|---|---|---|---|
| v614b | v613 mixed, 4 envs, 400 steps | strict `0/1600`; best centered positive depth `s=1.319 mm`, `r=0.403 mm`, `theta=0.05097`, module `s=-22.311 mm`; largest tip `s=3.181 mm` was lateral bypass with `r=38.452 mm` | reject; reset lateral quality still too poor |
| v617 | v616 calibrated mixed, 4 envs, 400 steps | strict `0/1600`; best tip `s=4.326 mm`, `r=1.244 mm`, `theta=0.05798`, module `s=-19.298 mm`; final env2 retained `s=4.310 mm`, module `s=-19.315 mm` | partial axial/module improvement, not success; orientation/lateral/module still fail |
| v618 | v616 calibrated mixed, continue from v617 checkpoint 400, stronger module-depth progress/loss and lower gated axial reward | strict `0/1600`; best/final env2 `s=4.352 mm`, `r=1.242 mm`, `theta=0.05797`, module `s=-19.273 mm` | stable but not meaningfully better than v617; module-depth pressure did not solve lateral/orientation gate |

Current best from this randomized branch is v618 by a tiny margin, but it is far from strict full insertion: roughly
`42.5 mm` short of the `46.864 mm` seated tip-depth target, theta is about `0.058 rad` rather than `<0.030 rad`,
lateral is above the strict `0.5 mm` target, and module depth is still negative. This is not a visual or metric
success.

Next tuning decision:

- Continue from v617 only with stricter gated learning: increase depth-gated module consistency pressure, tighten
  lateral/offgate penalties, and reduce the chance that axial reward wins when `r/theta/module` are poor.
- Do not re-enable hard servo overrides yet. The v616 curriculum is now plausible enough for reward-only training
  smoke tests, but any promotion still requires strict post-step geometry and visual evidence.

## Update - v850 Tip-Preserving Randomization and v855/v856 Smokes

The v701/v849 audit found a specific reset-generation risk: train-mixed episodes reset `gripper_tcp`, but semantic
reward/success uses `sfp_tip_link`, and nonzero orientation perturbations were not marked tip-preserving. I updated the
randomized curriculum builder so it reconstructs the measured reset-body-to-tip vector from the calibrated
`lowtheta_metric_reset` metadata and rotates around the semantic tip when adding orientation perturbations.

New generated curriculum:

```text
outputs/agentic_reward_curriculum_20260529/generated_episode_configs/randomized_curriculum_v850_tip_preserving_from_v642
```

Validation:

```text
outputs/agentic_reward_curriculum_20260529/reset_settle_validation/2026-05-30_20-53-17_v852_v850_tip_preserving_train_mixed_8env_zeroaction_repoout
```

The v852 reset/settle validation accepted 8/8 sampled train-mixed episodes. Step-2 tip metrics were
`s=-19.90..+1.17 mm`, `r=0.265..2.532 mm`, theta `0.0162..0.0275 rad`; module metrics were
`s=-43.54..-22.47 mm`, `r=0.157..1.791 mm`. This is a better randomized training substrate than the single shallow
reset because it varies final/near/bridge starts and no longer applies orientation perturbations as pure gripper-body
rotations.

Short training probes:

| run | setup | strict success | result | decision |
|---|---|---:|---|---|
| v855 | no-guard ConvNeXt/history smoke from v706 on v852 accepted episodes | 0 | final tip `s` max `3.604 mm`, mean `r=0.318 mm`, mean theta `0.0194 rad`; module `s` mean `-21.818 mm`, final axial consistency error mean `43.982 mm` | partial alignment signal; not promotion |
| v856 | continue v855, larger translation clip and stronger module-depth/axial rewards | 0 | final tip `s` max `5.833 mm`, mean `r=0.383 mm`, mean theta `0.0273 rad`, theta max `0.0352 rad`; module consistency gate mean worsened to `0.434`, force mean `11.18 N`, reward/Q unstable | reject |

This reinforces the earlier diagnosis. Randomization and stronger visual/history architecture help alignment and small
shallow tip progress, but they do not create the module-following trajectory needed for strict full insertion. v856 is
especially important because it tested the obvious "more axial authority plus more module reward" patch and produced
more shallow tip depth while making strict-theta/module consistency worse.

Current recommendation:

1. Keep v850/v852 as the validated randomized curriculum substrate for future short probes.
2. Do not continue the v856-style larger-clip reward-only branch.
3. Prioritize the cheatcode/contact teacher path or imitation-heavy/HIL-SERL from valid module-following trajectories.
4. Treat any positive tip `s` with module `s` still around `-20 mm` as a false-positive risk, regardless of strict
   lateral/orientation alignment.
- After v618, the next bounded change should target the orientation floor directly. Pure module-depth reward pressure
  was stable but did not move theta below `0.05 rad` or bring `r` under `0.5 mm` during positive-depth steps.

Validation commands used after code changes:

```bash
python -m py_compile \
  aic_utils/aic_isaac/scripts/validate_serl_reset_settle.py \
  aic_utils/aic_isaac/scripts/calibrate_randomized_curriculum_from_reset_metrics.py
.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py
```

Result: `31 passed`.

## Update - Full Randomized Curriculum Validation v620-v623

The single-reset shallow training path remains stopped. I expanded the curriculum validation beyond the earlier
8-episode v616 smoke set:

| config / run | purpose | post-step reset distribution | decision |
|---|---|---|---|
| `randomized_curriculum_v620_phase_curriculum_poststepcal_envtarget_v613/shallow_final` | 24 calibrated shallow/final episodes | step 20 `s=-3.479..+4.731 mm`, `r=0.035..2.531 mm`, `theta=0.052..0.067` | usable for curriculum, not success |
| `.../near_gate` | 24 calibrated near-gate episodes | step 20 `s=-10.582..-2.003 mm`, `r=0.040..3.496 mm`, `theta=0.050..0.073` | usable with lateral filtering |
| `.../bridge` | 24 calibrated bridge episodes | step 20 `s=-20.539..-9.491 mm`, `r=0.171..5.436 mm`, `theta=0.048..0.088` | usable with lateral filtering |
| `.../heldout_40x10` | held-out eval starts | step 20 `s=-40.461..-39.829 mm`, `r=9.075..10.205 mm`, `theta=0.050..0.102` | held-out eval only |
| `.../train_mixed_50_30_20` | 48-episode mixed curriculum | step 20 `s=-20.033..+5.156 mm`, `r=0.127..4.750 mm`, `theta=0.049..0.087` | too broad for current actor |
| `randomized_curriculum_v622_tight_lowr_50_33_17_from_v620` | 36-episode low-lateral filtered set | validation accepted 35/36; rejected one `r=4.645 mm` episode | promote accepted subset only |

Training and policy-only checks:

| run | setup | best post-step metric | decision |
|---|---|---|---|
| v621 | no-guard v620 mixed, 8 envs, from v618 ckpt400 | no centered positive depth; best `s=-2.393 mm`, `r=2.207 mm`, `theta=0.05868`, module `s=-26.011 mm` | stop at ckpt100; `no_axial_progress` plus lateral drift |
| v622 | no-guard accepted v622 low-r set, 8 envs, lower Q/lr/action clip, stronger lateral penalty | no centered positive depth; best `s=-1.002 mm`, `r=1.608 mm`, `theta=0.05592`, module `s=-24.621 mm` | reject; tighter curriculum did not unlock insertion |
| v623-v483 | policy-only v483 ckpt400 on v622 accepted starts | best centered `s=3.059 mm`, `r=0.851 mm`, `theta=0.05339`, module `s=-20.567 mm`; strict 0 | v483 is best of compared checkpoints but still not trainable success |
| v623-v514 | policy-only v514 ckpt400 on v622 accepted starts | best centered `s=1.059 mm`, `r=0.977 mm`, `theta=0.05411`, module `s=-22.565 mm`; strict 0 | reject as worse than v483 |
| v623-v618 | policy-only v618 ckpt400 on v622 accepted starts | best centered `s=2.984 mm`, `r=0.751 mm`, `theta=0.05209`, module `s=-20.642 mm`; strict 0 | similar to v483, still not success |
| v624 | v483 ckpt400, v622 accepted starts, no-execute phase guide imitation, guide weight 40, selected phase repeat 4 | guide loss active (`target_action_guide_loss_weighted≈0.02`, selected fraction `0.25`); best centered `s=0.881 mm`, `r=0.988 mm`, `theta=0.05346`, module `s=-22.744 mm`; strict 0 | reject; constructive guide loss as configured regressed depth |

Interpretation:

- The randomized reset problem is fixed enough for experiments: there are now validated shallow/final, near-gate,
  bridge, and held-out 40/10 distributions, plus a 35-episode accepted low-r training subset.
- Training from the current v618 checkpoint does not improve under no-guard reward-only updates; it widens lateral
  error before making axial progress.
- Rechecking older v483/v514 checkpoints on the same validated randomized starts did not recover the old rare
  `8-17 mm` insertion transient. The best policy-only v623 row is still only `s=3.059 mm` with theta above strict and
  module depth far behind.
- The first constructive-target smoke v624 confirmed the phase guide loss is wired, but it did not improve geometry.
  It likely teaches alignment/hover corrections without providing a repeatable module-consistent insertion trajectory.
- Strict success remains false for every run. Positive tip depth is still a false-positive risk because `theta`,
  strict lateral, and module consistency fail.

## Update - Low-Theta Reset Local-Origin Fix and v643 No-Guard Continuation

The single-reset shallow branch remains stopped. I tested whether the low-theta full-depth contact state seen in the
v637 controller/contact diagnostic could be converted into valid randomized final-window reset episodes.

Code fix:

- `build_lowtheta_reset_from_metric.py` now supports selected-env metric extraction with an explicit selected-env
  origin. The previous v641 attempt subtracted only the env8 clone delta and left the env0 world origin in episode-local
  YAML coordinates; that produced `~13 m` TCP reset errors and meter-scale lateral errors.
- The fixed v642 generator derives env0 origin from
  `post_step_insertion_geometry.entrance_world_env0 - start_near_gate.target_gate_position`, then adds the rounded
  selected clone delta before converting world poses back to episode-local coordinates.
- Py-compile passed for the changed script.

Validation:

| run | purpose | result | decision |
|---|---|---|---|
| v641 | first env8 low-theta reset generation from v637 step 2 | invalid local/world conversion; reset YAML had clone-world coordinates around `[10, -9, ...]`; validation showed meter-scale lateral errors | reject; generator bug |

## Update - Randomized-Curriculum Plan Refresh v653-v656

The current no-guard insertion plan is updated to stop single-reset shallow training. Recent v529/v530 and earlier
shallow runs used the same one-YAML folder:

- `outputs/agentic_reward_curriculum_20260529/generated_episode_configs/full_depth_start2x0_v464_settle_centered_from_v462`
- `num_envs=1`
- step-1 post-step reset in both v529 and v530: tip `s=-2.053 mm`, `r=0.082 mm`, `theta=0.05073 rad`,
  module `s=-25.678 mm`, module `r=1.010 mm`, strict false.

That setup repeatedly presents one entrance-hover state. It is insufficient for robust axial/module-consistent
insertion and should not be resumed.

The later randomized no-guard smokes also remain rejected:

| run | episode config | envs | result | decision |
|---|---|---:|---|---|
| v653 | `randomized_curriculum_v652_lowtheta_from_v642/train_mixed_50_30_20` | 4 | first row mean `s=-2.228 mm`, `r=1.060 mm`, `theta=0.04457`; final mean `s=+0.369 mm`, `r=3.573 mm`, `theta=0.04793`, module `s=-23.254 mm` | reject: lateral bypass / tip-depth false-positive risk |
| v654 | `randomized_curriculum_v652_lowtheta_from_v642/shallow_final/episodes` | 4 | first row mean `s=+0.434 mm`, `r=0.652 mm`, `theta=0.04345`; stopped at step 33 after reward collapse to `-55.88`; strict false | reject: unstable reward scaling / actor drift |

Validated v652 reset buckets show useful axial spread but not strict-orientation starts:

| bucket | validation run | post-step reset distribution | decision |
|---|---|---|---|
| shallow/final | `validate_v652_shallow_final_zeroaction_reset20_env24_serl_probe` | step 1 `s=-2.813..+1.999 mm`, `r=0.267..2.773 mm`, `theta=0.04097..0.04595`; step 20 mean `r=1.944 mm`, theta mean `0.04486` | reject for strict final-window training until lower-theta starts exist |
| near-gate | `validate_v652_near_gate_zeroaction_reset20_env24_serl_probe` | step 1 `s=-9.984..-1.681 mm`, `r=0.345..5.078 mm`, `theta=0.04286..0.04641`; step 20 mean `r=3.541 mm` | reject as too laterally loose for long training |
| bridge | `validate_v652_bridge_zeroaction_reset20_env24_serl_probe` | step 1 `s=-20.039..-8.739 mm`, `r=0.415..11.416 mm`, `theta=0.04231..0.05444`; step 20 mean `r=5.781 mm` | held-out/probe only until near/final is stable |

I ran two reset-orientation calibration sweeps before any further training:

| config / run | source | validation | result | decision |
|---|---|---|---|---|
| `lowtheta_orisweep_v655_world_from_v642` | deep v642 low-theta metric reset | `validate_v655_orisweep_world_from_v642_zeroaction_reset10_env75` | 75 variants, accepted `0/75`; step 10 theta `0.03470..0.81737 rad`, `r=0.149..26.169 mm`; best theta row still had `r=8.828 mm` | reject |
| `shallow_orisweep_v656_world_from_v652` | v652 shallow/final bucket | `validate_v656_shallow_orisweep_world_from_v652_zeroaction_reset10_env75` | 75 variants, accepted `0/75`; step 10 `s=-3.957..-0.086 mm`, `r=1.055..10.827 mm`, `theta=0.04076..0.12739 rad` | reject |

Commands used for the new validation pass:

```bash
.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/sweep_near_gate_reset_orientation.py \
  --input-dir outputs/agentic_reward_curriculum_20260529/generated_episode_configs/lowtheta_metric_v642_from_v637_env8_step2_localfix \
  --output-dir outputs/agentic_reward_curriculum_20260529/generated_episode_configs/lowtheta_orisweep_v655_world_from_v642 \
  --rx-values=-0.08,-0.04,0.0,0.04,0.08 --ry-values=-0.08,-0.04,0.0,0.04,0.08 \
  --rz-values=-0.04,0.0,0.04 --composition world --limit 75 --overwrite

.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/validate_serl_reset_settle.py \
  --episode-config-dir outputs/agentic_reward_curriculum_20260529/generated_episode_configs/lowtheta_orisweep_v655_world_from_v642 \
  --output-root outputs/agentic_reward_curriculum_20260529/reset_validation \
  --train-output-root outputs/agentic_reward_curriculum_20260529/policy_train_runs \
  --run-name validate_v655_orisweep_world_from_v642_zeroaction_reset10_env75 \
  --num-envs 75 --steps 10 --zero-action --max-lateral-m 0.003 --max-theta-rad 0.030 \
  --min-s-m 0.0 --max-s-m 0.050 --max-wall-time-minutes 30 --seed 65530

.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/sweep_near_gate_reset_orientation.py \
  --input-dir outputs/agentic_reward_curriculum_20260529/generated_episode_configs/randomized_curriculum_v652_lowtheta_from_v642/shallow_final \
  --output-dir outputs/agentic_reward_curriculum_20260529/generated_episode_configs/shallow_orisweep_v656_world_from_v652 \
  --rx-values=-0.06,-0.03,0.0,0.03,0.06 --ry-values=-0.06,-0.03,0.0,0.03,0.06 \
  --rz-values=-0.03,0.0,0.03 --composition world --limit 75 --overwrite

.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/validate_serl_reset_settle.py \
  --episode-config-dir outputs/agentic_reward_curriculum_20260529/generated_episode_configs/shallow_orisweep_v656_world_from_v652 \
  --output-root outputs/agentic_reward_curriculum_20260529/reset_validation \
  --train-output-root outputs/agentic_reward_curriculum_20260529/policy_train_runs \
  --run-name validate_v656_shallow_orisweep_world_from_v652_zeroaction_reset10_env75 \
  --num-envs 75 --steps 10 --zero-action --max-lateral-m 0.003 --max-theta-rad 0.030 \
  --min-s-m -0.004 --max-s-m 0.004 --max-wall-time-minutes 30 --seed 65630
```

Current decision:

- Do not continue v529/v530-style single-reset training.
- Do not launch long no-guard randomized training from v652/v655/v656. The reset distributions either start outside the
  strict orientation gate or drift laterally during settle; the previous short randomized training then exploited tip
  depth while module consistency remained poor.
- Next bounded code path should repair the reset/controller orientation floor or contact/asset realization issue before
  training. Reward/curriculum tuning alone has repeatedly produced shallow hover, lateral bypass, or tip-depth
  false positives.

## Update - Evaluator-Target Orientation Calibration v657-v659

I added an optional orientation calibration path to `calibrate_randomized_curriculum_from_reset_metrics.py`:

- `--calibrate-orientation` now left-multiplies each reset-body quaternion by
  `target_tip_quat_from_strict_evaluator * inverse(actual_post_step_tip_quat)`.
- The first implementation incorrectly used the YAML `reference_reward_body_start_orientation_wxyz`, which is
  identity-like for these episodes and not the strict evaluator's semantic target quaternion for `sfp_tip_link`.
  v657 exposed that mismatch by rotating resets into meter-scale invalid poses.
- The corrected implementation reads `post_step_insertion_geometry.target_orientation_wxyz_by_env` from the strict
  evaluator metrics row. This keeps old behavior unchanged unless `--calibrate-orientation` is passed.

Validation after the code change:

```bash
.pixi/envs/default/bin/python -m py_compile \
  aic_utils/aic_isaac/scripts/calibrate_randomized_curriculum_from_reset_metrics.py
.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py
```

Result: `31 passed`.

Calibration/validation outcomes:

| config / run | calibration source | post-step reset distribution | decision |
|---|---|---|---|
| `shallow_orical_v657_from_v656_step10` | v656 step 10, using YAML reference quaternion by mistake | step 10 `s=-277.202..-76.691 mm`, `r=12.037..645.894 mm`, `theta=1.358..3.131 rad` | reject; documents why YAML reference quat is not usable as the strict semantic target |
| `shallow_orical_v658_evaltarget_from_v656_step10` | v656 step 10, using strict evaluator target quaternion | step 1 theta improved to `0.0152..0.0560 rad`, but step 10 drifted to `theta=0.0397..0.0571 rad`; step 10 `r=0.323..18.795 mm`; accepted `0/75` | reject |
| `shallow_orical_v659_from_v658_step1` | v658 step 1, second per-env calibration iteration | step 1 theta `0.0165..0.0537 rad`; step 10 drifted to `theta=0.0384..0.0750 rad`; step 10 `r=0.872..39.137 mm`; accepted `0/75` | reject |

Commands:

```bash
.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/calibrate_randomized_curriculum_from_reset_metrics.py \
  --input-config-dir outputs/agentic_reward_curriculum_20260529/generated_episode_configs/shallow_orisweep_v656_world_from_v652 \
  --validation-run outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-30_08-56-43_validate_v656_shallow_orisweep_world_from_v652_zeroaction_reset10_env75_serl_probe \
  --output-config-dir outputs/agentic_reward_curriculum_20260529/generated_episode_configs/shallow_orical_v658_evaltarget_from_v656_step10 \
  --step 10 --calibrate-orientation --overwrite

.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/validate_serl_reset_settle.py \
  --episode-config-dir outputs/agentic_reward_curriculum_20260529/generated_episode_configs/shallow_orical_v658_evaltarget_from_v656_step10 \
  --output-root outputs/agentic_reward_curriculum_20260529/reset_validation \
  --train-output-root outputs/agentic_reward_curriculum_20260529/policy_train_runs \
  --run-name validate_v658_shallow_orical_evaltarget_zeroaction_reset10_env75 \
  --num-envs 75 --steps 10 --zero-action --max-lateral-m 0.003 --max-theta-rad 0.030 \
  --min-s-m -0.004 --max-s-m 0.004 --max-wall-time-minutes 30 --seed 65830

.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/calibrate_randomized_curriculum_from_reset_metrics.py \
  --input-config-dir outputs/agentic_reward_curriculum_20260529/generated_episode_configs/shallow_orical_v658_evaltarget_from_v656_step10 \
  --validation-run outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-30_09-06-23_validate_v658_shallow_orical_evaltarget_zeroaction_reset10_env75_serl_probe \
  --output-config-dir outputs/agentic_reward_curriculum_20260529/generated_episode_configs/shallow_orical_v659_from_v658_step1 \
  --step 1 --calibrate-orientation --overwrite

.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/validate_serl_reset_settle.py \
  --episode-config-dir outputs/agentic_reward_curriculum_20260529/generated_episode_configs/shallow_orical_v659_from_v658_step1 \
  --output-root outputs/agentic_reward_curriculum_20260529/reset_validation \
  --train-output-root outputs/agentic_reward_curriculum_20260529/policy_train_runs \
  --run-name validate_v659_shallow_orical_from_v658_step1_zeroaction_reset10_env75 \
  --num-envs 75 --steps 10 --zero-action --max-lateral-m 0.003 --max-theta-rad 0.030 \
  --min-s-m -0.004 --max-s-m 0.004 --max-wall-time-minutes 30 --seed 65930
```

Decision:

Do not train from v657, v658, or v659. The calibration can make some first post-step theta values look strict, but the
state does not remain physically settled: theta rebounds above `0.030 rad`, lateral error remains too large, and module
depth is still around `-24 mm`. The blocker is now classified as reset/controller/contact realization drift, not lack of
randomized YAML generation. The next single code path should be a reset-settle controller/asset diagnostic that measures
why zero-action settle changes semantic tip pose this much, including the known oversized `body_sdf_collision` contact
ablation, before any further long reward-only training.

## Update - Reset-Settle Contact/Controller Diagnostics v660-v663

I added the same diagnostic-only collision toggle used by `wrist_contact_realization.py` to the actual SERL
`train.py` reset/eval path:

- New flag: `--disable_collision_prim_regex`
- It is repeatable, config-driven, and defaults to off.
- It writes `collision_toggle_report.json` and embeds the report in `train_config.json`.
- `validate_serl_reset_settle.py` now forwards this through `--disable-collision-prim-regex`.

Validation after the code change:

```bash
.pixi/envs/default/bin/python -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py \
  aic_utils/aic_isaac/scripts/validate_serl_reset_settle.py
.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py
```

Result: `31 passed`.

Diagnostic runs:

| run | path | setup | result | decision |
|---|---|---|---|---|
| v660 | `contact_realization_runs/2026-05-30_09-15-14_v660_v659_shallow_orical_posehold_collision_on_env12` | standalone wrist diagnostic, v659 reset, zero pose-hold, collisions on | no strict rows; best row `s=-1.275 mm`, `r=0.462 mm`, `theta=0.0211`, module consistency `0`; wrist force proxy around `63k` | collision-on baseline remains invalid |
| v661 | `contact_realization_runs/2026-05-30_09-16-04_v661_v659_shallow_orical_posehold_disable_body_sdf_env12` | same, but disabled `sfp_module_link/collisions/body_sdf_collision` | no strict rows; best row `s=-0.764 mm`, `r=0.250 mm`, `theta=0.0220`, module consistency `0`; force proxy still around `64k` | disabling only body SDF does not fix this reset family |
| v662 | `reset_validation/2026-05-30_09-19-05_validate_v662_v659_disable_body_sdf_zeroaction_reset10_env12` | actual `train.py` reset-settle path, v659, body SDF disabled | accepted `0/12`; step 10 `s=-0.790..-0.270 mm`, `r=1.354..12.250 mm`, `theta=0.0396..0.0485`, module `s=-24.42..-23.90 mm` | reject |
| v663 | `reset_validation/2026-05-30_09-22-10_validate_v663_v659_disable_all_sfp_module_collisions_zeroaction_reset10_env12` | actual `train.py` path, all SFP module/tip collision prims disabled | accepted `0/12`; step 10 `s=4.646..7.205 mm`, but `r=21.74..34.15 mm`, `theta=0.0574..0.0746`, module `s=-18.98..-16.42 mm`; reward collapsed | reject; disabling all module collisions causes lateral escape |

The v662 actual `train.py` diagnostic confirms the collision toggle was applied: `matched_count=12` for
`/Robot/cable/sfp_module/sfp_module_link/collisions/body_sdf_collision$`.

Important observation:

The actual `train.py` diagnostics show that with zero commanded TCP action, the gripper/semantic bodies still move
millimeters during the first environment step. This is passive reset-settle/contact/constraint motion, not actor
learning and not a reward-design issue. The standalone wrist diagnostic confirms the same family is not stabilized by
removing only `body_sdf_collision`; disabling all SFP module collisions creates a different false positive pattern
where tip depth increases while lateral error explodes.

Current decision:

- Do not train from v659 or its collision-disabled variants.
- Do not make `body_sdf_collision` removal a default training fix. It helped a separate scripted deep-contact
  diagnostic (v651) but did not stabilize randomized shallow/final resets in the actual training path.
- Classify the current blocker as `reset_settle_passive_contact_or_constraint_drift`.
- Next recommended code path: build accepted robot-joint-state resets from already-settled post-step rows only if those
  rows have low `r/theta` after settle; otherwise stop reset tuning and return to a controller/asset-level diagnostic
  that measures why the cable/module is preloaded by the IK reset.

## Update - Direct Semantic Tip Reset Diagnostic v664

I tested whether the v659 shallow/final reset drift was mainly caused by targeting `gripper_tcp` instead of the
semantic tip. This was a reset-only diagnostic, not a training-mode change, because direct `sfp_tip_link` IK was already
a known risk.

Generated diagnostic config:

- `outputs/agentic_reward_curriculum_20260529/generated_episode_configs/shallow_tipreset_v664_from_v659_diag`
- source: first 12 v659 shallow/final episodes
- change: `start_near_gate.reset_body_name=sfp_tip_link`, reset position from
  `reference_reward_body_start_position_world`, reset orientation from the strict target pose orientation

Validation command:

```bash
.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/validate_serl_reset_settle.py \
  --episode-config-dir outputs/agentic_reward_curriculum_20260529/generated_episode_configs/shallow_tipreset_v664_from_v659_diag \
  --output-root outputs/agentic_reward_curriculum_20260529/reset_validation \
  --train-output-root outputs/agentic_reward_curriculum_20260529/policy_train_runs \
  --run-name validate_v664_tipreset_from_v659_zeroaction_reset10_env12 \
  --num-envs 12 --steps 10 --zero-action --max-lateral-m 0.003 --max-theta-rad 0.030 \
  --min-s-m -0.004 --max-s-m 0.004 --max-wall-time-minutes 30 --seed 66430
```

Result:

| run | result | decision |
|---|---|---|
| v664 | accepted `0/12`; step 1 `s=-225.872..40.121 mm`, `r=3.462..468.276 mm`, theta `1.498..3.138 rad`; step 10 `s=-230.456..39.020 mm`, `r=7.461..519.812 mm`, theta `1.428..3.081 rad` | reject |

This closes the direct-tip-reset hypothesis. Directly targeting `sfp_tip_link` at reset is dramatically unstable in the
actual SERL reset path, matching the earlier prior that direct tip IK should not be used for training. The viable reset
path remains wrist/gripper IK plus explicit tip-motion compensation, but the current compensation is not enough to
produce stable low-theta/module-consistent randomized shallow starts.

I also attempted to re-run the old `robot_joint_state_v638_full_depth_strict_from_v637` reset family with the body SDF
collision disabled, but the saved config directory now contains no YAML episodes, so there was nothing reproducible to
validate from that path without regenerating the source episodes.

## Update - Reset/Action Body Consistency Diagnostics v666-v669

The v662/v663 first-step diagnostics showed millimeter-scale motion even with zero commanded TCP action. Comparing
selected body poses from the first logged step showed:

- v662 (`body_sdf_collision` disabled, default IK body `wrist_3_link`): step-1 `gripper_tcp` moved
  `2.983/4.895/8.569 mm` min/mean/max; `sfp_tip_link` moved `1.241/3.766/12.751 mm`.
- v663 (all SFP module collisions disabled): step-1 drift became much worse, with `sfp_tip_link` moving
  `28.407/31.437/36.255 mm`, confirming collision removal creates a lateral-escape artifact.
- v664 (direct `sfp_tip_link` reset): step-1 `sfp_tip_link` moved `4.097/62.908/126.438 mm`, confirming direct tip IK
  is not viable.

The gripper-tip distance stayed fixed in these diagnostics, so the first-step drift is not primarily flexible cable
stretch. The more likely issue is reset/action/controller consistency: the near-gate reset was placing `gripper_tcp`,
while the default differential IK action controls `wrist_3_link`.

I added a validation-only option to `validate_serl_reset_settle.py`:

- `--ik-body-name <body>` exports `AIC_ISAAC_IK_BODY_NAME=<body>` before launching the underlying `train.py` probe.
- Default behavior is unchanged.

Validation:

```bash
.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/scripts/validate_serl_reset_settle.py
```

I then added a generic validation-only passthrough:

- `--extra-env KEY=VALUE` exports additional environment variables for the underlying `train.py` probe.
- This was used to test whether stronger implicit actuator hold reduces reset-settle drift.

Reset/action-body outcomes:

| run | setup | step 1 post-step distribution | step 10 post-step distribution | accepted | decision |
|---|---|---|---|---:|---|
| v666 | v659 shallow/final episodes, `AIC_ISAAC_IK_BODY_NAME=gripper_tcp` | `s=-1.532..-0.047 mm`, `r=1.762..13.448 mm`, theta `0.035..0.042` | `s=-1.450..-0.346 mm`, `r=0.165..10.696 mm`, theta `0.041..0.045` | 0/12 | partial stabilization only |
| v667 | orientation-calibrated from v666 step 10, gripper IK body | `s=-2.624..0.112 mm`, `r=2.279..17.529 mm`, theta `0.013..0.051` | `s=-2.401..-0.043 mm`, `r=1.590..18.409 mm`, theta `0.040..0.056` | 0/12 | reject; calibration worsened lateral drift |
| v668 | reset body changed to `wrist_3_link`, using measured v659 pre-step wrist pose; default IK body remains `wrist_3_link` | `s=-0.845..-0.133 mm`, `r=1.847..13.392 mm`, theta `0.034..0.039` | `s=-0.811..-0.291 mm`, `r=1.370..10.681 mm`, theta `0.040..0.043` | 0/12 | best controller-consistent reset, still not trainable |
| v669 | orientation-calibrated from v668 step 10 | `s=-1.617..-0.357 mm`, `r=3.218..24.396 mm`, theta `0.032..0.069` | `s=-3.488..-0.130 mm`, `r=0.872..24.971 mm`, theta `0.037..0.070` | 0/12 | reject |
| v670 | v668 wrist reset with `AIC_ISAAC_ARM_ACTUATOR_STIFFNESS=8000`, `AIC_ISAAC_ARM_ACTUATOR_DAMPING=400` | `s=-1.098..-0.018 mm`, `r=1.483..12.647 mm`, theta `0.034..0.042` | `s=-1.507..-0.012 mm`, `r=0.837..10.415 mm`, theta `0.040..0.045` | 0/12 | reject; stronger arm hold does not solve the theta floor |
| v671 | v668 with 25% position and 25% orientation post-step calibration | `s=-1.076..-0.312 mm`, `r=2.750..9.207 mm`, theta `0.033..0.041` | `s=-0.787..-0.327 mm`, `r=2.424..8.695 mm`, theta `0.039..0.045` | 0/12 | reject; lower max r but worse mean r and still theta-blocked |
| v672 | v668 with 25% position-only post-step calibration | `s=-2.989..-0.194 mm`, `r=1.492..21.340 mm`, theta `0.035..0.056` | `s=-2.697..0.602 mm`, `r=0.971..24.222 mm`, theta `0.040..0.062` | 0/12 | reject; position-only correction destabilizes |

I then made this branch reproducible instead of continuing one-off shell edits:

- `build_action_body_reset_from_metrics.py` builds reset configs that target a measured controller/action body pose
  such as `wrist_3_link`.
- `run_wrist_reset_compensation_sweep.py` runs bounded gain sweeps by calling the calibration and validation scripts,
  writing one JSON record per candidate plus `summary.json`.

Validation:

```bash
.pixi/envs/default/bin/python -m py_compile \
  aic_utils/aic_isaac/scripts/build_action_body_reset_from_metrics.py \
  aic_utils/aic_isaac/scripts/run_wrist_reset_compensation_sweep.py \
  aic_utils/aic_isaac/scripts/calibrate_randomized_curriculum_from_reset_metrics.py \
  aic_utils/aic_isaac/scripts/validate_serl_reset_settle.py
.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py
```

Result: `31 passed`.

Scripted v673 sweep:

- sweep folder:
  `outputs/agentic_reward_curriculum_20260529/reset_compensation_sweeps/2026-05-30_09-49-40_v673_wristcomp_from_v668`
- base config: `shallow_wristreset_v668_from_v659_prestep`
- source validation: v668 step 10
- gains tested: position `{0.0, 0.1}` x orientation `{0.0, 0.1}`

| candidate | accepted | final `s` mm min/mean/max | final `r` mm min/mean/max | final theta min/mean/max | decision |
|---|---:|---:|---:|---:|---|
| `p0_o0` | 0/12 | `-1.244 / -0.710 / -0.372` | `1.589 / 4.784 / 12.803` | `0.0399 / 0.0421 / 0.0458` | reject |
| `p0_o0.1` | 0/12 | `-2.680 / -0.990 / -0.396` | `1.937 / 5.556 / 13.388` | `0.0391 / 0.0415 / 0.0441` | reject |
| `p0.1_o0` | 0/12 | `-1.172 / -0.655 / -0.252` | `1.038 / 5.114 / 13.882` | `0.0402 / 0.0431 / 0.0504` | reject |
| `p0.1_o0.1` | 0/12 | `-1.481 / -0.750 / -0.429` | `1.907 / 5.716 / 12.824` | `0.0397 / 0.0425 / 0.0459` | reject |

Decision:

- Do not launch long randomized no-guard training from v666-v673. None produce accepted shallow/final-window resets.
- v668 is the least bad reset/action-body configuration, but its post-settle theta remains around `0.040-0.043 rad`
  and lateral error is still too large for strict final insertion learning.
- The randomized-curriculum branch is blocked by reset/controller realization, not by lack of curriculum YAMLs.
- Next recommended code change: implement a controller-level reset compensation/hold diagnostic that makes the reset
  body and action body identical (`wrist_3_link` by default), then optimizes the wrist reset pose against post-step
  semantic tip `s/r/theta` after a one-step settle. Do not use direct `sfp_tip_link` IK.
| v642 | same source after selected-env-origin fix | step 1 `s=28.591..47.999 mm`, `r=0.099..2.453 mm`, `theta=0.0389..0.0478 rad`, module `s=4.956..24.363 mm`; step 10 drifted to `s=14.161..55.751 mm`, `r=0.244..14.609 mm`, `theta=0.0364..0.1293 rad`; accepted count `0/15` | reject for strict training; physically closer but theta/module/settle drift still fail |

The v642 result is useful evidence: the coordinate bug is fixed, but replaying a measured low-theta contact pose does
not create stable strict full-insertion starts. Even the best early rows remain above the strict `theta < 0.030 rad`
threshold or fail module/body consistency after settle.

Bounded randomized no-guard continuation:

| run | setup | best post-step metric | decision |
|---|---|---|---|
| v643 | accepted v622 randomized low-r curriculum, no guard/servo overrides, resumed from v639 checkpoint 100; stronger gated axial/module-depth reward and bypass penalty | strict `0`; best centered positive-depth row at step 180 env 5: `s=8.233 mm`, `r=0.401 mm`, `theta=0.05790 rad`, final axial consistency error `37.564 mm` | reject as `tip_depth_false_positive` |

v643 was stopped after checkpoint 200 because the first checkpoint interval reproduced the same false-positive failure:
tip signed depth increased while the module/body remained far behind and semantic orientation stayed well above the
strict threshold. The run folder contains metrics/config/command/checkpoints/snapshots/git status:

- `outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-30_07-27-45_train_v643_noguard_randomized_v622_continue_from_v639_ckpt100`
- Decision record:
  `outputs/agentic_reward_curriculum_20260529/agent_decisions/agent_decision_v641_v643_randomized_lowtheta_and_noguard.json`

Current recommendation:

- Do not continue reward-only no-guard training on v622/v643 without a new source of module-consistent low-theta
  final-window trajectories.
- The remaining blocker is not the single-reset overfit anymore; that has been fixed by validated randomized
  curricula. The blocker is now controller/contact realization or missing teacher data for module-consistent full-depth
  insertion.
- The next bounded change should generate or recover a teacher/residual trajectory that advances the module with the
  tip while keeping semantic theta below `0.030 rad`; then use imitation or conservative residual learning. More axial
  reward by itself is producing tip-depth false positives.

Next bounded decision:

Do not continue long reward-only training from v618/v622 as configured. Do not continue the current v624 guide-loss
setup either. The next useful branch should collect or synthesize explicit module-consistent final-insertion residual
targets; if those cannot be generated, the blocker is likely the absence of repeatable module-consistent final-insertion
examples rather than randomized reset coverage.

## Update - Randomized Teacher Probe v625

I ran one short privileged teacher/data-collection probe on the validated v622 accepted randomized starts before any
longer training:

- command: `outputs/agentic_reward_curriculum_20260529/commands/collect_v625_teacher_randomized_v622_smoke120_env4.txt`
- run: `outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-30_06-18-14_collect_v625_teacher_randomized_v622_smoke120_env4`
- episode source: `outputs/agentic_reward_curriculum_20260529/reset_validation/2026-05-30_05-46-14_validate_v622_tight_lowr_zeroaction_reset20_env36/accepted_episodes`
- envs/steps: 4 envs, 120 steps
- hard insertion action guard: disabled
- executed privileged target-tip guide: enabled only for data-collection diagnosis
- strict checker: unchanged

Result:

| run | best centered post-step row | module/body state | strict_success | decision |
|---|---|---|---|---|
| v625 | env0 step120: tip `s=9.456 mm`, `r=0.234 mm`, theta `0.05444 rad` | module `s=-14.174 mm`, module `r=0.790 mm`; still `37.408 mm` short of full tip depth | false | reject as teacher; module-depth blocked and theta above strict |

Residual-target audit for v625:

| label | count |
|---|---:|
| `module_depth_blocked_centered_high_theta` | 100 |
| `lateral_bypass` | 17 |
| `contact_spike` | 3 |

There were zero `safe_centered_progress` rows. The run saved separate high-quality center/left/right videos, snapshots,
CSV metrics, command/config, git diff/status, and a markdown summary in the run folder.

Interpretation:

The randomized curriculum itself is no longer the immediate blocker. Even an executed privileged target-tip guide on
the validated randomized starts produces shallow tip progress while the module remains behind and semantic-tip theta
stays around `0.054 rad`. Continuing no-guard reward-only training, or the current no-execute guide imitation branch,
would be expected to keep learning entrance hover/partial tip insertion rather than strict full insertion.

Next recommendation:

Stop this bounded randomized-curriculum training branch. The next single code change should debug controller/asset/IK
realization for module-following axial insertion: command pure port-axis wrist motion from the best partial state and
measure whether both `s` and `module_s` advance without contact spike or lateral sweep. If pure axis motion cannot
advance the module from the `module_s=-14..-6 mm` blocked regime, the task should be treated as a controller/contact
realization blocker before more reward or curriculum tuning.

## Update - Direct Axis and Zero-Velocity Reset Probes v626-v630

The pure-axis/controller diagnostic has now been run.

- v626 direct inward axis from a partial robot-state reset briefly reached `s=17.293 mm`, module `s=-6.328 mm`, then
  backed out to final `s=9.018 mm`, module `s=-14.607 mm`.
- v627 mirrored/backout sign confirmed the axis sign was not simply reversed; it mostly backed out and hit a clipped
  `35 N` force row.
- v628 regenerated the same partial robot-state reset family with omitted joint velocities. Validation accepted 4/4
  cleaner reset episodes, and runtime diagnostics reported `full_joint_velocities_zeroed=true`.
- v629 reran direct inward axis on the v628 accepted resets. It still only reached best `s=17.326 mm`, `r=0.443 mm`,
  theta `0.05157`, module `s=-6.295 mm`, then ended at `s=8.963..11.192 mm` and module `s=-14.664..-12.435 mm`.
- v630 tried to find a lower-theta robot-state source. The only candidate failed settle validation: final `s=-0.228 mm`,
  `r=2.088 mm`, theta `0.07615`, module `s=-23.823 mm`.

Decision:

Do not resume long reward-only/randomized training yet. The randomized curriculum and zero-velocity reset fixes improve
experimental hygiene, but direct inward controller motion still cannot sustain module-consistent insertion from the best
partial states. The next useful work is a contact/IK/asset realization diagnostic that can prove a seated or near-seated
module pose is physically holdable before collecting more policy data.

## Update - Randomized Curriculum Re-Audit and v639/v640 No-Guard Smoke

The active plan was updated to stop continuing the single-reset shallow branch. I confirmed no stale Isaac/SERL jobs
were running, then re-audited the reset sources:

| run/config | reset source | envs | reset/eval result | decision |
|---|---|---:|---|---|
| v529/v530 | `full_depth_start2x0_v464_settle_centered_from_v462` | 1 | identical first post-step state: about `s=-2.053 mm`, `r=0.082 mm`, theta `0.0507 rad` | stop; single-reset shallow overfit risk confirmed |
| v620 shallow/final | randomized `s=-3..+2 mm` bucket | 24 | step20 `s=-3.479..+4.731 mm`, `r=0.035..2.531 mm`, theta `0.0517..0.0673` | physically plausible, but reject as strict-orientation curriculum |
| v620 near/bridge/40x10 | randomized near/bridge/heldout buckets | 24/24/6 | `s/r` roughly target ranges; heldout 40/10 remains around `s=-40 mm`, `r≈10 mm`; theta floor remains `>0.050 rad` | useful for held-out eval, not strict final training |
| v622 tight-low-r | accepted low-lateral mixed curriculum | 36 | step20 `s=-20.148..+4.395 mm`, `r=0.076..4.645 mm`, theta `0.0480..0.0833` | smoke-train only; still missing low-theta starts |

Generated audit artifacts:

- `outputs/agentic_reward_curriculum_20260529/reset_audits/randomized_curriculum_v620_shallow_final.md`
- `outputs/agentic_reward_curriculum_20260529/reset_audits/randomized_curriculum_v620_heldout40x10.md`
- `outputs/agentic_reward_curriculum_20260529/reset_audits/randomized_curriculum_v622_tight_lowr.md`

I also checked whether the old full-depth reset sweep can serve as low-theta final-window data in the SERL path:

| run | setup | result | decision |
|---|---|---|---|
| v637 | wrist-contact diagnostic, full-depth reset sweep, saved env8/env35 videos | env8 step2 metric-strict: `s=45.762 mm`, `r=0.191 mm`, theta `0.02083`, consistency `0.884`; falls out by step3 | useful visual/geometry evidence only; not policy success and not stable |
| v638 | SERL zero-action validation of 64 full-depth reset sweep episodes | zero accepted strict episodes; step1 strict count `0`, step5 strict count `0`; best depth-aligned step5 env35 `s=45.809 mm`, `r=0.110 mm`, theta `0.04709` | reject as strict final-window reset source in training runtime |

Then I ran the requested randomized no-guard branch from the latest v622 checkpoint:

| run | setup | best post-step row | visual/video | strict_success | decision |
|---|---|---|---|---|---|
| v639 | no-guard randomized v622 accepted episodes, 8 envs, stopped after checkpoint100 | step111/env3 `s=5.096 mm`, `r=0.405 mm`, theta `0.05716`, module `s=-18.529 mm`, module `r=0.697 mm` | training snapshots only | false | reject for long continuation; theta/module still blocked |
| v640 | policy-only rollout from v639 checkpoint100, 4 envs, separate 30 fps center/left/right videos | step140/env1 `s=6.393 mm`, `r=0.500 mm`, theta `0.05572`, module `s=-17.230 mm`, module `r=0.856 mm` | final images show shallow tip progress, module/cable outside | false | reject; not full insertion |

v640 artifacts:

- command: `outputs/agentic_reward_curriculum_20260529/commands/eval_v640_policyonly_v639_ckpt100_v622_tightlowr_video_env4.txt`
- run: `outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-30_07-09-55_eval_v640_policyonly_v639_ckpt100_v622_tightlowr_video_env4`
- videos: `env000{0..3}_{center,left,right}_full_episode_30fps_quality.mp4`
- snapshots: `step_images/step_000000`, `step_000035`, `step_000070`, `step_000105`, `step_000140`
- metrics: `metrics.jsonl`, `metrics_summary_codex.json`, `cheatcode_phase_summary.json`

Conclusion for this branch:

The randomized curriculum requirement is satisfied for reset coverage, but not for strict-orientation coverage. The
policy can learn shallow tip progress from varied starts, yet it remains in the same failure class: semantic-tip theta
stays around `0.05-0.06 rad`, module depth lags by roughly `23-24 mm`, and video sanity shows the module is not seated.
Do not resume long training from v639 without first adding stable low-theta/module-following final-window data or fixing
the controller/contact realization that prevents such data from being generated.

## Update - Deep Final-Window Teacher/Replay v645-v648

I followed the recommendation to generate explicit module-consistent final-window evidence rather than continue
single-reset or reward-only shallow training.

| run | setup | result | strict_success | decision |
|---|---|---|---:|---|
| v645 | privileged target-tip teacher from v642 low-theta starts, 4 envs, 140 steps, high-quality videos | reached near-full module-consistent rows; best centered row step20/env2: `s=45.845 mm`, `r=0.083 mm`, theta `0.04221 rad`, module `s=22.210 mm`, module `r=0.885 mm`, consistency error `0.015 mm` | 0 | useful diagnostic, not success |
| v646 | robot joint reset from v645, explicit zero velocities | step1 mean `s=40.680 mm`, mean `r=0.401 mm`, mean theta `0.04446 rad`; step10 mean `s=36.011 mm`, mean consistency error `11.146 mm` | 0 | reject; near-full state is not replay-stable with zero velocities |
| v647 | robot joint reset from v645, recorded post-step velocities | step1 best `s=47.872 mm`, `r=0.530 mm`, theta `0.03938 rad`, module `s=24.231 mm`, consistency error `0.009 mm`; step10 mean `s=27.090 mm`, mean consistency error `18.703 mm` | 0 | better immediate replay, but still unstable |
| v648 | final-window orientation/contact teacher from v647 recorded-velocity resets, 8 envs, 100 steps | best metric-like row step2/env1: `s=49.055 mm`, `r=0.299 mm`, theta `0.04007 rad`, module `s=25.416 mm`, module `r=0.818 mm`, consistency error `0.011 mm`; center/left snapshots remain visually ambiguous | 0 | reject; near-full orientation/contact blocked |

Artifacts:

- v645 run: `outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-30_07-49-12_train_v645_teacher_v642_deep_module_probe_env4`
- v647 reset validation: `outputs/agentic_reward_curriculum_20260529/reset_validation/2026-05-30_07-56-17_validate_v647_robotstate_v645_recordedvel_zeroaction_reset10`
- v648 run: `outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-30_07-58-34_train_v648_teacher_v647_recordedvel_final_orientation_probe`

Current diagnosis:

The randomized-curriculum branch is no longer blocked by lack of reset coverage alone. The best evidence now points to a
final-window orientation/contact stability blocker: the system can briefly reach near-full module-consistent geometry,
but theta remains around `0.039-0.045 rad`, the state tends to back out under zero action, and video sanity is not clean.
Do not claim success from these rows. The next change should add explicit failure classification for this near-full
orientation/contact case and then try only a bounded micro-backoff/micro-orientation final-window probe before any more
learning.

Classifier follow-up:

- Updated `extract_privileged_residual_targets.py` to label all envs in multi-env runs and distinguish
  `near_full_orientation_blocked` / `near_full_orientation_contact_blocked`.
- Re-extracted v645+v648 to
  `outputs/agentic_reward_curriculum_20260529/residual_target_audits/v648_multienv_nearfull_reextract`.
- Label counts: `tip_depth_false_positive=886`, `contact_spike=444`, `lateral_bypass=15`,
  `near_full_orientation_blocked=6`, `near_full_orientation_contact_blocked=4`,
  `prejump_realization_mismatch=5`.
- Validation passed: `python -m py_compile aic_utils/aic_isaac/scripts/extract_privileged_residual_targets.py` and
  `python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py` (`31 passed`).

## Update - Micro-Backoff Final-Window Probe v649

I ran one bounded final-window micro-backoff/orientation probe from the v647 recorded-velocity reset family:

- command: `outputs/agentic_reward_curriculum_20260529/commands/train_v649_teacher_v647_microbackoff_orientation_probe.txt`
- run: `outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-30_08-08-20_train_v649_teacher_v647_microbackoff_orientation_probe`
- config: target depth `45.8 mm`, axial step `3 um`, lateral step `30 um`, rotation step `0.00004 rad`, realized-r recovery enabled.

Result:

| run | best row | strict_success | decision |
|---|---|---:|---|
| v649 | step21/env6: `s=49.741 mm`, `r=0.384 mm`, theta `0.03890 rad`, module `s=26.104 mm`, module `r=0.399 mm`, consistency error `0.013 mm`, force `27.99 N` | 0 | reject |

The best theta row was still `0.03740 rad`, above the strict `0.030 rad` threshold, and it occurred with high force.
Extracted center/left/right snapshots remain visually ambiguous. This confirms that simply lowering target depth and
adding tiny recovery does not solve the final orientation/contact block.

## Update - Semantic Visual Overlay Fix v650

While reviewing v648/v649 snapshots, I found that the center-camera overlay printed `o≈3.11 rad` for `sfp_tip_link`
while post-step metrics reported theta around `0.04 rad`. The mismatch was in the overlay text: it computed raw
body-quaternion error and ignored the configured semantic body orientation offset / orientation mode.

Fix:

- `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py` now uses `_body_target_orientation_error_from_config` in
  `_overlay_insertion_debug`.
- The strict success checker and reward definitions were not changed.

Validation:

- `python -m py_compile` passed for `serl/train.py` and `extract_privileged_residual_targets.py`.
- Insertion reward geometry tests passed (`31 passed`).
- v650 2-step smoke saved new images in
  `outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-30_08-14-22_train_v650_overlay_semantic_theta_smoke`.
- The fixed overlay now reports `sfp_tip_link ... o=0.04rad`, matching the metric scale.

## Update - Contact Geometry and Randomized Curriculum v651-v654

### Contact geometry diagnostic

I compared local SDF assets against upstream `intrinsic-dev/aic` and found no local SDF drift for the SFP module,
NIC card, or SFP cable assets. The full-depth target remains the NIC cage/entrance geometry, not the old 8 mm
criterion. However, the converted Isaac USD still contains a large module collision mesh:
`/Robot/cable/sfp_module/sfp_module_link/collisions/body_sdf_collision`, with an approximate bbox of
`13.75 x 47.70 x 21.91 mm`. This is much taller than the nominal port opening and is a plausible final-window
contact blocker.

I added a diagnostic-only `--disable_collision_prim_regex` option to
`aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py` and ran v651:

- run: `outputs/agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_08-20-56_v651_disable_sfp_body_sdf_collision_posehold`
- disabled: 36 matching `body_sdf_collision` prims
- result: tip strict rows increased from 15/216 in v637 to 45/216 in v651, and env10 stayed tip-strict through
  step5 (`s=45.962 mm`, `r=0.278 mm`, theta `0.00461 rad`, consistency gate `0.840`)
- caveat: module/body strict rows were still false and forces were still very high, so this is not a success claim.

Decision: this supports a contact-geometry blocker hypothesis, but official success must still use unchanged strict
geometry plus video sanity. Collision disabling is a diagnostic ablation only until the asset/contact model is justified.

### Randomized curriculum rebuild

The earlier v529/v530 runs used `num_envs=1` and
`generated_episode_configs/full_depth_start2x0_v464_settle_centered_from_v462`, so they were effectively repeated
single-reset shallow training. Existing v613 randomized validation improved coverage but still started with
theta around `0.05-0.07 rad`.

I generated v652 from the later low-theta v642 seed:

- config root: `outputs/agentic_reward_curriculum_20260529/generated_episode_configs/randomized_curriculum_v652_lowtheta_from_v642`
- phases: `shallow_final`, `near_gate`, `bridge`, `heldout_40x10`, `train_mixed_50_30_20`
- base settled signed depth: `43 mm`

Post-step zero-action validation:

| bucket | run | step1 s mm min/mean/max | step1 r mm min/mean/max | step1 theta rad min/mean/max | step20 drift summary | decision |
|---|---|---:|---:|---:|---|---|
| shallow_final | `2026-05-30_08-27-25_validate_v652_shallow_final_zeroaction_reset20_env24_serl_probe` | -2.813 / -0.579 / 1.999 | 0.267 / 1.101 / 2.773 | 0.04097 / 0.04317 / 0.04595 | r mean grows to 1.944 mm, theta mean to 0.04486 | usable for short orientation/lateral learning only |
| near_gate | `2026-05-30_08-30-58_validate_v652_near_gate_zeroaction_reset20_env24_serl_probe` | -9.984 / -5.643 / -1.681 | 0.345 / 2.301 / 5.078 | 0.04286 / 0.04452 / 0.04641 | r mean grows to 3.541 mm | usable as harder near-gate exposure after retention improves |
| bridge | `2026-05-30_08-33-41_validate_v652_bridge_zeroaction_reset20_env24_serl_probe` | -20.039 / -14.979 / -8.739 | 0.415 / 4.589 / 11.416 | 0.04231 / 0.04815 / 0.05444 | r mean grows to 5.781 mm | too broad for early training except low-probability mix |

### Training probes

| run | setup | outcome | failure label | decision |
|---|---|---|---|---|
| v653 | no-guard, v652 mixed 50/30/20, 4 envs, from v614b checkpoint, checkpointed at step100 then stopped at step123 | s moved forward (`mean -2.23 mm` to `+0.37 mm`), but r mean grew `1.06 mm` to `3.57 mm`; module remained far behind (`module_s mean -23.25 mm` at step123) | `lateral_bypass` + `tip_depth_false_positive` | reject; do not continue mixed training |
| v654 | no-guard, v652 shallow-only, stricter gates/lower axial reward, from v614b checkpoint | reward collapsed immediately (`-22.8` step1 to `-55.9` step33) before useful checkpoint | `unstable_learning_or_actor_drift` / reward-scale collapse | reject; stricter scalar reward made the medium-theta starts unlearnable |

Current recommendation:

1. Do not resume v653 or v654.
2. Keep v652 as a documented randomized reset family, but do not use bridge until shallow retention is stable.
3. Next bounded change should be either:
   - build a lower-theta shallow reset family by orientation-sweep calibration and revalidate before training, or
   - address the Isaac converted `body_sdf_collision` / contact geometry issue so final-window module-consistent states are physically stable.
4. Further reward-only training from medium-theta starts is unlikely to produce strict insertion without one of those changes.

## Update - Near-Full Controller/Contact Probes v675-v677

I did not resume single-reset shallow training. Instead I ran bounded controller/contact probes from the v649
teacher/final-window run because v649 briefly entered a near-full tip-depth state but did not satisfy strict success.

Key results:

| run | setup | best strict-like row | final row | strict_success | decision |
|---|---|---|---|---:|---|
| v675 | semantic-axis forward `100 um`, audit start step 105 | step20 `s=45.809 mm`, `r=0.202 mm`, theta `0.04220`, consistency `0.708` | step125 `s=20.680 mm`, `r=0.274 mm`, theta `0.05267`, consistency `0` | false | reject |
| v676 | semantic-axis forward `40 um`, audit start step 20 | same step20 transient | step60 `s=20.682 mm`, `r=0.382 mm`, theta `0.05377`, consistency `0` | false | reject |
| v677 | semantic-axis forward `40 um`, action-chunk start step 17 | same step20 transient | step60 `s=20.682 mm`, `r=0.382 mm`, theta `0.05377`, consistency `0` | false | reject |
| v677 | rotation axes `4 mrad`, action-chunk start step 17 | best theta `0.03876 rad` only after dropping to `s=42.207 mm`, `r=1.252 mm`, consistency `0` | step60 `s=20.682 mm`, `r=0.382 mm`, theta `0.05377`, consistency `0` | false | reject |

Artifacts:

- `outputs/agentic_reward_curriculum_20260529/controller_contact_v675_from_v649`
- `outputs/agentic_reward_curriculum_20260529/controller_contact_v676_nearfull_from_v649`
- `outputs/agentic_reward_curriculum_20260529/controller_contact_v677_chunkstart_from_v649`
- `outputs/agentic_reward_curriculum_20260529/agent_decisions/2026-05-30_v675_v677_controller_contact_nearfull.json`

Conclusion:

The near-full row is not a successful insertion. It has full tip depth and tight lateral error, but theta is still above
the strict `0.030 rad` threshold and module consistency is only `0.708`. It then collapses under contact to about
`20.7 mm` tip depth with zero module consistency. Tiny semantic-axis commands and bounded rotations do not stabilize
the state. This reinforces the current stop decision for reward-only training until the contact/controller path can
produce stable, non-inserted-to-inserted final-window trajectories.

False-positive guardrail:

The v637 diagnostic has step-0 already-seated reset rows that numerically pass strict thresholds, such as
`s=45.801 mm`, `r=0.201 mm`, theta `0.0100 rad`, consistency `0.913`. These are reset states, not policy-driven
insertions from a non-contact start, and should only be used to debug reset/contact realization.

## Update - Collision Ablation v678-v679

I fetched `upstream/main` and compared the local authored SFP/NIC SDF source files used for Gazebo-style geometry.
The checked files match upstream; the local SDF was not accidentally edited. The NIC cage collision depth remains
`48.72 mm`. The likely problem is the converted Isaac USD/contact representation, not the official SDF source.

I then ran diagnostic-only collision toggles on the same v649 near-full replay:

| run | change | outcome | decision |
|---|---|---|---|
| v678 | disable exactly one `body_sdf_collision` prim and run 40 um semantic-axis forward probe | depth no longer collapses; final tip `s=49.942 mm`, but `r=0.589 mm`, theta `0.05551`, module consistency `0` | reject as tip-depth false positive |
| v679 | same collision toggle with 4 mrad rotation-axis probe | no strict theta improvement; same module-consistency failure | reject |

Collision toggle report:

- matched prim: `/World/envs/env_0/Robot/cable/sfp_module/sfp_module_link/collisions/body_sdf_collision`
- matched count: `1`

Conclusion:

The collision ablation strongly suggests the converted body SDF collision participates in the near-full ejection.
However, disabling it is not a solution: it stabilizes an over-depth tip state while semantic theta and module
consistency fail. The next credible step is to replace/override that converted mesh collision with a collision model
consistent with the Gazebo SDF primitive boxes, then revalidate from non-contact randomized starts before any training.

## Update - Runtime SDF Body Box Replacement v680-v681

I implemented the next contact diagnostic as an off-by-default train flag:

- `--replace_sfp_body_sdf_collision_with_sdf_boxes`
- disables converted SFP module `body_sdf_collision` mesh prims,
- creates four runtime USD cube collision prims matching the official SDF `body_collider_box*` entries.

Validation probe results:

| run | setup | best row | final row | decision |
|---|---|---|---|---|
| v680 | SDF box replacement + 40 um forward probe | `s=45.809 mm`, `r=0.202 mm`, theta `0.04220`, module `s=22.174 mm`, module `r=0.705 mm`, consistency `0.708` | `s=43.987 mm`, `r=0.183 mm`, theta `0.04474`, consistency `0.004` | reject, but better diagnostic than disabling collisions |
| v681 | SDF box replacement + 4 mrad rotation probe | same best row; best theta `0.03988 rad` only with lateral `2.232 mm` and consistency `0` | same final row | reject |

This replacement avoids the collision-disabled over-depth false positive and reduces the hard ejection pattern, but it
still does not solve strict insertion. The current best path is now:

1. keep the SDF box replacement off by default,
2. use it for a bounded non-contact final-window reset validation,
3. only resume training if reset/settle plus tiny-action probes preserve full depth, `r <= 0.5 mm`, theta `<0.030 rad`,
   and module consistency together.

## Update - Reward-Manager Orientation Source v689

I kept single-reset shallow training stopped and patched the reward-manager path so
`--episode_target_orientation_source reference_reward_body_start` is no longer only a diagnostic convention. It now
flows into the target orientation, orientation gates, strict success bonus, and target-success/failure terms. The
historical `target_pose` behavior remains the default.

Validation:

- `python -m py_compile` passed for `train.py`, `validate_serl_reset_settle.py`, and `rewards.py`.
- `aic_utils/aic_isaac/test/test_insertion_reward_geometry.py`: `31 passed`.

Bounded smoke:

- command: `outputs/agentic_reward_curriculum_20260529/commands/train_v689_rewardmanager_refbody_targettip_step4.txt`
- run: `outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-30_11-09-00_train_v689_rewardmanager_refbody_targettip_step4`
- decision: reject for training

| step | max tip `s` | `r` at max `s` | module `s` at max `s` | module `r` at max `s` | strict |
|---:|---:|---:|---:|---:|---:|
| 1 | `66.389 mm` | `4.403 mm` | `42.765 mm` | `3.560 mm` | 0 |
| 2 | `66.661 mm` | `5.027 mm` | `43.054 mm` | `3.624 mm` | 0 |
| 3 | `66.992 mm` | `3.701 mm` | `43.355 mm` | `2.955 mm` | 0 |
| 4 | `67.157 mm` | `4.567 mm` | `43.532 mm` | `3.600 mm` | 0 |

Interpretation:

The orientation-source mismatch is fixed for the reward/success path, but this does not make the target-tip guide
usable for training. The smoke still creates a tip-depth false positive: the tip over-inserts while lateral error is
millimeters off and the module trails the tip. This reinforces the current rule: do not resume long randomized or
single-reset training until the reset/controller/contact path can produce module-consistent, low-theta final-window
transitions from non-contact starts.

## Update - Randomized Current-Depth Curriculum v690-v699

I stopped the single-reset shallow path and validated current-depth reset acceptance before training:

| run | bucket | acceptance | post-step reset distribution | decision |
|---|---|---:|---|---|
| v690 | v652 shallow, target-depth consistency | 0/12 | `s` mean `0.029 mm`, `r` mean `0.963 mm`, theta mean `0.0221 rad`, module `s` mean `-23.603 mm` | reject for shallow reset acceptance because final-depth consistency is the wrong acceptance criterion outside final seating |
| v691 | v652 shallow, current-depth consistency | 12/12 | same geometry as v690, current-depth consistency gate `0.829..0.972` | accept shallow subset |
| v693 | v652 near-gate, current-depth consistency | 5/12 | step20 `s` mean `-5.010 mm`, `r` mean `3.508 mm`, theta mean `0.0275 rad` | accept filtered subset only |
| v694 | v652 bridge, current-depth consistency | 3/12 | step20 `s` mean `-13.201 mm`, `r` mean `6.221 mm`, theta mean `0.0264 rad` | accept filtered subset only |
| v698 | v697 interleaved accepted mix | 12/12 | step20 `s` range `-19.043..2.647 mm`, `r` mean `1.497 mm`, theta mean `0.0225 rad` | accept for short training |

I created:

- `outputs/agentic_reward_curriculum_20260529/generated_episode_configs/randomized_curriculum_v697_interleaved_currentdepth_mix_from_v691_v693_v694`
- file mix: 12 shallow, 7 near-gate, 5 bridge (`50% / 29% / 21%`)
- ordering is interleaved so sorted/env-indexed sampling does not silently use shallow-only starts.

Short no-guard training:

- command: `outputs/agentic_reward_curriculum_20260529/commands/train_v699_noguard_interleaved_v697_currentdepth_bodyboxes_smoke320_env4_from_v692ckpt120.txt`
- run: `outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-30_11-32-15_train_v699_noguard_interleaved_v697_currentdepth_bodyboxes_smoke320_env4_from_v692ckpt120`
- checkpoint seed: v692 `checkpoint_000120.pt`
- no hard insertion guard, no target-tip servo, current-depth consistency reward mode, reference-body target orientation, SDF body box diagnostic replacement enabled

Result:

| row | tip `s` mean/max | tip `r` mean/max | theta mean/max | module `s` mean/max | module `r` mean/max | final consistency error mean | strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| v692 final step120 | `3.427 / 6.193 mm` | `0.305 / n/a mm` | `0.0223 / n/a rad` | `-20.204 / n/a mm` | `0.776 / n/a mm` | `~42.4 mm` | 0 |
| v699 best axial step221 | `4.668 / 13.252 mm` | `0.325 / 0.385 mm` | `0.0328 / 0.0506 rad` | `-18.955 / -10.351 mm` | `1.035 / 1.189 mm` | `41.119 mm` | 0 |
| v699 final step320 | `2.834 / 5.003 mm` | `0.282 / 0.455 mm` | `0.0395 / 0.0522 rad` | `-20.790 / -18.618 mm` | `1.163 / 1.290 mm` | `42.954 mm` | 0 |

Interpretation:

The randomized no-guard curriculum is no longer overfitting to one shallow reset and it does create real axial motion
while keeping lateral error much tighter than the prior tip-depth false positives. It is still not close to strict full
insertion. The current blocker is module/body consistency plus theta drift: the tip advances a few to 13 mm in the
best row, but the module remains around `19 mm` outside the entrance on average and strict final consistency is still
about `41-43 mm` short.

Decision:

Do not promote v699 as success. The next bounded step is a policy-only visual/metric rollout from v699
`checkpoint_000160.pt` and `checkpoint_000320.pt` on shallow, near-gate, and held-out starts. If the video agrees with
the metrics, tune the reward/curriculum toward depth-gated module following and theta retention before longer training.
