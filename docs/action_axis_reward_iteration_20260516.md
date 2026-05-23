# Action-Axis Reward Iteration - 2026-05-16

## Context

Starting checkpoint:

```text
outputs/one_day_insertion_pipeline/gpu2_50k_mixed200_timeout_restart_20260516/runs/2026-05-16_01-30-18_gpu2_50k_mixed200_timeout_from_002500/checkpoints/checkpoint_005000.pt
```

Episode mix used for the latest experiments:

```text
outputs/analysis/mixed_20_30_40_6mm_action_gate_20260516/episode_configs_interleaved
```

This interleaves 20mm/6mm, 30mm/8mm, 40mm/10mm, and 6mm/0.7mm starts so 2-env runs see the near-gate cases quickly.

## Code Changes

Updated `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`:

- Added per-env `cheatcode_phase_reward` diagnostics for small env counts.
- Added insertion action guard controls:
  - `--insertion_action_guard_activation_depth_m`
  - `--insertion_action_guard_lateral_direction_sign`
  - `--insertion_action_guard_blocked_axial_step_m`
- Added orientation gating to the action guard so inward axial motion is blocked unless tip orientation is within the guide threshold.
- Changed centered axial guard behavior to keep using an explicit centerline correction rather than preserving raw policy lateral motion.

These are runtime guard/diagnostic controls only; model architecture and ACT remain unchanged.

## Experiments

### 1. Raw action-axis reward

Run root:

```text
outputs/one_day_insertion_pipeline/insertion_action_gate_bodydelta_interleaved800_short_ep_from_005000_20260516
```

Finding:

- Body-delta action-axis source worked better than raw `action_manager.action`.
- Near-gate starts briefly reached good lateral alignment:
  - best `r` around 0.19-0.52mm at `s=-4.3` to `-5.4mm`
  - theta stayed around 0.065 rad
- The policy then moved toward the entrance while lateral error grew to 9-15mm.
- The reward correctly gave no positive axial reward while misaligned, so the failure was in the guide/action path, not the reward formula.

### 2. Orientation/centered guard

Run root:

```text
outputs/one_day_insertion_pipeline/insertion_action_gate_bodydelta_interleaved800_short_ep_guardfix_from_005000_20260516
```

Finding:

- Blocking axial motion by orientation worked mechanically, but the guard overrode far-start lateral motion too aggressively.
- Far-start lateral error grew immediately to 20-30mm in several variants.
- This showed the guard must not blindly own lateral correction throughout the whole approach.

### 3. Near-only guard with direction sign

Run roots:

```text
outputs/one_day_insertion_pipeline/insertion_action_gate_bodydelta_interleaved800_short_ep_nearguard_from_005000_20260516
outputs/one_day_insertion_pipeline/insertion_action_gate_bodydelta_interleaved800_short_ep_nearguard_signfix_from_005000_20260516
```

Finding:

- The empirical lateral correction sign for the guard is `-1.0`.
- With sign `-1.0`, near-gate behavior initially stayed controlled:
  - `r=0.72-0.91mm` at `s=-6mm`
  - no immediate bad near-gate samples
- But over tens of steps, lateral error still grew while approaching the entrance.
- The guard was clamping positive axial action, but lateral/IK motion still leaked axial progress.

### 4. Hold/back-out guard

Run root:

```text
outputs/one_day_insertion_pipeline/insertion_action_gate_bodydelta_interleaved800_holdfix_from_005000_20260516
```

Finding:

- Active outward hold helped prevent entrance crossing while misaligned.
- But with 3s episodes and guard inactive until -10mm, far-start lateral drift became large before the near-gate phase.
- Best behavior was still pre-insertion alignment only; no strict success.

### 5. Far-active sign-fixed guard

Run root:

```text
outputs/one_day_insertion_pipeline/insertion_action_gate_bodydelta_interleaved800_faractive_signfix_from_005000_20260516
```

Best early near-gate samples:

| run | step/env | s mm | r mm | theta rad |
| --- | --- | ---: | ---: | ---: |
| `faractive_hold_g0` | 43/env1 | -5.68 | 0.30 | 0.0637 |
| `faractive_insert_g1` | 45/env1 | -5.36 | 0.46 | 0.0651 |
| `faractive_soft_g3` | 47/env1 | -5.87 | 0.26 | 0.0645 |
| `faractive_align_g2` | 43/env1 | -5.76 | 0.44 | 0.0635 |

Later behavior:

- `faractive_align_g2` improved orientation to theta 0.0546 at step 123, but lateral error had already drifted to 3.34mm.
- Insertion variants still drifted laterally before entry.
- Max `s` was sometimes near or slightly past zero, but with `r` tens of mm, so this is a bypass/failure, not insertion.

Visual check:

```text
outputs/one_day_insertion_pipeline/insertion_action_gate_bodydelta_interleaved800_faractive_signfix_from_005000_20260516/runs/2026-05-16_14-34-49_faractive_insert_g1/step_images/step_000100/env_0001_center_camera.png
outputs/one_day_insertion_pipeline/insertion_action_gate_bodydelta_interleaved800_faractive_signfix_from_005000_20260516/runs/2026-05-16_14-34-49_faractive_insert_g1/step_images/step_000100/env_0001_left_camera.png
outputs/one_day_insertion_pipeline/insertion_action_gate_bodydelta_interleaved800_faractive_signfix_from_005000_20260516/runs/2026-05-16_14-34-49_faractive_insert_g1/step_images/step_000100/env_0001_right_camera.png
```

The three-camera view agrees with the metrics: the plug is visibly off-center and not inserted. The overlay for that frame reports approximately `s=-27.6mm`, `r=18.6mm`.

## Current Conclusion

The action-axis reward and guard are directionally useful but not sufficient yet.

What is working:

- Reward correctly refuses positive axial reward while lateral/orientation/action alignment are bad.
- Body-delta action-axis diagnostics are meaningful.
- Near-gate lateral alignment can briefly reach sub-mm `r`.
- Visual frames agree with reported `s/r/theta`.

What is failing:

- The guide/action path still lets lateral error grow as the plug approaches the entrance.
- Orientation improvement and lateral improvement are not synchronized; theta can improve only after `r` has already drifted.
- Guarded lateral correction is very sensitive to sign and activation depth.
- No strict success candidate was produced.

Best current fallback:

Use the original best checkpoint for submission fallback, not the latest action-axis runs:

```text
outputs/one_day_insertion_pipeline/gpu2_50k_mixed200_timeout_restart_20260516/runs/2026-05-16_01-30-18_gpu2_50k_mixed200_timeout_from_002500/checkpoints/checkpoint_005000.pt
```

Best next technical change:

Make the guide itself phase-aware in world/port coordinates instead of relying on an external action guard:

1. Align lateral and orientation while holding `s` near the start/hover depth.
2. Only emit inward axial guide action once both `r` and theta are inside thresholds.
3. During inward motion, project the final TCP delta onto the insertion axis plus a small centerline correction, instead of blending arbitrary policy lateral action.
4. Keep the action-axis reward as validation and scoring, not as the only mechanism preventing bypass.

## 2026-05-17 Semantic Consistency Progress Reward

Run root:

```text
outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517
```

Starting checkpoint:

```text
outputs/one_day_insertion_pipeline/multgate_deeper_consistency_20260517/runs/2026-05-17_12-43-46_guard_loose1_align_g0/checkpoint_latest.pt
```

Episode config mix:

```text
outputs/analysis/mixed_10_20_30_40_x_4_6_8_10_multgate_20260516/episode_configs_interleaved
```

Code changes:

- Added semantic consistency progress/loss terms to the cheatcode insertion reward.
- The consistency gate now ramps with insertion depth, so early lateral alignment is not blocked by `sfp_module_link`, but deeper insertion must also move the visible module consistently.
- Exposed CLI knobs:
  - `--target_reward_cheatcode_semantic_progress_scale`
  - `--target_reward_cheatcode_semantic_progress_weight`
  - `--target_reward_cheatcode_semantic_loss_weight`
- Fixed a reward-manager validation bug where `body_to_object_cheatcode_phase_reward` did not accept the new semantic progress parameters.
- Verified formula tests after the fix:

```bash
LC_USER_ID=yoonjung zsh -lc 'docker exec isaac-lab-base bash -lc "cd /workspace/isaaclab && ./isaaclab.sh -p -m pytest -q aic/aic_utils/aic_isaac/test/test_insertion_reward_geometry.py"'
```

Result:

```text
28 passed
```

### Best Semantic Progress Runs

| run | best step/env | s mm | r mm | theta rad | consistency gate | strict success |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `semantic_progress3_balanced_g0` | 102/env1 | 7.95 | 0.139 | 0.042 | 0.830 | no |
| `semantic_progress3_seat_g2` | 100/env1 | 7.92 | 0.134 | 0.042 | 0.827 | no |
| `semantic_progress5_recover_g2` | 98/env1 | 7.44 | 0.134 | 0.042 | 0.784 | no |

The best checkpoint preserved so far is:

```text
outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517/best_semantic_progress3_seat_step100_checkpoint.pt
```

Source run:

```text
outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517/runs/2026-05-17_17-42-48_semantic_progress3_seat_g2/checkpoint_latest.pt
```

Visual frames for the best run:

```text
outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517/runs/2026-05-17_17-42-48_semantic_progress3_seat_g2/step_images/step_000100/env_0001_center_camera.png
outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517/runs/2026-05-17_17-42-48_semantic_progress3_seat_g2/step_images/step_000100/env_0001_left_camera.png
outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517/runs/2026-05-17_17-42-48_semantic_progress3_seat_g2/step_images/step_000100/env_0001_right_camera.png
```

Center-camera H.264 video generated from the best recorded sequence:

```text
outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517/videos/semantic_progress3_seat_env1_center_1440w_30fps_h264_android.mp4
```

Note: this video is upscaled/interpolated from the 256x288 Isaac debug frame sequence, so it is Android-playable H.264 but still limited by the noisy source render.

Interpretation:

- Semantic consistency progress is the first reward variant that produced module-consistent insertion-like depth near the target depth.
- Lateral alignment is now excellent during insertion-like motion: `r` is around `0.13-0.14mm`.
- The remaining failure is orientation. The best samples have `theta` around `0.042rad`, while strict success requires `theta < 0.03rad`.
- No strict success candidate has been logged, because success is correctly not tip-depth-only.
- The visual frames are noisy and hard to judge, but they are consistent with the logs: the tip is centered/entering, while full seated insertion is not visually clean enough to claim success.

### Failed Follow-Up: Aggressive Orientation

Runs:

```text
outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517/runs/2026-05-17_18-00-28_semantic_progress4_orient_tight_g0
outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517/runs/2026-05-17_18-00-35_semantic_progress4_slow_seat_g2
```

Finding:

- Increasing rotation step and tightening the orientation switch made behavior worse.
- Lateral error grew before insertion:
  - `semantic_progress4_slow_seat_g2`: env0 reached `r=6.96mm`, env1 `r=2.68mm`.
  - `semantic_progress4_orient_tight_g0`: env0 reached `r=4.75mm`, env1 `r=1.57mm`.
- Conclusion: aggressive rotation steals authority from lateral control. Orientation improvement must stay modest unless the guide also compensates the tip sweep perfectly.

### Active Recovery Run

Run:

```text
outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517/runs/2026-05-17_18-05-41_semantic_progress5_recover_g2
```

Best status:

| env | s mm | r mm | theta rad | consistency gate | success |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 7.99 | 0.137 | 0.057 | 0.678 | no |
| 1 | 7.92 | 0.134 | 0.042 | 0.827 | no |

This recovery run reproduced the `progress3_seat` best but did not clearly exceed it.

Preserved checkpoint:

```text
outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517/best_semantic_progress5_recover_step100_checkpoint.pt
```

### Failed Follow-Up: Final Orientation Tightening

Run:

```text
outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517/runs/2026-05-17_18-18-58_semantic_progress6_final_orient_g0
```

Settings changed from the best semantic-progress runs:

- Kept guide rotation step mild at `0.0015`.
- Tightened scheduled orientation reward tolerance from `0.140/0.200` to `0.075/0.140`.
- Increased orientation-progress weight from `0.8` to `1.4`.
- Reduced axial push slightly to avoid overshooting while orientation was refined.

Finding:

- This worsened lateral control before insertion.
- At step 45:
  - env0: `s=-5.60mm`, `r=1.63mm`, `theta=0.057`
  - env1: `s=-5.28mm`, `r=3.03mm`, `theta=0.041`
- Because `r` grew above the insertion corridor before entry, this run was stopped early.
- Conclusion: tightening orientation reward without a better tip-sweep compensated orientation controller trades away lateral alignment. The current bottleneck is not reward scale alone.

### Current Recommendation

Use the preserved semantic-progress checkpoint as the current best submission/training seed:

```text
outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517/best_semantic_progress3_seat_step100_checkpoint.pt
```

For further training, avoid aggressive orientation settings. The best next attempt should keep the `progress3_seat` guide settings and add only a small final orientation refinement near `s=6-8mm`, while preserving lateral correction and semantic consistency loss.

The final-orientation test above shows that simply tightening orientation gates is not enough. A better next code change would be a phase-specific guide controller that, once `s>5mm` and `r<0.2mm`, rotates about the port axis while explicitly compensating the SFP tip center position. Without that compensation, orientation corrections increase lateral error and lose the port.

## 2026-05-16 Compensated Rotation Guide

A follow-up issue was found in the `cheatcode_transform` guide: rotating the TCP
changes the world position of `sfp_tip_link`, because the tip is offset from the
controlled TCP. The guide was commanding translation to the desired tip pose and
rotation independently, so the rotation-induced tip sweep could increase lateral
error even when the translational command was correct.

Code change:

- `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
- Added a small quaternion-from-rotvec helper.
- In `_cheatcode_transform_guided_policy_action`, estimate the world-space tip
  displacement induced by the guide rotation and subtract it from the
  translational guide target.

Validation command:

```bash
python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py
```

### Compensated Guide Runs

Run root:

```text
outputs/one_day_insertion_pipeline/phaseguide_compensated_interleaved800_from_005000_20260516
```

Best samples:

| run | step/env | s mm | r mm | theta rad | note |
| --- | --- | ---: | ---: | ---: | --- |
| `comp_insert_rot003_g2` | 283/env1 | -5.66 | 0.181 | 0.0687 | best near-gate lateral sample |
| `comp_insert_rot003_g2` | 240/env1 | -2.41 | 2.64 | 0.0646 | closest approach, lateral drift |
| `comp_insert_rot006_g3` | 282/env1 | -5.86 | 0.231 | 0.0688 | low r near gate |
| `comp_insert_rot006_g3` | 400/env1 | -1.17 | 9.32 | 0.0606 | stronger rotation drifted laterally |

Visual check:

```text
outputs/one_day_insertion_pipeline/phaseguide_compensated_interleaved800_from_005000_20260516/runs/2026-05-16_15-33-23_comp_insert_rot003_g2/step_images/step_000060/env_0001_center_camera.png
outputs/one_day_insertion_pipeline/phaseguide_compensated_interleaved800_from_005000_20260516/runs/2026-05-16_15-33-23_comp_insert_rot003_g2/step_images/step_000060/env_0001_left_camera.png
outputs/one_day_insertion_pipeline/phaseguide_compensated_interleaved800_from_005000_20260516/runs/2026-05-16_15-33-23_comp_insert_rot003_g2/step_images/step_000060/env_0001_right_camera.png
```

The visual frames agree with the metrics: around `s=-5.0mm`, `r=1.4mm`, the tip
is plausibly near the port centerline. The stronger rotation run is visibly
off-axis and should not be used.

### Relaxed Orientation / Slow Approach Runs

Run root:

```text
outputs/one_day_insertion_pipeline/orient_compensated_from_005000_20260516
```

Settings:

- 2 envs per GPU.
- `target_action_guide_rotation_step_size=0.003`.
- compensated guide translation enabled.
- `target_reward_cheatcode_sigma_theta_insert=0.075`.
- pure-guide and guide-imitation variants from checkpoint `checkpoint_005000.pt`.

Interim metrics:

| run | rows | best near sample `(s,r,theta)` | closest approach `(s,r,theta)` | strict good sample? |
| --- | ---: | --- | --- | --- |
| `pure_hold_orient_g0` | 500 | `(-5.66mm, 0.18mm, 0.0687)` | `(-2.41mm, 2.64mm, 0.0646)` | no |
| `pure_slow_insert_relaxed_g1` | 500 | `(-5.62mm, 0.19mm, 0.0687)` | `(-2.36mm, 5.18mm, 0.0606)` | no |
| `imitate_slow_insert_relaxed_g2` | 197 | `(-5.97mm, 0.50mm, 0.0689)` | `(-3.46mm, 4.82mm, 0.0541)` | no |
| `imitate_orient_first_g3` | 179 | `(-6.00mm, 0.51mm, 0.0689)` | `(-3.28mm, 4.54mm, 0.0544)` | no |

Visual check:

```text
outputs/one_day_insertion_pipeline/orient_compensated_from_005000_20260516/runs/2026-05-16_16-15-47_pure_slow_insert_relaxed_g1/step_images/step_000040/env_0001_center_camera.png
outputs/one_day_insertion_pipeline/orient_compensated_from_005000_20260516/runs/2026-05-16_16-15-47_pure_slow_insert_relaxed_g1/step_images/step_000040/env_0001_left_camera.png
outputs/one_day_insertion_pipeline/orient_compensated_from_005000_20260516/runs/2026-05-16_16-15-47_pure_slow_insert_relaxed_g1/step_images/step_000040/env_0001_right_camera.png
```

At step 40 the center image reports approximately `s=-6.0mm`, `r=0.7mm`,
`theta=0.09rad`. Visually the tip is close to the entrance and roughly centered.
This is decent pre-insertion progress, but not enough for strict insertion
because orientation remains above the target and later approach loses lateral
alignment.

### Orientation Sign / Hover-at-6mm Diagnostic

Run root:

```text
outputs/one_day_insertion_pipeline/orient_sign_hover6_from_005000_20260516
```

Settings:

- Hold/hover target moved outward to `s=-6mm`.
- Rotation step `0.004`.
- Compared positive and negative rotation signs.

Results:

| run | rows | best near sample `(s,r,theta)` | closest approach `(s,r,theta)` | conclusion |
| --- | ---: | --- | --- | --- |
| `hover6_rotpos_g0` | 500 | `(-5.83mm, 0.27mm, 0.0694)` | `(-2.30mm, 7.48mm, 0.0608)` | positive sign is usable but still drifts |
| `hover6_rotneg_g1` | 320 | `(-5.94mm, 0.57mm, 0.0707)` | `(-0.85mm, 33.63mm, 0.1624)` | negative sign is wrong; stopped |

### Updated Conclusion

The compensated guide is a real improvement: it removes the worst rotation-
induced lateral blow-up and can repeatedly reach sub-mm lateral error near
`s=-6mm`. However, the policy/guide still has not produced a state where
lateral and orientation requirements are both satisfied:

- best lateral samples have `r < 0.5mm` but `theta` around `0.069rad`;
- best orientation samples have `theta around 0.054rad` but `r` around
  `3.7-4.8mm`;
- no run produced positive `s`;
- no strict success candidate occurred.

Do not promote this to strong insertion reward yet. The safest current fallback
is still the checkpoint:

```text
outputs/one_day_insertion_pipeline/gpu2_50k_mixed200_timeout_restart_20260516/runs/2026-05-16_01-30-18_gpu2_50k_mixed200_timeout_from_002500/checkpoints/checkpoint_005000.pt
```

Recommended next change if time remains:

1. Keep the compensated positive-sign guide.
2. Use a smaller rotation step, likely `0.001-0.002`, while holding around
   `s=-6mm`.
3. Do not allow approach toward `s=-4mm` until both `r < 1.0mm` and
   `theta < 0.060rad`.
4. Once both gates are simultaneously true, use a very small axial step and keep
   lateral correction active during insertion.

## 2026-05-17 Multiplicative Reward and Aggressive Near-Entrance Runs

### Episode Distribution

Generated and used the mixed 800-episode curriculum:

```text
outputs/analysis/mixed_10_20_30_40_x_4_6_8_10_multgate_20260516/episode_configs_interleaved
```

Distribution:

- axial starts: `10, 20, 30, 40mm`
- lateral starts: `4, 6, 8, 10mm`
- 50 episodes for each axial/lateral pair
- total: 800 episodes
- 2 envs per GPU

### Code/Reward Changes

Changed `cheatcode_insertion_phase_reward` to make positive insertion reward
conjunctive instead of additive. High insertion reward now requires all of:

- low lateral error `r`
- low tip orientation error `theta`
- action aligned with the port axis

The positive insertion gate is effectively:

```text
G_insert_combined = G_lateral * G_orientation * G_action_axis
```

Positive axial/corridor credit is multiplied by this combined gate. Lateral,
orientation, near-misalignment, retreat, and inside-alignment penalties remain
available as shaping/guardrails.

Validation:

```text
./isaaclab.sh -p -m pytest -q aic/aic_utils/aic_isaac/test/test_insertion_reward_geometry.py
24 passed
```

### Guide and Guard Fixes

Two sign/frame issues were found during randomized lateral-direction episodes:

1. Fixed-sign `cheatcode_transform` guide could reduce `r` for some episodes
   and increase it for others.
2. The insertion action guard had the same issue after reset, sometimes pushing
   an already near-centered state from about `r=1.5mm` back out toward `5mm`.

Fixes:

- added `--target_action_guide_adaptive_lateral_sign`
- changed insertion action guard adaptive sign to choose the one-step lateral
  direction that predicts lower `r`

### Intermediate Runs and Findings

Run root:

```text
outputs/one_day_insertion_pipeline/multgate_adaptive_sign_20260517
```

Finding: adaptive guide sign improved the first episode, but `r` still rebounded
as `s` moved inward. Example: env0 improved from `r=4.0mm` to `1.86mm`, then
rebounded to about `4.47mm`.

Run root:

```text
outputs/one_day_insertion_pipeline/multgate_adaptive_guard_loose1_20260517
```

Finding: action guard helped, with best `r` around `0.71-0.76mm`, but after
reset the guard could still push `r` upward. This motivated the guard adaptive
sign fix.

Run root:

```text
outputs/one_day_insertion_pipeline/multgate_adaptive_guard_nohover_20260517
```

Important settings:

- hover disabled:
  - `--target-action-guide-preinsert-hover-depth nan`
  - `--insertion-cheatcode-hover-weight 0.0`
- action guard enabled:
  - `--insertion-action-guard`
  - `--insertion-action-guard-adaptive-lateral-sign`
  - `--insertion-action-guard-lateral-threshold-m 0.0015`
  - `--insertion-action-guard-lateral-step-m 0.00020`
- weak insertion:
  - axial progress `0.08`
  - corridor `0.12`

Result: stable alignment/approach but no insertion. Best run reached about
`s=-6.0mm` with `r` around `1.4mm`; env0 repeatedly reached sub-mm lateral
error.

### Aggressive Gated Insertion

Run root:

```text
outputs/one_day_insertion_pipeline/multgate_aggressive_near_entrance_20260517
```

Important settings:

- resumed from:

```text
outputs/one_day_insertion_pipeline/multgate_adaptive_guard_nohover_20260517/runs/2026-05-17_01-12-34_guard_loose2_insert_g3/checkpoint_latest.pt
```

- aggressive but still gated insertion reward:
  - axial progress `0.80`
  - corridor `2.00`
  - inside alignment `1.00`
  - retreat `0.40`
- action guard:
  - lateral threshold `0.0015m`
  - lateral step `0.00020m`
  - centered axial step `0.00004m`
- guide:
  - adaptive lateral sign
  - no hover
  - axial step `0.000012m`
- episode length initially `2.0s`

Finding: strong improvement inside each episode, but `2.0s` episodes cut off
near `s=-5mm`. The following reset then caused a backward/outward transition.
This was not a reward failure; it was a timeout/curriculum issue.

### Long-Episode Aggressive Run

Run root:

```text
outputs/one_day_insertion_pipeline/multgate_aggressive_long_episode_20260517
```

Same aggressive gated insertion settings as above, but:

- `--episode-length-s 4.0`
- steps `2400`
- updates `2000`

Current metrics snapshot:

| run | rows/last step | max s mm | r at max s mm | theta at max s | positive-s samples | success candidates | best r mm | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `2026-05-17_09-27-26_guard_loose2_insert_g3` | 31/31 | -6.86 | 1.422 | 0.0505 | 0 | 0 | 0.052 | early/stalled |
| `2026-05-17_09-27-27_guard_loose2_align_g1` | 31/31 | -6.46 | 0.160 | 0.0569 | 0 | 0 | 0.053 | early/stalled |
| `2026-05-17_09-28-26_guard_loose1_align_g0` | 296/296 | 4.65 | 0.133 | 0.0414 | 40 | 0 | 0.053 | best so far |
| `2026-05-17_09-28-34_guard_loose1_insert_g2` | 280/280 | 3.27 | 0.134 | 0.0414 | 19 | 0 | 0.052 | second best |

Best current checkpoint:

```text
outputs/one_day_insertion_pipeline/multgate_aggressive_long_episode_20260517/runs/2026-05-17_09-28-26_guard_loose1_align_g0/checkpoint_latest.pt
```

Best visual frames inspected:

```text
outputs/one_day_insertion_pipeline/multgate_aggressive_long_episode_20260517/runs/2026-05-17_09-28-26_guard_loose1_align_g0/step_images/step_000060/env_0000_center_camera.png
outputs/one_day_insertion_pipeline/multgate_aggressive_long_episode_20260517/runs/2026-05-17_09-28-26_guard_loose1_align_g0/step_images/step_000060/env_0000_left_camera.png
outputs/one_day_insertion_pipeline/multgate_aggressive_long_episode_20260517/runs/2026-05-17_09-28-26_guard_loose1_align_g0/step_images/step_000060/env_0000_right_camera.png
```

The saved debug overlay at step 60 reports approximately:

```text
s=-0.9mm, r=0.2mm, theta=0.00 display-rounded / axis theta around 0.057 in metrics
sfp_module_link remains outside: about -24.5mm axial, 1.2mm lateral
```

### Current Conclusion

This is the first sequence that produced real signed-depth insertion progress:

- `s` crossed positive, up to `+4.65mm`
- `r` stayed tight at about `0.13mm`
- `theta` stayed tight at about `0.041rad`
- success candidate stayed false, correctly, because full seated depth and
  module-consistency checks are not yet satisfied

The remaining gap is full seated insertion and consistency-body progress. The
tip can enter while `sfp_module_link` still reports far outside, so do not claim
final success from tip `s` alone. The next training direction should keep this
long-episode aggressive gated setup and increase seated-depth/consistency
pressure only after shallow insertion is stable.

## 2026-05-17 Late Iteration: Post-Step Diagnostics and Final Orientation Hold

Run root:

```text
outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517
```

### Diagnostic Fix

Saved camera images are captured after `env.step`, but the metrics row only
contained `pre_step_insertion_geometry`. This caused a misleading comparison:
the row at `step=120` could report a well-centered pre-action state while the
saved `step_000120` image overlay showed the post-action/reset state.

Code change:

- `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
  - added `post_step_insertion_geometry`
  - added `post_step_all_body_insertion_geometry`
  - changed `cheatcode_phase_summary.json` generation to prefer post-step
    geometry when available

Validation:

```text
python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py
./isaaclab.sh -p -m pytest -q aic/aic_utils/aic_isaac/test/test_insertion_reward_geometry.py
28 passed
```

### Final-Orientation Guide/Guard Changes

Added final-orientation phase knobs to the guide:

- `--target_action_guide_final_orientation_depth_m`
- `--target_action_guide_final_orientation_lateral_m`
- `--target_action_guide_final_orientation_threshold_rad`
- `--target_action_guide_final_orientation_axial_step_size`
- `--target_action_guide_final_orientation_rotation_step_size`

Also extended insertion action guard to recognize the same final-orientation
hold condition and log:

```text
insertion_action_guard_final_orientation_hold_fraction
```

The launcher now also accepts:

- episode length as arg 28
- TCP rotation clip as arg 29

### Iteration Results

| run | key settings | best post-step state | result |
| --- | --- | --- | --- |
| `semantic_progress10_poststep_finalrot_g0` | final hold at `s>=4.5mm`, lateral `<0.6mm`, final rot `0.006`, 10s episode | env1 step 145/146: `s≈8.0mm`, `r≈0.276mm`, `theta≈0.0367rad`, consistency `≈0.91` | best current candidate; not strict success because `theta>0.03` |
| `semantic_progress11_rotclip8_g0` | same but `tcp_rotation_action_clip=0.008`, final axial `-0.00005` | env1: `theta≈0.0359rad`, but only `s≈5.2mm`, consistency `≈0.26` | orientation improved slightly but axial hold too strong; not better |
| `semantic_progress12_rotclip8_relaxaxial_g0` | `tcp_rotation_action_clip=0.008`, final axial `0.0` | env1: `theta≈0.0356rad`, but `s≈4.3-4.8mm`, `r≈0.46mm`, consistency `≤0.14` | not better; higher rotation authority widened lateral/limited depth |

Best candidate checkpoint preserved:

```text
outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517/best_semantic_progress10_poststep_finalrot_latest_checkpoint.pt
```

Earlier strong checkpoint still relevant:

```text
outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517/best_semantic_progress7_final_hold_checkpoint.pt
```

Best inspected frames for the current best candidate:

```text
outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517/runs/2026-05-17_23-09-40_semantic_progress10_poststep_finalrot_g0/step_images/step_000140/env_0001_center_camera.png
outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517/runs/2026-05-17_23-09-40_semantic_progress10_poststep_finalrot_g0/step_images/step_000140/env_0001_left_camera.png
outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517/runs/2026-05-17_23-09-40_semantic_progress10_poststep_finalrot_g0/step_images/step_000140/env_0001_right_camera.png
```

Visual inspection of those three cameras: the center overlay reports
approximately `s=+7.7mm`, `r=0.3mm`, centered-depth `c=0.93`, but the side views
still show a visibly awkward/angled plug pose. This should be treated as
near-seated semantic progress, not a reliable final insertion.

### Updated Conclusion

The best reliable trend is still:

- lateral alignment becomes tight before entrance
- tip signed depth can reach the seated-depth neighborhood
- consistency gate can become high (`≈0.9`)
- orientation remains the blocker (`theta≈0.036-0.037rad`, strict success needs
  `<0.03rad`)

Increasing rotation authority alone slightly reduced `theta`, but it either
stalled axial progress or widened lateral error. The best submission fallback is
still the guided hybrid around `semantic_progress10_poststep_finalrot_g0`, not
the more aggressive rotation-clip variants.

## 2026-05-18 Orientation Guide Validation

Goal: validate whether the remaining insertion failure is due to orientation
using no-learning guide rollouts before spending more RL time.

Baseline validation used 2 envs/GPU, mixed 10/20/30/40 mm axial and 4/6/8/10 mm
lateral episode configs, `cheatcode_transform`, post-step diagnostics, and no
actor/critic updates.

Key findings:

- No insertion action guard is not viable. `validate2_cheatcode_transform_guided_no_guard_g0`
  let lateral error grow to `r=13-26mm`.
- The guarded transform keeps lateral alignment tight and reaches positive
  depth, but full-quaternion orientation stalls around `theta=0.036-0.055rad`.
- A bug was found in the final-orientation hold: the guard held TCP translation
  while allowing rotation, but because the tip is offset from TCP, rotation moved
  the tip axially/laterally anyway. The guard now compensates induced tip motion
  and logs `insertion_action_guard_final_orientation_induced_tip_delta_m_mean`.
- With compensation, strict holds prevent overshoot but can stall too early if
  orientation does not keep improving.
- Added `--target_action_guide_axis_only_orientation` as an ablation that rotates
  the configured tip insertion axis onto the port axis.

Representative no-learning runs:

| run | key settings | best post-step state | interpretation |
| --- | --- | --- | --- |
| `validate3_final_early_d25_rot6_g2` | full-quat final hold from `s>=2.5mm`, final rot `0.006`, no compensation | env1 `s=8.06mm`, `r=0.276mm`, `theta=0.0366`, consistency `0.91` | repeats prior near-seated semantic progress but misses strict orientation |
| `validate4_comp_hold_d25_rot6_g2` | same with induced-tip compensation | env1 only `s=3.25mm`, `r=0.275mm`, `theta=0.0363`, consistency `0.023` | compensation prevents drift/overshoot but stalls because theta remains above threshold |
| `validate5b_axisonly_comp_d25_rot8_g3` | axis-only orientation, compensated hold, release threshold `0.032` | env1 `s=8.08mm`, `r=0.107mm`, `theta=0.0317`, consistency `0.891`; 12 near-seated samples with `theta<=0.0325` | best orientation improvement so far, but still not strict `<0.030` and not robust across both envs |
| `validate6_axisonly_strict030_rot10_g3` | axis-only, threshold `0.030`, final rot `0.010` | env1 `s=2.68mm`, `r=0.267mm`, `theta=0.0309` | stricter hold blocks insertion before seated depth |
| `validate7_axisonly_strict030_crawl5um_rot10_g4` | axis-only, threshold `0.030`, final axial crawl `5um/step` | env1 `s=2.69mm`, `r=0.266mm`, `theta=0.0309` | crawl did not overcome the orientation plateau |

Best visual frame from the axis-only ablation:

```text
outputs/one_day_insertion_pipeline/cheatcode_policy_validation_20260518/runs/2026-05-18_10-12-50_validate5b_axisonly_comp_d25_rot8_g3/step_images/step_000160/env_0001_center_camera.png
outputs/one_day_insertion_pipeline/cheatcode_policy_validation_20260518/runs/2026-05-18_10-12-50_validate5b_axisonly_comp_d25_rot8_g3/step_images/step_000160/env_0001_left_camera.png
outputs/one_day_insertion_pipeline/cheatcode_policy_validation_20260518/runs/2026-05-18_10-12-50_validate5b_axisonly_comp_d25_rot8_g3/step_images/step_000160/env_0001_right_camera.png
```

Visual inspection: center camera reports deep, centered tip geometry
(`s≈8.5mm`, `r≈0.1mm`, tip centered-depth `c=1.00`), while side cameras still
show an awkward angled plug pose. Treat this as strong evidence that insertion
axis alignment improved, but not as proof of robust full insertion.

Current conclusion:

- Force is not the limiting factor in the best near-seated samples.
- Lack of axial reward is not the main issue; the guide can reach seated tip
  depth.
- The remaining blocker is orientation/pose quality near the entrance. Axis-only
  guidance improves measured insertion-axis error from roughly `0.036-0.037rad`
  to `0.0315-0.0317rad`, but strict `0.030rad` success is still not reliably
  reached.
- Next best experiment is a hybrid guide: use full-quaternion orientation during
  approach for robustness, then switch to axis-only final refinement only after
  lateral error is tight and `s` is positive. The all-axis-only guide hurt one
  env's approach stability, so it should not replace the full transform guide
  globally.

Follow-up hybrid implementation:

- Added `--target_action_guide_final_axis_only_orientation`.
- This keeps full-quaternion `cheatcode_transform` during approach and switches
  only the final-orientation window to insertion-axis alignment.
- Validation:
  - `validate8_hybrid_finalaxis_thr032_g2`
  - `validate8_hybrid_finalaxis_thr030_g3`
- Result: both hybrid runs preserved lateral alignment but stalled around
  `s≈3.4mm`, `r≈0.44mm`, `theta≈0.0359rad` on env1 and `theta≈0.055rad` on
  env0. This was worse than the all-axis-only seated-depth sample.

Revised recommendation:

- Keep the induced-tip compensation patch.
- Do not switch the main guide globally to all-axis-only; it improves one env's
  `theta` but hurts another env's approach.
- Do not use the current hybrid final-axis-only implementation as the default;
  it stalls too early.
- The most useful next change, if more time is available, is to debug why the
  rotation guide has a two-mode plateau (`theta≈0.055rad` in env0,
  `theta≈0.031-0.036rad` in env1). This likely points to target-frame or
  controller-frame calibration rather than reward scale alone.

## 2026-05-18 Continued Orientation Iterations

Additional no-learning guide/station experiments:

| run | change | outcome |
| --- | --- | --- |
| `validate9_fullquat_late_d6_rot12_crawl10_g1` | full-quaternion final orientation station only after `s>=6mm`, final rot `0.012`, slow crawl | stable but did not reduce theta: env0 `s=6.17mm`, `r=0.23mm`, `theta=0.0548`, consistency `0.425`; env1 `s=6.37mm`, `r=0.26mm`, `theta=0.0358`, consistency `0.568` |
| `validate9_axisonly_release033_d25_rot8_g2` | global axis-only guide, release threshold `0.033` | best env1 repeated: `s=8.02mm`, `r=0.116mm`, `theta=0.0321`, consistency `0.887`; env0 still failed approach |
| `validate9_axisonly_strict030_rot15_crawl20_g3` | global axis-only guide, strict `0.030`, rot `0.015`, crawl `20um` | env1 improved to `theta=0.0305` but stalled at `s=2.70mm`; env0 still failed approach |
| `validate9_fullquat_nohold_rot4_clip10_g4` | stronger full-quaternion rotation with no final hold | lateral blew up (`r>20mm`), not viable |

Frame/sign diagnostics:

| run | change | outcome |
| --- | --- | --- |
| `validate10b_axisonly_nofix_iksign_g2` | no Isaac IK xy sign fix | both envs stalled outside; env0 could reach `r=0.036mm` outside but did not approach |
| `validate10b_axisonly_bycard_iksign_g1` | target-card-dependent IK sign fix | similar outside stall |
| `validate10b_axisonly_noadaptive_latsign_pos_g3` | fixed lateral sign `+1` | env1 repeated best behavior (`s=8.02mm`, `r=0.116mm`, `theta=0.0321`, consistency `0.887`); env0 failed |
| `validate10b_axisonly_noadaptive_latsign_neg_g4` | fixed lateral sign `-1` | lateral diverged; not viable |

Slower approach / lateral-retention experiments:

| run | change | outcome |
| --- | --- | --- |
| `validate11_slowaxial_5um_lateralhold_g1` | `5um` axial, tight lateral hold | preserved env0 lateral better but stalled outside: env0 `s=-0.66mm`, `r=1.17mm`, `theta=0.0553` |
| `validate11_strict_gate_5um_g3` | tighter orientation gate and `5um` axial | best env0 near entrance: `s=0.24mm`, `r=0.62mm`, `theta=0.0553`; no insertion |
| `validate12_long_slow10_lateralhold_g1` | longer wall time, `10um` axial | env0 `s=-0.45mm`, `r≈1.1mm`; no seated progress |
| `validate12_long_strictgate5_lateralhold_g2` | longer wall time, strict gate | env0 crossed entrance slightly (`s=0.24mm`) with `r≈0.62mm`, but theta remained `0.055` |

Short learning attempts from
`best_semantic_progress10_poststep_finalrot_latest_checkpoint.pt`:

| run | change | early result |
| --- | --- | --- |
| `orientlearn2_axis_teacher_rewardhi_g3` | axis-only guide, high orientation reward, small Q (`actor_q_weight=0.02`) | worse lateral alignment (`r≈2-3mm` outside), not promising |
| `orientlearn2_axis_teacher_imitation_g4` | axis-only guide, imitation-heavy, Q disabled | better than Q variant: env0 kept `r<0.5mm` outside but theta stayed `≈0.054`; env1 improved to `theta=0.0339`, `r=0.58mm`, `s=-1.65mm` early |

Updated diagnosis:

- The guide can make the tip laterally accurate outside the gate, but env0 loses
  lateral alignment during approach unless axial motion is slowed substantially.
- Slowing axial motion preserves lateral alignment better but does not solve the
  env0 orientation plateau.
- The best env1 axis-only result is still the closest to strict success, but it
  remains just above threshold (`theta≈0.0315-0.0321rad`).
- Reward-only changes did not overcome the plateau in short runs; the likely
  blocker is controller/target-frame authority for rotating the physical tip,
  especially for mirrored episode/card assignments.

## 2026-05-19 Orientation Controller Debug

Question investigated: whether the remaining orientation plateau is caused by
the reward/guide requesting the wrong rotation, or by the Isaac controller not
realizing that rotation at the physical SFP tip.

Code changes:

- Added per-step `axis_alignment_realization` diagnostics in
  `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`.
  These log requested guide rotation, predicted axis-error reduction if that
  rotation were applied to `sfp_tip_link`, realized before/after tip-axis error,
  and the dot product between requested rotation and the true corrective axis.
- Added `--isaac-ik-body-name` in
  `aic_utils/aic_isaac/scripts/train_isaac_online_serl.py`, backed by
  `AIC_ISAAC_IK_BODY_NAME` in `aic_task_env_cfg.py`, so the IK action body can
  be switched for diagnostics without editing the environment config.
- Added `--target_action_guide_separate_rotation_compensation` and
  `--target_action_guide_rotation_compensation_clip_m` so guide translation
  can reserve motion for compensating the tip displacement induced by wrist
  rotation. This avoids throwing away the compensation when the normal guide
  translation step is intentionally small.

Main diagnostic runs:

| run | IK body / compensation | result |
| --- | --- | --- |
| `2026-05-19_20-12-30_axis_realization_probe_g1` | normal guide, wrist IK | requested rotation direction was correct (`dot=1.0`); env action received the requested rotation magnitude; realized axis-error reduction was much smaller than predicted and approached zero near the plateau |
| `2026-05-19_20-21-18_rotation_only_axis_probe_rot10_g1_2026-05-19_20-20-24` | wrist IK, pure rotation, no compensation | env0 `theta 0.0647 -> 0.0548`, env1 `0.0647 -> 0.0309`; lateral error exploded to `59.6mm` / `49.2mm` because wrist-frame rotation moves the tip laterally |
| `2026-05-19_20-29-33_rotation_only_tipik_axis_probe_rot10_g1_2026-05-19_20-28-37` | `sfp_tip_link` IK body | not viable: lateral error exploded worse (`184mm` / `180mm`) and theta was not better; controlling the semantic tip body directly with the current 6-DOF IK setup is unstable |
| `2026-05-19_20-41-11_rotation_comp2_axis_probe_rot10_comp1mm_g1_2026-05-19_20-40-17` | wrist IK, pure rotation, separate 1mm compensation | partial improvement: lateral drift reduced to `26.0mm` / `15.0mm`; env1 reached `theta=0.0299`, but env0 still plateaued near `0.054rad` |

Interpretation:

- The guide is not rotating in the wrong sign or target frame. The requested
  rotation is aligned with the corrective axis in the diagnostic (`dot=1.0`).
- The action magnitude reaches Isaac, and `wrist_3_link`, `gripper_tcp`, and
  `sfp_tip_link` all rotate. The problem is that the realized rotation is not
  an effective tip-axis correction after the first few steps.
- Directly controlling `sfp_tip_link` via the Isaac IK action is worse, so the
  right short-term path is still wrist IK plus explicit tip-motion compensation,
  not switching the action body for training.
- Separate rotation compensation is a real improvement for lateral drift, but
  it is only a partial fix. It does not remove the env/card-dependent orientation
  plateau, especially the env0 `theta≈0.054rad` mode.

Validation:

- `python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
- `python -m py_compile aic_utils/aic_isaac/scripts/train_isaac_online_serl.py aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py`
- Container formula tests:
  `./isaaclab.sh -p -m pytest -q aic/aic_utils/aic_isaac/test/test_insertion_reward_geometry.py`
  passed: `28 passed`.

Recommendation after this debug:

- Do not use `--isaac-ik-body-name sfp_tip_link` for training unless a separate
  IK/action-body calibration is done; it is unstable in the current setup.
- For guide-based insertion runs, keep wrist IK and enable
  `--target_action_guide_separate_rotation_compensation` with a compensation
  clip around `0.001m` so orientation corrections do not push the tip as far
  off the port axis.
- Treat the remaining `theta≈0.054rad` plateau as a controller/asset realization
  issue, not a reward-sign issue. Reward tuning alone is unlikely to solve it.

## 2026-05-19 Calibration Sweep and Relaxed Insertion Probe

Ran a short controller/action calibration sweep, using no learning and
2 envs/GPU settings. Attempted parallel 4-GPU launch first, but multiple Isaac
Kit processes on this host exited before environment creation with key-value DB
locking warnings. Re-ran the sweep serially to get valid metrics.

Calibration settings:

| run | rotation step | compensation clip | lateral step | result |
| --- | ---: | ---: | ---: | --- |
| `serial_rot04_comp0500_lat0200` | `0.004` | `0.0005m` | `0.0002m` | best of sweep; env0 plateaued `theta≈0.054`, env1 reached `theta≈0.0295`; final `r≈32.2mm/11.1mm` after 120 steps |
| `serial_rot06_comp1000_lat0200` | `0.006` | `0.0010m` | `0.0002m` | more lateral drift; final `r≈41.0mm/15.5mm` |
| `serial_rot08_comp1000_lat0300` | `0.008` | `0.0010m` | `0.0003m` | worse lateral drift; final `r≈56.2mm/27.4mm` |
| `serial_rot08_comp1500_lat0300` | `0.008` | `0.0015m` | `0.0003m` | same as prior; no improvement from larger compensation |

Conclusion from calibration:

- The gentlest setting is best: `rot=0.004`, `compensation=0.0005m`,
  `lateral=0.0002m`.
- Increasing rotation step or compensation does not break the env0
  `theta≈0.054rad` plateau and increases lateral drift.
- The env/card-dependent plateau remains a controller/physical realization
  blocker.

Relaxed insertion probes from
`best_semantic_progress10_poststep_finalrot_latest_checkpoint.pt`:

| run | settings | outcome |
| --- | --- | --- |
| `guided_insert_relaxed_rot04_comp0500` | `theta` gate relaxed to `0.055`, axial step `50um`, lateral release `1.5mm` | false axial progress: env0 reached `s=9.62mm` but lateral error grew to `53.3mm`; no success or consistency gate |
| `guided_insert_tightlat_slowaxial_rot04_comp0500` | lateral release tightened to `0.5mm`, axial step `20um`, no retention guard | still false axial progress: env0 reached `s=5.40mm` with `r=73.3mm`; env1 reached `s=30.6mm` with `r=34.9mm`; no success |
| `guided_insert_projectcomp_tightlat` | same as tight run plus projected compensation to remove positive axial component while off-center | unchanged relative to tight run through step 220; forward drift is not caused by positive axial compensation |

Additional code change:

- When separate rotation compensation is enabled, compensation is clipped
  separately from intended translation.
- Added a guard that removes positive insertion-axis component from the
  compensation vector while the tip is not laterally/orientationally aligned.
  This compiled, but the validation trajectory was unchanged, so the remaining
  false insertion is not caused by compensation's axial component.

Current blocker:

- The guide can briefly reduce `r` below `1mm`, but as wrist IK continues to
  rotate/approach, the physical SFP tip moves off the port axis. Positive
  `s` after this point is not insertion; it is bypass/side motion with
  consistency gate near zero.
- Relaxing `theta` alone is unsafe. It allows forward motion, but the module is
  not aligned and no strict success/consistency signal is achieved.

Recommended next engineering direction:

- Stop trying to solve this with reward or scalar guide tuning.
- Add an explicit closed-loop tip-frame servo/action guard that predicts the
  next tip pose from the requested wrist delta and rejects any action whose
  predicted next lateral error increases or whose positive axial component
  occurs while `r > 0.5-1.0mm`.
- Longer term, calibrate or replace the Isaac task-space controller so the
  controlled frame and semantic insertion frame are the same physical objective,
  without relying on wrist-frame rotations to indirectly move the tip.

## 2026-05-19 Closed-loop Rotation Guard Probe

Implemented and tested an explicit near-entrance rotation guard:

- `--insertion_action_guard_zero_rotation_when_offcenter`
- `--insertion_action_guard_rotation_lateral_threshold_m`
- wrapper forwarding through `train_isaac_online_serl.py`

The guard zeros rotational TCP commands while the semantic `sfp_tip_link` is
laterally off the port centerline. This directly targets the observed failure
mode where wrist rotation improved orientation metrics but physically swept the
SFP tip sideways before insertion.

Validation:

- `python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
- `python -m py_compile aic_utils/aic_isaac/scripts/train_isaac_online_serl.py`
- Container formula tests:
  `./isaaclab.sh -p -m pytest -q aic/aic_utils/aic_isaac/test/test_insertion_reward_geometry.py`
  passed: `28 passed`.

No-learning guard probes from the same checkpoint:

| run | rotation guard | final orientation hold | outcome |
| --- | --- | --- | --- |
| `2026-05-19_23-03-05_guard_zero_rot_tightlat_2026-05-19_23-02-15` | zero rotation while `r > 0.5mm` | off | env0 reached shallow positive `s=0.15mm`, `r=0.45mm`, `theta=0.052`; env1 stayed outside with `r=0.98mm`, `theta=0.122`; no success |
| `2026-05-19_23-08-31_guard_zero_rot_thresh12_tightlat_2026-05-19_23-07-40` | zero rotation while `r > 1.2mm` | off | env1 improved orientation strongly and reached `s=0.09mm`, `r=0.96mm`, `theta=0.029`; env0 drifted late to `r=3.70mm`; no success |
| `2026-05-19_23-22-46_guard_final_orientation_hold_full_2026-05-19_23-22-00` | zero rotation while `r > 1.2mm` | on near entrance | best stability tradeoff: env0 ended `s=-0.18mm`, `r=0.95mm`, `theta=0.052`; env1 ended `s=0.09mm`, `r=0.95mm`, `theta=0.030`; no success |

Visual inspection:

- Checked center, left, and right camera images for the final-orientation run.
- The plug is visibly closer and no longer bypasses sideways like the earlier
  relaxed insertion probes.
- The SFP module is still outside the NIC cage. The small positive `s` in env1
  should be treated as shallow tip entrance/contact, not full insertion.

Conclusion:

- The rotation guard fixes the biggest regression: lateral error no longer
  explodes while the guide rotates near the port.
- The best current no-learning behavior is a stable pre-insertion pose around
  `r ~= 0.95mm`, with `theta` between `0.03` and `0.052rad`.
- This is still not enough for strict success because success requires tight
  lateral alignment (`<0.5mm`), orientation, axial depth, and
  `sfp_module_link` consistency. The visual module remains outside.

Current recommendation:

- Use `guard_final_orientation_hold_full` as the safest guided fallback setting
  if a submission needs a conservative near-gate policy.
- The next actual insertion attempt should add a retention/servo phase after
  `r < 1mm`: keep lateral correction active, hold/trim orientation, then permit
  only very small positive axial steps while monitoring `sfp_module_link`
  consistency. Do not loosen success thresholds or trust `sfp_tip_link` depth
  alone.
