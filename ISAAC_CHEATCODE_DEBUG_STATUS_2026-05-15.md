# Isaac Cheatcode / Insertion Debug Status - 2026-05-15

## Current Repo State

- Repo: `Team-Sprinkle/aic`
- Branch: `feat/hybrid-train`
- HEAD: `5cd4cd625c460c9c3002c46e3905f4127ad62ad3`
- Working tree: dirty; this report was written without committing changes.
- Isaac container: `a7134d41ab70`
- Isaac Sim / Isaac Lab observed earlier in this session:
  - Isaac Sim `5.1.0-rc.19+release.26219.9c81211b.gl`
  - Isaac Lab `2.3.2`
- Safety convention used for recent runs:
  - `--num-envs 2`
  - short smoke tests only
  - `--max-wall-time-minutes`
  - `--ram-watchdog-min-available-gb 25`
  - no long training after the geometry/control issue became evident

## Current Answer to "Do We Have a Working Cheatcode Policy?"

No. We do not currently have a robust working cheatcode policy.

What we have is a set of privileged/no-learning smoke tests that can move the semantic `sfp_tip_link` forward along the insertion axis for a forced card0 near-gate setup, but they do not produce reliable visible insertion of the SFP module. The visible module frame `sfp_module_link` remains outside the port in the runs below.

The most important current finding is that a reward/metric based on `sfp_tip_link` can look like partial insertion while the visible `sfp_module_link` is still physically before the entrance.

## Files Changed During This Debugging Pass

Important modified files:

- `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
- `aic_utils/aic_isaac/scripts/train_isaac_online_serl.py`
- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py`
- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py`
- `aic_utils/aic_isaac/scripts/isaac_episode_configs.py`
- `aic_utils/aic_isaac/test/test_isaac_online_serl.py`
- `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/probe_target_reward.py`

Important new files:

- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/insertion_geometry.py`
- `aic_utils/aic_isaac/scripts/audit_insertion_geometry_synthetic.py`
- `aic_utils/aic_isaac/scripts/audit_insertion_reward_geometry.py`
- `aic_utils/aic_isaac/scripts/plot_isaac_insertion_trajectory.py`
- `aic_utils/aic_isaac/test/test_insertion_reward_geometry.py`

Untracked generated/submission artifacts also exist under:

- `docker/aic_submission/`
- `submission_handoff/`

## Implementation Changes Made

### Reward / Geometry Diagnostics

- Added shared insertion-geometry logic with explicit signed depth, target depth, lateral error, lateral gate, depth fraction, centered depth fraction, and bypass metrics.
- Added fail-fast style validation for target/entrance/axis consistency so invalid geometry does not silently collapse the seated target to the entrance.
- Removed or reduced reliance on unsafe target projection/clamping patterns that could hide axis/frame errors.
- Added pure formula-level tests for desired reward topology:
  - on-center forward motion positive
  - off-center forward motion negative
  - lateral improvement positive
  - lateral degradation negative
  - seated/on-center highest
  - beside-port/deep penalized
- Added overlay diagnostics in saved rendered frames:
  - entrance marker
  - target marker
  - insertion-axis marker/line
  - `sfp_tip_link`
  - `sfp_module_link`
  - `gripper_tcp`
  - numeric overlay for signed depth, lateral error, centered fraction, orientation error, target depth

### Action / IK Sign Debugging

- Confirmed in smoke testing that Isaac root-frame IK x/y behaves opposite the expected convention for the near-gate setup.
- The currently preferred flags for the near-gate smoke tests were:
  - `--fix-isaac-ik-xy-sign`
  - `--no-isaac-ik-xy-sign-by-target-card`
- A per-card x/y sign rule appears wrong for this near-gate setup.

### Guide / Cheat Controller Changes

- Fixed one real bug in `_target_guided_policy_action`:
  - It previously did not apply `body_position_offset` when computing the body point to guide.
  - This could make a guide using `sfp_module_link + SFP_TIP_LOCAL` inconsistent with rewards/diagnostics.
- Removed a double-application issue in the target guide path:
  - The guide path had pre-applied the Isaac x/y sign fix, and the normal action conversion then applied it again.
  - After the fix, sign conversion is applied in one place.
- Added an experimental rotation step option:
  - `--target-action-guide-rotation-step-size`
  - This was an experiment, not a proven fix.
  - In the root action-frame smoke test it made behavior much worse, indicating that rotation command frame/sign handling is still suspect.

## Key Semantic Frame Finding

The body offset between `sfp_module_link` and `sfp_tip_link` is about 23.65 mm, matching the constant in code:

- `SFP_TIP_LOCAL = (0.0, -0.02365, 0.0)`

This matters because many metrics report `sfp_tip_link` depth. In recent smoke tests:

- `sfp_tip_link` can reach +8 to +11 mm signed depth.
- `sfp_module_link` can still be -12 to -16 mm before the entrance.

So `sfp_tip_link` progress does not mean visible module insertion. Any success claim must be visually verified and must include module/body/contact evidence, not only semantic-tip depth.

## Target Geometry Finding

For the current near-gate episode configs, the target geometry itself appears numerically healthy:

- target depth is about 24.0 mm
- target/entrance/axis residual is near micron scale in the validated runs
- the insertion axis is approximately world `-Z`:
  - `[0.0, 0.012642, -0.999920]`

This means the current failure is probably not the original "target collapsed to entrance" bug for these configs. The more likely current problem is control/reference-frame/semantic-body inconsistency, possibly combined with contact/jamming and orientation mismatch.

## Smoke Tests and Findings

All recent smoke tests used `num_envs=2`.

### R90 - Overlay Axis Smoke

Run directory:

- `/workspace/isaaclab/outputs/train/isaac_online_serl/2026-05-15_08-38-07_r90_overlay_axis_global_xy_sign_seed236_2env_7`

Visual sheets:

- `outputs/video_audits/r90_overlay_axis_global_xy_sign/env0_center_camera_overlay_sheet.jpg`
- `outputs/video_audits/r90_overlay_axis_global_xy_sign/env1_center_camera_overlay_sheet.jpg`

Finding:

- Geometry markers looked sane.
- Target depth was about 24 mm.
- Target projection/residual was very small.
- This supported continuing with control smoke tests rather than reward tuning.

### R94 - Constant Root `-Z`, Original Mixed Card Episodes

Run directory:

- `/workspace/isaaclab/outputs/train/isaac_online_serl/2026-05-15_08-49-26_r94_constant_root_minus_z_inward_global_xy_seed236_2env_35`

Visual sheets:

- `outputs/video_audits/r94_constant_root_minus_z_inward/env0_center_camera_overlay_sheet.jpg`
- `outputs/video_audits/r94_constant_root_minus_z_inward/env1_center_camera_overlay_sheet.jpg`

Representative final metrics at step 35:

- env0:
  - `sfp_tip_link`: depth `+0.322 mm`, lateral `0.933 mm`
  - `sfp_module_link`: depth `-23.268 mm`, lateral `1.196 mm`
- env1:
  - `sfp_tip_link`: depth `+8.690 mm`, lateral `0.177 mm`
  - `sfp_module_link`: depth `-14.903 mm`, lateral `1.501 mm`

Finding:

- env1 semantic tip progressed.
- env0 mostly stalled.
- Neither env showed full visible insertion.

### R95 - Target Guide After Single-Sign Fix

Run directory:

- `/workspace/isaaclab/outputs/train/isaac_online_serl/2026-05-15_08-51-24_r95_target_guide_root_single_sign_seed236_2env_40`

Visual sheets:

- `outputs/video_audits/r95_target_guide_root_single_sign/env0_center_camera_overlay_sheet.jpg`
- `outputs/video_audits/r95_target_guide_root_single_sign/env1_center_camera_overlay_sheet.jpg`

Representative final metrics at step 40:

- env0:
  - `sfp_tip_link`: depth `+0.306 mm`, lateral `0.943 mm`
  - `sfp_module_link`: depth `-23.289 mm`
- env1:
  - `sfp_tip_link`: depth `+10.084 mm`, lateral `0.177 mm`
  - `sfp_module_link`: depth `-13.510 mm`

Finding:

- Guide improved semantic tip depth in env1 but did not produce visible insertion.
- env0 still stalled.

### R96 - Forced Card0 Pair, Constant Root `-Z`

Temporary config:

- `outputs/episode_configs/tmp_card0_pair/episodes/episode_000001.yaml`
- `outputs/episode_configs/tmp_card0_pair/episodes/episode_000002.yaml`

Both files were copied from the card0 episode config to isolate card-specific behavior.

Run directory:

- `/workspace/isaaclab/outputs/train/isaac_online_serl/2026-05-15_08-58-14_r96_card0_pair_root_minus_z_seed236_2env_35`

Visual sheets:

- `outputs/video_audits/r96_card0_pair_root_minus_z/env0_center_camera_overlay_sheet.jpg`
- `outputs/video_audits/r96_card0_pair_root_minus_z/env1_center_camera_overlay_sheet.jpg`

Representative final metrics at step 35:

- env0:
  - `sfp_tip_link`: depth `+7.28 mm`, lateral `0.20 mm`
  - `sfp_module_link`: depth `-16.31 mm`, lateral `1.55 mm`
- env1:
  - `sfp_tip_link`: depth `+8.74 mm`, lateral `0.18 mm`
  - `sfp_module_link`: depth `-14.85 mm`, lateral `1.50 mm`

Finding:

- Forcing both envs to card0 made both envs progress similarly.
- The prior env0 issue was likely card/start/contact-specific, not merely env index.
- Still no full visible insertion.

### R97 - Forced Card0 Pair, Longer Constant Root `-Z`

Run directory:

- `/workspace/isaaclab/outputs/train/isaac_online_serl/2026-05-15_08-59-34_r97_card0_pair_root_minus_z_long_seed236_2env_120`

Visual sheets:

- `outputs/video_audits/r97_card0_pair_root_minus_z_long/env0_center_camera_overlay_sheet.jpg`
- `outputs/video_audits/r97_card0_pair_root_minus_z_long/env1_center_camera_overlay_sheet.jpg`

Representative metrics:

- step 50:
  - env0 `sfp_tip_link`: depth `+7.53 mm`, lateral `0.26 mm`
  - env1 `sfp_tip_link`: depth `+11.21 mm`, lateral `0.25 mm`
  - env0 `sfp_module_link`: depth `-16.06 mm`
  - env1 `sfp_module_link`: depth `-12.38 mm`
- step 120:
  - env0 `sfp_tip_link`: depth `+8.50 mm`, lateral `0.42 mm`
  - env1 `sfp_tip_link`: depth `+9.09 mm`, lateral `0.31 mm`
  - env0 `sfp_module_link`: depth `-15.00 mm`, lateral `2.38 mm`
  - env1 `sfp_module_link`: depth `-14.44 mm`, lateral `2.18 mm`

Finding:

- Longer constant-axis push did not seat.
- It peaked around 9-11 mm semantic-tip depth and then drifted/worsened.

### R98 - Forced Card0 Pair, Stronger Constant Root `-Z` at 2 mm/step

Run directory:

- `/workspace/isaaclab/outputs/train/isaac_online_serl/2026-05-15_09-03-16_r98_card0_pair_root_minus_z_2mm_seed236_2env_80`

Visual sheets:

- `outputs/video_audits/r98_card0_pair_root_minus_z_2mm/env0_center_camera_overlay_sheet.jpg`
- `outputs/video_audits/r98_card0_pair_root_minus_z_2mm/env1_center_camera_overlay_sheet.jpg`

Representative metrics:

- step 40:
  - env0 `sfp_tip_link`: depth `+10.73 mm`, lateral `0.37 mm`
  - env1 `sfp_tip_link`: depth `+11.19 mm`, lateral `0.29 mm`
  - env0 `sfp_module_link`: depth `-12.85 mm`
  - env1 `sfp_module_link`: depth `-12.39 mm`
- step 80:
  - env0 `sfp_tip_link`: depth `+7.88 mm`, lateral `0.45 mm`
  - env1 `sfp_tip_link`: depth `+7.42 mm`, lateral `0.42 mm`
  - env0 `sfp_module_link`: depth `-15.57 mm`, lateral `2.77 mm`
  - env1 `sfp_module_link`: depth `-16.00 mm`, lateral `2.98 mm`

Finding:

- Stronger push still did not seat.
- It caused lateral drift and regression after initially improving depth.

### R99 - Forced Card0 Pair, Insertion Action Guard

Run directory:

- `/workspace/isaaclab/outputs/train/isaac_online_serl/2026-05-15_09-05-15_r99_card0_pair_guarded_axis_seed236_2env_90`

Visual sheets:

- `outputs/video_audits/r99_card0_pair_guarded_axis/env0_center_camera_overlay_sheet.jpg`
- `outputs/video_audits/r99_card0_pair_guarded_axis/env1_center_camera_overlay_sheet.jpg`

Representative metrics:

- step 40:
  - env0 `sfp_tip_link`: depth `+8.84 mm`, lateral `0.18 mm`
  - env1 `sfp_tip_link`: depth `+10.17 mm`, lateral `0.17 mm`
  - env0 `sfp_module_link`: depth `-14.75 mm`
  - env1 `sfp_module_link`: depth `-13.43 mm`
- step 90:
  - env0 `sfp_tip_link`: depth `+7.91 mm`, lateral `0.29 mm`
  - env1 `sfp_tip_link`: depth `+10.61 mm`, lateral `0.26 mm`
  - env0 `sfp_module_link`: depth `-15.66 mm`, lateral `1.81 mm`
  - env1 `sfp_module_link`: depth `-12.96 mm`, lateral `1.76 mm`

Finding:

- Lateral guard did not resolve the depth ceiling.
- Visible module remained outside.

### R100 - Forced Card0 Pair, Experimental Rotation Guide

Run directory:

- `/workspace/isaaclab/outputs/train/isaac_online_serl/2026-05-15_09-08-33_r100_card0_pair_rot_guide_seed236_2env_110`

Visual sheets:

- `outputs/video_audits/r100_card0_pair_rot_guide/env0_center_camera_overlay_sheet.jpg`
- `outputs/video_audits/r100_card0_pair_rot_guide/env1_center_camera_overlay_sheet.jpg`

Representative metrics:

- step 1:
  - `sfp_tip_link` depth around `-0.1 mm`
  - lateral already `0.23-0.38 mm`
- step 40:
  - env0 `sfp_tip_link`: depth `+6.39 mm`, lateral `85.74 mm`
  - env1 `sfp_tip_link`: depth `-0.17 mm`, lateral `34.69 mm`
- step 110:
  - env0 `sfp_tip_link`: depth `+15.12 mm`, lateral `178.46 mm`
  - env1 `sfp_tip_link`: depth `+16.83 mm`, lateral `175.20 mm`

Finding:

- This made behavior much worse.
- Rotation delta was almost certainly expressed in the wrong command frame/sign for root-frame IK, or the target/body orientation offsets are wrong for control even if useful for reward diagnostics.
- This experiment should not be treated as a fix.

## Important Visual Conclusion

The visual overlays and center-camera sheets do not show reliable insertion in any of R94-R100.

The most relevant sheets to review are:

- `outputs/video_audits/r97_card0_pair_root_minus_z_long/env0_center_camera_overlay_sheet.jpg`
- `outputs/video_audits/r97_card0_pair_root_minus_z_long/env1_center_camera_overlay_sheet.jpg`
- `outputs/video_audits/r98_card0_pair_root_minus_z_2mm/env0_center_camera_overlay_sheet.jpg`
- `outputs/video_audits/r99_card0_pair_guarded_axis/env0_center_camera_overlay_sheet.jpg`
- `outputs/video_audits/r100_card0_pair_rot_guide/env0_center_camera_overlay_sheet.jpg`

## Current Diagnosis

Most likely problems, in order:

1. The currently used semantic success/reward body can produce misleading progress.
   - `sfp_tip_link` is a semantic point about 23.65 mm ahead of `sfp_module_link`.
   - It can move into positive depth while the visible SFP module remains outside.

2. The current cheat/control path does not have a correct closed-loop 6D alignment controller.
   - Pure root-frame axial translation reaches a depth ceiling and then drifts.
   - Lateral guarding does not fix the ceiling.
   - The first attempted rotation guide has the wrong command-frame/sign behavior and explodes lateral error.

3. Card/start-specific behavior exists.
   - Original mixed-card runs had one env stall.
   - Forced card0 pair made both envs behave similarly.
   - This suggests target card index/start pose/contact geometry may matter.

4. The target entrance/axis/target metadata for the current near-gate card0 config does not look collapsed or obviously wrong.
   - Target depth is about 24 mm.
   - Residual is tiny.
   - Axis is visually/numerically plausible.

5. Reward tuning should remain paused until a privileged smoke controller can visibly insert.
   - Training cannot be trusted while the cheat/smoke controller cannot command visible insertion.

## What May Have Gone Wrong Earlier

Potential earlier implementation risks to audit before more changes:

- The reward and guide originally used different body points when `body_position_offset` was involved.
- The guide path had a sign-fix interaction that could double-apply the Isaac x/y correction.
- Metrics previously treated semantic-tip axial depth as "partial insertion" too readily.
- Orientation error in the overlays is around `3.1 rad` for the semantic tip frame in the translation-only runs, but translation-only guide ignored orientation entirely.
- The first attempt to add rotation showed that orientation correction is nontrivial and probably needs to mirror the known Gazebo/ROS cheatcode transform math exactly, not an ad hoc root-frame axis-angle command.

## Recommended Next Step

Do not continue reward tuning or SERL training yet.

Next recommended debug path:

1. Revert or isolate the experimental rotation-guide change before relying on the code path.
2. Implement a separate, minimal privileged controller script that mirrors the existing Gazebo/ROS cheatcode transform:
   - choose a physical/visible reference body, likely `sfp_module_link` plus an explicit local nose offset
   - compute the rigid transform that aligns the plug frame to the port frame
   - apply that transform to the controlled `gripper_tcp` or `wrist_3_link`
   - command a clipped 6D delta in the same frame the Isaac IK action expects
3. Validate it with `num_envs=2`, no learning, no randomization, and overlays.
4. First prove visible card0 near-gate insertion.
5. Then prove card1 near-gate insertion.
6. Only then return to policy/reward training.

## Exact Representative Command Pattern

The most useful no-learning smoke pattern was:

```bash
AIC_ISAAC_RANDOMIZATION_PROFILE=none TERM=xterm timeout 420 \
/workspace/isaaclab/aic/.pixi/envs/default/bin/python \
aic/aic_utils/aic_isaac/scripts/train_isaac_online_serl.py \
  --isaaclab /workspace/isaaclab/isaaclab.sh \
  --num-envs 2 \
  --steps 120 \
  --updates 999 \
  --update-every-steps 100000 \
  --warmup-steps 240 \
  --seed 236 \
  --episode-config-dir aic/outputs/episode_configs/tmp_card0_pair/episodes \
  --reward-preset near_gate_corridor_v1 \
  --target-reward-body sfp_tip_link \
  --actor-update-start-steps 100000 \
  --n-action-steps 1 \
  --tcp-action-frame root \
  --tcp-translation-action-clip 0 \
  --tcp-rotation-action-clip 0 \
  --debug-diagnostics \
  --diagnostics-every 1 \
  --debug-audit-steps 120 \
  --debug-audit-constant-action 0 0 -0.001 0 0 0 \
  --save-step-images \
  --debug-visual-overlays \
  --image-log-every 5 \
  --max-logged-image-steps 30 \
  --max-wall-time-minutes 7 \
  --ram-watchdog-min-available-gb 25 \
  --fix-isaac-ik-xy-sign \
  --no-isaac-ik-xy-sign-by-target-card
```

This is not a success command. It is only a reproducible smoke test showing semantic-tip progress without visible seating.

