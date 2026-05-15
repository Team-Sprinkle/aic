# Isaac Tip Orientation Reward Findings - 2026-05-15

## Summary

The previous `cheatcode_insertion_v1` / `cheatcode_alignment_v1` reward used full quaternion error for `theta`.
That blocked the guide at roughly `0.073-0.080 rad` even when the SFP tip looked close to coaxial with the NIC port.

I changed the cheatcode insertion/alignment presets to use a semantic tip-axis orientation error by default:

- SFP-to-NIC: reward body `sfp_tip_link`, body-local insertion axis `+Z`
- SC-to-SC: reward body `sc_tip_link`, body-local insertion axis `+Y`

The reward still uses the configured tip body, not `gripper_tcp`.

## Code Changes

- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py`
  - Added `orientation_error_mode={quat,axis}` and `orientation_axis_local`.
  - Routed axis-mode orientation through:
    - `body_to_object_cheatcode_phase_reward`
    - `body_to_object_axial_progress`
    - `body_to_object_insertion_corridor`
    - `body_to_object_orientation_tanh`
    - `body_to_object_orientation_gated_exp`
    - success and once-only success bonus

- `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
  - Added CLI:
    - `--target_reward_orientation_error_mode`
    - `--target_reward_orientation_axis_local`
  - Cheatcode presets default to `axis`.
  - Auto axis default is SFP `+Z`, SC `+Y`.
  - Guide orientation gating now uses the same semantic axis mode.

- `aic_utils/aic_isaac/scripts/train_isaac_online_serl.py`
  - Forwarded the new orientation CLI knobs.

- `aic_utils/aic_isaac/scripts/run_one_day_insertion_pipeline.py`
  - Uses axis-mode orientation for one-day SFP sweeps.
  - Alignment pass criteria now allow the intended move from 6 mm outside to the safe hover around 4 mm outside, as long as `max_s` remains outside the entrance.

## Why SFP Uses Local +Z

The episode config generator sets:

```python
body_orientation = target_orientation * inverse(quat_from_rpy(0, pi, 0))
```

The reward then applies `body_orientation_offset = quat_from_rpy(0, pi, 0)` for `sfp_tip_link`.
After that offset, the SFP tip frame local `+Z` aligns with the port insertion axis.

Using local `+Y` was wrong for SFP: the audit reported `theta ~= 1.52 rad`.

## Visual Inspection

Run:

```text
outputs/one_day_insertion_pipeline/one_day_20260515_axis_orient_sweep/runs/2026-05-15_16-32-50_stage2_guide_w2_from_1_20260515_163226_gpu0
```

Final inspected camera frames:

```text
step_images/step_000258/env_0000_center_camera.png
step_images/step_000258/env_0000_left_camera.png
step_images/step_000258/env_0000_right_camera.png
```

H.264 videos:

```text
videos/env0_center_camera_h264.mp4
videos/env0_left_camera_h264.mp4
videos/env0_right_camera_h264.mp4
```

Visual result: the tip is held outside the entrance and is roughly coaxial with the port, but it is still slightly offset/twisted. This is good enough to proceed to alignment-imitation; it is not a completed insertion.

## Metrics

Best corrected-axis guide run:

| metric | value |
| --- | ---: |
| initial `s` | `-5.918 mm` |
| final `s` | `-3.164 mm` |
| max `s` | `-2.851 mm` |
| initial `r` | `5.999 mm` |
| final `r` | `1.291 mm` |
| best `r` | `1.046 mm` |
| initial `theta` | `0.0707 rad` |
| final `theta` | `0.0596 rad` |
| best `theta` | `0.0596 rad` |
| force clipped fraction | `0.015` |

This crosses the alignment orientation gate (`theta < 0.06`) while staying outside the entrance.

## Remaining Caveat

The SC-to-SC default uses `sc_tip_link` and local `+Y` because the SC episode metadata defines `SC_INSERTION_AXIS_WORLD = (0, 1, 0)` and does not define a target orientation. This should be verified with the same three-camera audit before SC training; if the SC tip USD frame differs, use:

```bash
--target_reward_orientation_axis_local <x> <y> <z>
```

to override it.

