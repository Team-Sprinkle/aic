# Agent Teacher CheatCode Gym Audit

## What Was Added

New script:

- `aic_gym_gz/run_cheatcode_gym.py`

This runs a thin gym-native adapter inspired by the official CheatCode policy and writes:

- rollout summary json
- 4 default videos

## Official CheatCode vs Gym Adapter

Official policy source:

- `aic_example_policies/aic_example_policies/ros/CheatCode.py`

Official CheatCode behavior:

- uses ground-truth TF lookups
- works in `base_link`
- computes absolute gripper pose targets
- sends `MotionUpdate` pose commands into the official controller path

Gym adapter behavior:

- uses `aic_gym_gz` observation only
- outputs native 6D gym action
- preserves the staged intent:
  - move above entrance
  - descend while keeping XY aligned
  - insert toward target

## Why Direct Reuse Was Not Straightforward

Direct reuse of official CheatCode inside `aic_gym_gz` was not practical because the official implementation depends on:

- ROS node lifecycle
- TF buffer and transform listener
- `MoveRobotCallback`
- `Task` ROS messages
- official controller `MotionUpdate` interface

`aic_gym_gz` instead exposes:

- synchronous Gym env
- already-aggregated observation dict
- native 6D Cartesian action

So a thin semantic adapter was the lowest-risk path.

## Semantic Mismatches

### Observation mismatch

Official CheatCode expects:

- TF frames for port, plug, and gripper TCP
- continuous transform queries
- direct access to official controller space

Gym provides:

- `plug_pose`
- `target_port_pose`
- `target_port_entrance_pose`
- `tcp_pose`
- no TF tree
- no controller pose-target interface

### Action mismatch

Official CheatCode sends:

- absolute pose targets

Gym expects:

- native 6D linear/angular velocity-like command
- held for `hold_action_ticks`

## Current Adapter Logic

`CheatCodeGymAdapter` phases:

1. `move_above_entrance`
   - drive plug above entrance by `+0.20 m` in z
2. `descend`
   - reduce z while maintaining XY alignment toward entrance
3. `insert`
   - drive toward final target pose

This is intentionally simple and faithful to the broad CheatCode strategy, not a byte-for-byte reproduction.

## Smoke Result in This Environment

Artifact:

- `aic_gym_gz/artifacts/cheatcode_gym_smoke.json`

Observed mock result:

- 1 episode
- return: `31.780057720873245`
- length: `512`
- termination: `truncated`
- gym_final_score: `62.14367177315697`

So the current adapter runs end-to-end in the gazebo gym stack, but it is not yet good enough to finish the task on the mock rollout.

## Video Artifacts

Generated:

- `aic_gym_gz/artifacts/videos/cheatcode_gym_ep0_seed123_trial_1/`

Containing:

- `camera_left.mp4`
- `camera_center.mp4`
- `camera_right.mp4`
- `overview_top_down_xy.mp4`
- `metadata.json`

## Recommendation

If the goal is "can CheatCode semantics be exercised inside gazebo gym?", the answer is yes.

If the goal is parity with official CheatCode, the next work item should be a stronger adapter that operates on:

- explicit plug-to-port relative transforms
- explicit insertion-axis frame
- absolute TCP target pose interface above the current native 6D action layer

That would be a better match than planning directly in native 6D action space.
