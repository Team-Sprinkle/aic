# Gazebo-Gym Runtime Parity

This document describes how the current `aic_gym_gz` implementation relates to
the official AIC rollout surface and where parity is still approximate.

## Current parity goal

`aic_gym_gz` is trying to be:

- good enough for gym-style RL training
- explicit about what is local versus official
- close enough to the official rollout surface that policies and teacher systems
  can be developed against it

It is not trying to claim full official parity unless that has been explicitly
measured.

## Score and reward labels

Use these labels exactly:

- `rl_step_reward`: dense local RL training reward returned by `env.step()`
- `gym_final_score`: local episode-level score returned by `final_evaluation()`
- `official_eval_score`: official toolkit score, only when the official toolkit
  is actually run
- `teacher_official_style_score`: teacher-side approximation outside
  `aic_gym_gz`

Important:

- `rl_step_reward` is for RL optimization
- `gym_final_score` is for local trajectory-level evaluation
- `gym_final_score` is not `official_eval_score`

## What is aligned today

- explicit target-port and target-port-entrance geometry are propagated into the
  observation and score geometry
- controller-state semantics are flattened into the public observation when the
  ROS topic exists
- current wrench semantics are propagated into the observation when the live ROS
  wrench topic exists
- image timestamps and `CameraInfo` are propagated in image mode when the sidecar
  ROS topics exist
- local episode scoring uses the strongest available local geometry and traces
- mock checkpoint/restore is exact

## Observation parity notes

The public observation schema includes:

- current `wrench`
- `wrench_timestamp`
- `tcp_pose`
- `tcp_velocity`
- flattened controller-state fields
- images, image timestamps, and `camera_info` in image mode
- score geometry such as:
  - `distance_to_target`
  - `distance_to_entrance`
  - `orientation_error`
  - `insertion_progress`
  - `lateral_misalignment`
  - `partial_insertion`

### F/T and temporal behavior

- The policy sees the current wrench sample only.
- No built-in wrench history is provided.
- If `ticks_per_step > 1`, the observation contains the final sample for the
  held-action window, not a max or window summary.
- That means transient force/contact spikes can be missed at the policy level.

This is an important parity boundary. A policy that needs temporal memory should
build it explicitly.

## Reward parity notes

`rl_step_reward` is intentionally not a per-step decomposition of the final
score. It is a dense local training reward with Isaac-Lab-style shaping.

That means:

- reward parity to the official toolkit is not the goal for `rl_step_reward`
- behavior alignment and optimization usefulness are the goals

`gym_final_score` is the local score path that is closer in shape to the
official scoring decomposition, but it still remains local to `aic_gym_gz`.

## Final-score parity notes

`gym_final_score` uses:

- target-port geometry
- target-port-entrance geometry
- plug path
- TCP path
- timing trace
- wrench trace when available
- off-limit contact trace when available

It stays approximate relative to the official toolkit because:

- `official_eval_score` is not run here
- jerk still comes from env-side velocity history
- live wrench/contact exactness still depends on ROS topics

## Live dependencies

In live mode, the following fields depend on ROS topics being available:

- `wrench` / `wrench_timestamp`
- controller-state fields
- `off_limit_contact`
- images
- image timestamps
- `camera_info`

The schema remains stable when those topics are missing. The missing live fields
are zero-filled or false-filled rather than dropped.

## Checkpoint parity notes

- mock backend checkpoint/restore: exact
- live checkpoint export: approximate reset-and-rerun artifact only
- live midpoint restore: not available

## Official compatibility statement

`aic_gym_gz` is compatible with the official rollout surface at the level of:

- explicit observation fields
- fixed-rollout parity tooling
- local score-shape analysis

It is not a replacement for final validation with the official toolkit.

Before making any official-quality claim, run the official toolkit and report
`official_eval_score`.
