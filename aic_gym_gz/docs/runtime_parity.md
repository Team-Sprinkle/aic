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

### Auxiliary within-step summaries

`aic_gym_gz` now also emits an explicitly non-official
`auxiliary_force_contact_summary` in `step_info`.

This summary is not part of the official-compatible observation contract. It is
an auxiliary training/debugging surface that can preserve within-step evidence
such as:

- `wrench_current`
- `wrench_max_abs_recent`
- `wrench_mean_recent`
- `wrench_max_force_abs_recent`
- `wrench_max_torque_abs_recent`
- `had_contact_recent`
- `max_contact_indicator_recent`
- `first_wrench_recent`
- `last_wrench_recent`
- `time_of_peak_within_step`

Important:

- `observation["wrench"]` remains the current sample only
- `observation["off_limit_contact"]` remains the current sample only
- the auxiliary summary must not be described as an official observation field

Source quality depends on backend:

- mock backend: exact internal sub-step aggregation
- live backend: best-effort aggregation from real ROS callback history when
  available
- if exact sub-step access is not available on a live path, the summary records
  that limitation instead of claiming exact simulator parity

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
- current-sample-only wrench/contact observation semantics
- fixed-rollout parity tooling
- local score-shape analysis

It is not a replacement for final validation with the official toolkit.

Before making any official-quality claim, run the official toolkit and report
`official_eval_score`.

## Transient-contact validation

Use `aic_gym_gz.validate_force_transients` when you want to audit coarse-step
aliasing directly.

That validator checks:

- obstacle/contact transient case where the final sample can be quiet while the
  auxiliary within-step summary still detects contact
- no-contact control case to guard against false positives
- repeated coarse boundary-crossing to confirm consistent surfacing of hidden
  transients

The validator also reports:

- whether direct Isaac Lab parity was tested
- if not, which conceptual Isaac-Lab-style expectations were checked instead
- whether direct official AIC parity was tested
- if not, what limitation prevented it
