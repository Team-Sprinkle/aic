# AIC Gym-GZ Architecture

## Goal

Build a standalone RL training path with synchronous `reset()` / `step()`
semantics that stays as behaviorally aligned as practical with the official AIC
evaluation stack, while keeping the training hot loop independent of the full
ROS orchestration layer wherever possible.

## Official source of truth

The new path reuses these official definitions directly:

- Scene schema: `aic_engine/config/sample_config.yaml`
- Task board and cable geometry:
  - `aic_description/urdf/task_board.urdf.xacro`
  - `aic_description/urdf/cable.sdf.xacro`
- World export:
  - `aic_description/world/aic.sdf`
- Action semantics:
  - `aic_interfaces/aic_control_interfaces/msg/MotionUpdate.msg`
  - `docs/aic_controller.md`
  - `aic_bringup/config/aic_ros2_controllers.yaml`
- Observation semantics:
  - `aic_interfaces/aic_model_interfaces/msg/Observation.msg`
  - `aic_interfaces/aic_control_interfaces/msg/ControllerState.msg`
  - `aic_adapter/src/aic_adapter.cpp`
- Score semantics:
  - `docs/scoring.md`
  - `aic_scoring/src/ScoringTier2.cc`

## Architectural split

The implementation separates:

1. Env API
2. Task and reward logic
3. Observation and action IO
4. Runtime stepping
5. Backend simulator integration

That split allows:

- a deterministic mock backend for local tests and reward validation
- a live Gazebo-backed path without rewriting env/task logic
- parity tooling that compares fixed rollouts without changing the RL interface

## Core components

- `AicInsertionEnv`
  - public Gymnasium API
  - returns `rl_step_reward` from `step()`
  - attaches `final_evaluation()` output at termination/truncation
- `AicInsertionTask`
  - owns action space and observation space
  - computes `rl_step_reward`
  - records episode traces for `gym_final_score`
- `AicGazeboRuntime`
  - owns exact synchronous stepping
  - delegates to the active backend
- `AicGazeboIO`
  - converts internal runtime state to the public observation schema
- `AicParityHarness`
  - compares rollouts and local score reports

## Observation architecture

The public observation contract is explicit and stable.

State fields include:

- robot state
- TCP state
- plug and target geometry
- current wrench and timestamp
- current contact flag
- flattened controller-state fields
- score geometry fields such as:
  - `distance_to_target`
  - `distance_to_entrance`
  - `orientation_error`
  - `insertion_progress`
  - `lateral_misalignment`
  - `partial_insertion`

Image mode adds:

- wrist RGB images
- image timestamps
- `camera_info`

### Temporal semantics

The observation model is current-sample-based.

- one `env.step()` produces one policy observation
- no built-in observation history is provided
- if `ticks_per_step > 1`, the policy sees the final sample for that held action
  window

This is important for F/T interpretation. Policies that need temporal memory
must build it themselves.

## Reward and score architecture

The implementation deliberately separates training reward from episode scoring.

### `rl_step_reward`

`rl_step_reward` is:

- dense
- per-step
- local
- shaped for optimization

It follows Isaac-Lab-style reward shaping rather than trying to exactly
reconstruct the final score.

### `gym_final_score`

`gym_final_score` is:

- computed at episode end
- based on the accumulated episode trace
- useful for local evaluation and analysis

### `official_eval_score`

`official_eval_score` is external to `aic_gym_gz`.

The architecture intentionally leaves it outside the local training loop because
it requires the official toolkit path.

## Live-path dependencies

The live path still depends on ROS topics for some fields:

- wrench
- controller state
- off-limit contacts
- images
- image timestamps
- `camera_info`

When these are absent, the observation schema stays stable and the missing live
fields are zero-filled or false-filled.

## Checkpoint architecture

- mock backend checkpoint/restore is exact
- live checkpoint export is a reset-and-rerun artifact only
- live midpoint restore is not currently available

## What remains approximate

- the default tested backend in this shell is mock, not Gazebo-backed
- live parity depends on external ROS/Gazebo availability
- local final scoring is still distinct from the official toolkit
- policy-level F/T currently exposes only the final sample per step
