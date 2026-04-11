# AIC Gym-GZ Architecture

## Goal

Build a standalone RL training path with synchronous `reset()` / `step()` semantics
that stays behaviorally aligned with the official AIC evaluation stack, while
keeping ROS out of the hot loop wherever possible.

## Official source of truth

The new path reuses these official definitions directly:

- Scene schema: `aic_engine/config/sample_config.yaml`
- Task board and cable geometry knobs:
  - `aic_description/urdf/task_board.urdf.xacro`
  - `aic_description/urdf/cable.sdf.xacro`
- World export / scene container:
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

## What is replaced

The training path does not use these in the hot loop:

- `aic_engine`
- `aic_adapter`
- `ros_gz_bridge`
- ROS lifecycle / action orchestration

Instead it introduces:

- `AicGazeboRuntime`
  - exact synchronous stepping
  - deterministic `reset(seed=...)`
  - one action held for `K` sim ticks
- `AicInsertionTask`
  - simulator-agnostic reward / termination / evaluation logic
- `AicGazeboIO`
  - Gazebo-native observation extraction and command application interface
- `AicEnvRandomizer`
  - official-schema-aligned reset sampling
- `AicParityHarness`
  - rollout comparison tooling

## Runtime shape

The current implementation deliberately separates:

1. Env API
2. Task / reward logic
3. Observation / action IO
4. Runtime stepping
5. Backend simulator integration

That split allows:

- a deterministic fake backend for local tests
- a future ScenarIO + gym-gz backend without rewriting env logic
- a Gazebo Transport image/state bridge without coupling reward logic to Gazebo APIs

## State-only observation contract

Phase 1 exposes:

- `joint_positions`
- `joint_velocities`
- `gripper_state`
- `tcp_pose`
- `tcp_velocity`
- `plug_pose`
- `target_port_pose`
- `plug_to_port_relative`
- `wrench`
- `off_limit_contact`

This is intentionally close to the union of:

- `Observation.msg`
- `ControllerState.msg`
- scoring-time plug / port TFs

## Image observation plan

Planned order:

1. Gazebo Transport subscription to the three wrist image topics
2. Gazebo system plugin for direct frame extraction if transport alone is insufficient
3. ROS sidecar only as an isolated fallback

The current package includes only the IO seam for this work.

## Reward and score alignment

Per-step reward uses named terms:

- `success_reward`
- `wrong_port_penalty`
- `partial_insertion_reward`
- `proximity_reward`
- `progress_reward`
- `duration_penalty`
- `path_efficiency_term`
- `smoothness_term`
- `excessive_force_penalty`
- `off_limit_contact_penalty`

Final evaluation mirrors the official decomposition shape:

- Tier 2:
  - duration
  - trajectory smoothness
  - trajectory efficiency
  - insertion force
  - contacts
- Tier 3:
  - success / wrong-port / partial insertion / proximity

## What remains approximate

- The tested backend is not Gazebo-backed in this shell.
- The live ScenarIO + gym-gz backend is represented as `ScenarioGymGzBackend`
  but cannot be exercised until those dependencies are available.
- The image path is unimplemented.
- Parity against the official ROS flow still needs live trace capture in a Gazebo-enabled environment.

## TODO

- [x] Load official scenario YAML and expose a clean scenario model.
- [x] Implement deterministic reset and exact-tick synchronous stepping.
- [x] Add state-only observation mode with a stable schema.
- [x] Add named reward terms and official-like final score summary.
- [x] Add random-policy demo and unit tests.
- [x] Add parity harness for offline rollout comparisons.
- [x] Add benchmark helper for the standalone env.
- [ ] Implement real ScenarIO + gym-gz runtime backend.
- [ ] Implement Gazebo Transport observation extraction for state signals.
- [ ] Add Gazebo-native three-camera RGB ingestion.
- [ ] Capture official ROS rollout traces and finish parity validation.
- [ ] Measure throughput versus the official ROS evaluation path on the same machine.
