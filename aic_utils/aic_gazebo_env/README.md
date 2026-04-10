# AIC Gazebo Env

`aic_gazebo_env` is a ROS-free Gazebo gym-like runtime intended for training and
experimentation under `aic_utils/`.

It is **not** part of the official evaluation path.

The official evaluation flow remains:

- Gazebo + ROS + `aic_engine`
- launched from the `aic_eval` container
- started via `/entrypoint.sh ground_truth:=false start_aic_engine:=true`

This package exists to provide a separate training-only integration surface
without modifying `aic_engine`, `aic_bringup`, or the official eval container
workflow.

Current milestone scope:

- implemented Python env/runtime/backend modules
- real Gazebo CLI runtime/client path alongside the fake backend
- unit, smoke, parity, and live probe coverage under `tests/` and `scripts/`
- no direct ROS dependency in the package API

Current implementation notes:

- The branch is no longer "skeleton only".
- `GazeboRuntime` and `GazeboCliClient` provide a real no-ROS Gazebo path.
- `GazeboTransportClient` plus the repo-local `aic_gz_transport_bridge`
  helper are now the primary live transport path for Gazebo subscriptions and
  service calls.
- The C++ helper keeps persistent Gazebo Transport subscriptions to
  `/world/<world>/state` and `/world/<world>/pose/info`, issues transport
  service requests for `/world/<world>/control`, `/world/<world>/set_pose`,
  and `/world/<world>/joint_target`, and reports observation generations back
  to Python.
- Helper startup now uses a two-stage readiness check: liveness via `ping`,
  then sample readiness via helper status / `wait_until_ready`, so startup does
  not report success before at least one valid state sample has been received.
- The older Gazebo CLI observation path remains available as an explicit
  fallback / diagnostic path when the live runtime does not yield a fresh
  transport sample within the bounded timeout window.
- `reset()` now performs an explicit bounded post-reset world advance before it
  demands a fresh observation, and action paths perform explicit bounded world
  advancement before post-action reads instead of relying on passive simulator
  progress.
- The env-facing `observation["step_count"]` is a logical episode step count.
  Raw Gazebo iteration counts remain available in info payloads as
  `sim_step_count_raw` / `state_text_raw` for diagnostics.
- The package still remains training-oriented and separate from the official
  ROS + `aic_engine` evaluation flow.

Backend protocol notes:

- The package includes backend-agnostic Python request/response schema for
  `reset`, `step`, and `get_observation`.
- The schema is transport-neutral and intentionally does not depend on ROS or a
  Gazebo plugin.
- The current live implementation already maps these messages onto Gazebo
  Transport request/reply services without changing the public env API.

Preferred real env action interface:

- The preferred env-facing action for RL-style control is the cleaner
  tracked-source format:
  `{"position_delta": [dx, dy, dz], "orientation_delta": [x, y, z, w], "multi_step": N}`
- A more robot-relevant adapter is also supported:
  `{"ee_delta_action": {"position_delta": [dx, dy, dz], "orientation_delta": [x, y, z, w], "frame": "world|local", "max_position_delta_norm": M}, "multi_step": N}`
- This top-level tracked-source action always targets the configured tracked
  source entity.
- `ee_delta_action` is a pose-based tracked-source adapter, not joint control.
- `ee_delta_action.frame` defaults to `"world"` and supports:
  - `"world"`: `position_delta` is already in the world frame.
  - `"local"`: `position_delta` is interpreted in the current tracked-source
    local frame and rotated into the world frame before bridge handoff.
- `ee_delta_action.position_delta` is interpreted as a world-frame Cartesian
  translation delta for the tracked source entity after any local-to-world
  frame conversion.
- `ee_delta_action.orientation_delta` is interpreted as a quaternion pose delta
  composed against the current tracked-source orientation by the existing
  pose-delta bridge path.
- `ee_delta_action.orientation_delta` is normalized to unit length at the env
  adapter layer. Zero quaternions are rejected.
- `ee_delta_action.max_position_delta_norm` is optional. When provided, the
  translation delta is clipped to that Euclidean norm before any local-to-world
  conversion and bridge handoff.
- The adapter remains pose-based control over the tracked source entity. It
  does not introduce joint control, actuator control, or robot dynamics.
- Preferred action contract:
  - At least one of `position_delta` or `orientation_delta` must be present.
  - `position_delta` is an optional 3-element numeric list `[dx, dy, dz]`.
  - `orientation_delta` is an optional 4-element numeric quaternion list
    `[x, y, z, w]`.
  - `multi_step` is optional, must be a positive integer, and defaults to `1`
    when omitted.
  - Omitted optional fields stay omitted during normalization. For example,
    a position-only env action becomes a position-only `policy_action`.
- `GazeboEnv.action_spec` exposes this preferred contract as a lightweight
  descriptor for RL-style callers.
- The env normalizes that cleaner form into the canonical bridge-facing policy
  action:
  `{"policy_action": {"position_delta": [...], "orientation_delta": [...]}, "multi_step": N}`
- The `ee_delta_action` adapter first normalizes into the preferred tracked-
  source env action, which then normalizes into the same `policy_action` path.
- At the real bridge layer, `policy_action` is translated into
  `delta_source_pose`, which is then resolved against the latest real source
  pose and sent through Gazebo's `/world/<world>/set_pose` service.
- Older bridge-level action payloads are still supported for backward
  compatibility:
  - `policy_action`
  - `set_entity_position`
  - `set_entity_pose`
  - `delta_source_pose`

Gymnasium-compatible wrapper:

- `GymnasiumGazeboEnv` is a thin wrapper around `GazeboEnv`.
- It exposes:
  - `reset(seed=None, options=None)`
  - `step(action)`
  - `close()`
  - `action_space`
- `action_space` is based on the preferred tracked-source action contract:
  - `position_delta`: 3D continuous delta
  - `orientation_delta`: 4D continuous quaternion delta
  - `multi_step`: positive integer boxed as a 1D integer field
- `GazeboEnv.observation_spec` exposes the current stable structured
  observation contract for RL-style use.
- `GazeboEnv.flatten_observation(observation)` converts the stable tracked-pair
  subset into a deterministic numeric vector for RL pipelines.
- Stable observation contract currently includes:
  - top-level: `world_name`, `step_count`, `entity_count`, `entity_names`
  - derived geometry: `task_geometry["tracked_entity_pair"]`
  - tracked source/target pose fields in `entities_by_name`
  - tracked-pair success fields: `distance_success`,
    `orientation_success`, `success`
- `GymnasiumGazeboEnv.observation_space` now exposes a partial stable numeric
  subset of that contract:
  - `step_count`
  - `entity_count`
  - `tracked_entity_pair.relative_position`
  - `tracked_entity_pair.distance`
  - `tracked_entity_pair.source_orientation`
  - `tracked_entity_pair.target_orientation`
  - `tracked_entity_pair.orientation_error`
  - `tracked_entity_pair.distance_success`
  - `tracked_entity_pair.orientation_success`
  - `tracked_entity_pair.success`
- The raw observation dict is still passed through unchanged; the observation
  space is intentionally partial rather than a full flattening of the nested
  observation.
- Flattened observation field order is fixed as:
  1. `step_count`
  2. `entity_count`
  3. `tracked_entity_pair.relative_position[0]`
  4. `tracked_entity_pair.relative_position[1]`
  5. `tracked_entity_pair.relative_position[2]`
  6. `tracked_entity_pair.distance`
  7. `tracked_entity_pair.source_orientation[0]`
  8. `tracked_entity_pair.source_orientation[1]`
  9. `tracked_entity_pair.source_orientation[2]`
  10. `tracked_entity_pair.source_orientation[3]`
  11. `tracked_entity_pair.target_orientation[0]`
  12. `tracked_entity_pair.target_orientation[1]`
  13. `tracked_entity_pair.target_orientation[2]`
  14. `tracked_entity_pair.target_orientation[3]`
  15. `tracked_entity_pair.orientation_error`
  16. `tracked_entity_pair.distance_success`
  17. `tracked_entity_pair.orientation_success`
  18. `tracked_entity_pair.success`
- Flattened booleans are encoded as `0.0` or `1.0`.
- `GymnasiumGazeboEnv(flatten_observation=True)` returns this flattened vector
  from `reset()` and `step()`, and exposes a flat `(18,)` `observation_space`.
- In flattened wrapper mode, the raw structured observation remains available in
  `info["raw_observation"]`.
- If `gymnasium` is installed, the wrapper uses real Gymnasium `spaces`.
  Otherwise it falls back to lightweight compatible space descriptors so the
  wrapper remains importable in minimal environments.

Manual smoke procedure:

1. Start the existing eval container using the documented quickstart flow.
   Example:
   `docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest`
   `distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval`
   `distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true`
2. From the host Pixi workspace, run the documented WaveArm policy.
   Example:
   `pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.WaveArm`
3. In a separate host shell from the same repo, verify the new package imports.
   Example:
   `PYTHONPATH=aic_utils/aic_gazebo_env python3 -c "import aic_gazebo_env; print(aic_gazebo_env.__all__)"`

This smoke procedure is intentionally non-invasive:

- It does not modify `aic_engine`.
- It does not modify `aic_bringup`.
- It does not modify `aic_model`.
- It verifies that the new package can coexist with the existing workspace
  assumptions while remaining training-only.

Alignment notes against official eval:

- Coordinate frames:
  - Training env uses `world` as the global frame.
  - Training env names the robot base frame `base_link`.
  - Training env uses `gripper/tcp` as the end-effector frame.
- Robot naming:
  - Training env uses `ur5e`, matching the Gazebo bringup entity name.
- Action semantics:
  - Training env currently exposes `joint_position_delta`.
  - Official eval policies send joint or Cartesian targets through `aic_controller`.
- Success criteria:
  - Training env currently uses a simplified object-to-target distance criterion
    plus optional orientation gating.
  - Official eval success is task-specific cable insertion and scoring via
    `aic_engine` + `aic_scoring`.
  - The current gym reward is a training heuristic only and is not the
    authoritative evaluation score.
- Observation differences:
  - Training env allows privileged observations and omits cameras and wrench data.
  - Official eval observations are assembled by `aic_adapter` and include richer ROS data.

Manual smoke validation:

Run these from the repo root:

1. Reset/step/close smoke:
   `PYTHONPATH=aic_utils/aic_gazebo_env python3 aic_utils/aic_gazebo_env/scripts/smoke_reset_step.py`
   Validates:
   - package import
   - env construction
   - reset/step/close API contract
   - basic reward / termination / info fields
   Expected output:
   - one reset line
   - one step line
   - a final `smoke_reset_step: OK`

2. Random rollout smoke:
   `PYTHONPATH=aic_utils/aic_gazebo_env python3 aic_utils/aic_gazebo_env/scripts/smoke_random_rollout.py`
   Validates:
   - repeated stepping does not crash
   - rollout state evolves over time
   - termination/truncation paths are reachable
   Expected output:
   - a short rollout summary with step count, reward, and terminal flags
   - a final `smoke_random_rollout: OK`

3. Scripted-policy smoke:
   `PYTHONPATH=aic_utils/aic_gazebo_env python3 aic_utils/aic_gazebo_env/scripts/smoke_constant_action.py`
   Validates:
   - constant small actions produce stable directional state change
   - repeated steps remain well-formed and deterministic enough for manual inspection
   Expected output:
   - a summary with start/end x position and step count
   - a final `smoke_constant_action: OK`

4. Tiny training smoke:
   - not added

Live benchmark setup:

- Live benchmarking needs:
  - a Gazebo CLI executable resolvable as `gz`
  - the helper binary `aic_gz_transport_bridge` built into a workspace install
  - a running world such as the official eval container world
- The helper is built by colcon as package `aic_gazebo_transport_bridge` and is
  installed at:
  `install/aic_gazebo_transport_bridge/lib/aic_gazebo_transport_bridge/aic_gz_transport_bridge`
- The benchmark now performs a preflight before probing the live runtime. It
  reports:
  - repo root
  - resolved `gz` path or `null`
  - resolved helper path or `null`
  - nearby setup script, if one was found
  - whether the helper directory is already on `PATH`
  - searched locations for `gz` and the helper
- If binaries are missing but a nearby workspace setup script exists, the
  benchmark prints an exact `bash -lc "source ... && python3 ..."` command to
  rerun inside the sourced environment.

Common live benchmark flow:

1. Build the helper in a sourced workspace:
   `source /opt/ros/kilted/setup.bash`
   `colcon build --packages-select aic_gazebo_transport_bridge`
2. Source the workspace:
   `source install/setup.bash`
3. Run the benchmark:
   `PYTHONPATH=aic_utils/aic_gazebo_env python3 aic_utils/aic_gazebo_env/scripts/live_transport_benchmark.py`

If you want the benchmark to reuse the same automatic prep path as the e2e runner:

- `PYTHONPATH=aic_utils/aic_gazebo_env python3 aic_utils/aic_gazebo_env/scripts/live_transport_benchmark.py --auto-build --auto-launch`

Canonical live e2e workflow:

- Preferred command:
  `PYTHONPATH=aic_utils/aic_gazebo_env python3 aic_utils/aic_gazebo_env/scripts/run_live_e2e.py --auto-build --auto-launch`
- What it does automatically when possible:
  - discovers repo/workspace/container context
  - builds `aic_gz_transport_bridge` with a targeted `colcon build --packages-select aic_gazebo_transport_bridge`
  - attaches to an existing `aic_eval` distrobox container when available
  - launches the official world headlessly via the supported entrypoint or launch file when requested
  - waits for live health before smoke/parity checks
  - runs smoke stages and a small parity-oriented action sequence
- The canonical live command does not replace the official eval path. It reuses the supported launch path (`/entrypoint.sh` or sourced `aic_bringup`) to exercise the separate training-only runtime against a known-good live world.

Live health/e2e stages:

- `gz` reachable
- helper reachable
- world control service reachable
- state topic live
- first observation succeeds
- reset succeeds
- no-op step succeeds
- smoke sequence:
  - get observation
  - reset
  - no-op step
  - bounded pose-delta step
  - bounded joint-delta step
- parity sequence:
  - validates world/entity/tracked-pair sanity
  - checks success flag presence and types
  - checks logical/raw step counts are monotonic
  - checks repeated no-op stability within tolerance

Gated live pytest:

- Live e2e tests are skipped by default.
- Enable them explicitly:
  - `AIC_GAZEBO_ENV_RUN_LIVE_E2E=1 python3 -m pytest aic_utils/aic_gazebo_env/tests/test_live_e2e.py -q`
- Optional automation flags for the test subprocess:
  - `AIC_GAZEBO_ENV_AUTO_BUILD=1`
  - `AIC_GAZEBO_ENV_AUTO_LAUNCH=1`

Expected successful live e2e output:

- JSON with:
  - `preflight`
  - `context`
  - `result.health`
  - `result.smoke`
  - `result.parity`
- On success:
  - `result.health.no_op_step_ok == true`
  - `result.smoke.ok == true`
  - `result.parity.ok == true`
   - reason: the package does not expose a dedicated training entrypoint yet, and this validation pass is limited to smoke scripts over the existing public env API.
