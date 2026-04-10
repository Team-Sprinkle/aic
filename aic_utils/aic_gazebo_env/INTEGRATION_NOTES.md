# Gazebo RL Integration Notes

This note captures the current conclusion of the Gazebo RL-integration work and
ties it back to the official toolkit behavior on `main`.

## Official evaluation flow

The official evaluation path is not the new gym package.

The authoritative launch path is:

- `docs/getting_started.md`
- `aic_bringup/launch/aic_gz_bringup.launch.py`
- `docker/aic_eval/Dockerfile`
- `/entrypoint.sh ground_truth:=false start_aic_engine:=true`

The official runtime stack is:

1. `rmw_zenoh_cpp` middleware
2. Zenoh router plus session identity
3. `aic_gz_bringup.launch.py`
4. `aic_adapter`
5. `aic_engine`
6. participant model via `aic_model` or equivalent ROS node

Important consequences:

- the engine owns trial orchestration
- the engine spawns the task board and cable
- the engine tares the FT sensor before cable spawn
- the controller is driven through ROS topics, not Gazebo-native ad hoc RPCs
- official scoring is computed after recording the official ROS topics for the trial

## ACL and reachability

`docs/access_control.md` and the Zenoh session scripts define a split between
the `eval` identity and the `model` identity.

Important consequences:

- the `model` identity is intentionally blocked from services such as
  `/gz_server/get_entities_states`
- parity against the official toolkit must be run from the correct evaluation
  session context
- plain `docker exec` in the wrong session is not a reliable parity setup

## Authoritative score path

The authoritative score implementation is:

- `aic_engine/src/aic_engine.cpp`
- `aic_engine/config/sample_config.yaml`
- `aic_scoring/src/ScoringTier2.cc`
- `aic_scoring/include/aic_scoring/ScoringTier2.hh`
- `aic_bringup/config/ros_gz_bridge_config.yaml`

The engine starts `aic_scoring::ScoringTier2`, records the required topics into
a bag for each trial, then computes Tier 2 and Tier 3 after the task ends.

Required score inputs include:

- `/joint_states`
- `/tf`
- `/tf_static`
- `/scoring/tf`
- `/aic/gazebo/contacts/off_limit`
- `/fts_broadcaster/wrench`
- `/aic_controller/joint_commands`
- `/aic_controller/pose_commands`
- `/scoring/insertion_event`
- `/aic_controller/controller_state`

The official Tier 2 / Tier 3 logic is:

- insertion success: `75` for correct port, `-12` for wrong port
- partial insertion: `38` to `50`
- proximity score: `0` to `25`
- duration bonus: `0` to `12`, only if Tier 3 is positive
- trajectory smoothness bonus: `0` to `6`, only if Tier 3 is positive
- trajectory efficiency bonus: `0` to `6`, only if Tier 3 is positive
- force penalty: `-12` if force exceeds `20 N` for more than `1 s`
- off-limit contact penalty: `-24` on first prohibited contact

This is the scoring authority. The current gym reward is not.

## Current gym reward status

`aic_utils/aic_gazebo_env/aic_gazebo_env/reward.py` and
`GazeboCliClient._compute_step_outcome()` implement a small training heuristic:

- negative tracked distance
- optional orientation success gate
- fixed success bonus

That logic is acceptable as a temporary dense shaping signal for experiments,
but it is not the official toolkit score and should not be treated as the final
RL reward definition.

## Transport findings

The current CLI-backed live path proved the control concept but also exposed the
main architectural bottleneck.

Confirmed:

- `/world/<world>/joint_target` can drive real joint motion
- world name must be resolved from the running world, not hardcoded
- `/world/aic_world/state` contains usable live joint and link state

Observed problems:

- repeated `gz topic -e -n 1` subprocess calls are slow
- repeated one-shot subscriptions are unreliable for idle or change-driven topics
- `pose/info` is especially unreliable in the current usage pattern
- the current path samples observations at mismatched times relative to command
  application, which is now the main parity blocker on the reduced slice

Recommendation:

- do not treat repeated `gz` CLI subprocesses as the final RL transport
- keep the current path only as a temporary validation harness
- move the final RL stepping path into a persistent process with persistent
  subscribers and direct request/reply transport

## Recommended architecture

Short-term:

1. keep the direct Gazebo joint-target bridge for validation
2. keep the state-topic fallback for narrow observation slices
3. align parity sampling time on a minimal direct-Gazebo vs env slice
4. run toolkit parity only from the proper eval-session middleware context

Final transport recommendation:

1. implement a persistent Gazebo-side bridge instead of repeated CLI calls
2. use native Gazebo transport subscribers/services, or a small C++ bridge if
   Python transport access is still too fragile
3. support deterministic multi-tick stepping in that persistent process
4. expose a gym-style API on top of that persistent bridge, not on top of shell
   subprocesses

Reasons:

- lower latency
- lower sampling jitter
- fewer lost observations
- better parity with official timing and scoring signals
- easier integration of official score inputs over time

## Reward alignment recommendation

Use a two-layer reward story.

Episode score authority:

- keep `aic_scoring` as the source of truth for official end-of-trial score

Dense RL shaping:

- if dense reward is needed, make it explicit that it is shaping
- derive shaping from signals that are compatible with the official score path
  when possible, such as plug-port geometry, insertion progress, contact flags,
  and FT history

Current bridge step:

- the real client now supports an opt-in `official_tier3` reward mode for the
  stable tracked-pair slice
- this mirrors the official Tier 3 proximity-score shape using the tracked
  pair's current distance and reset-time initial distance
- it is still only a partial alignment layer, not a replacement for the full
  official `aic_scoring` path

Do not let the current distance-only gym reward become the de facto definition
of success.

## Parity plan

Smallest stable comparable slice first:

1. same start state
2. same joint-target semantics
3. same settle rule after command
4. same observation sample time
5. same derived geometry fields
6. same reward function for the slice being compared

Then:

1. compare direct Gazebo bridge vs env on one-step and short-rollout traces
2. compare toolkit controller path vs direct Gazebo path from the correct
   eval-session context
3. only after timing alignment, compare score-related signals

## Docker build-time conclusion

The slow local `aic_eval_local` build is expected for a source image build.

Evidence:

- `docs/getting_started.md` recommends `docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest`
- `docs/build_eval.md` marks source builds as advanced and heavier-weight
- `docker/aic_eval/Dockerfile` performs:
  - `vcs import` of the workspace
  - apt dependency installation
  - `rosdep install`
  - `GZ_BUILD_FROM_SOURCE=1 colcon build --executor sequential`

Local image observations:

- image created on `2026-04-10T13:25:41Z`
- image size about `2.63 GB` compressed image size from `docker image inspect`
- largest Docker history layers:
  - dependency install layer: about `4.93 GB`
  - workspace build layer: about `611 MB`
  - source copy layer: about `216 MB`

Conclusion:

- the quick README experience is fast because it usually pulls a prebuilt image
- the local branch workflow is slow because it rebuilds the evaluation
  workspace from source inside Docker
- the slow inner loop is therefore expected unless cached layers are reused

Recommended inner loop:

1. avoid rebuilding the full eval image for routine RL iteration
2. prefer a pulled eval image or one locally built base image reused across runs
3. bind-mount or host-build the few packages under active development when possible
4. reserve full `aic_eval_local` rebuilds for container-level validation

## Current blocker

In the current shell context used for this work:

- no built workspace is present at `/home/ubuntu/ws_aic/install`
- `ros2` is not on `PATH`
- `gz` is not on `PATH`

Therefore:

- Python-side API and bridge tests can be completed locally
- live Gazebo parity, live eval-session parity, and real latency benchmarks
  cannot be executed from this exact environment until a built or containerized
  runtime is available
