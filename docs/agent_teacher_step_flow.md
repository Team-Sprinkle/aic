# Agent Teacher Step Flow

## End-to-End Flow

```text
TeacherPlan (planner-facing waypoint payload + horizon)
  -> MinimumJerkSmoother.smooth()
      - preserve current plug-to-TCP offset
      - convert planner waypoints according to planner_output_mode
      - supported modes:
          * absolute_cartesian_waypoint
          * delta_cartesian_waypoint
          * native_6d_action
      - finite-difference target poses at base_dt=0.02 s
      - emit DenseTrajectoryPoint(action[6], target_tcp_pose[7])
  -> run_teacher_rollout()
      - iterate dense points
      - env.step(action)
  -> AicInsertionEnv.step()
      - sanitize_action()
      - runtime.step(action, ticks=hold_action_ticks)
  -> AicGazeboRuntime.step()
      - backend.apply_action(action)
      - backend.step_ticks(tick_count)
  -> RuntimeBackend
      live:
        ee_delta_action.position_delta = linear_vel * 0.002 * tick_count
        ee_delta_action.orientation_delta = quat(angular_vel * 0.002 * tick_count)
        frame = world
      mock:
        integrate linear/angular action every sim tick
  -> new RuntimeState
  -> io.observation_from_state()
  -> task.evaluate_step()
      - reward
      - termination/truncation
```

## Exact Timestep Math

### Simulation timestep

`AicGazeboRuntime.sim_dt = 0.002`

So one simulator tick is:

- `0.002 s`
- `500 Hz`

### `env.step()` timestep

Default `ticks_per_step = 8`, wired through:

- `make_default_env(... ticks_per_step=8)`
- `make_live_env(... ticks_per_step=8)`
- `AicInsertionTask.hold_action_ticks = ticks_per_step`

So one `env.step()` holds the action for:

- `8 * 0.002 = 0.016 s`

That is:

- `62.5 Hz` env step rate

### Teacher dense trajectory timestep

`MinimumJerkSmoother.base_dt = 0.02`

So the smoother emits dense points at:

- `0.02 s`
- `50 Hz`

This does **not** match the env step exactly:

- smoother point dt: `0.02 s`
- env step hold dt: `0.016 s`

The code records `DenseTrajectoryPoint.dt=0.02`, but execution still uses `env.step()` with the task/runtime hold of `0.016 s`.

## Exact Action Semantics

### Native env action

Action shape is `(6,)`.

`sanitize_action()` in `io.py` only clips:

- first 3 dims to `[-0.25, 0.25]`
- last 3 dims to `[-2.0, 2.0]`

No frame conversion. No normalization. No pose interpretation.

### Live backend meaning

In `ScenarioGymGzBackend.step_ticks()`:

- `action[:3]` is treated as translational velocity-like Cartesian command
- `action[3:]` is treated as angular velocity-like command
- integrated over `tick_count * 0.002`
- sent as:
  - `position_delta`
  - `orientation_delta`
  - `frame = "world"`

So in practice the action means:

- linear velocity in world-frame units of m/s-like scale
- angular velocity in rad/s-like scale

held constant for `tick_count` simulator ticks.

### Mock backend meaning

In `MockStepperBackend.step_ticks()`:

- `linear = action[:3]`
- `angular = action[3:]`
- per-tick integration:
  - `next_tcp_pose[:3] += linear * sim_dt`
  - `next_tcp_pose[5] += angular[2] * sim_dt`

This is also velocity-like, but with simplified yaw-only orientation integration.

## Where Motion Conversion Happens

### Planner output

`TeacherPlan.waypoints[*].position_xyz`

These are interpreted by the smoother according to `MinimumJerkSmoother.planner_output_mode`.

### Waypoint to TCP conversion

In `MinimumJerkSmoother.smooth()`:

- `absolute_cartesian_waypoint`
  - compute current `plug_to_tcp_offset = tcp_pose[:3] - plug_pose[:3]`
  - set `target_tcp_pose[:3] = waypoint.position_xyz + plug_to_tcp_offset`
  - set `target_tcp_pose[5] = waypoint.yaw`

- `delta_cartesian_waypoint`
  - treat `waypoint.position_xyz` as a delta from current plug position
  - convert delta plug target into an absolute TCP target pose
  - treat `waypoint.yaw` as a delta yaw

- `native_6d_action`
  - treat `position_xyz` as native translational action components
  - treat `yaw` as native yaw-rate-like action component
  - bypass finite-difference-derived action and reuse the native action directly

So the current system now contains an explicit small adapter layer:

- planner-facing waypoint/native action mode
-> execution-facing absolute TCP target pose or direct native 6D action

### TCP target pose to low-level action

Still in `MinimumJerkSmoother.smooth()`:

- minimum-jerk blend between current and target TCP pose
- finite difference consecutive poses using `base_dt=0.02`
- clip to `max_linear_speed=0.25`, `max_angular_speed=2.0`
- emit native 6D action

So for the default mode the planner is **not** directly emitting native 6D commands. The smoother generates those. In `native_6d_action` mode, the smoother records that pass-through in `TrajectorySegment.conversion_metadata`.

## Reward / termination path

After each `env.step()`:

- `io.observation_from_state()` builds observation
- `task.evaluate_step()` computes RL-style step reward
- termination if:
  - success
  - wrong port
  - off-limit contact
- truncation if:
  - `step_count >= max_episode_steps`

Final score comes from `task.final_evaluation()`.

## Reference Frames: Exact Recoverable State

### Explicit in code

- live action application frame: `world`
- official CheatCode reference frame: `base_link`

### Implicit / ambiguous

- planner payload pose frames are not declared
- runtime entity poses come from Gazebo entities and behave like world-frame poses
- controller-state fields likely come from controller topics, but frame semantics are not declared in the payload

## Answering the 6 Requested Questions

### What is the sim timestep?

- `0.002 s`

### What is the `env.step` timestep?

- default `8` ticks
- `0.016 s`

### What action is held over how many sim ticks?

- one sanitized 6D action
- held constant for `hold_action_ticks` / `ticks_per_step`
- default `8` simulator ticks

### Where is the action converted to motion?

- live: `ScenarioGymGzBackend.step_ticks()`
- mock: `MockStepperBackend.step_ticks()`

### How are teacher waypoints converted into actual low-level actions?

- `TeacherPlan.waypoints`
-> `MinimumJerkSmoother.smooth()`
-> dense absolute TCP targets
-> finite-difference velocity-like 6D actions

### Are these absolute targets, deltas, velocities, or twists?

In practice:

- planner waypoints, default mode: absolute plug-space targets
- planner waypoints, delta mode: plug-space deltas from current state
- planner waypoints, native mode: native action components encoded in waypoint fields
- smoother intermediate target poses: absolute TCP targets
- env/runtime action: velocity-like Cartesian linear/angular command
- live backend transport command: integrated Cartesian delta in world frame

## Recommendation

Planner-facing native 6D action is not the natural interface.

Better planner-facing interfaces:

1. absolute Cartesian waypoint
2. delta Cartesian waypoint
3. keep native 6D only as the execution-layer interface

The current code already approximates option 1 through the smoother. A full configurable action-interface adapter was **not** implemented in this pass because it would change planner contracts and deserved a more careful rollout.
