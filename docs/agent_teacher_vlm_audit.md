# Agent Teacher VLM Audit

## Scope

This audit traces the current VLM/planner payload path in `aic_gym_gz`:

`demo_teacher_rollout.py` / `run_teacher_search.py`
-> `teacher.runner.run_teacher_rollout()`
-> `teacher.policy.AgentTeacherController.select_plan()`
-> `teacher.context.TeacherContextExtractor.build_planning_state()`
-> `planners.openai_backend.OpenAIPlannerBackend.build_request_payload()`

The current payload is a **short-horizon segment-planning** payload, not a whole-episode plan.

There is now also an optional additive global-guidance path:

`teacher.policy.AgentTeacherController._maybe_refresh_global_guidance()`
-> `planners.base.PlannerBackend.plan_global_guidance()`
-> `planners.openai_backend.OpenAIPlannerBackend.build_global_guidance_request_payload()`

That path is disabled by default and returns milestone/phase guidance metadata, not executable segment actions.

## Exact Current Payload Behavior

### Current observation included

`TeacherContextExtractor.build_planning_state()` currently places these into `policy_context`, and `OpenAIPlannerBackend._planner_state_payload()` forwards a subset into `current_observation`:

- `tcp_pose`
- `plug_pose`
- `target_port_pose`
- `target_port_entrance_pose`
- `tcp_velocity`
- `wrench`
- `wrench_timestamp`
- `distance_to_target`
- `distance_to_entrance`
- `lateral_misalignment`
- `orientation_error`
- `insertion_progress`
- `off_limit_contact`
- `relative_geometry`
- `frame_context`

Notably, `score_geometry` and `auxiliary_force_contact_summary` are in `policy_context` but are still **not forwarded into the `current_observation` block**. Some of that information is only represented indirectly elsewhere.

### History included

History comes from `TemporalObservationBuffer.teacher_memory_summary()`. The backend currently forwards only:

- `dynamics_summary`
- `geometry_progress_summary`
- `wrench_contact_trend_summary`
- `compact_signal_samples`
- `auxiliary_history_summary`
- `phase_guidance`
- `global_guidance`

Raw history such as:

- `action_history`
- `wrench_history`
- `wrench_timestamp_history`
- `tcp_velocity_history`
- `controller_state_history`

is present in `TeacherPlanningState.temporal_context`, but only a compact sampled subset is now forwarded through `compact_signal_samples`.

### Past images

Past official camera images are included only through:

- `TemporalObservationBuffer.recent_visual_frames()`
- `teacher.visual_context.build_recent_visual_observations()`
- `OpenAIPlannerBackend._visual_content()`

They are sent as `input_image` items with `image_url=data:image/png;base64,...` and `detail="low"`.

Current behavior:

- history source: recent visual frames from teacher history
- current configured upstream pool: now up to `TeacherContextExtractor.max_recent_visual_frames` frames, default `10`
- per-request selected subset: `select_visual_context_items()`
- per-episode budget: `OpenAIPlannerConfig.max_visual_images_per_episode`, default `30`

Important detail: these images are **not embedded inside the JSON text payload**. They are attached as separate Responses API `input_image` items.

### Non-camera Gazebo overview images

By default there is no true Gazebo rendered overview camera in the planner payload.

There is now an additive opt-in teacher path for live fixed overview images:

- `TeacherContextExtractor(prefer_live_scene_overview=True)`
- `teacher.visual_context.latest_live_overview_images()`
- live topics:
  - `/overview_camera/image` for `top_down_xy`
  - `/overview_front_camera/image` for `front_xz`

If that flag is enabled and the topics are available, `top_down_xy` and `front_xz` can be sourced from live overview cameras. Otherwise the planner falls back to the existing schematic renders.

Current "overview images" are teacher-side schematic renders from:

- `teacher.visual_context.build_scene_overview_images()`

They are:

- generated from scenario geometry plus current state
- views: `top_down_xy`, `front_xz`, `side_yz`
- encoded as PNG data URLs
- attached as `input_image`
- explicitly described in the prompt as **non-official teacher-side schematic views**

Today the default planner path still uses schematic overview images unless the live-overview preference is explicitly enabled.

A dedicated runtime probe now exists at:

- `aic_gym_gz/probe_overview_cameras.py`

In this shell it reported both fixed overview topics as present in configuration but not delivering frames (`ready=false`, zero timestamps), so live planner-side overview validation remains blocked by runtime availability rather than payload code.

### Sensor data actually sent

#### Wrench / contact

- `wrench` and `wrench_timestamp` exist in `policy_context`
- `wrench` and `wrench_timestamp` are now sent in `current_observation`
- `off_limit_contact` is sent in `current_observation`
- `signal_quality_context.wrench` is sent
- auxiliary contact summaries are sent only as aggregated history in `temporal_context_summary.auxiliary_history_summary`

What is **not** sent:

- wrench history
- full auxiliary per-step force/contact summaries

#### Joint states

- `joint_positions` and `joint_velocities` exist in env observation
- they do **not** appear in `TeacherPlanningState.policy_context`
- they are **not sent** to the VLM

#### TCP pose / plug pose / target poses

Sent:

- `tcp_pose`
- `plug_pose`
- `target_port_pose`
- `target_port_entrance_pose`

Not sent:

- explicit relative transforms like `tcp_to_port`, `plug_to_entrance`, `plug_to_tcp`

#### Controller state / reference tcp / tcp error / target mode

`TeacherPlanningState.controller_context` contains:

- `controller_state` summary
- `reference_tcp_pose`
- `tcp_error`
- `controller_target_mode`

But the backend currently sends only:

- `controller_state_available`
- `reference_tcp_pose`
- `tcp_error`
- `controller_target_mode`

#### Velocities

`tcp_velocity` is now sent in `current_observation` and is also represented in compact sampled temporal history.

No joint velocity history is sent.

### Obstacle coordinates / geometry / oracle context

The planner does get substantial teacher-only geometry:

- `obstacle_summary[:6]`
- `scene_geometry_context.board_pose_xyz_rpy`
- `scene_geometry_context.target_port_pose`
- `scene_geometry_context.target_port_entrance_pose`
- `scene_geometry_context.plug_pose`
- `scene_geometry_context.clearance_summary`
- `scene_geometry_context.scene_layout_summary`
- `scene_geometry_context.cable_context`

This is teacher-side privileged context, not official participant observation.

### Frame conventions

Frame conventions are **not explicit enough** in the planner payload.

Observed facts from code:

- live backend sends `ee_delta_action.frame = "world"` in `runtime.py`
- official CheatCode policy works in `base_link`
- `task.frame = "base_link"` exists in `task.py` but is not used in payload construction
- planner payload does not declare what frame `tcp_pose`, `plug_pose`, `target_port_pose`, or `target_port_entrance_pose` are in
- `scene_layout_summary.present_obstacles[*].approximate_world_xyz` does imply world frame for obstacle summaries

So the payload currently mixes world-like geometry with no explicit global frame contract.

### Data quality / missingness flags

Yes. The planner receives `signal_quality_context = state.data_quality`.

Current signals tracked:

- `wrench`
- `controller_state`
- `camera_info`
- `target_port_entrance_pose`
- `partial_insertion_depth`
- `tier1_validity`

This is one of the strongest parts of the current payload.

### Important information not currently included

Most important omissions:

- raw current wrench vector and short wrench history
- joint states
- tcp velocity and motion-command trend
- explicit relative transforms in task-relevant frames
- exact obstacle extents / geometry envelopes
- explicit frame tree and insertion-axis definition
- live camera frames in the common no-image path
- per-step contact localization

## Compact Field Table

| Field name | Source file/function | Current/planned? | Current timestep granularity | Sent to VLM | Notes |
|---|---|---:|---|---:|---|
| `tcp_pose` | `teacher.context.build_planning_state` | current | current step | yes | In `current_observation`; frame not explicit |
| `plug_pose` | `teacher.context.build_planning_state` | current | current step | yes | In `current_observation`; world-like absolute pose |
| `target_port_pose` | `teacher.context.build_planning_state` | current | static/current | yes | In `current_observation` and `scene_geometry_context` |
| `target_port_entrance_pose` | `teacher.context.build_planning_state` | current | static/current | yes | In `current_observation` and `scene_geometry_context` |
| `distance_to_target` | `teacher.context.build_planning_state` | current | current step | yes | In `current_observation` |
| `distance_to_entrance` | `teacher.context.build_planning_state` | current | current step | yes | In `current_observation` |
| `lateral_misalignment` | `teacher.context.build_planning_state` | current | current step | yes | In `current_observation` |
| `orientation_error` | `teacher.context.build_planning_state` | current | current step | yes | In `current_observation`; frame semantics unclear |
| `insertion_progress` | `teacher.context.build_planning_state` | current | current step | yes | In `current_observation`; current mock semantics are questionable |
| `off_limit_contact` | `teacher.context.build_planning_state` | current | current step | yes | Only coarse boolean contact state |
| `tcp_velocity` | `teacher.context.build_planning_state` | current | current step | yes | Forwarded in `current_observation` |
| `wrench` | `teacher.context.build_planning_state` | current | current step | yes | Forwarded in `current_observation` |
| `wrench_timestamp` | `teacher.context.build_planning_state` | current | current step | yes | Forwarded in `current_observation` |
| `official_current_wrench` | `teacher.context.build_planning_state` | current | current step | no | Stored only in `policy_context` |
| `score_geometry` | `teacher.context.build_planning_state` | current | current step | no | Only selected derived fields reach VLM |
| `auxiliary_force_contact_summary` | `teacher.context.build_planning_state` | current | current step / within-step aggregate | no | Only summarized history form is sent |
| `dynamics_summary` | `teacher.history.teacher_memory_summary` | current | recent history summary | yes | Sent in `temporal_context_summary` |
| `geometry_progress_summary` | `teacher.history.teacher_memory_summary` | current | recent history summary | yes | Sent in `temporal_context_summary` |
| `auxiliary_history_summary` | `teacher.history.teacher_memory_summary` | current | recent history summary | yes | Sent in `temporal_context_summary` |
| `phase_guidance` | `teacher.planning.phase_guidance_from_state` | current | current + recent history summary | yes | Strong steering signal |
| `action_history` | `teacher.history.teacher_memory_summary` | current | recent history | no | Present in planning state only |
| `wrench_history` | `teacher.history.teacher_memory_summary` | current | recent history | no | Present in planning state only |
| `tcp_velocity_history` | `teacher.history.teacher_memory_summary` | current | recent history | no | Present in planning state only |
| `controller_state_history` | `teacher.history.teacher_memory_summary` | current | recent history | no | Present in planning state only |
| `compact_signal_samples` | `teacher.history._compact_signal_samples` | current | sparse recent history | yes | Compact sampled actions, wrench, tcp velocity, contact flags, timestamps |
| `joint_positions` | env/io observation only | current | current step | no | Never enters planning state |
| `joint_velocities` | env/io observation only | current | current step | no | Never enters planning state |
| `controller_state_available` | `planners.openai_backend._planner_state_payload` | current | current step | yes | Boolean only |
| `tcp_error` | `teacher.context.build_planning_state` | current | current step | yes | Sent if runtime controller data exists |
| `controller_target_mode` | `teacher.context.build_planning_state` | current | current step | yes | Sent if runtime controller data exists |
| `reference_tcp_pose` | `teacher.context.build_planning_state` | current | current step | yes | Now forwarded in `controller_context` |
| `relative_geometry` | `teacher.context._relative_geometry_summary` | current | current step | yes | Explicit relative transforms and insertion-axis features |
| `frame_context` | `teacher.context._frame_context` | current | static contract metadata | yes | Declares world/base_link conventions and planner spaces |
| `wrench_contact_trend_summary` | `teacher.history._wrench_contact_trend_summary` | current | recent history summary | yes | Compact force/contact trend summary |
| `global_guidance` | `teacher.policy._maybe_refresh_global_guidance` / `openai_backend.plan_global_guidance` | current optional | low-frequency per segment interval | yes | Additive milestone/phase strategy path with separate budget |
| `obstacle_summary` | `teacher.context._obstacle_summary` | current | static scenario | yes | First 6 entries only |
| `scene_layout_summary` | `teacher.context._scene_layout_summary` | current | static/current | yes | Teacher-only geometry with approximate world coordinates |
| `clearance_summary` | `teacher.context._clearance_summary` | current | static scenario | yes | Coarse only |
| `task_board_pose_xyz_rpy` | `teacher.context.build_planning_state` | current | static scenario | yes | Sent as `board_pose_xyz_rpy` |
| recent wrist-camera images | `teacher.visual_context.build_recent_visual_observations` | current | sparse recent frames | yes | Separate `input_image`, low detail |
| scene overview images | `teacher.visual_context.build_scene_overview_images` | current | current state image | yes | Separate `input_image`; default schematic, optional live top-down via `/overview_camera/image` |
| image budget across episode | `planners.openai_backend._visual_content` | current | episode budget | yes | Added in this task, default `30` |
| `planner_output_mode` | `teacher.policy.AgentTeacherController.select_plan` | current optional | run config | yes | Logged in planning metadata; selects absolute, delta, or native-6D interpretation |
| explicit frame tree | not implemented | planned only | n/a | no | Recommended |
| exact obstacle extents | not implemented | planned only | n/a | no | Recommended |
| relative transforms in port frame | not implemented | planned only | n/a | no | Recommended |

## Why Score Is Low: Code-Grounded Causes

### 1. The planner is only local segment planning

The OpenAI backend prompt explicitly asks for "one short segment", and the controller replans segment-by-segment. There is no true low-frequency global plan artifact.

### 2. Rollout-mode OpenAI budgets are internally inconsistent

`demo_teacher_rollout.py` sets `TeacherConfig(candidate_plan_count=3)`, and `AgentTeacherController.select_plan()` calls the planner once per candidate. With the OpenAI backend default `max_calls_per_episode=8`, rollout mode can exhaust budget after roughly:

- 2 full segments at 3 calls/segment = 6 calls
- then only 2 more candidate calls on the third segment

That is a concrete failure mode for OpenAI rollout mode.

### 3. Search diversity is weak

`evaluate_teacher_search` on the smoke artifact reported many near-duplicate candidates and `value_over_single_plan = 0.0`. The refinement family is often producing effectively identical phase/waypoint signatures.

### 4. Frame semantics are unclear and likely mixed

Live actuation uses `world` frame deltas, while official CheatCode uses `base_link`, and the planner payload does not state frames explicitly.

### 5. The planner does not see the actual native control semantics

The planner emits absolute plug-space waypoints, but execution is via a velocity-like 6D action held for `hold_action_ticks`. That adapter is reasonable, but the planner payload never sees that contract explicitly.

### 6. Critical execution telemetry is missing from the VLM payload

Missing:

- raw wrench / wrench trend
- reference TCP pose
- controller tracking status
- relative geometry in explicit task frames
- live visual feedback in no-image mode

### 7. Current mock `score_geometry` can be misleading

In the payload dump, `insertion_progress=1.0` while `distance_to_entrance` is still large. The missing-context VLM audit also flagged this contradiction. That weakens phase reasoning and prompt trustworthiness.

## Structured Missing-Context Recommendations

Actual structured audit artifact:

- `aic_gym_gz/artifacts/context_audit/missing_context_audit_vlm.json`

Top recommendations from the VLM audit:

- Critical missing context:
  - live controller state and mode detail
  - controller `tcp_error`
  - real wrench force/torque history
- High-value extra images:
  - current wrist/TCP live view
  - close frontal port-region view
  - oblique view along the approach axis
- High-value geometry/frame additions:
  - explicit `world -> board -> entrance -> port` frame tree
  - board axes and insertion-axis direction
  - exact obstacle extents
  - explicit plug-to-TCP transform
- Ambiguities flagged:
  - `insertion_progress=1.0` contradicts large entrance distance
  - `in_insertion_zone=true` while far from target
  - `orientation_error` definition/frame unclear
  - scene overviews are schematic and can be over-interpreted

## Current vs Recommended Planning Split

Current:

- only local segment planning is implemented
- search acts like multi-rollout local-plan selection, not true global planning

Recommended additive split:

- global planner:
  - around 5 calls / trajectory
  - scene-overview heavy
  - phase + corridor + milestone waypoints
- local planner:
  - every segment
  - wrist-camera heavy
  - short-horizon corrections

This split is **designed in this audit but not fully implemented**. Current code remains segment-local by default.
