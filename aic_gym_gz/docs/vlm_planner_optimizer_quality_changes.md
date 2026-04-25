# VLM Planner And Optimizer Quality Changes

This note documents the trajectory-quality changes made after the stored GPT-5
failure analysis for
`outputs/trajectory_datasets/sfp_to_nic/vlm_planner/nic_cards_2/n1/trial2_2026_0425_193412`.

## Feedback Addressed

The feedback identified these primary trajectory defects:

- Planning and commands were effectively world-frame only, with no enforced
  port-entrance-frame sequence.
- The VLM/optimizer could advance toward insertion while lateral and yaw errors
  were still large.
- TCP and plug-tip roles were easy to conflate.
- Yaw near wraparound could be interpolated the long way.
- The trajectory lacked explicit align, entry, and insert gates.

## Planner Changes

The OpenAI planner prompt now explicitly requires the following near-port rule:

1. Use `current_observation.relative_geometry.port_frame_error` as the primary
   near-port coordinate system.
2. Treat waypoint `position_xyz` values as plug-tip targets in Gazebo world
   coordinates, not TCP targets.
3. Do not choose `guarded_insert` unless both lateral error and yaw error are
   below threshold.
4. If the alignment gate is not satisfied, choose `pre_insert_align` at
   `pre_insertion_waypoint_world_xyz` with `target_yaw` and a low speed scale.

The planner backend also applies a deterministic post-parse guard:

- If the VLM emits `guarded_insert` or another near-port insertion segment while
  lateral/yaw alignment is not satisfied, the backend rewrites it to a cautious
  `pre_insert_align` plan.
- The rewrite is recorded in `decision_diagnostics.assumptions_used` and
  `decision_diagnostics.blocking_gaps`, so traces show that the guard acted.

This keeps the VLM responsible for coarse/global planning while preventing a
single bad local segment from bypassing the port-frame gates.

## Optimizer Changes

`MinimumJerkSmoother` now has a port-frame alignment gate for near-port phases.
It uses:

- `target_port_entrance_pose` as the port-frame origin.
- The vector from `target_port_entrance_pose` to `target_port_pose` as the
  positive insertion axis.
- The observed plug-to-TCP offset to convert plug-tip targets into TCP targets.

For `pre_insert_align` and `guarded_insert` segments, the optimizer rewrites
sparse targets into a gated sequence:

1. If lateral or yaw error is too large, target the pre-insertion standoff:
   `entrance - 0.025 * insertion_axis`.
2. If already aligned and the phase is `guarded_insert`, target guarded entry:
   `entrance - 0.005 * insertion_axis`.
3. Then advance only a small amount along the positive insertion axis.

The optimizer also normalizes yaw interpolation with shortest-path wraparound,
so yaw near `pi` does not produce a large rotation in the wrong direction.

Near the final target, the guarded insertion speed limit is reduced to
`0.010 m/s`. This implements the feedback request for slow axial motion during
the actual insertion portion.

Gated `pre_insert_align` segments also enforce a minimum transit speed scale of
`0.8` after the optimizer rewrites a bad near-port sparse target. This prevents
a cautious VLM speed scale from consuming the entire rollout budget before the
plug reaches the pre-insert standoff.

After `trial10_2026_0425_210851`, GPT-5 feedback and the live rollout showed a
different failure mode: the optimizer produced the right port-frame standoff,
but the open-loop TCP delta reached `0.12 m/s` near the held cable and the
measured plug tip swung out of the corridor. To address that, unaligned
port-frame standoff motion now has its own speed cap:

- `port_alignment_unaligned_speed_limit = 0.030 m/s`
- this cap is used both when computing the dense horizon and when clipping each
  Cartesian action
- the cap is recorded as `segment_linear_speed_limit_mps`

This keeps the VLM at coarse Cartesian planning intervals while making the
optimizer responsible for slow, smooth, stable plug-tip alignment before any
guarded insertion or cheatcode handoff.

After `trial14_2026_0425_221105`, the optimized target path was correct in
port-frame terms but live Gazebo still under-tracked the open-loop velocity
deltas. The rollout executor now applies a closed-loop Cartesian tracker for
optimizer-rewritten port-frame alignment segments:

- each `env.step()` compares the latest observed `state.tcp_pose` with the
  current dense `target_tcp_pose`
- the executed action is a clipped velocity toward that dense target
- the original smoother action is retained as `planned_action`
- the dataset action is the actual `executed_action`
- tracker metadata is stored under `trajectory_point.feedback_tracker`

This keeps the VLM sparse plan and minimum-jerk dense target sequence intact,
while making execution robust to live Gazebo cable/tool-link under-tracking.

## Hybrid Handoff Changes

The deterministic close-range policy is now gated to the actual insertion
neighborhood:

- distance to entrance must be below `0.055 m` or distance to target below
  `0.060 m`
- lateral misalignment must be below `0.010 m`
- orientation error must be below `0.12 rad`

This keeps the VLM planner plus trajectory optimizer responsible for coarse and
pre-insert alignment, and reserves the cheatcode-like policy for the near-target
insertion phase.

## Artifact Metadata

Each generated `TrajectorySegment` now records:

- `conversion_metadata.port_frame_alignment_gate.active`
- `gate_action`
- `coordinate_frame`
- `origin_world_xyz`
- `axis_unit_world_xyz`
- lateral error, axial depth, yaw error, and thresholds used
- final post-gate target lateral error, axial depth, and yaw error in the
  `port_entrance_frame`
- `guarded_insert_speed_limit_mps`, so summaries distinguish the guarded
  0.010 m/s insertion clamp from the less restrictive near-entrance approach
  clamp

This makes it possible to audit whether a run used VLM-only sparse targets or
whether the optimizer rewrote those targets through the port-frame gate.

## Current Limitation

These changes improve planner and optimizer quality, but live Gazebo plug-tip
dynamics are still the main residual risk. Dataset-quality conclusions must be
based on real Gazebo runs with wrist and overview videos, not schematic debug
renders.
