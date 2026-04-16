# Agent Teacher

This document describes the training-time agent-teacher stack added on top of
the existing `feat/gazebo-gym-scn` architecture.

## Design goals

- keep changes additive and namespaced
- expose normal rollout-time context plus teacher-only oracle context
- plan in segments instead of per-tick actions
- use temporal windows rather than single frames
- support active probing to estimate cable dynamics
- export replayable artifacts for later data generation and comparison

## Package layout

- `aic_gym_gz/teacher/history.py`
  Rolling history of runtime state, action history, optional image references,
  timestamps, and derived cable-dynamics metrics.
- `aic_gym_gz/teacher/context.py`
  Builds a compact `TeacherPlanningState` with:
  policy-visible context, task metadata, oracle scene context, obstacle
  summaries, recent probe results, and dynamics summaries.
- `aic_gym_gz/probes/library.py`
  Safe micro-probes:
  `hold_settle`, `micro_sweep_xy`, `yaw_wiggle`, `lift_and_hold`.
- `aic_gym_gz/planners/base.py`
  Structured planner interface.
- `aic_gym_gz/planners/mock.py`
  Deterministic backend for tests and smoke runs.
- `aic_gym_gz/planners/openai_backend.py`
  OpenAI backend scaffold using `OPENAI_API_KEY`.
- `aic_gym_gz/trajectory/smoothing.py`
  Minimum-jerk densification from sparse waypoints to low-jerk dense segments.
- `aic_gym_gz/teacher/policy.py`
  Hierarchical controller with phase tracking and branch-and-evaluate candidate selection.
- `aic_gym_gz/teacher/replay.py`
  Replay artifact format, serializer, replay runner, and comparison helper.
- `aic_gym_gz/teacher/runner.py`
  End-to-end teacher rollout driver.

## Rollout flow

1. Reset `AicInsertionEnv` with a deterministic seed and optional `trial_id`.
2. Seed `TemporalObservationBuffer` from the initial `RuntimeState`.
3. Build `TeacherPlanningState` from:
   rollout context, oracle context, obstacle summaries, dynamics summaries, and recent probe results.
4. Query one or more planner candidates through the planner backend.
5. Smooth each sparse plan to a dense low-jerk segment.
6. Score the candidates and keep the best segment.
7. If cable settling looks poor or the planner requests it, execute a probe.
8. Execute the selected dense segment, logging timing and dynamics information.
9. Save the rollout as a replay artifact for deterministic mock replay or best-effort live replay.

## Temporal metrics

The temporal buffer computes:

- plug oscillation magnitude
- cable settling score
- recent motion energy
- quasi-static detection
- time since last significant cable motion
- wrench energy

These metrics are intentionally compact. They are designed for planner prompts
and diagnostics rather than direct policy learning tensors.

## Planner contract

Input is a compact structured planning state, not raw image tensors or full
simulator dumps.

Required output fields:

- `next_phase`
- `waypoints`
- `motion_mode`
- `caution_flag`
- `should_probe`
- `segment_horizon_steps`
- `segment_granularity`
- `rationale_summary`

The planner does not emit per-tick actions directly.

## OpenAI integration scaffold

The current OpenAI backend is intentionally a scaffold.

- Secrets are read from `OPENAI_API_KEY` only.
- No keys are written to config files, artifacts, or logs.
- `OpenAIPlannerBackend.build_request_payload(...)` serializes the compact planning state.
- The actual API invocation and strict response parsing are left as a follow-up.

Recommended next step:

1. Add a small adapter that calls the OpenAI Responses API.
2. Enforce a strict JSON schema for planner output.
3. Map the response into `TeacherPlan`.
4. Keep planner prompts and tool settings in runtime config, not committed secrets.

## Replay and determinism

Current replay guarantees:

- deterministic mock backend:
  exact reset plus replay of dense segments
- live Gazebo path:
  same scenario/settings replay and comparison, but not exact arbitrary mid-rollout restore

Current limitation:

- exact simulator checkpoint/restore at arbitrary intermediate states is not
  implemented in the current live architecture

Practical approximation implemented now:

- deterministic reset from seed and scenario
- action/segment replay
- deterministic planner candidate branching
- explicit limitation recorded in replay artifacts

## Commands

```bash
pixi run python -m aic_gym_gz.demo_teacher_rollout \
  --output aic_gym_gz/artifacts/teacher_rollout.json

pixi run python -m aic_gym_gz.record_teacher_temporal_diagnostics --steps 8
pixi run python -m aic_gym_gz.run_teacher_probe_experiment --probe micro_sweep_xy
pixi run python -m aic_gym_gz.replay_teacher_artifact \
  --artifact aic_gym_gz/artifacts/teacher_rollout.json
pixi run python -m aic_gym_gz.compare_teacher_replay \
  --artifact aic_gym_gz/artifacts/teacher_rollout.json
pixi run python -m aic_gym_gz.benchmark_teacher_planner --episodes 3
pixi run python -m unittest discover -s aic_gym_gz/tests -p 'test_teacher*.py'
```
