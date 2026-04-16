# Gazebo-Gym Runtime Parity

This branch upgrades the foundational `aic_gym_gz` path so it is closer to the
official runtime and scoring interfaces before teacher-layer features are stacked
on top.

## Fixed in this branch

- Live wrench parity is improved by subscribing to `/fts_broadcaster/wrench` and subtracting the controller FT tare offset when available.
- Controller-state semantics are propagated into the gym observation path from `/aic_controller/controller_state`.
- CameraInfo semantics are propagated alongside wrist images when `include_images=True`.
- The runtime now attempts to resolve the actual task port and port-entrance entities instead of treating `tabletop` as the scoring target.
- `rl_step_reward` is now a dense Isaac-Lab-style local training reward with explicit per-step reward terms.
- `gym_final_score` remains a separate trajectory-level local score path using duration, efficiency, contact, insertion-force, and partial-insertion logic as closely as the local runtime allows.
- Mock runtime checkpoint/restore is exact. Live runtime checkpoint export is reset-and-rerun only and is explicitly labeled approximate.

## Remaining gaps

- `official_eval_score` is still not computed inside `aic_gym_gz`; that requires running the official toolkit path.
- Live midpoint restore is still unavailable because the Gazebo training transport path does not expose a world snapshot / restore service.
- Jerk parity is improved but still uses env-side velocity history rather than the official scorer's TF-buffer implementation.
- Observation parity still depends on ROS topics being present for wrench, controller state, contacts, and camera_info.

## Score labels

- `rl_step_reward`: dense local RL training reward returned by `env.step()`
- `gym_final_score`: local gazebo-gym final episode score reported by `final_evaluation()`
- `gym_reward`: umbrella label for the local gazebo-gym reward/scoring family in `aic_gym_gz`
- `teacher_official_style_score`: higher-level teacher-side approximation on `feat/agent-teacher`
- `official_eval_score`: actual official toolkit evaluation path
