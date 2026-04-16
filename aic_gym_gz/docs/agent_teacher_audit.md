# Agent Teacher Audit

This audit summarizes parity gaps between the current teacher path and the
official evaluation path, plus dataset-export compatibility.

## Observation parity

| Official observation item | Current gym/teacher availability | Exact source file/function | Status |
|---|---|---|---|
| Wrist camera images | Live teacher path receives images through ROS sidecar; mock path uses placeholders | `aic_gym_gz/io.py:RosCameraSidecarIO.observation_from_state`, `aic_gym_gz/teacher/runner.py:_observation_summary` | Matched |
| Image timestamps | Stored in env obs, temporal buffer, and planner state | `aic_gym_gz/io.py:RosCameraSidecarIO.observation_from_state`, `aic_gym_gz/teacher/context.py:TeacherContextExtractor.build_planning_state` | Matched |
| Wrist wrench | Synthetic in mock path; live gym runtime currently does not yet propagate official wrench end-to-end | `aic_gym_gz/runtime.py:ScenarioGymGzBackend._runtime_state_from_observation`, `aic_gym_gz/runtime.py:MockStepperBackend.step_ticks` | Approximate |
| TCP pose | Present in teacher obs and replay logs | `aic_gym_gz/runtime.py:_runtime_state_from_observation`, `aic_gym_gz/io.py:_base_observation` | Matched |
| TCP velocity | Present in teacher obs and replay logs | `aic_gym_gz/runtime.py:_runtime_state_from_observation`, `aic_gym_gz/io.py:_base_observation` | Matched |
| Joint states | Present in teacher obs and replay logs | `aic_gym_gz/runtime.py:_runtime_state_from_observation`, `aic_gym_gz/io.py:_base_observation` | Matched |
| Official `controller_state` | Not exposed directly in planner state | `aic_interfaces/aic_model_interfaces/msg/Observation.msg`, `aic_adapter/src/aic_adapter.cpp` | Missing |
| CameraInfo | Not exposed in planner state | `aic_interfaces/aic_model_interfaces/msg/Observation.msg`, `aic_adapter/src/aic_adapter.cpp` | Missing |
| Teacher oracle board/plug/target context | Explicitly exposed only in teacher mode | `aic_gym_gz/teacher/context.py` | Matched |

Notes:

- The smallest end-to-end extension needed for the teacher planner path was to
  preserve image timestamps and compact image summaries in the temporal buffer
  and planning state. That is now implemented.
- Full official observation parity would require bringing controller-state and
  camera-info semantics into the teacher observation pipeline, not just the gym
  state dict.

## Reward / scoring parity

| Official metric/term | Current gym implementation | Exact source file/function | Status |
|---|---|---|---|
| Tier 2 jerk / smoothness | Teacher selection uses official-style quadratic-window jerk; env reward still uses a simpler approximation | `aic_scoring/src/ScoringTier2.cc:GetTrajectoryJerkScore`, `aic_gym_gz/teacher/scoring.py`, `aic_gym_gz/reward.py:_average_linear_jerk` | Approximate |
| Duration | Thresholds and interpolation match docs / scorer | `aic_scoring/src/ScoringTier2.cc:GetTaskDurationScore`, `aic_gym_gz/teacher/scoring.py` | Matched |
| Trajectory efficiency | Thresholds and interpolation match docs / scorer | `aic_scoring/src/ScoringTier2.cc:GetTrajectoryEfficiencyScore`, `aic_gym_gz/teacher/scoring.py` | Matched |
| Insertion force penalty | Formula matches, but live accuracy depends on wrench availability | `aic_scoring/src/ScoringTier2.cc:GetInsertionForceScore`, `aic_gym_gz/teacher/scoring.py` | Approximate |
| Off-limit contact penalty | Matches | `aic_scoring/src/ScoringTier2.cc:GetContactsScore`, `aic_gym_gz/teacher/scoring.py` | Matched |
| Successful insertion / wrong port | Matches | `aic_scoring/src/ScoringTier2.cc:ComputeTier3Score`, `aic_gym_gz/task.py:evaluate_step`, `aic_gym_gz/teacher/scoring.py` | Matched |
| Partial insertion depth | Approximated from target pose and configured depth because official port entrance TF is unavailable in gym artifacts | `aic_scoring/src/ScoringTier2.cc:GetDistanceScore`, `aic_gym_gz/teacher/scoring.py` | Approximate |
| Tier 1 validity | Assumed locally, still official-only in real submission loop | `docs/scoring.md`, `aic_model/aic_model/aic_model.py` | Approximate |

Notes:

- Exact parity is not currently possible for partial insertion because the gym
  teacher path does not expose the official port entrance transform used by
  `aic_scoring`.
- Candidate search now uses the most official-faithful local score available and
  records parity notes inside each ranked candidate.

## Dataset export compatibility

| Target export path | Current status | Exact source file/function | Status |
|---|---|---|---|
| exp/data LeRobot schema family | Teacher exporter uses the same observation/action key family as `policy_recorder` from `origin/exp/data` | `origin/exp/data:aic_utils/lerobot_robot_aic/lerobot_robot_aic/policy_recorder.py`, `aic_gym_gz/teacher/dataset_export.py` | Matched |
| Rich teacher metadata | Exported as sidecar metadata JSON | `aic_gym_gz/teacher/dataset_export.py` | Matched |
| Controller error fields | Synthesized from commanded target vs observed TCP | `origin/exp/data:aic_utils/lerobot_robot_aic/lerobot_robot_aic/policy_recorder.py`, `aic_gym_gz/teacher/dataset_export.py` | Approximate |
| Teleop/policy dataset merge checks | Compatibility tool restored into current branch | `aic_utils/lerobot_robot_aic/lerobot_robot_aic/validate_dataset_compatibility.py` | Matched |

Notes:

- `aic_training_interfaces` / `aic_training_utils` from `exp/data` appear to be
  bringup helpers, not dataset schema owners, so the direct reuse point is the
  `lerobot_robot_aic` dataset path.
- Teacher export now supports:
  1. JSONL with full teacher metadata per step
  2. Native LeRobot-compatible dataset export with sidecar metadata

## Official replay limitations

- The selected replay path is `aic_model`-compatible and publishes normal
  `MotionUpdate` velocity commands through `TeacherReplayPolicy`.
- This is robust for replay-style execution, but it is still approximate with
  respect to exact original controller timing because the official controller
  closes the loop on its own state and scheduling.
- Intermediate live simulator checkpoint/restore remains unavailable, so search
  ranking is exact only on deterministic reset/replay paths.
