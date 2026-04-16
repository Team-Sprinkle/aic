# aic_gym_gz

`aic_gym_gz` is a standalone Gym-style training path for the AIC cable insertion
challenge.

It is separate from the official ROS-first evaluation flow. The package reuses:

- the official trial YAML schema from `aic_engine/config/sample_config.yaml`
- the official task board and cable geometry definitions from `aic_description`
- the official scoring shape from `aic_scoring/src/ScoringTier2.cc`

Current status:

- Milestone 1: implemented
- Milestone 2: implemented with a deterministic state-only backend
- Milestone 3: implemented for the current live fixed-rollout path
- Milestone 4: implemented with named reward terms and official-like score summary
- Milestone 5: implemented with live wrist-camera ingestion through an isolated ROS sidecar fallback
- Milestone 6: implemented for fixed-rollout parity against the official toolkit
- Milestone 7: implemented with live benchmark reports under `artifacts/`

The live target architecture is:

- `AicGazeboRuntime`: owns synchronous stepping and exact tick advancement
- `AicInsertionTask`: action space, observation space, reward, done, truncation
- `AicGazeboIO`: Gazebo-native observations and action application
- `AicEnvRandomizer`: official-schema-aligned scenario randomization
- `AicParityHarness`: rollout comparison tooling
- `teacher/`: training-time agent-teacher orchestration, replay, and temporal diagnostics
- `planners/`: planner backend contracts plus deterministic and OpenAI scaffolds
- `trajectory/`: dense segment smoothing from sparse subgoals
- `probes/`: safe micro-probes for cable-dynamics estimation

## Quick start

```bash
pixi run python -m aic_gym_gz.demo_random_policy
pixi run python -m unittest discover -s aic_gym_gz/tests
pixi run python -m aic_gym_gz.benchmark
pixi run python -m aic_gym_gz.live_benchmark
pixi run python -m aic_gym_gz.deterministic_policy_parity
pixi run python -m aic_gym_gz.live_training_smoke
pixi run python -m aic_gym_gz.demo_teacher_rollout --output aic_gym_gz/artifacts/teacher_rollout.json
pixi run python -m aic_gym_gz.replay_teacher_artifact --artifact aic_gym_gz/artifacts/teacher_rollout.json
pixi run python -m aic_gym_gz.compare_teacher_replay --artifact aic_gym_gz/artifacts/teacher_rollout.json
pixi run python -m aic_gym_gz.run_teacher_audit --output aic_gym_gz/artifacts/teacher_audit.json
pixi run python -m aic_gym_gz.run_teacher_search --output aic_gym_gz/artifacts/teacher_search.json
pixi run python -m aic_gym_gz.export_teacher_official_replay --search-artifact aic_gym_gz/artifacts/teacher_search.json --output aic_gym_gz/artifacts/teacher_selected_replay.json
pixi run python -m aic_gym_gz.export_teacher_dataset --search-artifact aic_gym_gz/artifacts/teacher_search.json --output-dir aic_gym_gz/artifacts/teacher_dataset --format jsonl
```

## How to use for RL

Recommended workflow:

1. Launch the official Gazebo+ROS stack in the container and wait until `aic_controller` is active.
2. Run the deterministic parity gate once.
3. Run the training smoke script.
4. Start your learner against `make_live_env(...)`.

Recommended live configuration today:

- `attach_to_existing=True`
- `transport_backend="cli"`
- `include_images=False` for first training runs
- enable images only after state-only training is stable

Minimal Gym-style usage:

```python
import numpy as np

from aic_gym_gz.env import make_live_env

env = make_live_env(
    include_images=False,
    enable_randomization=False,
    attach_to_existing=True,
    transport_backend="cli",
    ticks_per_step=8,
)

observation, info = env.reset(seed=123)

done = False
while not done:
    action = np.zeros(6, dtype=np.float32)
    observation, reward, terminated, truncated, step_info = env.step(action)
    done = terminated or truncated

env.close()
```

Observation keys in state-only mode:

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
- `sim_tick`
- `sim_time`

Additional keys in image mode:

- `images["left"]`
- `images["center"]`
- `images["right"]`
- `image_timestamps`

Useful pre-training checks:

```bash
pixi run python -m aic_gym_gz.deterministic_policy_parity
pixi run python -m aic_gym_gz.live_training_smoke
pixi run python -m aic_gym_gz.live_training_smoke --include-images
```

Artifacts worth checking:

- deterministic parity: `artifacts/deterministic_policy_state/`
- training smoke: `artifacts/training_smoke/`
- live benchmark: `artifacts/live_benchmark_state.json` and `artifacts/live_benchmark_image.json`
- teacher rollout replay: `artifacts/teacher_rollout.json`

## Agent teacher stack

The teacher stack is intentionally additive and namespaced under `aic_gym_gz`.
It is meant for trajectory generation, diagnostics, and future dataset creation,
not direct leaderboard submission.

Current components:

- `teacher/history.py`: rolling temporal buffer with derived cable-dynamics metrics
- `teacher/context.py`: teacher-mode oracle context assembly separate from policy observations
- `probes/library.py`: safe hold, sweep, wiggle, and lift probes
- `planners/mock.py`: deterministic planner backend for tests and smoke runs
- `planners/openai_backend.py`: OpenAI planner scaffold using `OPENAI_API_KEY`
- `trajectory/smoothing.py`: minimum-jerk segment densification
- `teacher/policy.py`: hierarchical teacher controller with branch-and-evaluate plan selection
- `teacher/scoring.py`: official-style teacher candidate scorer
- `teacher/search.py`: candidate generation, rollout search, ranking, and near-perfect selection
- `teacher/dataset_export.py`: JSONL and LeRobot-compatible export adapters
- `teacher/official_replay.py`: selected-trajectory loader for official replay
- `teacher/replay.py`: replay artifact serialization and comparison helpers
- `teacher/runner.py`: end-to-end rollout driver

Minimal demo path:

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
pixi run python -m aic_gym_gz.run_teacher_audit --output aic_gym_gz/artifacts/teacher_audit.json
pixi run python -m aic_gym_gz.run_teacher_search --output aic_gym_gz/artifacts/teacher_search.json
pixi run python -m aic_gym_gz.export_teacher_official_replay \
  --search-artifact aic_gym_gz/artifacts/teacher_search.json \
  --output aic_gym_gz/artifacts/teacher_selected_replay.json
pixi run python -m aic_gym_gz.export_teacher_dataset \
  --search-artifact aic_gym_gz/artifacts/teacher_search.json \
  --output-dir aic_gym_gz/artifacts/teacher_dataset \
  --format jsonl
```

OpenAI planner notes:

- Do not commit API keys or planner secrets.
- The scaffold reads the key from `OPENAI_API_KEY` only.
- `OpenAIPlannerBackend.build_request_payload(...)` produces the compact planning payload.
- The actual API call and strict JSON response parsing are intentionally left as a follow-up integration.

Current determinism limitation:

- Exact reset-level replay works on the deterministic mock backend.
- Exact intermediate-state cloning is not available in the live Gazebo path today.
- Branch-and-evaluate therefore uses deterministic planner variants and reset/replay patterns, not true simulator forks.

Official replay policy path:

```bash
pixi run python -m aic_gym_gz.export_teacher_official_replay \
  --search-artifact aic_gym_gz/artifacts/teacher_search.json \
  --output aic_gym_gz/artifacts/teacher_selected_replay.json

export AIC_TEACHER_REPLAY_ARTIFACT=/home/ubuntu/ws_aic/src/aic/aic_gym_gz/artifacts/teacher_selected_replay.json
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.TeacherReplayPolicy
```

## What is real today

- deterministic `reset(seed=...)`
- exact synchronous `step(action)` over a configured number of ticks
- stable state-only observation dictionary
- reward decomposition with named terms
- official-like final score decomposition
- live fixed-rollout parity against the official toolkit in state-only and state+image modes
- live benchmark reports for the official control path versus the `aic_gym_gz` attached replay path
- deterministic state-only parity regression artifacts under `artifacts/deterministic_policy_state`
- state-only and image-mode live smoke artifacts under `artifacts/training_smoke`
- teacher rollout, replay, probe, and temporal diagnostics utilities for training-time teacher generation
- candidate search with ranked top-K and explicit near-perfect thresholding
- official-policy replay conversion through `TeacherReplayPolicy`
- dataset export to JSONL and LeRobot-compatible formats with sidecar metadata

## What is still approximate

- the current tested backend is deterministic and simulator-free
- the default `make_default_env()` path is still deterministic and simulator-free for tests
- the live backend is routed through `aic_utils/aic_gazebo_env`, not upstream ScenarIO / gym-gz
- image ingestion currently uses a dedicated ROS bridge sidecar fallback rather than pure Gazebo Transport
- the official reset metric is a readiness surrogate on this machine because `/gz_server/reset_simulation` still destabilizes the official bringup
- repeated env-style live training startup is still less stable than the fixed-rollout parity path; the deterministic parity gate is currently the stronger readiness check
- live branch-and-evaluate does not yet support exact simulator checkpoint/restore at arbitrary intermediate states
- the teacher planner path still lacks full official `controller_state` and `CameraInfo` parity
- partial insertion scoring remains approximate without the official port-entrance TF in gym artifacts
- LeRobot compatibility is schema-compatible, but controller-specific error fields are synthesized from teacher artifacts

See [docs/architecture.md](/home/ubuntu/ws_aic/src/aic/aic_gym_gz/docs/architecture.md).
See [docs/agent_teacher.md](/home/ubuntu/ws_aic/src/aic/aic_gym_gz/docs/agent_teacher.md).
See [docs/agent_teacher_audit.md](/home/ubuntu/ws_aic/src/aic/aic_gym_gz/docs/agent_teacher_audit.md).
