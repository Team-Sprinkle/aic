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

## Quick start

```bash
pixi run python -m aic_gym_gz.demo_random_policy
pixi run python -m unittest discover -s aic_gym_gz/tests
pixi run python -m aic_gym_gz.benchmark
pixi run python -m aic_gym_gz.live_benchmark
pixi run python -m aic_gym_gz.deterministic_policy_parity
pixi run python -m aic_gym_gz.live_training_smoke
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
- `target_port_entrance_pose`
- `plug_to_port_relative`
- `wrench`
- `wrench_timestamp`
- `off_limit_contact`
- `controller_tcp_pose`
- `controller_reference_tcp_pose`
- `controller_tcp_velocity`
- `controller_tcp_error`
- `controller_reference_joint_state`
- `controller_target_mode`
- `fts_tare_wrench`
- `score_geometry`
- `sim_tick`
- `sim_time`

Additional keys in image mode:

- `images["left"]`
- `images["center"]`
- `images["right"]`
- `image_timestamps`
- `camera_info`

Runtime parity audit:

```bash
pixi run python -m aic_gym_gz.audit_runtime \
  --output-json /tmp/aic_gym_runtime_audit.json \
  --output-markdown /tmp/aic_gym_runtime_audit.md
```

Runtime checkpoint export:

```bash
pixi run python -m aic_gym_gz.export_runtime_checkpoint \
  --output /tmp/aic_gym_checkpoint.json
```

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

## What is still approximate

- the current tested backend is deterministic and simulator-free
- the default `make_default_env()` path is still deterministic and simulator-free for tests
- the live backend is routed through `aic_utils/aic_gazebo_env`, not upstream ScenarIO / gym-gz
- image ingestion currently uses a dedicated ROS bridge sidecar fallback rather than pure Gazebo Transport
- the official reset metric is a readiness surrogate on this machine because `/gz_server/reset_simulation` still destabilizes the official bringup
- repeated env-style live training startup is still less stable than the fixed-rollout parity path; the deterministic parity gate is currently the stronger readiness check

See [docs/architecture.md](/home/ubuntu/ws_aic/src/aic/aic_gym_gz/docs/architecture.md).
