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

## What is real today

- deterministic `reset(seed=...)`
- exact synchronous `step(action)` over a configured number of ticks
- stable state-only observation dictionary
- reward decomposition with named terms
- official-like final score decomposition
- live fixed-rollout parity against the official toolkit in state-only and state+image modes
- live benchmark reports for the official control path versus the `aic_gym_gz` attached replay path
- deterministic state-only parity regression artifacts under `artifacts/deterministic_policy_state`

## What is still approximate

- the current tested backend is deterministic and simulator-free
- the default `make_default_env()` path is still deterministic and simulator-free for tests
- the live backend is routed through `aic_utils/aic_gazebo_env`, not upstream ScenarIO / gym-gz
- image ingestion currently uses a dedicated ROS bridge sidecar fallback rather than pure Gazebo Transport
- the official reset metric is a readiness surrogate on this machine because `/gz_server/reset_simulation` still destabilizes the official bringup
- repeated env-style live training startup is still less stable than the fixed-rollout parity path; the deterministic parity gate is currently the stronger readiness check

See [docs/architecture.md](/home/ubuntu/ws_aic/src/aic/aic_gym_gz/docs/architecture.md).
