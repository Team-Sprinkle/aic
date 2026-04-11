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
- Milestone 3: partial, via official-schema randomization
- Milestone 4: partial, via named reward terms and official-like score summary
- Milestones 5-7: interface scaffolding added, live Gazebo integration still pending

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
```

## What is real today

- deterministic `reset(seed=...)`
- exact synchronous `step(action)` over a configured number of ticks
- stable state-only observation dictionary
- reward decomposition with named terms
- official-like final score decomposition
- parity harness for offline rollout CSV comparison

## What is still approximate

- the current tested backend is deterministic and simulator-free
- ScenarIO + gym-gz integration is isolated behind `ScenarioGymGzBackend`
- Gazebo-native image extraction is defined as an interface, not yet wired
- parity against the official ROS path still needs live trace capture in a Gazebo-enabled environment

See [docs/architecture.md](/home/ubuntu/ws_aic/src/aic/aic_gym_gz/docs/architecture.md).
