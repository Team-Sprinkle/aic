# IsaacLab Stage 5 Bringup

## Host prerequisites

- Docker and NVIDIA Container Toolkit are available.
- IsaacLab `v2.3.2` is cloned at `/home/ubuntu/IsaacLab`.
- The AIC checkout is mounted into the container at `/workspace/isaaclab/aic`.
- `Intrinsic_assets` exists under `aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/Intrinsic_assets`.

On this EC2 host the AIC checkout is bind-mounted with `/home/ubuntu/IsaacLab/docker/aic-bind.yaml`.

## Enter container

```bash
cd /home/ubuntu/IsaacLab
./docker/container.py start base --files aic-bind.yaml
./docker/container.py enter base --files aic-bind.yaml
```

## Install and verify

Inside the container:

```bash
cd /workspace/isaaclab
aic/aic_utils/aic_isaac/aic_isaaclab/scripts/install_aic_task.sh
aic/aic_utils/aic_isaac/aic_isaaclab/scripts/smoke_import_aic_task.sh
```

If `isaaclab` is not importable in a fresh container, `install_aic_task.sh` preinstalls `flatdict==4.0.1` without build isolation and reinstalls `source/isaaclab`.

## Discovered task IDs

- `AIC-Task-v0`
  - Registered in `source/aic_task/aic_task/tasks/manager_based/aic_task/__init__.py`
  - Entry point: `isaaclab.envs:ManagerBasedRLEnv`
  - Env config: `aic_task.tasks.manager_based.aic_task.aic_task_env_cfg:AICTaskEnvCfg`
  - RSL-RL config: `aic_task.tasks.manager_based.aic_task.agents.rsl_rl_ppo_cfg:PPORunnerCfg`

## Smoke environment

```bash
cd /workspace/isaaclab
aic/aic_utils/aic_isaac/aic_isaaclab/scripts/smoke_aic_isaaclab_env.sh
```

The smoke defaults to camera-enabled observations. If `AIC_ISAAC_DISABLE_CAMERAS`
is set to `1`, the camera-required training and evaluation entry points fail
instead of silently falling back to low-dimensional observations.

## Smoke PPO training

```bash
cd /workspace/isaaclab
aic/aic_utils/aic_isaac/aic_isaaclab/scripts/train_aic_isaaclab_ppo_smoke.sh
```

Defaults:
- `TASK_ID=AIC-Task-v0`
- `NUM_ENVS=1`
- `MAX_ITERATIONS=1`
- `AIC_ISAAC_RANDOMIZATION_PROFILE=none`
- `AIC_ISAAC_DISABLE_CAMERAS=0`

## Checkpoint evaluation

Use the finite RSL-RL evaluator for headless checkpoint metrics:

```bash
cd /workspace/isaaclab
CHECKPOINT=/workspace/isaaclab/aic/outputs/train/stage5_aic_lowdim_ppo/aic_task/2026-04-30_09-08-49_stage5_aic_lowdim_ppo/model_200.pt \
  NUM_ENVS=4 NUM_EPISODES=4 MAX_STEPS=6500 \
  aic/aic_utils/aic_isaac/aic_isaaclab/scripts/eval_aic_isaaclab_ppo.sh
```

The evaluator prints a single `AIC_EVAL_METRICS` JSON line with average reward,
average episode length, reaching episode rate, reaching step rate, and video
status. The default AIC episode timeout is `200s`, or about `6000` env steps.

## Known limitations

- Isaac logs warn that referenced GLB visuals in the USD cannot be opened as USD-format assets. The environment and camera PPO smoke still run.
- IsaacLab `v2.3.2` image build can miss `isaaclab` because `flatdict==4.0.1` fails under isolated build dependencies; the helper script repairs this in-container.

## Next steps

Use the smoke command above for readiness checks. For longer PPO runs, increase `NUM_ENVS`, `MAX_ITERATIONS`, and choose `AIC_ISAAC_RANDOMIZATION_PROFILE=light` or `heavy` after confirming the camera smoke remains stable.
