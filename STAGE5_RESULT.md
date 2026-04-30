# Stage 5 Result

## What succeeded

- IsaacLab `v2.3.2` is available at `/home/ubuntu/IsaacLab`.
- The AIC checkout is available in the running `isaac-lab-base` container at `/workspace/isaaclab/aic`.
- `Intrinsic_assets` exists at the required AIC task path.
- `aic_task` installs in editable mode and imports inside IsaacLab Python.
- `AIC-Task-v0` is registered after importing `aic_task.tasks`.
- A one-env headless environment smoke test resets and steps successfully with low-dimensional observations.
- A one-iteration RSL-RL PPO smoke run completes successfully with `AIC_ISAAC_DISABLE_CAMERAS=1`.
- A sustained lowdim PPO run with 64 environments reached iteration `200/1000` and was stopped intentionally after the user confirmed that was sufficient.

## What failed

- The default camera-enabled path is not validated on this EC2 host. Headless rendering startup with cameras enabled was slow/stalled and was stopped.
- The canonical training command without `AIC_ISAAC_DISABLE_CAMERAS=1` fails because the task spawns cameras but the command does not pass `--enable_cameras`.
- Isaac logs warn that several `.glb` visual references inside `aic_unified_robot_cable_sdf.usd` cannot be opened as USD-format assets. This did not block lowdim env or PPO smoke.

## Exact commands run

```bash
cd /workspace/isaaclab
./isaaclab.sh -p -m pip install --no-build-isolation flatdict==4.0.1
./isaaclab.sh -p -m pip install -e source/isaaclab
./isaaclab.sh -p -m pip install -e aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task
./isaaclab.sh -p -c "import aic_task; print('aic_task import OK')"
aic/aic_utils/aic_isaac/aic_isaaclab/scripts/smoke_import_aic_task.sh
aic/aic_utils/aic_isaac/aic_isaaclab/scripts/smoke_aic_isaaclab_env.sh
RUN_NAME=stage5_helper_smoke OUTPUT_DIR=/workspace/isaaclab/aic/outputs/train/isaac_stage5_helper_smoke \
  aic/aic_utils/aic_isaac/aic_isaaclab/scripts/train_aic_isaaclab_ppo_smoke.sh
AIC_ISAAC_DISABLE_CAMERAS=1 NUM_ENVS=64 MAX_ITERATIONS=1000 RUN_NAME=stage5_aic_lowdim_ppo \
  OUTPUT_DIR=/workspace/isaaclab/aic/outputs/train/stage5_aic_lowdim_ppo \
  aic/aic_utils/aic_isaac/aic_isaaclab/scripts/train_aic_isaaclab_ppo_smoke.sh
```

## Discovered task IDs

- `AIC-Task-v0`
  - Registration: `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/__init__.py`
  - Entry point: `isaaclab.envs:ManagerBasedRLEnv`
  - Env config: `aic_task.tasks.manager_based.aic_task.aic_task_env_cfg:AICTaskEnvCfg`
  - RSL-RL config: `aic_task.tasks.manager_based.aic_task.agents.rsl_rl_ppo_cfg:PPORunnerCfg`

## Smoke test results

- Import smoke: success, printed `aic_task import OK` and `['AIC-Task-v0']`.
- Env smoke: success.
  - Observation type: `dict`
  - Info keys: `['log']`
  - Action dimension: `6`
  - Step result length: `5`
  - Lowdim policy observation shape reported by IsaacLab: `(154,)`

## Training smoke result

- Result: success.
- Command: `train_aic_isaaclab_ppo_smoke.sh`
- Settings:
  - `TASK_ID=AIC-Task-v0`
  - `NUM_ENVS=1`
  - `MAX_ITERATIONS=1`
  - `SEED=1`
  - `AIC_ISAAC_RANDOMIZATION_PROFILE=none`
  - `AIC_ISAAC_DISABLE_CAMERAS=1`
- PPO initialized actor/critic MLPs with `154` observation inputs and `6` actions.
- One learning iteration completed.
- Total timesteps: `24`
- Training time reported by RSL-RL: about `3` seconds.
- Logs were written under `outputs/train/isaac_stage5_helper_smoke/aic_task`.

## Sustained PPO run result

- Result: success, stopped intentionally after sufficient validation.
- Settings:
  - `TASK_ID=AIC-Task-v0`
  - `NUM_ENVS=64`
  - `MAX_ITERATIONS=1000`
  - `SEED=1`
  - `RUN_NAME=stage5_aic_lowdim_ppo`
  - `AIC_ISAAC_DISABLE_CAMERAS=1`
- Reached learning iteration: `200/1000`
- Total timesteps: `308736`
- Elapsed training time at stop: about `12m24s`
- Output size: about `29M`
- Logs/checkpoints:
  - `outputs/train/stage5_aic_lowdim_ppo/aic_task/2026-04-30_09-08-49_stage5_aic_lowdim_ppo/events.out.tfevents...`
  - `model_0.pt`
  - `model_50.pt`
  - `model_100.pt`
  - `model_150.pt`
  - `model_200.pt`
  - `params/agent.yaml`
  - `params/env.yaml`

## Files changed

- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/__init__.py`
- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py`
- `aic_utils/aic_isaac/aic_isaaclab/scripts/install_aic_task.sh`
- `aic_utils/aic_isaac/aic_isaaclab/scripts/smoke_import_aic_task.sh`
- `aic_utils/aic_isaac/aic_isaaclab/scripts/smoke_aic_isaaclab_env.sh`
- `aic_utils/aic_isaac/aic_isaaclab/scripts/train_aic_isaaclab_ppo_smoke.sh`
- `aic_utils/aic_isaac/aic_isaaclab/docs/isaaclab_stage5_bringup.md`
- `STAGE5_RESULT.md`

## Remaining blockers

- Camera-enabled training still needs separate validation on a headless-rendering-capable setup or a faster EC2 rendering configuration.
- The USD `.glb` visual-reference warnings should be inspected before visual/camera training is considered production-ready.
- The Docker helper needs the local IsaacLab override `docker/aic-bind.yaml` on this host to mount `/home/ubuntu/ws_aic/src/aic` into `/workspace/isaaclab/aic`.

## Recommended next command for real training

Resume or extend lowdim PPO from the validated path:

```bash
cd /workspace/isaaclab
TASK_ID=AIC-Task-v0 NUM_ENVS=64 MAX_ITERATIONS=1000 RUN_NAME=stage5_aic_lowdim_ppo_continued \
  AIC_ISAAC_DISABLE_CAMERAS=1 \
  OUTPUT_DIR=/workspace/isaaclab/aic/outputs/train/stage5_aic_lowdim_ppo \
  aic/aic_utils/aic_isaac/aic_isaaclab/scripts/train_aic_isaaclab_ppo_smoke.sh
```
