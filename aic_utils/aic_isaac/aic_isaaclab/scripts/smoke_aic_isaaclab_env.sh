#!/usr/bin/env bash
set -euo pipefail

cd "${ISAACLAB_ROOT:-/workspace/isaaclab}"

TASK_ID="${TASK_ID:-AIC-Task-v0}"
NUM_ENVS="${NUM_ENVS:-1}"
DEVICE="${DEVICE:-cuda:0}"
export AIC_ISAAC_RANDOMIZATION_PROFILE="${AIC_ISAAC_RANDOMIZATION_PROFILE:-none}"
export AIC_ISAAC_DISABLE_CAMERAS="${AIC_ISAAC_DISABLE_CAMERAS:-0}"

PYTHONUNBUFFERED=1 ./isaaclab.sh -p - "$TASK_ID" "$NUM_ENVS" "$DEVICE" <<'PY'
import sys

from isaaclab.app import AppLauncher

task_id = sys.argv[1]
num_envs = int(sys.argv[2])
device = sys.argv[3]

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app
status = 1
try:
    import gymnasium as gym
    import torch

    import aic_task.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(task_id, device=device, num_envs=num_envs)
    env_cfg.episode_length_s = 0.1
    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    obs, info = env.reset()
    action_dim = env.unwrapped.action_manager.total_action_dim
    action = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
    result = env.step(action)
    print(f"obs_type={type(obs).__name__} info_keys={list(info.keys()) if isinstance(info, dict) else type(info)}")
    print(f"action_dim={action_dim} step_result_len={len(result)}")
    env.close()
    print("AIC IsaacLab env smoke OK")
    status = 0
finally:
    simulation_app.close()

raise SystemExit(status)
PY
