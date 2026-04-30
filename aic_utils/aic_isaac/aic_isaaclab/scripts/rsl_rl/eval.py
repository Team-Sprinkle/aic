# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an RSL-RL checkpoint for the AIC IsaacLab task."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint for AIC.")
parser.add_argument("--video", action="store_true", default=False, help="Record one evaluation video.")
parser.add_argument("--video_length", type=int, default=300, help="Length of the recorded video in steps.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--num_episodes", type=int, default=8, help="Number of completed episodes to evaluate.")
parser.add_argument("--max_steps", type=int, default=4096, help="Maximum vector-env steps before stopping.")
parser.add_argument("--task", type=str, default="AIC-Task-v0", help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--seed", type=int, default=1, help="Seed used for the environment.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import aic_task.tasks  # noqa: F401


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _get_reaching_term(base_env):
    reward_manager = getattr(base_env.unwrapped, "reward_manager", None)
    if reward_manager is None:
        return None
    try:
        return reward_manager.get_term_cfg("reaching_bonus")
    except Exception:
        return None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Evaluate an RSL-RL policy checkpoint."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(base_env.unwrapped, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "eval"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording evaluation video.")
        print_dict(video_kwargs, nesting=4)
        base_env = gym.wrappers.RecordVideo(base_env, **video_kwargs)

    reaching_term = _get_reaching_term(base_env)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    obs = env.get_observations()
    num_envs = args_cli.num_envs
    current_rewards = torch.zeros(num_envs, device=env.unwrapped.device)
    current_lengths = torch.zeros(num_envs, dtype=torch.long, device=env.unwrapped.device)
    current_reached = torch.zeros(num_envs, dtype=torch.bool, device=env.unwrapped.device)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_reached: list[float] = []
    reaching_step_hits = 0
    reaching_step_total = 0

    steps = 0
    while simulation_app.is_running() and steps < args_cli.max_steps and len(episode_rewards) < args_cli.num_episodes:
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, _ = env.step(actions)
            policy_nn.reset(dones)

            rewards = rewards.reshape(-1)
            dones = dones.reshape(-1).bool()
            current_rewards += rewards
            current_lengths += 1

            if reaching_term is not None:
                reached = reaching_term.func(base_env.unwrapped, **reaching_term.params).reshape(-1).bool()
                current_reached |= reached
                reaching_step_hits += int(reached.sum().item())
                reaching_step_total += int(reached.numel())

            done_ids = torch.nonzero(dones, as_tuple=False).flatten()
            for env_id in done_ids.tolist():
                episode_rewards.append(float(current_rewards[env_id].item()))
                episode_lengths.append(int(current_lengths[env_id].item()))
                episode_reached.append(1.0 if bool(current_reached[env_id].item()) else 0.0)
                current_rewards[env_id] = 0.0
                current_lengths[env_id] = 0
                current_reached[env_id] = False
                if len(episode_rewards) >= args_cli.num_episodes:
                    break

        steps += 1
        if args_cli.video and steps >= args_cli.video_length:
            break

    metrics = {
        "checkpoint": resume_path,
        "task": args_cli.task,
        "num_envs": num_envs,
        "target_episodes": args_cli.num_episodes,
        "completed_episodes": len(episode_rewards),
        "vector_env_steps": steps,
        "average_reward": _mean(episode_rewards),
        "average_episode_length": _mean([float(length) for length in episode_lengths]),
        "reaching_episode_rate": _mean(episode_reached),
        "reaching_step_rate": (reaching_step_hits / reaching_step_total if reaching_step_total else None),
        "video_recorded": bool(args_cli.video),
    }
    print("AIC_EVAL_METRICS " + json.dumps(metrics, sort_keys=True))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
