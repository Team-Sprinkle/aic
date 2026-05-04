#!/usr/bin/env python3
"""Online ACT-adapter SERL/SAC-style training loop for AIC Isaac Lab."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="AIC-Task-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--act_torchscript", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="outputs/train/isaac_stage5_serl")
parser.add_argument("--run_name", default="stage5_online_serl")
parser.add_argument("--steps", type=int, default=64)
parser.add_argument("--updates", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--replay_capacity", type=int, default=10000)
parser.add_argument("--max_wall_time_minutes", type=float, default=0.0)
parser.add_argument("--log_every", type=int, default=10)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--adapter_lr", type=float, default=1e-4)
parser.add_argument("--critic_lr", type=float, default=1e-4)
parser.add_argument("--act_lr", type=float, default=1e-5)
parser.add_argument("--freeze_act", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--adapter_penalty_weight", type=float, default=1e-3)
parser.add_argument("--act_preservation_weight", type=float, default=1e-2)
parser.add_argument("--adapter_delta_clip", type=float, default=0.05)
parser.add_argument("--action_clip", type=float, default=0.05)
parser.add_argument("--bc_weight", type=float, default=0.0)
parser.add_argument("--cql_weight", type=float, default=0.0)
parser.add_argument("--state_source", choices=["lerobot_compatible", "policy_prefix"], default="lerobot_compatible")
parser.add_argument("--task_family", choices=["sfp_to_nic", "sc_to_sc"], default="sfp_to_nic")
parser.add_argument("--target_port_index", type=int, default=0)
parser.add_argument("--target_card_index", type=int, default=0)
parser.add_argument("--target_card_valid", type=int, default=1)
parser.add_argument("--gripper_joint_position", type=float, default=0.0035405)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
PROCESS_START_TIME = time.monotonic()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import aic_task.tasks  # noqa: F401


from torch import nn
from torch.nn import functional as F


CAMERA_KEYS = [
    "observation.images.center_camera",
    "observation.images.left_camera",
    "observation.images.right_camera",
]

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def _canonical_task_vector(args: argparse.Namespace, *, device: torch.device, batch_size: int) -> torch.Tensor:
    if args.task_family == "sfp_to_nic":
        family = [1.0, 0.0]
        if args.target_card_valid != 1:
            raise ValueError("sfp_to_nic Isaac task context requires target_card_valid=1")
        if args.target_card_index < 0 or args.target_card_index >= 5:
            raise ValueError("sfp_to_nic target_card_index must be in 0..4")
    elif args.task_family == "sc_to_sc":
        family = [0.0, 1.0]
        if args.target_card_valid != 0 or args.target_card_index != -1:
            raise ValueError("sc_to_sc Isaac task context requires target_card_index=-1 and target_card_valid=0")
    else:
        raise ValueError(f"Unsupported task_family: {args.task_family}")
    if args.target_port_index not in {0, 1}:
        raise ValueError("target_port_index must be 0 or 1")
    port = [1.0, 0.0] if args.target_port_index == 0 else [0.0, 1.0]
    card = [0.0] * 5
    if args.target_card_valid:
        card[args.target_card_index] = 1.0
    vector = torch.tensor(
        family + port + card + [float(args.target_card_valid)],
        dtype=torch.float32,
        device=device,
    )
    return vector.unsqueeze(0).expand(batch_size, -1)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data = deque(maxlen=capacity)

    def append(self, transition: dict[str, Any]) -> None:
        self.data.append({key: self._detach(value) for key, value in transition.items()})

    def _detach(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: self._detach(v) for k, v in value.items()}
        return value

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, Any]:
        indices = torch.randint(len(self.data), (batch_size,)).tolist()
        items = [self.data[i] for i in indices]
        return {
            "obs": self._stack_obs([item["obs"] for item in items], device),
            "next_obs": self._stack_obs([item["next_obs"] for item in items], device),
            "action": torch.stack([item["action"] for item in items]).to(device),
            "reward": torch.stack([item["reward"] for item in items]).to(device),
            "done": torch.stack([item["done"] for item in items]).to(device),
        }

    def _stack_obs(self, obs_items: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
        return {
            "state": torch.stack([item["state"] for item in obs_items]).to(device),
            "images": {
                key: torch.stack([item["images"][key] for item in obs_items]).to(device)
                for key in CAMERA_KEYS
            },
        }


def _policy_tensor(obs: Any) -> torch.Tensor:
    if isinstance(obs, dict):
        obs = obs.get("policy", obs)
    if not isinstance(obs, torch.Tensor):
        raise TypeError(f"Expected Isaac policy observation tensor, got {type(obs)}")
    return obs


def _camera_tensor(env, sensor_name: str, *, device: torch.device) -> torch.Tensor:
    sensor = env.unwrapped.scene.sensors[sensor_name]
    output = sensor.data.output
    if "rgb" not in output:
        raise RuntimeError(f"Camera sensor {sensor_name!r} does not expose rgb output keys: {sorted(output)}")
    image = output["rgb"].to(device)
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    else:
        image = image.float()
        if image.max() > 2.0:
            image = image / 255.0
    if image.ndim != 4:
        raise RuntimeError(f"Camera sensor {sensor_name!r} rgb output has unexpected shape: {tuple(image.shape)}")
    if image.shape[-1] in (3, 4):
        image = image[..., :3].permute(0, 3, 1, 2).contiguous()
    elif image.shape[1] != 3:
        raise RuntimeError(f"Camera sensor {sensor_name!r} rgb output has unexpected shape: {tuple(image.shape)}")
    return F.interpolate(image, size=(256, 288), mode="bilinear", align_corners=False)


def _raw_camera_images(env, *, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "observation.images.center_camera": _camera_tensor(env, "center_camera", device=device),
        "observation.images.left_camera": _camera_tensor(env, "left_camera", device=device),
        "observation.images.right_camera": _camera_tensor(env, "right_camera", device=device),
    }


def _to_act_obs(policy_obs: torch.Tensor, images: dict[str, torch.Tensor], *, state_dim: int) -> dict[str, Any]:
    state = policy_obs[:, :state_dim]
    return {"state": state, "images": images}


def _named_index(names: list[str], target: str) -> int:
    try:
        return names.index(target)
    except ValueError as exc:
        raise RuntimeError(f"Isaac robot does not expose {target!r}; available names: {names}") from exc


def _isaac_lerobot_state(env, args: argparse.Namespace, *, device: torch.device, state_dim: int) -> torch.Tensor:
    robot = env.unwrapped.scene["robot"]
    body_names = list(getattr(robot, "body_names", []))
    joint_names = list(getattr(robot, "joint_names", []))
    wrist_index = _named_index(body_names, "wrist_3_link")
    joint_indices = [_named_index(joint_names, name) for name in ARM_JOINT_NAMES]
    data = robot.data
    batch_size = int(data.body_pos_w.shape[0])

    tcp_pos = data.body_pos_w[:, wrist_index]
    # Isaac Lab stores quaternions in wxyz order. The LeRobot dataset feature
    # names say xyzw, but the recorded identity rows are numerically wxyz, so
    # keep the numeric order used by ACT training.
    tcp_quat = data.body_quat_w[:, wrist_index]
    tcp_lin_vel = getattr(data, "body_lin_vel_w", torch.zeros(batch_size, len(body_names), 3, device=device))[
        :, wrist_index
    ]
    tcp_ang_vel = getattr(data, "body_ang_vel_w", torch.zeros(batch_size, len(body_names), 3, device=device))[
        :, wrist_index
    ]
    tcp_error = torch.zeros(batch_size, 6, dtype=torch.float32, device=device)
    joint_pos = data.joint_pos[:, joint_indices]
    gripper = torch.full(
        (batch_size, 1),
        float(args.gripper_joint_position),
        dtype=torch.float32,
        device=device,
    )
    wrench = torch.zeros(batch_size, 6, dtype=torch.float32, device=device)
    base_state = torch.cat(
        [tcp_pos, tcp_quat, tcp_lin_vel, tcp_ang_vel, tcp_error, joint_pos, gripper, wrench],
        dim=-1,
    )
    if base_state.shape[1] != 32:
        raise RuntimeError(f"Expected Isaac LeRobot-compatible base state dim 32, got {base_state.shape[1]}")
    task_vector = _canonical_task_vector(args, device=device, batch_size=batch_size)
    state = torch.cat([base_state, task_vector], dim=-1)
    if state.shape[1] < state_dim:
        raise RuntimeError(f"Isaac LeRobot-compatible state has {state.shape[1]} dims, checkpoint expects {state_dim}")
    return state[:, :state_dim]


def _act_obs_from_env(
    env,
    policy_obs: torch.Tensor,
    images: dict[str, torch.Tensor],
    args: argparse.Namespace,
    *,
    device: torch.device,
    state_dim: int,
) -> dict[str, Any]:
    if args.state_source == "policy_prefix":
        return _to_act_obs(policy_obs, images, state_dim=state_dim)
    return {"state": _isaac_lerobot_state(env, args, device=device, state_dim=state_dim), "images": images}


def _repeat_first_action(action: torch.Tensor, *, action_horizon: int, single_action_dim: int) -> torch.Tensor:
    first = action[:, :single_action_dim]
    return first.repeat(1, action_horizon)


def _mlp(input_dim: int, hidden_dim: int, num_layers: int, output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = input_dim
    for _ in range(num_layers):
        layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU()])
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class IsaacACTAdapterActor(nn.Module):
    """TorchScript ACT base plus the offline-trained adapter."""

    def __init__(
        self,
        *,
        act_base: torch.jit.ScriptModule,
        state_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int,
        num_layers: int,
        adapter_scale: float,
        adapter_delta_clip: float | None,
        action_clip: float | None,
    ):
        super().__init__()
        self.act_base = act_base
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.adapter_scale = float(adapter_scale)
        self.adapter_delta_clip = None if adapter_delta_clip is None else float(adapter_delta_clip)
        self.action_clip = None if action_clip is None else float(action_clip)
        self.adapter = _mlp(state_dim + action_dim, hidden_dim, num_layers, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -2.0))

    def action_components(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        chunk = self.act_base(
            obs["state"],
            obs["images"]["observation.images.center_camera"],
            obs["images"]["observation.images.left_camera"],
            obs["images"]["observation.images.right_camera"],
        )
        base_action = chunk[:, : self.action_horizon, :].reshape(obs["state"].shape[0], -1)
        raw_delta_action = self.adapter(torch.cat([obs["state"], base_action], dim=-1))
        if self.adapter_delta_clip is not None and self.adapter_delta_clip > 0.0:
            delta_action = raw_delta_action.clamp(-self.adapter_delta_clip, self.adapter_delta_clip)
        else:
            delta_action = raw_delta_action
        unclipped_final_action = base_action + self.adapter_scale * delta_action
        if self.action_clip is not None and self.action_clip > 0.0:
            final_action = unclipped_final_action.clamp(-self.action_clip, self.action_clip)
        else:
            final_action = unclipped_final_action
        return {
            "base_action": base_action,
            "raw_delta_action": raw_delta_action,
            "delta_action": delta_action,
            "unclipped_final_action": unclipped_final_action,
            "final_action": final_action,
        }

    def mean_action(self, obs: dict[str, Any]) -> torch.Tensor:
        return self.action_components(obs)["final_action"]


class ImageStateEncoder(nn.Module):
    def __init__(self, *, state_dim: int, camera_keys: list[str], feature_dim: int = 256):
        super().__init__()
        self.camera_keys = list(camera_keys)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=4, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        self.proj = nn.Sequential(nn.Linear(state_dim + 64 * len(self.camera_keys), feature_dim), nn.ReLU())

    def forward(self, obs: dict[str, Any]) -> torch.Tensor:
        image_features = [self.image_encoder(obs["images"][key]) for key in self.camera_keys]
        return self.proj(torch.cat([obs["state"], *image_features], dim=-1))


class VisionCritic(nn.Module):
    def __init__(self, *, state_dim: int, camera_keys: list[str], action_dim: int):
        super().__init__()
        self.encoder = ImageStateEncoder(state_dim=state_dim, camera_keys=camera_keys)
        self.q = nn.Sequential(
            nn.Linear(256 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs: dict[str, Any], action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([self.encoder(obs), action], dim=-1))


class OnlineSERLTrainer:
    def __init__(
        self,
        *,
        actor: IsaacACTAdapterActor,
        critic1: VisionCritic,
        critic2: VisionCritic,
        gamma: float,
        tau: float,
        adapter_lr: float,
        critic_lr: float,
        adapter_penalty_weight: float,
        act_preservation_weight: float,
        device: torch.device,
    ):
        self.actor = actor.to(device)
        self.critic1 = critic1.to(device)
        self.critic2 = critic2.to(device)
        self.target_critic1 = VisionCritic(state_dim=critic1.encoder.proj[0].in_features - 64 * len(CAMERA_KEYS), camera_keys=CAMERA_KEYS, action_dim=actor.action_dim).to(device)
        self.target_critic2 = VisionCritic(state_dim=critic2.encoder.proj[0].in_features - 64 * len(CAMERA_KEYS), camera_keys=CAMERA_KEYS, action_dim=actor.action_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.gamma = gamma
        self.tau = tau
        self.adapter_penalty_weight = adapter_penalty_weight
        self.act_preservation_weight = act_preservation_weight
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=adapter_lr)
        self.critic_opt = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr)

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        obs, next_obs = batch["obs"], batch["next_obs"]
        action, reward, done = batch["action"], batch["reward"], batch["done"]
        with torch.no_grad():
            next_action = self.actor.mean_action(next_obs)
            target_q = torch.minimum(self.target_critic1(next_obs, next_action), self.target_critic2(next_obs, next_action))
            td_target = reward + self.gamma * (1.0 - done) * target_q
        q1 = self.critic1(obs, action)
        q2 = self.critic2(obs, action)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        components = self.actor.action_components(obs)
        actor_action = components["final_action"]
        base_action = components["base_action"]
        delta_action = components["delta_action"]
        actor_q = torch.minimum(self.critic1(obs, actor_action), self.critic2(obs, actor_action))
        adapter_penalty = delta_action.square().mean()
        act_preservation_loss = F.mse_loss(actor_action, base_action)
        actor_loss = -actor_q.mean() + self.adapter_penalty_weight * adapter_penalty + self.act_preservation_weight * act_preservation_loss
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        self._soft_update()
        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "critic_loss": float(critic_loss.detach().cpu()),
            "q_mean": float(torch.minimum(q1, q2).mean().detach().cpu()),
            "adapter_delta_norm": float(delta_action.norm(dim=-1).mean().detach().cpu()),
            "raw_adapter_delta_norm": float(
                components.get("raw_delta_action", delta_action).norm(dim=-1).mean().detach().cpu()
            ),
            "adapter_penalty": float(adapter_penalty.detach().cpu()),
            "act_preservation_loss": float(act_preservation_loss.detach().cpu()),
            "final_minus_act_norm": float((actor_action - base_action).norm(dim=-1).mean().detach().cpu()),
            "unclipped_final_minus_act_norm": float(
                (components.get("unclipped_final_action", actor_action) - base_action)
                .norm(dim=-1)
                .mean()
                .detach()
                .cpu()
            ),
            "log_std_mean": float(self.actor.log_std.mean().detach().cpu()),
        }

    def _soft_update(self) -> None:
        for target, source in ((self.target_critic1, self.critic1), (self.target_critic2, self.critic2)):
            for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * source_param.data)


def _infer_adapter_shape(actor_state: dict[str, torch.Tensor]) -> tuple[int, int]:
    linear_weights = [(key, value) for key, value in actor_state.items() if key.startswith("adapter.") and key.endswith(".weight")]
    linear_weights.sort(key=lambda item: int(item[0].split(".")[1]))
    hidden_dim = int(linear_weights[0][1].shape[0])
    num_layers = len(linear_weights) - 1
    return hidden_dim, num_layers


def _load_adapter_actor(
    checkpoint: dict[str, Any],
    *,
    act_torchscript: Path,
    state_dim: int,
    action_dim: int,
    action_horizon: int,
    device: torch.device,
    adapter_delta_clip: float | None,
    action_clip: float | None,
) -> IsaacACTAdapterActor:
    actor_state = checkpoint["actor"]
    hidden_dim, num_layers = _infer_adapter_shape(actor_state)
    adapter_scale = float((checkpoint.get("warmstart_report") or {}).get("adapter_scale", 1.0))
    act_base = torch.jit.load(str(act_torchscript), map_location=device).eval()
    for param in act_base.parameters():
        param.requires_grad = False
    actor = IsaacACTAdapterActor(
        act_base=act_base,
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        adapter_scale=adapter_scale,
        adapter_delta_clip=adapter_delta_clip,
        action_clip=action_clip,
    ).to(device)
    own_state = actor.state_dict()
    compatible = {key: value for key, value in actor_state.items() if key in own_state and tuple(value.shape) == tuple(own_state[key].shape)}
    own_state.update(compatible)
    actor.load_state_dict(own_state, strict=True)
    return actor


def _save_checkpoint(path: Path, trainer: OnlineSERLTrainer, train_config: dict[str, Any], step: int) -> None:
    torch.save(
        {
            "actor": trainer.actor.state_dict(),
            "critic1": trainer.critic1.state_dict(),
            "critic2": trainer.critic2.state_dict(),
            "target_critic1": trainer.target_critic1.state_dict(),
            "target_critic2": trainer.target_critic2.state_dict(),
            "actor_optimizer": trainer.actor_opt.state_dict(),
            "critic_optimizer": trainer.critic_opt.state_dict(),
            "online_serl_config": train_config,
            "step": step,
        },
        path,
    )


def _checkpoint_training_context(checkpoint: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if "vision_offline_serl_config" in checkpoint:
        return (
            checkpoint["vision_offline_serl_config"],
            checkpoint.get("dataset_summary") or {},
            checkpoint.get("warmstart_report") or {},
        )
    online_cfg = checkpoint.get("online_serl_config") or {}
    source = online_cfg.get("checkpoint") or {}
    offline_cfg = source.get("vision_offline_serl_config")
    if offline_cfg is None:
        raise KeyError(
            "Checkpoint does not contain vision_offline_serl_config and does not look like "
            "an online SERL checkpoint with online_serl_config.checkpoint.vision_offline_serl_config."
        )
    return (
        offline_cfg,
        source.get("dataset_summary") or {},
        source.get("warmstart_report") or {},
    )


def main() -> None:
    torch.manual_seed(args_cli.seed)
    checkpoint_path = Path(args_cli.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    offline_cfg, dataset_summary, warmstart = _checkpoint_training_context(checkpoint)
    state_dim = int(offline_cfg["state_dim"])
    action_horizon = int(offline_cfg["action_horizon"])
    single_action_dim = int(offline_cfg["action_dim"] // action_horizon)

    device = torch.device(args_cli.device)
    actor = _load_adapter_actor(
        checkpoint,
        act_torchscript=Path(args_cli.act_torchscript),
        state_dim=state_dim,
        action_dim=int(offline_cfg["action_dim"]),
        action_horizon=action_horizon,
        device=device,
        adapter_delta_clip=args_cli.adapter_delta_clip,
        action_clip=args_cli.action_clip,
    )
    critic1 = VisionCritic(state_dim=state_dim, camera_keys=CAMERA_KEYS, action_dim=int(offline_cfg["action_dim"]))
    critic2 = VisionCritic(state_dim=state_dim, camera_keys=CAMERA_KEYS, action_dim=int(offline_cfg["action_dim"]))
    critic1.load_state_dict(checkpoint["critic1"], strict=True)
    critic2.load_state_dict(checkpoint["critic2"], strict=True)
    trainer = OnlineSERLTrainer(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        gamma=args_cli.gamma,
        tau=args_cli.tau,
        adapter_lr=args_cli.adapter_lr,
        critic_lr=args_cli.critic_lr,
        adapter_penalty_weight=args_cli.adapter_penalty_weight,
        act_preservation_weight=args_cli.act_preservation_weight,
        device=device,
    )

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    # The SERL actor consumes raw camera tensors directly from the sensors. Avoid
    # computing the PPO-specific ResNet feature observation terms, but keep the
    # camera sensors enabled and rendered.
    env_cfg.observations.policy.center_rgb = None
    env_cfg.observations.policy.left_rgb = None
    env_cfg.observations.policy.right_rgb = None
    print("[AIC SERL] Creating Isaac env", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[AIC SERL] Isaac env created", flush=True)
    replay = ReplayBuffer(args_cli.replay_capacity)

    run_dir = Path(args_cli.output_dir) / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args_cli.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    train_config = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint": {
            "vision_offline_serl_config": offline_cfg,
            "dataset_summary": dataset_summary,
            "warmstart_report": warmstart,
        },
        "isaac_adapter": {
            "state_source": args_cli.state_source,
            "state_contract": (
                "LeRobot-compatible 32D state plus canonical 10D task vector"
                if args_cli.state_source == "lerobot_compatible"
                else f"first_{state_dim}_dims"
            ),
            "image_source": "raw_isaac_camera_sensor_rgb_resized_to_3x256x288",
            "act_torchscript": str(args_cli.act_torchscript),
            "action_executed": "first_action_from_flattened_chunk",
            "adapter_delta_clip": args_cli.adapter_delta_clip,
            "action_clip": args_cli.action_clip,
            "ppo_resnet_observation_terms_disabled": True,
            "camera_sensors_enabled": True,
            "task_context": {
                "task_family": args_cli.task_family,
                "target_port_index": args_cli.target_port_index,
                "target_card_index": args_cli.target_card_index,
                "target_card_valid": args_cli.target_card_valid,
            },
            "gripper_joint_position": args_cli.gripper_joint_position,
        },
        "args": vars(args_cli),
    }
    (run_dir / "train_config.json").write_text(json.dumps(train_config, indent=2, sort_keys=True), encoding="utf-8")

    print("[AIC SERL] Resetting Isaac env", flush=True)
    obs, _ = env.reset()
    print("[AIC SERL] Isaac env reset complete", flush=True)
    policy_obs = _policy_tensor(obs).to(device)
    print(f"[AIC SERL] Policy obs shape: {tuple(policy_obs.shape)}", flush=True)
    current_images = _raw_camera_images(env, device=device)
    print("[AIC SERL] Initial raw camera read complete", flush=True)
    updates_done = 0
    last_metrics: dict[str, float] = {}
    stop_reason = "max_steps"
    for step in range(1, args_cli.steps + 1):
        if args_cli.max_wall_time_minutes > 0.0:
            elapsed_minutes = (time.monotonic() - PROCESS_START_TIME) / 60.0
            if elapsed_minutes >= args_cli.max_wall_time_minutes:
                stop_reason = "max_wall_time"
                print(
                    f"[AIC SERL] Wall-time limit reached at step {step - 1}: "
                    f"{elapsed_minutes:.2f} min",
                    flush=True,
                )
                break
        act_obs = _act_obs_from_env(
            env,
            policy_obs,
            current_images,
            args_cli,
            device=device,
            state_dim=state_dim,
        )
        with torch.no_grad():
            action_chunk = trainer.actor.mean_action(act_obs)
        env_action = action_chunk[:, :single_action_dim]
        next_obs, reward, terminated, truncated, _ = env.step(env_action)
        next_policy_obs = _policy_tensor(next_obs).to(device)
        next_images = _raw_camera_images(env, device=device)
        next_act_obs = _act_obs_from_env(
            env,
            next_policy_obs,
            next_images,
            args_cli,
            device=device,
            state_dim=state_dim,
        )
        done = torch.logical_or(terminated, truncated).float().reshape(-1, 1).to(device)
        reward = reward.reshape(-1, 1).to(device)
        action_for_critic = _repeat_first_action(env_action, action_horizon=action_horizon, single_action_dim=single_action_dim)
        for env_index in range(policy_obs.shape[0]):
            replay.append(
                {
                    "obs": {"state": act_obs["state"][env_index], "images": {k: v[env_index] for k, v in act_obs["images"].items()}},
                    "next_obs": {
                        "state": next_act_obs["state"][env_index],
                        "images": {k: v[env_index] for k, v in next_act_obs["images"].items()},
                    },
                    "action": action_for_critic[env_index],
                    "reward": reward[env_index],
                    "done": done[env_index],
                }
            )
        policy_obs = next_policy_obs
        current_images = next_images

        if len(replay) >= args_cli.batch_size and updates_done < args_cli.updates:
            batch = replay.sample(args_cli.batch_size, device)
            last_metrics = trainer.train_step(batch)
            updates_done += 1

        row = {
            "step": step,
            "updates_done": updates_done,
            "replay_size": len(replay),
            "reward_mean": float(reward.mean().detach().cpu()),
            **last_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        if args_cli.log_every > 0 and (step == 1 or step % args_cli.log_every == 0):
            print(
                f"[AIC SERL] step={step} updates={updates_done} replay={len(replay)} "
                f"reward={row['reward_mean']:.6f}",
                flush=True,
            )
        if updates_done >= args_cli.updates:
            stop_reason = "target_updates"
            break

    train_config["result"] = {
        "stop_reason": stop_reason,
        "steps_completed": step if stop_reason != "max_wall_time" else max(step - 1, 0),
        "updates_done": updates_done,
        "elapsed_minutes": (time.monotonic() - PROCESS_START_TIME) / 60.0,
    }
    (run_dir / "train_config.json").write_text(json.dumps(train_config, indent=2, sort_keys=True), encoding="utf-8")
    _save_checkpoint(run_dir / "checkpoint_latest.pt", trainer, train_config, train_config["result"]["steps_completed"])
    print(f"Wrote online SERL checkpoint: {run_dir / 'checkpoint_latest.pt'}")
    print(f"Wrote metrics: {metrics_path}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
