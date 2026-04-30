from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from gazebo_rl.gym_env import GazeboRLEnv
from gazebo_rl.serl_policy import ACTAdapterSERLGazeboPolicy, ACT_CAMERA_KEYS
from gazebo_rl.train import add_recording_args


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
    def __init__(self, *, state_dim: int, camera_keys: list[str], action_dim: int, feature_dim: int = 256):
        super().__init__()
        self.encoder = ImageStateEncoder(state_dim=state_dim, camera_keys=camera_keys, feature_dim=feature_dim)
        self.q = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs: dict[str, Any], action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([self.encoder(obs), action], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data = deque(maxlen=int(capacity))

    def append(self, transition: dict[str, Any]) -> None:
        self.data.append({key: self._detach(value) for key, value in transition.items()})

    def _detach(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: self._detach(item) for key, item in value.items()}
        return value

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, Any]:
        if len(self.data) < batch_size:
            raise ValueError(f"Cannot sample batch_size={batch_size} from replay_size={len(self.data)}")
        indices = torch.randint(len(self.data), (batch_size,)).tolist()
        items = [self.data[index] for index in indices]
        return {
            "obs": self._stack_obs([item["obs"] for item in items], device),
            "next_obs": self._stack_obs([item["next_obs"] for item in items], device),
            "action": torch.stack([item["action"] for item in items]).to(device),
            "reward": torch.stack([item["reward"] for item in items]).to(device),
            "done": torch.stack([item["done"] for item in items]).to(device),
        }

    def _stack_obs(self, obs_items: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
        return {
            "state": torch.cat([item["state"] for item in obs_items], dim=0).to(device),
            "images": {
                key: torch.cat([item["images"][key] for item in obs_items], dim=0).to(device)
                for key in ACT_CAMERA_KEYS
            },
        }


class GazeboOnlineSERLTrainer:
    def __init__(
        self,
        *,
        policy: ACTAdapterSERLGazeboPolicy,
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
        self.policy = policy
        self.actor = policy.actor.to(device)
        self.critic1 = critic1.to(device)
        self.critic2 = critic2.to(device)
        self.target_critic1 = VisionCritic(
            state_dim=policy.state_dim,
            camera_keys=ACT_CAMERA_KEYS,
            action_dim=policy.action_dim,
        ).to(device)
        self.target_critic2 = VisionCritic(
            state_dim=policy.state_dim,
            camera_keys=ACT_CAMERA_KEYS,
            action_dim=policy.action_dim,
        ).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.adapter_penalty_weight = float(adapter_penalty_weight)
        self.act_preservation_weight = float(act_preservation_weight)
        self.actor_opt = torch.optim.Adam(
            [param for param in self.actor.parameters() if param.requires_grad],
            lr=float(adapter_lr),
        )
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=float(critic_lr),
        )

    def obs_to_actor(self, obs: dict[str, Any]) -> dict[str, Any]:
        return self.policy._obs_to_actor(obs)

    def action_for_env(self, actor_obs: dict[str, Any]) -> tuple[list[float], dict[str, float]]:
        with torch.no_grad():
            components = self.actor.action_components(actor_obs)
            action = components["final_action"].squeeze(0)
            metrics = {
                "base_action_norm": float(components["base_action"].norm(dim=-1).mean().detach().cpu()),
                "adapter_delta_norm": float(components["delta_action"].norm(dim=-1).mean().detach().cpu()),
                "raw_adapter_delta_norm": float(
                    components.get("raw_delta_action", components["delta_action"])
                    .norm(dim=-1)
                    .mean()
                    .detach()
                    .cpu()
                ),
                "final_action_norm": float(components["final_action"].norm(dim=-1).mean().detach().cpu()),
                "unclipped_final_action_norm": float(
                    components.get("unclipped_final_action", components["final_action"])
                    .norm(dim=-1)
                    .mean()
                    .detach()
                    .cpu()
                ),
            }
        return action[: self.policy.single_action_dim].detach().cpu().numpy().astype(float).tolist(), metrics

    def action_chunk_from_env_action(self, action: list[float]) -> torch.Tensor:
        first = torch.as_tensor(action, dtype=torch.float32, device=self.policy.device).reshape(1, -1)
        return first.repeat(1, self.policy.action_horizon).squeeze(0)

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        action = batch["action"]
        reward = batch["reward"]
        done = batch["done"]
        with torch.no_grad():
            next_action = self.actor.mean_action(next_obs)
            target_q = torch.minimum(
                self.target_critic1(next_obs, next_action),
                self.target_critic2(next_obs, next_action),
            )
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
        act_preservation_loss = F.mse_loss(actor_action, base_action.detach())
        actor_loss = (
            -actor_q.mean()
            + self.adapter_penalty_weight * adapter_penalty
            + self.act_preservation_weight * act_preservation_loss
        )
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
            "final_minus_act_norm": float((actor_action - base_action.detach()).norm(dim=-1).mean().detach().cpu()),
            "unclipped_final_minus_act_norm": float(
                (components.get("unclipped_final_action", actor_action) - base_action.detach())
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

    def save_checkpoint(self, path: Path, *, train_config: dict[str, Any], step: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "target_critic1": self.target_critic1.state_dict(),
                "target_critic2": self.target_critic2.state_dict(),
                "actor_optimizer": self.actor_opt.state_dict(),
                "critic_optimizer": self.critic_opt.state_dict(),
                "online_gazebo_serl_config": train_config,
                "step": step,
            },
            path,
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def load_trainer(args: argparse.Namespace) -> tuple[GazeboOnlineSERLTrainer, dict[str, Any]]:
    device = torch.device(args.device)
    policy = ACTAdapterSERLGazeboPolicy(
        args.checkpoint,
        act_torchscript=args.act_torchscript,
        device=args.device,
        allow_zero_images=args.allow_zero_images,
        adapter_delta_clip=args.adapter_delta_clip,
        action_clip=args.action_clip,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    critic1 = VisionCritic(state_dim=policy.state_dim, camera_keys=ACT_CAMERA_KEYS, action_dim=policy.action_dim)
    critic2 = VisionCritic(state_dim=policy.state_dim, camera_keys=ACT_CAMERA_KEYS, action_dim=policy.action_dim)
    critic1.load_state_dict(checkpoint["critic1"], strict=True)
    critic2.load_state_dict(checkpoint["critic2"], strict=True)
    trainer = GazeboOnlineSERLTrainer(
        policy=policy,
        critic1=critic1,
        critic2=critic2,
        gamma=args.gamma,
        tau=args.tau,
        adapter_lr=args.adapter_lr,
        critic_lr=args.critic_lr,
        adapter_penalty_weight=args.adapter_penalty_weight,
        act_preservation_weight=args.act_preservation_weight,
        device=device,
    )
    train_config = {
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint": {
            "vision_offline_serl_config": policy.dataset_summary.get("vision_offline_serl_config")
            or (checkpoint.get("vision_offline_serl_config") or {}),
            "dataset_summary": policy.dataset_summary,
            "warmstart_report": policy.warmstart_report,
        },
        "gazebo_adapter": {
            "act_torchscript": str(Path(args.act_torchscript).resolve()),
            "image_source": "gazebo_bridge_live_rgb_jpeg_resized_to_3x256x288",
            "action_executed": "first_action_from_flattened_chunk",
            "adapter_delta_clip": args.adapter_delta_clip,
            "action_clip": args.action_clip,
            "allow_zero_images": bool(args.allow_zero_images),
            "include_images": bool(args.include_images),
        },
        "args": _jsonable(vars(args)),
    }
    if not train_config["checkpoint"]["vision_offline_serl_config"]:
        source = (checkpoint.get("online_serl_config") or checkpoint.get("online_gazebo_serl_config") or {}).get(
            "checkpoint", {}
        )
        train_config["checkpoint"]["vision_offline_serl_config"] = source.get("vision_offline_serl_config") or {}
        train_config["checkpoint"]["dataset_summary"] = source.get("dataset_summary") or policy.dataset_summary
        train_config["checkpoint"]["warmstart_report"] = source.get("warmstart_report") or policy.warmstart_report
    return trainer, train_config


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    trainer, train_config = load_trainer(args)
    if args.dry_run:
        summary = {
            "status": "dry_run",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "act_torchscript": str(Path(args.act_torchscript).resolve()),
            "state_dim": trainer.policy.state_dim,
            "action_dim": trainer.policy.action_dim,
            "single_action_dim": trainer.policy.single_action_dim,
            "action_horizon": trainer.policy.action_horizon,
            "adapter_delta_clip": args.adapter_delta_clip,
            "action_clip": args.action_clip,
            "include_images": bool(args.include_images),
            "allow_zero_images": bool(args.allow_zero_images),
        }
        (output_dir / "dry_run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary

    replay = ReplayBuffer(args.replay_capacity)
    updates_done = 0
    total_steps = 0
    episodes: list[dict[str, Any]] = []
    last_metrics: dict[str, float] = {}
    max_seconds = float(args.max_minutes) * 60.0 if args.max_minutes > 0 else None

    for episode in range(args.max_episodes):
        if max_seconds is not None and time.monotonic() - started >= max_seconds:
            break
        results_dir = output_dir / "results" / f"episode_{episode:03d}"
        env = GazeboRLEnv(
            workspace_dir=args.workspace_dir,
            engine_config=args.engine_config,
            sim_distrobox=args.sim_distrobox,
            ground_truth=args.ground_truth,
            gazebo_gui=args.gazebo_gui,
            launch_rviz=args.launch_rviz,
            max_steps=args.max_steps,
            per_trial_timeout_sec=args.per_trial_timeout_sec,
            results_dir=results_dir,
            record_lerobot=args.record_lerobot,
            record_root=args.record_root,
            record_repo_id=args.record_repo_id,
            record_single_task=args.record_single_task,
            record_video=args.record_video,
            record_fps=args.record_fps,
            record_resume=args.record_resume or episode > 0,
            record_drain_sec=args.record_drain_sec,
            record_image_writer_processes=args.record_image_writer_processes,
            record_image_writer_threads_per_camera=args.record_image_writer_threads_per_camera,
            record_video_encoding_batch_size=args.record_video_encoding_batch_size,
            include_images=args.include_images,
        )
        episode_reward = 0.0
        episode_steps = 0
        try:
            obs, _ = env.reset()
            for _ in range(args.max_steps):
                actor_obs = trainer.obs_to_actor(obs)
                action, action_metrics = trainer.action_for_env(actor_obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_actor_obs = trainer.obs_to_actor(next_obs)
                replay.append(
                    {
                        "obs": actor_obs,
                        "next_obs": next_actor_obs,
                        "action": trainer.action_chunk_from_env_action(action),
                        "reward": torch.tensor([float(reward)], dtype=torch.float32),
                        "done": torch.tensor([float(terminated or truncated)], dtype=torch.float32),
                    }
                )
                total_steps += 1
                episode_steps += 1
                episode_reward += float(reward)
                row: dict[str, Any] = {
                    "episode": episode,
                    "step": total_steps,
                    "episode_step": episode_steps,
                    "updates_done": updates_done,
                    "replay_size": len(replay),
                    "reward": float(reward),
                    **action_metrics,
                    **last_metrics,
                }
                if len(replay) >= args.batch_size and updates_done < args.updates:
                    last_metrics = trainer.train_step(replay.sample(args.batch_size, trainer.policy.device))
                    updates_done += 1
                    row.update(last_metrics)
                    row["updates_done"] = updates_done
                with metrics_path.open("a", encoding="utf-8") as metrics_file:
                    metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
                obs = next_obs
                if terminated or truncated or updates_done >= args.updates:
                    break
            episodes.append(
                {
                    "episode": episode,
                    "steps": episode_steps,
                    "reward": episode_reward,
                    "results_dir": str(results_dir),
                }
            )
        finally:
            env.close()
        if updates_done >= args.updates:
            break

    train_config["result"] = {
        "episodes_completed": len(episodes),
        "steps_completed": total_steps,
        "updates_done": updates_done,
        "elapsed_sec": time.monotonic() - started,
    }
    (output_dir / "train_config.json").write_text(json.dumps(train_config, indent=2, sort_keys=True), encoding="utf-8")
    trainer.save_checkpoint(checkpoint_path, train_config=train_config, step=total_steps)
    summary = {
        **train_config["result"],
        "episodes": episodes,
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "train_config_path": str(output_dir / "train_config.json"),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ACT-adapter SERL online through GazeboRLEnv.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--act-torchscript", required=True)
    parser.add_argument("--output-dir", default="outputs/gazebo_rl/online_serl/latest")
    parser.add_argument("--workspace-dir", default=".")
    parser.add_argument("--engine-config", default=None)
    parser.add_argument("--sim-distrobox", default=None)
    parser.add_argument("--ground-truth", type=_bool, default=True)
    parser.add_argument("--gazebo-gui", type=_bool, default=False)
    parser.add_argument("--launch-rviz", type=_bool, default=False)
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--replay-capacity", type=int, default=512)
    parser.add_argument("--max-minutes", type=float, default=10.0)
    parser.add_argument("--per-trial-timeout-sec", type=float, default=300.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--adapter-penalty-weight", type=float, default=1e-2)
    parser.add_argument("--act-preservation-weight", type=float, default=1e-1)
    parser.add_argument("--adapter-delta-clip", type=float, default=0.05)
    parser.add_argument("--action-clip", type=float, default=0.05)
    parser.add_argument("--include-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-zero-images", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    add_recording_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(json.dumps(run_training(args), indent=2, sort_keys=True))
