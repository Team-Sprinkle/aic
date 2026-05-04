"""Vision-based offline SERL components backed by a LeRobot ACT actor."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import OBS_IMAGES

from .act_warmstart import inspect_act_checkpoint, resolve_act_checkpoint_dir

RewardMode = Literal["dataset", "final_success", "zero"]


def _unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def _read_dataframes(dataset_root: Path) -> pd.DataFrame:
    files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No LeRobot parquet files found under {dataset_root / 'data'}")
    df = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    required = {"action", "observation.state", "episode_index", "frame_index"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    sort_keys = ["episode_index", "frame_index"]
    if "index" in df.columns:
        sort_keys.append("index")
    return df.sort_values(sort_keys).reset_index(drop=True)


def _rewards(df: pd.DataFrame, reward_mode: RewardMode) -> np.ndarray:
    if reward_mode == "dataset" and "reward" in df.columns:
        return df["reward"].to_numpy(dtype=np.float32)
    if reward_mode == "zero":
        return np.zeros(len(df), dtype=np.float32)
    rewards = np.zeros(len(df), dtype=np.float32)
    last_indices = df.groupby("episode_index", sort=False).tail(1).index.to_numpy()
    rewards[last_indices] = 1.0
    return rewards


def _stack_actions(df: pd.DataFrame) -> np.ndarray:
    values = [np.asarray(v, dtype=np.float32).reshape(-1) for v in df["action"]]
    if not values:
        raise ValueError("Dataset action column is empty")
    dim = values[0].shape[0]
    if any(v.shape[0] != dim for v in values):
        raise ValueError("Dataset action column has inconsistent action dimensions")
    return np.stack(values, axis=0)


def _action_chunk(actions: np.ndarray, episodes: np.ndarray, idx: int, horizon: int) -> np.ndarray:
    values = []
    episode = episodes[idx]
    last_valid = actions[idx]
    for offset in range(horizon):
        src = idx + offset
        if src < len(actions) and episodes[src] == episode:
            last_valid = actions[src]
        values.append(last_valid)
    return np.concatenate(values, axis=0).astype(np.float32)


@dataclass(frozen=True)
class VisionDatasetSummary:
    dataset_root: str
    num_frames: int
    num_episodes: int
    state_dim: int
    single_action_dim: int
    action_dim: int
    action_horizon: int
    camera_keys: list[str]


class VisionOfflineSERLDataset(Dataset[dict[str, Any]]):
    """LeRobot-backed transition dataset using official video/image loading."""

    def __init__(
        self,
        dataset_root: Path,
        *,
        camera_keys: list[str],
        action_horizon: int,
        reward_mode: RewardMode = "dataset",
        video_backend: str = "pyav",
    ):
        if action_horizon < 1:
            raise ValueError("action_horizon must be >= 1")
        self.dataset_root = Path(dataset_root)
        self.camera_keys = list(camera_keys)
        self.action_horizon = int(action_horizon)
        self.lerobot = LeRobotDataset(
            repo_id=f"local/{self.dataset_root.name}",
            root=self.dataset_root,
            video_backend=video_backend,
        )
        self.df = _read_dataframes(self.dataset_root)
        if len(self.df) != len(self.lerobot):
            raise ValueError(
                f"Parquet rows ({len(self.df)}) do not match LeRobot frames ({len(self.lerobot)})"
            )
        self.actions = _stack_actions(self.df)
        self.rewards = _rewards(self.df, reward_mode)
        self.episodes = self.df["episode_index"].to_numpy(dtype=np.int64)
        self.frames = self.df["frame_index"].to_numpy(dtype=np.int64)
        first = self.lerobot[0]
        self.state_dim = int(first["observation.state"].reshape(-1).shape[0])
        self.single_action_dim = int(first["action"].reshape(-1).shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.single_action_dim * self.action_horizon)

    @property
    def summary(self) -> VisionDatasetSummary:
        return VisionDatasetSummary(
            dataset_root=str(self.dataset_root),
            num_frames=len(self),
            num_episodes=int(len(set(int(v) for v in self.episodes))),
            state_dim=self.state_dim,
            single_action_dim=self.single_action_dim,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            camera_keys=self.camera_keys,
        )

    def __len__(self) -> int:
        return len(self.lerobot)

    def _obs_at(self, idx: int) -> dict[str, Any]:
        item = self.lerobot[idx]
        return {
            "state": item["observation.state"].float(),
            "images": {key: item[key].float() for key in self.camera_keys},
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        next_idx = idx + 1 if idx + 1 < len(self) and self.episodes[idx + 1] == self.episodes[idx] else idx
        return {
            "obs": self._obs_at(idx),
            "action": torch.as_tensor(_action_chunk(self.actions, self.episodes, idx, self.action_horizon)),
            "reward": torch.as_tensor([self.rewards[idx]], dtype=torch.float32),
            "next_obs": self._obs_at(next_idx),
            "done": torch.as_tensor([0.0 if next_idx != idx else 1.0], dtype=torch.float32),
        }


class ACTChunkActor(nn.Module):
    """ACT policy wrapper exposing a flattened action chunk for SERL."""

    def __init__(self, act_policy: ACTPolicy, *, action_horizon: int):
        super().__init__()
        self.act_policy = act_policy
        self.action_horizon = int(action_horizon)
        self.single_action_dim = int(next(iter(act_policy.config.output_features.values())).shape[0])
        self.action_dim = self.single_action_dim * self.action_horizon
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -2.0))

    def _act_batch(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        batch: dict[str, torch.Tensor] = {"observation.state": obs["state"]}
        for key, value in obs["images"].items():
            batch[key] = value
        return batch

    def act_action(self, obs: dict[str, Any]) -> torch.Tensor:
        batch = self._act_batch(obs)
        if self.act_policy.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.act_policy.config.image_features]
        chunk = self.act_policy.model(batch)[0]
        if chunk.shape[1] < self.action_horizon:
            raise ValueError(
                f"ACT chunk_size={chunk.shape[1]} is smaller than action_horizon={self.action_horizon}"
            )
        return chunk[:, : self.action_horizon, :].reshape(chunk.shape[0], -1)

    def mean_action(self, obs: dict[str, Any]) -> torch.Tensor:
        return self.act_action(obs)

    def forward(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        return self.action_components(obs)

    def action_components(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        action = self.mean_action(obs)
        return {
            "base_action": action.detach(),
            "delta_action": torch.zeros_like(action),
            "final_action": action,
        }

    def sample_action(self, obs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_action(obs)
        std = self.log_std.exp().unsqueeze(0)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob


def _mlp(input_dim: int, hidden_dim: int, num_layers: int, output_dim: int) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    layers: list[nn.Module] = []
    dim = input_dim
    for _ in range(num_layers):
        layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU()])
        dim = hidden_dim
    final = nn.Linear(dim, output_dim)
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


class ACTAdapterSERLActor(ACTChunkActor):
    """Frozen-or-finetuned ACT actor plus a small trainable action correction."""

    def __init__(
        self,
        act_policy: ACTPolicy,
        *,
        action_horizon: int,
        state_dim: int,
        adapter_hidden_dim: int = 256,
        adapter_num_layers: int = 2,
        adapter_scale: float = 1.0,
        freeze_act: bool = True,
        adapter_delta_clip: float | None = None,
        action_clip: float | None = None,
    ):
        super().__init__(act_policy, action_horizon=action_horizon)
        self.adapter_scale = float(adapter_scale)
        self.freeze_act = bool(freeze_act)
        self.adapter_delta_clip = None if adapter_delta_clip is None else float(adapter_delta_clip)
        self.action_clip = None if action_clip is None else float(action_clip)
        self.adapter = _mlp(
            input_dim=state_dim + self.action_dim,
            hidden_dim=adapter_hidden_dim,
            num_layers=adapter_num_layers,
            output_dim=self.action_dim,
        )
        self.set_act_frozen(self.freeze_act)

    def set_act_frozen(self, frozen: bool) -> None:
        self.freeze_act = bool(frozen)
        for param in self.act_policy.parameters():
            param.requires_grad = not self.freeze_act
        self.act_policy.eval() if self.freeze_act else self.act_policy.train()

    def act_action(self, obs: dict[str, Any]) -> torch.Tensor:
        if self.freeze_act:
            with torch.no_grad():
                return super().act_action(obs).detach()
        return super().act_action(obs)

    def delta_action(self, obs: dict[str, Any], base_action: torch.Tensor) -> torch.Tensor:
        return self.adapter(torch.cat([obs["state"], base_action], dim=-1))

    def _clamp_delta(self, delta_action: torch.Tensor) -> torch.Tensor:
        if self.adapter_delta_clip is None or self.adapter_delta_clip <= 0.0:
            return delta_action
        return delta_action.clamp(-self.adapter_delta_clip, self.adapter_delta_clip)

    def _clamp_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_clip is None or self.action_clip <= 0.0:
            return action
        return action.clamp(-self.action_clip, self.action_clip)

    def action_components(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        base_action = self.act_action(obs)
        raw_delta_action = self.delta_action(obs, base_action)
        delta_action = self._clamp_delta(raw_delta_action)
        unclipped_final_action = base_action + self.adapter_scale * delta_action
        final_action = self._clamp_action(unclipped_final_action)
        return {
            "base_action": base_action,
            "raw_delta_action": raw_delta_action,
            "delta_action": delta_action,
            "unclipped_final_action": unclipped_final_action,
            "final_action": final_action,
        }

    def mean_action(self, obs: dict[str, Any]) -> torch.Tensor:
        return self.action_components(obs)["final_action"]

    def forward(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        return self.action_components(obs)

    def adapter_parameters(self):
        yield from self.adapter.parameters()
        yield self.log_std


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
        self.proj = nn.Sequential(
            nn.Linear(state_dim + 64 * len(self.camera_keys), feature_dim),
            nn.ReLU(),
        )

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


@dataclass
class VisionOfflineSERLConfig:
    state_dim: int
    action_dim: int
    action_horizon: int
    camera_keys: list[str]
    gamma: float = 0.99
    tau: float = 0.005
    bc_weight: float = 1.0
    cql_weight: float = 0.0
    adapter_penalty_weight: float = 1e-3
    act_preservation_weight: float = 1e-2
    smoothness_weight: float = 0.0
    adapter_delta_clip: float | None = None
    action_clip: float | None = None
    lr: float = 1e-4
    adapter_lr: float = 1e-4
    act_lr: float = 1e-5
    critic_lr: float = 1e-4
    actor_mode: str = "act_adapter"
    freeze_act: bool = True


class VisionOfflineSERLTrainer:
    def __init__(self, *, config: VisionOfflineSERLConfig, actor: ACTChunkActor, device: str | torch.device):
        self.config = config
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.critic1 = VisionCritic(
            state_dim=config.state_dim,
            camera_keys=config.camera_keys,
            action_dim=config.action_dim,
        ).to(self.device)
        self.critic2 = VisionCritic(
            state_dim=config.state_dim,
            camera_keys=config.camera_keys,
            action_dim=config.action_dim,
        ).to(self.device)
        self.target_critic1 = VisionCritic(
            state_dim=config.state_dim,
            camera_keys=config.camera_keys,
            action_dim=config.action_dim,
        ).to(self.device)
        self.target_critic2 = VisionCritic(
            state_dim=config.state_dim,
            camera_keys=config.camera_keys,
            action_dim=config.action_dim,
        ).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_opt = torch.optim.Adam(self._actor_param_groups(), lr=config.lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=config.critic_lr,
        )

    def actor_components(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        return self.actor(obs)

    def actor_mean_action(self, obs: dict[str, Any]) -> torch.Tensor:
        return self.actor_components(obs)["final_action"]

    def _actor_param_groups(self) -> list[dict[str, Any]]:
        if isinstance(self.actor, ACTAdapterSERLActor):
            groups: list[dict[str, Any]] = [
                {
                    "params": [p for p in self.actor.adapter_parameters() if p.requires_grad],
                    "lr": self.config.adapter_lr,
                }
            ]
            act_params = [p for p in self.actor.act_policy.parameters() if p.requires_grad]
            if act_params:
                groups.append({"params": act_params, "lr": self.config.act_lr})
            return [group for group in groups if group["params"]]
        params = [p for p in self.actor.parameters() if p.requires_grad]
        return [{"params": params, "lr": self.config.lr}]

    def _obs_to_device(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {
            "state": obs["state"].to(self.device),
            "images": {key: value.to(self.device) for key, value in obs["images"].items()},
        }

    def _batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            "obs": self._obs_to_device(batch["obs"]),
            "next_obs": self._obs_to_device(batch["next_obs"]),
            "action": batch["action"].to(self.device),
            "reward": batch["reward"].to(self.device),
            "done": batch["done"].to(self.device),
        }

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        batch = self._batch_to_device(batch)
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        action = batch["action"]
        reward = batch["reward"]
        done = batch["done"]

        with torch.no_grad():
            next_action = self.actor_mean_action(next_obs)
            target_q = torch.minimum(
                self.target_critic1(next_obs, next_action),
                self.target_critic2(next_obs, next_action),
            )
            td_target = reward + self.config.gamma * (1.0 - done) * target_q

        q1 = self.critic1(obs, action)
        q2 = self.critic2(obs, action)
        td_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        conservative_loss = torch.zeros((), device=self.device)
        if self.config.cql_weight > 0.0:
            random_action = torch.randn_like(action)
            conservative_loss = (
                self.critic1(obs, random_action).mean()
                + self.critic2(obs, random_action).mean()
                - q1.detach().mean()
                - q2.detach().mean()
            )
        critic_loss = td_loss + self.config.cql_weight * conservative_loss
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        components = self.actor_components(obs)
        delta_action = components["delta_action"]
        base_action = components["base_action"]
        actor_action = components["final_action"]
        actor_q = torch.minimum(self.critic1(obs, actor_action), self.critic2(obs, actor_action))
        bc_loss = F.mse_loss(actor_action, action)
        adapter_penalty = delta_action.square().mean()
        act_preservation_loss = F.mse_loss(actor_action, base_action.detach())
        smoothness_loss = self._smoothness_loss(actor_action)
        actor_loss = (
            -actor_q.mean()
            + self.config.bc_weight * bc_loss
            + self.config.adapter_penalty_weight * adapter_penalty
            + self.config.act_preservation_weight * act_preservation_loss
            + self.config.smoothness_weight * smoothness_loss
        )
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        self.soft_update_targets()

        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "bc_loss": float(bc_loss.detach().cpu()),
            "critic_loss": float(critic_loss.detach().cpu()),
            "td_loss": float(td_loss.detach().cpu()),
            "conservative_loss": float(conservative_loss.detach().cpu()),
            "q_mean": float(torch.minimum(q1, q2).mean().detach().cpu()),
            "reward_mean": float(reward.mean().detach().cpu()),
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
            "smoothness_loss": float(smoothness_loss.detach().cpu()),
            "log_std_mean": float(_unwrap_module(self.actor).log_std.mean().detach().cpu()),
        }

    @torch.no_grad()
    def eval_step(self, batch: dict[str, Any]) -> dict[str, float]:
        batch = self._batch_to_device(batch)
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        action = batch["action"]
        reward = batch["reward"]
        done = batch["done"]

        next_action = self.actor_mean_action(next_obs)
        target_q = torch.minimum(
            self.target_critic1(next_obs, next_action),
            self.target_critic2(next_obs, next_action),
        )
        td_target = reward + self.config.gamma * (1.0 - done) * target_q

        q1 = self.critic1(obs, action)
        q2 = self.critic2(obs, action)
        td_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        conservative_loss = torch.zeros((), device=self.device)
        if self.config.cql_weight > 0.0:
            random_action = torch.randn_like(action)
            conservative_loss = (
                self.critic1(obs, random_action).mean()
                + self.critic2(obs, random_action).mean()
                - q1.detach().mean()
                - q2.detach().mean()
            )
        critic_loss = td_loss + self.config.cql_weight * conservative_loss

        components = self.actor_components(obs)
        delta_action = components["delta_action"]
        base_action = components["base_action"]
        actor_action = components["final_action"]
        actor_q = torch.minimum(self.critic1(obs, actor_action), self.critic2(obs, actor_action))
        bc_loss = F.mse_loss(actor_action, action)
        adapter_penalty = delta_action.square().mean()
        act_preservation_loss = F.mse_loss(actor_action, base_action.detach())
        smoothness_loss = self._smoothness_loss(actor_action)
        actor_loss = (
            -actor_q.mean()
            + self.config.bc_weight * bc_loss
            + self.config.adapter_penalty_weight * adapter_penalty
            + self.config.act_preservation_weight * act_preservation_loss
            + self.config.smoothness_weight * smoothness_loss
        )

        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "bc_loss": float(bc_loss.detach().cpu()),
            "critic_loss": float(critic_loss.detach().cpu()),
            "td_loss": float(td_loss.detach().cpu()),
            "conservative_loss": float(conservative_loss.detach().cpu()),
            "q_mean": float(torch.minimum(q1, q2).mean().detach().cpu()),
            "reward_mean": float(reward.mean().detach().cpu()),
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
            "smoothness_loss": float(smoothness_loss.detach().cpu()),
            "log_std_mean": float(_unwrap_module(self.actor).log_std.mean().detach().cpu()),
        }

    def _smoothness_loss(self, action: torch.Tensor) -> torch.Tensor:
        single_action_dim = max(self.config.action_dim // max(self.config.action_horizon, 1), 1)
        if self.config.action_horizon <= 1 or self.config.action_dim % single_action_dim != 0:
            return torch.zeros((), device=self.device)
        chunk = action.reshape(action.shape[0], self.config.action_horizon, single_action_dim)
        return (chunk[:, 1:, :] - chunk[:, :-1, :]).square().mean()

    def soft_update_targets(self) -> None:
        for target, source in ((self.target_critic1, self.critic1), (self.target_critic2, self.critic2)):
            for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
                target_param.data.mul_(1.0 - self.config.tau)
                target_param.data.add_(self.config.tau * source_param.data)

    def save_checkpoint(
        self,
        path: Path,
        *,
        train_config: dict[str, Any],
        dataset_summary: dict[str, Any],
        warmstart_report: dict[str, Any],
        step: int,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": _unwrap_module(self.actor).state_dict(),
                "critic1": _unwrap_module(self.critic1).state_dict(),
                "critic2": _unwrap_module(self.critic2).state_dict(),
                "target_critic1": self.target_critic1.state_dict(),
                "target_critic2": self.target_critic2.state_dict(),
                "actor_optimizer": self.actor_opt.state_dict(),
                "critic_optimizer": self.critic_opt.state_dict(),
                "vision_offline_serl_config": self.config.__dict__,
                "train_config": train_config,
                "dataset_summary": dataset_summary,
                "warmstart_report": warmstart_report,
                "step": step,
            },
            path,
        )


def load_act_actor(
    checkpoint: Path,
    *,
    action_horizon: int,
    device: str | torch.device,
    actor_mode: str = "act_adapter",
    state_dim: int | None = None,
    adapter_hidden_dim: int = 256,
    adapter_num_layers: int = 2,
    adapter_scale: float = 1.0,
    freeze_act: bool = True,
    adapter_delta_clip: float | None = None,
    action_clip: float | None = None,
) -> tuple[ACTChunkActor, dict[str, Any]]:
    checkpoint_dir = resolve_act_checkpoint_dir(checkpoint)
    policy = ACTPolicy.from_pretrained(checkpoint_dir, local_files_only=True)
    policy.to(device)
    if actor_mode == "act_adapter":
        if state_dim is None:
            raise ValueError("state_dim is required when actor_mode='act_adapter'")
        actor = ACTAdapterSERLActor(
            policy,
            action_horizon=action_horizon,
            state_dim=state_dim,
            adapter_hidden_dim=adapter_hidden_dim,
            adapter_num_layers=adapter_num_layers,
            adapter_scale=adapter_scale,
            freeze_act=freeze_act,
            adapter_delta_clip=adapter_delta_clip,
            action_clip=action_clip,
        )
    elif actor_mode == "act_direct":
        actor = ACTChunkActor(policy, action_horizon=action_horizon)
    else:
        raise ValueError(f"Unsupported actor_mode: {actor_mode}")
    warmstart = inspect_act_checkpoint(checkpoint_dir)
    try:
        from safetensors.torch import load_file

        tensors = load_file(str(checkpoint_dir / "model.safetensors"))
        policy_params = dict(policy.named_parameters())
        compatible_param_keys = [
            key for key, value in tensors.items()
            if key in policy_params and tuple(value.shape) == tuple(policy_params[key].shape)
        ]
        params_loaded = int(sum(policy_params[key].numel() for key in compatible_param_keys))
        tensors_loaded = len(compatible_param_keys)
        state_tensors_loaded = int(
            sum(
                1
                for key, value in tensors.items()
                if key in policy.state_dict() and tuple(value.shape) == tuple(policy.state_dict()[key].shape)
            )
        )
        skipped_tensors = sorted(
            key
            for key, value in tensors.items()
            if key in policy.state_dict() and tuple(value.shape) != tuple(policy.state_dict()[key].shape)
        )
    except Exception:
        params_loaded = int(sum(p.numel() for p in policy.parameters()))
        tensors_loaded = len(policy.state_dict())
        state_tensors_loaded = len(policy.state_dict())
        skipped_tensors = []
    actor_params = int(sum(p.numel() for p in actor.parameters()))
    act_params = int(sum(p.numel() for p in policy.parameters()))
    adapter_params = int(sum(p.numel() for p in actor.adapter.parameters())) if isinstance(actor, ACTAdapterSERLActor) else 0
    warmstart.update(
        {
            "mode": actor_mode,
            "tensors_loaded": tensors_loaded,
            "state_tensors_loaded": state_tensors_loaded,
            "parameters_loaded": params_loaded,
            "actor_parameters": actor_params,
            "act_parameters": act_params,
            "adapter_parameters": adapter_params,
            "act_frozen": bool(isinstance(actor, ACTAdapterSERLActor) and actor.freeze_act),
            "adapter_scale": float(adapter_scale) if isinstance(actor, ACTAdapterSERLActor) else 0.0,
            "adapter_delta_clip": adapter_delta_clip if isinstance(actor, ACTAdapterSERLActor) else None,
            "action_clip": action_clip if isinstance(actor, ACTAdapterSERLActor) else None,
            "percent_actor_parameters_loaded": 100.0 * params_loaded / max(actor_params, 1),
            "skipped_tensors": skipped_tensors,
        }
    )
    return actor, warmstart


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
