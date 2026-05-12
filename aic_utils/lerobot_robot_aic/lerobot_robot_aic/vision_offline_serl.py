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
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import OBS_IMAGES

from .act_warmstart import inspect_act_checkpoint, resolve_act_checkpoint_dir

RewardMode = Literal["dataset", "final_success", "zero"]
ActorUpdateMode = Literal["q_bc", "bc_only", "critic_only"]


def _unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def _activation(name: str) -> nn.Module:
    normalized = name.strip().lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name!r}")


class FourierStateEncoding(nn.Module):
    """Append sinusoidal features for selected scalar state coordinates."""

    def __init__(
        self,
        *,
        state_dim: int,
        indices: list[int] | tuple[int, ...],
        num_bands: int,
        max_freq: float,
        scale: float,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.indices = tuple(int(i) for i in indices)
        if any(i < 0 or i >= self.state_dim for i in self.indices):
            raise ValueError(f"State encoding indices must be within [0, {self.state_dim}); got {self.indices}")
        self.num_bands = int(num_bands)
        if self.num_bands < 1:
            raise ValueError("num_bands must be >= 1")
        self.max_freq = float(max_freq)
        self.scale = float(scale)
        freqs = torch.logspace(0.0, torch.log10(torch.tensor(self.max_freq)), self.num_bands)
        self.register_buffer("freqs", freqs.float(), persistent=False)
        self.output_dim = self.state_dim + len(self.indices) * self.num_bands * 2

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        selected = state[:, self.indices] * self.scale
        angles = selected.unsqueeze(-1) * self.freqs.to(device=state.device, dtype=state.dtype).view(1, 1, -1)
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(start_dim=1)
        return torch.cat([state, encoded], dim=-1)


def _make_state_encoder(
    *,
    state_dim: int,
    state_encoding: str,
    state_encoding_indices: list[int] | tuple[int, ...],
    state_encoding_num_bands: int,
    state_encoding_max_freq: float,
    state_encoding_scale: float,
) -> tuple[nn.Module, int]:
    if state_encoding == "none":
        return nn.Identity(), int(state_dim)
    if state_encoding == "fourier":
        encoder = FourierStateEncoding(
            state_dim=state_dim,
            indices=state_encoding_indices,
            num_bands=state_encoding_num_bands,
            max_freq=state_encoding_max_freq,
            scale=state_encoding_scale,
        )
        return encoder, int(encoder.output_dim)
    raise ValueError(f"Unsupported state_encoding: {state_encoding!r}")


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
    if reward_mode == "dataset":
        if "reward" not in df.columns:
            raise ValueError("reward_mode='dataset' requires a 'reward' column. Run add_offline_rewards.py first.")
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

    def __init__(self, act_policy: ACTPolicy, *, action_horizon: int, preprocessor=None, postprocessor=None):
        super().__init__()
        self.act_policy = act_policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
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
        output_device = obs["state"].device
        batch = self._act_batch(obs)
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
        if self.act_policy.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.act_policy.config.image_features]
        chunk = self.act_policy.model(batch)[0]
        if chunk.shape[1] < self.action_horizon:
            raise ValueError(
                f"ACT chunk_size={chunk.shape[1]} is smaller than action_horizon={self.action_horizon}"
            )
        chunk = chunk[:, : self.action_horizon, :]
        if self.postprocessor is not None:
            chunk = self.postprocessor(chunk)
            chunk = chunk.to(output_device)
        return chunk.reshape(chunk.shape[0], -1)

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


def _set_processor_pipeline_device(pipeline: Any, device: str | torch.device) -> None:
    """Keep loaded LeRobot processor DeviceProcessorSteps aligned with this trainer."""
    if pipeline is None:
        return
    target = str(torch.device(device))
    for step in getattr(pipeline, "steps", []):
        if hasattr(step, "device"):
            setattr(step, "device", target)
            post_init = getattr(step, "__post_init__", None)
            if callable(post_init):
                post_init()


def _mlp(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    output_dim: int,
    *,
    layer_norm: bool = False,
    zero_final: bool = True,
    activation: str = "relu",
) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    layers: list[nn.Module] = []
    dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(_activation(activation))
        dim = hidden_dim
    final = nn.Linear(dim, output_dim)
    if zero_final:
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


class GatedActionAdapter(nn.Module):
    """Small residual adapter whose gate starts near zero for ACT preservation."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        layer_norm: bool,
        activation: str,
    ):
        super().__init__()
        self.net = _mlp(
            input_dim,
            hidden_dim,
            num_layers,
            output_dim * 2,
            layer_norm=layer_norm,
            zero_final=True,
            activation=activation,
        )
        self.output_dim = int(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_delta, raw_gate = self.net(x).split(self.output_dim, dim=-1)
        return torch.tanh(raw_delta) * torch.sigmoid(raw_gate)


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
        adapter_arch: str = "mlp",
        adapter_layer_norm: bool = False,
        adapter_activation: str = "relu",
        state_encoding: str = "none",
        state_encoding_indices: list[int] | tuple[int, ...] = (),
        state_encoding_num_bands: int = 4,
        state_encoding_max_freq: float = 8.0,
        state_encoding_scale: float = 1.0,
        adapter_scale: float = 1.0,
        freeze_act: bool = True,
        adapter_delta_clip: float | None = None,
        action_clip: float | None = None,
    ):
        super().__init__(
            act_policy,
            action_horizon=action_horizon,
            preprocessor=getattr(act_policy, "_aic_preprocessor", None),
            postprocessor=getattr(act_policy, "_aic_postprocessor", None),
        )
        self.adapter_scale = float(adapter_scale)
        self.freeze_act = bool(freeze_act)
        self.adapter_delta_clip = None if adapter_delta_clip is None else float(adapter_delta_clip)
        self.action_clip = None if action_clip is None else float(action_clip)
        self.state_encoder, encoded_state_dim = _make_state_encoder(
            state_dim=state_dim,
            state_encoding=state_encoding,
            state_encoding_indices=state_encoding_indices,
            state_encoding_num_bands=state_encoding_num_bands,
            state_encoding_max_freq=state_encoding_max_freq,
            state_encoding_scale=state_encoding_scale,
        )
        adapter_input_dim = encoded_state_dim + self.action_dim
        if adapter_arch == "mlp":
            self.adapter = _mlp(
                input_dim=adapter_input_dim,
                hidden_dim=adapter_hidden_dim,
                num_layers=adapter_num_layers,
                output_dim=self.action_dim,
                layer_norm=adapter_layer_norm,
                zero_final=True,
                activation=adapter_activation,
            )
        elif adapter_arch == "gated":
            self.adapter = GatedActionAdapter(
                input_dim=adapter_input_dim,
                hidden_dim=adapter_hidden_dim,
                num_layers=adapter_num_layers,
                output_dim=self.action_dim,
                layer_norm=adapter_layer_norm,
                activation=adapter_activation,
            )
        else:
            raise ValueError(f"Unsupported adapter_arch: {adapter_arch!r}")
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
        return self.adapter(torch.cat([self.state_encoder(obs["state"]), base_action], dim=-1))

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
    def __init__(
        self,
        *,
        state_dim: int,
        camera_keys: list[str],
        feature_dim: int = 256,
        per_camera_dim: int = 64,
        image_encoder: str = "small_conv",
        layer_norm: bool = False,
        activation: str = "relu",
        state_encoding: str = "none",
        state_encoding_indices: list[int] | tuple[int, ...] = (),
        state_encoding_num_bands: int = 4,
        state_encoding_max_freq: float = 8.0,
        state_encoding_scale: float = 1.0,
    ):
        super().__init__()
        self.camera_keys = list(camera_keys)
        self.image_encoder_name = image_encoder
        self.state_dim = int(state_dim)
        self.feature_dim = int(feature_dim)
        self.per_camera_dim = int(per_camera_dim)
        self.state_encoder, encoded_state_dim = _make_state_encoder(
            state_dim=state_dim,
            state_encoding=state_encoding,
            state_encoding_indices=state_encoding_indices,
            state_encoding_num_bands=state_encoding_num_bands,
            state_encoding_max_freq=state_encoding_max_freq,
            state_encoding_scale=state_encoding_scale,
        )
        if image_encoder == "small_conv":
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=4, padding=2),
                _activation(activation),
                nn.Conv2d(16, 32, kernel_size=3, stride=4, padding=1),
                _activation(activation),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(32, per_camera_dim),
                _activation(activation),
            )
        elif image_encoder in {"resnet18", "resnet18_imagenet"}:
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.IMAGENET1K_V1 if image_encoder == "resnet18_imagenet" else None
            backbone = resnet18(weights=weights)
            backbone.fc = nn.Identity()
            self.image_encoder = nn.Sequential(backbone, nn.Linear(512, per_camera_dim), _activation(activation))
        elif image_encoder in {"convnext_tiny", "convnext_tiny_imagenet"}:
            from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

            weights = (
                ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if image_encoder == "convnext_tiny_imagenet" else None
            )
            backbone = convnext_tiny(weights=weights)
            backbone.classifier = nn.Identity()
            self.image_encoder = nn.Sequential(
                backbone,
                nn.Flatten(),
                nn.Linear(768, per_camera_dim),
                _activation(activation),
            )
        else:
            raise ValueError(f"Unsupported critic image encoder: {image_encoder!r}")
        proj_layers: list[nn.Module] = [
            nn.Linear(encoded_state_dim + per_camera_dim * len(self.camera_keys), feature_dim)
        ]
        if layer_norm:
            proj_layers.append(nn.LayerNorm(feature_dim))
        proj_layers.append(_activation(activation))
        self.proj = nn.Sequential(*proj_layers)

    def forward(self, obs: dict[str, Any]) -> torch.Tensor:
        image_features = [self.image_encoder(obs["images"][key]) for key in self.camera_keys]
        return self.proj(torch.cat([self.state_encoder(obs["state"]), *image_features], dim=-1))


class VisionCritic(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        camera_keys: list[str],
        action_dim: int,
        feature_dim: int = 256,
        image_encoder: str = "small_conv",
        arch: str = "concat",
        hidden_dim: int = 256,
        num_layers: int = 2,
        per_camera_dim: int = 64,
        layer_norm: bool = False,
        activation: str = "relu",
        state_encoding: str = "none",
        state_encoding_indices: list[int] | tuple[int, ...] = (),
        state_encoding_num_bands: int = 4,
        state_encoding_max_freq: float = 8.0,
        state_encoding_scale: float = 1.0,
    ):
        super().__init__()
        self.arch = arch
        self.action_dim = int(action_dim)
        self.encoder = ImageStateEncoder(
            state_dim=state_dim,
            camera_keys=camera_keys,
            feature_dim=feature_dim,
            image_encoder=image_encoder,
            per_camera_dim=per_camera_dim,
            layer_norm=layer_norm,
            activation=activation,
            state_encoding=state_encoding,
            state_encoding_indices=state_encoding_indices,
            state_encoding_num_bands=state_encoding_num_bands,
            state_encoding_max_freq=state_encoding_max_freq,
            state_encoding_scale=state_encoding_scale,
        )
        if arch == "concat":
            self.q = _mlp(
                feature_dim + action_dim,
                hidden_dim,
                num_layers,
                1,
                layer_norm=layer_norm,
                zero_final=False,
                activation=activation,
            )
        elif arch == "multiplicative":
            self.action_proj = nn.Sequential(nn.Linear(action_dim, feature_dim), _activation(activation))
            self.q = _mlp(
                feature_dim * 3,
                hidden_dim,
                num_layers,
                1,
                layer_norm=layer_norm,
                zero_final=False,
                activation=activation,
            )
        elif arch == "value_advantage":
            self.value = _mlp(
                feature_dim,
                hidden_dim,
                num_layers,
                1,
                layer_norm=layer_norm,
                zero_final=False,
                activation=activation,
            )
            self.advantage = _mlp(
                feature_dim + action_dim,
                hidden_dim,
                num_layers,
                1,
                layer_norm=layer_norm,
                zero_final=False,
                activation=activation,
            )
        else:
            raise ValueError(f"Unsupported critic architecture: {arch!r}")

    def forward(self, obs: dict[str, Any], action: torch.Tensor) -> torch.Tensor:
        obs_feature = self.encoder(obs)
        if self.arch == "concat":
            return self.q(torch.cat([obs_feature, action], dim=-1))
        if self.arch == "multiplicative":
            action_feature = self.action_proj(action)
            return self.q(torch.cat([obs_feature, action_feature, obs_feature * action_feature], dim=-1))
        if self.arch == "value_advantage":
            return self.value(obs_feature) + self.advantage(torch.cat([obs_feature, action], dim=-1))
        raise RuntimeError(f"Unsupported critic architecture at forward: {self.arch!r}")


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
    actor_update_mode: ActorUpdateMode = "q_bc"
    freeze_act: bool = True
    critic_image_encoder: str = "small_conv"
    critic_arch: str = "concat"
    critic_feature_dim: int = 256
    critic_hidden_dim: int = 256
    critic_num_layers: int = 2
    critic_per_camera_dim: int = 64
    critic_layer_norm: bool = False
    critic_activation: str = "gelu"
    adapter_arch: str = "mlp"
    adapter_layer_norm: bool = False
    adapter_activation: str = "gelu"
    state_encoding: str = "fourier"
    state_encoding_indices: tuple[int, ...] = ()
    state_encoding_num_bands: int = 4
    state_encoding_max_freq: float = 8.0
    state_encoding_scale: float = 10.0


class VisionOfflineSERLTrainer:
    def __init__(self, *, config: VisionOfflineSERLConfig, actor: ACTChunkActor, device: str | torch.device):
        self.config = config
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.critic1 = VisionCritic(
            state_dim=config.state_dim,
            camera_keys=config.camera_keys,
            action_dim=config.action_dim,
            image_encoder=config.critic_image_encoder,
            arch=config.critic_arch,
            feature_dim=config.critic_feature_dim,
            hidden_dim=config.critic_hidden_dim,
            num_layers=config.critic_num_layers,
            per_camera_dim=config.critic_per_camera_dim,
            layer_norm=config.critic_layer_norm,
            activation=config.critic_activation,
            state_encoding=config.state_encoding,
            state_encoding_indices=config.state_encoding_indices,
            state_encoding_num_bands=config.state_encoding_num_bands,
            state_encoding_max_freq=config.state_encoding_max_freq,
            state_encoding_scale=config.state_encoding_scale,
        ).to(self.device)
        self.critic2 = VisionCritic(
            state_dim=config.state_dim,
            camera_keys=config.camera_keys,
            action_dim=config.action_dim,
            image_encoder=config.critic_image_encoder,
            arch=config.critic_arch,
            feature_dim=config.critic_feature_dim,
            hidden_dim=config.critic_hidden_dim,
            num_layers=config.critic_num_layers,
            per_camera_dim=config.critic_per_camera_dim,
            layer_norm=config.critic_layer_norm,
            activation=config.critic_activation,
            state_encoding=config.state_encoding,
            state_encoding_indices=config.state_encoding_indices,
            state_encoding_num_bands=config.state_encoding_num_bands,
            state_encoding_max_freq=config.state_encoding_max_freq,
            state_encoding_scale=config.state_encoding_scale,
        ).to(self.device)
        self.target_critic1 = VisionCritic(
            state_dim=config.state_dim,
            camera_keys=config.camera_keys,
            action_dim=config.action_dim,
            image_encoder=config.critic_image_encoder,
            arch=config.critic_arch,
            feature_dim=config.critic_feature_dim,
            hidden_dim=config.critic_hidden_dim,
            num_layers=config.critic_num_layers,
            per_camera_dim=config.critic_per_camera_dim,
            layer_norm=config.critic_layer_norm,
            activation=config.critic_activation,
            state_encoding=config.state_encoding,
            state_encoding_indices=config.state_encoding_indices,
            state_encoding_num_bands=config.state_encoding_num_bands,
            state_encoding_max_freq=config.state_encoding_max_freq,
            state_encoding_scale=config.state_encoding_scale,
        ).to(self.device)
        self.target_critic2 = VisionCritic(
            state_dim=config.state_dim,
            camera_keys=config.camera_keys,
            action_dim=config.action_dim,
            image_encoder=config.critic_image_encoder,
            arch=config.critic_arch,
            feature_dim=config.critic_feature_dim,
            hidden_dim=config.critic_hidden_dim,
            num_layers=config.critic_num_layers,
            per_camera_dim=config.critic_per_camera_dim,
            layer_norm=config.critic_layer_norm,
            activation=config.critic_activation,
            state_encoding=config.state_encoding,
            state_encoding_indices=config.state_encoding_indices,
            state_encoding_num_bands=config.state_encoding_num_bands,
            state_encoding_max_freq=config.state_encoding_max_freq,
            state_encoding_scale=config.state_encoding_scale,
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
            random_q1 = self.critic1(obs, random_action)
            random_q2 = self.critic2(obs, random_action)
            conservative_loss = (
                torch.logsumexp(torch.cat([q1, random_q1], dim=-1), dim=-1).mean()
                - q1.mean()
                + torch.logsumexp(torch.cat([q2, random_q2], dim=-1), dim=-1).mean()
                - q2.mean()
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
        q_actor_loss = -actor_q.mean()
        if self.config.actor_update_mode == "q_bc":
            actor_loss = (
                q_actor_loss
                + self.config.bc_weight * bc_loss
                + self.config.adapter_penalty_weight * adapter_penalty
                + self.config.act_preservation_weight * act_preservation_loss
                + self.config.smoothness_weight * smoothness_loss
            )
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
        elif self.config.actor_update_mode == "bc_only":
            actor_loss = (
                self.config.bc_weight * bc_loss
                + self.config.adapter_penalty_weight * adapter_penalty
                + self.config.act_preservation_weight * act_preservation_loss
                + self.config.smoothness_weight * smoothness_loss
            )
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
        elif self.config.actor_update_mode == "critic_only":
            actor_loss = (
                q_actor_loss
                + self.config.bc_weight * bc_loss
                + self.config.adapter_penalty_weight * adapter_penalty
                + self.config.act_preservation_weight * act_preservation_loss
                + self.config.smoothness_weight * smoothness_loss
            ).detach()
        else:
            raise ValueError(f"Unsupported actor_update_mode: {self.config.actor_update_mode!r}")
        self.soft_update_targets()

        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "q_actor_loss": float(q_actor_loss.detach().cpu()),
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
            random_q1 = self.critic1(obs, random_action)
            random_q2 = self.critic2(obs, random_action)
            conservative_loss = (
                torch.logsumexp(torch.cat([q1, random_q1], dim=-1), dim=-1).mean()
                - q1.mean()
                + torch.logsumexp(torch.cat([q2, random_q2], dim=-1), dim=-1).mean()
                - q2.mean()
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
        q_actor_loss = -actor_q.mean()
        actor_loss = (
            -actor_q.mean()
            + self.config.bc_weight * bc_loss
            + self.config.adapter_penalty_weight * adapter_penalty
            + self.config.act_preservation_weight * act_preservation_loss
            + self.config.smoothness_weight * smoothness_loss
        )

        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "q_actor_loss": float(q_actor_loss.detach().cpu()),
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
    adapter_arch: str = "mlp",
    adapter_layer_norm: bool = False,
    adapter_activation: str = "gelu",
    state_encoding: str = "fourier",
    state_encoding_indices: list[int] | tuple[int, ...] = (0, 1, 2, 13, 14, 15),
    state_encoding_num_bands: int = 4,
    state_encoding_max_freq: float = 8.0,
    state_encoding_scale: float = 10.0,
    adapter_scale: float = 1.0,
    freeze_act: bool = True,
    adapter_delta_clip: float | None = None,
    action_clip: float | None = None,
) -> tuple[ACTChunkActor, dict[str, Any]]:
    checkpoint_dir = resolve_act_checkpoint_dir(checkpoint)
    policy = ACTPolicy.from_pretrained(checkpoint_dir, local_files_only=True)
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=checkpoint_dir)
    _set_processor_pipeline_device(preprocessor, device)
    _set_processor_pipeline_device(postprocessor, device)
    policy._aic_preprocessor = preprocessor
    policy._aic_postprocessor = postprocessor
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
            adapter_arch=adapter_arch,
            adapter_layer_norm=adapter_layer_norm,
            adapter_activation=adapter_activation,
            state_encoding=state_encoding,
            state_encoding_indices=state_encoding_indices,
            state_encoding_num_bands=state_encoding_num_bands,
            state_encoding_max_freq=state_encoding_max_freq,
            state_encoding_scale=state_encoding_scale,
            adapter_scale=adapter_scale,
            freeze_act=freeze_act,
            adapter_delta_clip=adapter_delta_clip,
            action_clip=action_clip,
        )
    elif actor_mode == "act_direct":
        actor = ACTChunkActor(policy, action_horizon=action_horizon, preprocessor=preprocessor, postprocessor=postprocessor)
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
            "adapter_arch": adapter_arch if isinstance(actor, ACTAdapterSERLActor) else None,
            "adapter_layer_norm": adapter_layer_norm if isinstance(actor, ACTAdapterSERLActor) else None,
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
