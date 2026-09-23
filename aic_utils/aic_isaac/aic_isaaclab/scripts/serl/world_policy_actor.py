"""Adapter for zero-shot Isaac rollout of the full-data world policy.

This module deliberately supports one environment and inference only.  The
deployed controller owns temporal KV caches and requires the four commands
actually executed between decisions.  Vector reset masks and 200 ms macro
replay must be implemented before this class is enabled for gradient updates.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CAMERA_KEYS = (
    "observation.images.center_camera",
    "observation.images.left_camera",
    "observation.images.right_camera",
)


class WorldPolicyRuntimeNormalizer:
    """Compatibility surface used by existing Isaac diagnostics."""

    def __init__(self, normalization: dict[str, Any], *, device: torch.device):
        self.state_mean = torch.tensor(normalization["state_mean"], device=device).view(1, -1)
        self.state_std = torch.tensor(normalization["state_std"], device=device).view(1, -1)
        self.action_mean = torch.tensor(normalization["action_map"]["center"], device=device).view(1, -1)
        self.action_std = torch.tensor(normalization["action_map"]["scale"], device=device).view(1, -1)

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != 42:
            raise ValueError(f"World policy expects base32 + task10 state, got {state.shape[-1]} values")
        base = state[:, :32].clone()
        base[:, 13:19] = 0.0
        quat = base[:, 3:7]
        quat = torch.where(quat[:, :1] < 0.0, -quat, quat)
        base[:, 3:7] = torch.nn.functional.normalize(quat, dim=-1)
        normalized = (base - self.state_mean) / self.state_std
        return torch.cat((normalized, state[:, 32:42]), dim=-1)

    @staticmethod
    def normalize_image(_key: str, image: torch.Tensor) -> torch.Tensor:
        return image


def exact_six_views(images: dict[str, torch.Tensor], *, size: int = 144) -> np.ndarray:
    """Apply the exact PIL preprocessing used by the Gazebo runtime.

    The frozen policy's tokenizer is not differentiated during zero-shot
    rollout, so retaining exact byte-level preprocessing is preferable to a
    faster but subtly different GPU resize at this gate.
    """

    arrays: list[np.ndarray] = []
    batch_size: int | None = None
    for key in CAMERA_KEYS:
        value = images[key].detach().cpu()
        if value.ndim != 4 or value.shape[1] != 3:
            raise ValueError(f"Expected BCHW RGB image for {key}, got {tuple(value.shape)}")
        batch_size = int(value.shape[0]) if batch_size is None else batch_size
        if int(value.shape[0]) != batch_size:
            raise ValueError("Camera batches have different sizes")
        if value.dtype == torch.uint8:
            byte = value
        else:
            byte = (value.float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        arrays.append(byte.permute(0, 2, 3, 1).contiguous().numpy())
    if batch_size != 1:
        raise ValueError(
            "World-policy temporal cache currently supports exactly one Isaac environment; "
            "batched reset masks are required before scaling"
        )

    from dreamer4.aic.data import camera_views

    return camera_views([array[0] for array in arrays], size=size, views=6)[None]


class IsaacWorldPolicyActor(nn.Module):
    """Exact current world-policy controller for bounded Isaac rollout."""

    actor_mode = "world_policy"
    state_dim = 42
    action_dim = 24
    action_horizon = 4
    single_action_dim = 6
    adapter_delta_clip = None
    tcp_translation_action_clip = None
    tcp_rotation_action_clip = None
    action_clip = None

    def __init__(self, *, checkpoint: Path, source: Path, device: torch.device):
        super().__init__()
        source = Path(source).resolve()
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from dreamer4.aic.models import AICController
        from dreamer4.aic.train import load

        model, saved = load(Path(checkpoint), device, control_only=True)
        if saved.get("options", {}).get("action_representation") != "tcp_delta_observation":
            raise ValueError("World policy checkpoint does not use observation-relative TCP deltas")
        if int(model.config.aic_views) != 6 or int(model.config.image_size) != 144:
            raise ValueError(
                f"Expected selected six-view RGB144 policy, got views={model.config.aic_views}, "
                f"size={model.config.image_size}"
            )
        model.eval().requires_grad_(False)
        # The future macro-RL path updates the complete policy head.  The
        # tokenizer/world remain frozen until a separately approved ablation.
        model.heads.policy.requires_grad_(True)
        self.model = model
        self.checkpoint = Path(checkpoint)
        self.act_torchscript_path = self.checkpoint
        self.source = source
        self.device = torch.device(device)
        self.normalization = saved["normalization"]
        self.act_normalizer = WorldPolicyRuntimeNormalizer(self.normalization, device=self.device)
        self.controller = AICController(self.model, self.normalization, self.device, "bf16")
        self.head = self.model.heads.policy
        self.register_buffer("_diagnostic_log_std", torch.tensor([-2.0], device=self.device), persistent=False)
        self._previous_chunk: np.ndarray | None = None
        self._decision = 0
        self._prefetched: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def log_std(self) -> torch.Tensor:
        return self._diagnostic_log_std

    def reset(self) -> None:
        self.controller.reset()
        self._previous_chunk = None
        self._decision = 0
        self._prefetched = None

    def _physical_action_from_feature(self, feature: torch.Tensor) -> torch.Tensor:
        """Run only the trainable policy head on a frozen causal feature."""
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=feature.device.type == "cuda",
        ):
            normalized = self.head(feature).sample(deterministic=True)["continuous"]
        center = torch.as_tensor(
            self.normalization["action_map"]["center"], device=feature.device, dtype=torch.float32
        ).repeat(4)
        scale = torch.as_tensor(
            self.normalization["action_map"]["scale"], device=feature.device, dtype=torch.float32
        ).repeat(4)
        return normalized.float().reshape(feature.shape[0], 24) * scale + center

    @torch.inference_mode()
    def _causal_action_and_feature(self, obs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the frozen tokenizer/world cache by one 200 ms decision."""
        from dreamer4.actions import empty_actions

        state = obs["state"]
        if state.shape != (1, 42):
            raise ValueError(f"World policy path requires state shape (1,42), got {tuple(state.shape)}")
        images = torch.as_tensor(exact_six_views(obs["images"], size=144), device=self.device).float() / 255.0
        base = state[:, :32].to(device=self.device, dtype=torch.float32).clone()
        base[:, 13:19] = 0.0
        base[:, 3:7] = torch.where(base[:, 3:4] < 0.0, -base[:, 3:7], base[:, 3:7])
        base[:, 3:7] = F.normalize(base[:, 3:7], dim=-1)
        mean = torch.as_tensor(self.normalization["state_mean"], device=self.device)
        std = torch.as_tensor(self.normalization["state_std"], device=self.device)
        normalized = (base - mean) / std
        task = state[:, 32:42].to(device=self.device, dtype=torch.float32)
        elapsed = torch.tensor([[min(self._decision * 0.2, 40.0) / 40.0]], device=self.device)
        context = torch.cat((task, elapsed), dim=-1)[:, None]
        incoming = empty_actions(self.model.config.actions, (1, 1), self.device)
        if self.controller.offset:
            if self._previous_chunk is None:
                raise RuntimeError("Causal world-policy history is missing the previously executed chunk")
            previous = torch.as_tensor(self._previous_chunk, device=self.device, dtype=torch.float32).reshape(1, 4, 6)
            center = torch.as_tensor(self.normalization["action_map"]["center"], device=self.device)
            scale = torch.as_tensor(self.normalization["action_map"]["scale"], device=self.device)
            incoming["continuous"] = ((previous - center) / scale).reshape(1, 1, 24)
        present = torch.full((1, 1), bool(self.controller.offset), dtype=torch.bool, device=self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            visual, self.controller.token_cache = self.model.encode_visual(
                images[:, None], self.controller.token_cache, self.controller.offset
            )
            clean = self.model.pack_observation(visual, normalized[:, None])
            features, self.controller.world_cache = self.model.observed_features(
                clean,
                incoming,
                present,
                context,
                self.controller.world_cache,
                self.controller.offset,
            )
        feature = features[:, -1].float().detach()
        action = self._physical_action_from_feature(feature).detach()
        self.controller.offset += 1
        self._previous_chunk = action.cpu().numpy().copy()
        self._decision += 1
        return action, feature

    def _components(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = torch.zeros_like(action)
        return {
            "base_action": action.detach(),
            "raw_delta_action": zero,
            "delta_action": zero,
            "unclipped_final_action": action,
            "translation_clipped_action": action,
            "rotation_clipped_action": action,
            "final_action": action,
        }

    def action_components(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        if "world_feature" in obs:
            return self._components(self._physical_action_from_feature(obs["world_feature"]))
        if self._prefetched is not None:
            action, feature = self._prefetched
            self._prefetched = None
        else:
            action, feature = self._causal_action_and_feature(obs)
        obs["world_feature"] = feature
        return self._components(action)

    @torch.inference_mode()
    def prefetch_next(self, obs: dict[str, Any]) -> None:
        """Encode the next decision once so replay receives its causal feature."""
        if self._prefetched is not None:
            raise RuntimeError("World-policy next decision was prefetched twice")
        action, feature = self._causal_action_and_feature(obs)
        obs["world_feature"] = feature
        self._prefetched = (action, feature)

    def record_executed_chunk(self, action: torch.Tensor) -> None:
        """Replace proposal history with the four commands actually sent to Isaac."""
        if action.shape != (1, 24):
            raise ValueError(f"Expected one executed 24D chunk, got {tuple(action.shape)}")
        self._previous_chunk = action.detach().to(dtype=torch.float32, device="cpu").numpy().copy()

    def mean_action(self, obs: dict[str, Any]) -> torch.Tensor:
        feature = obs.get("world_feature")
        if feature is None:
            raise RuntimeError("World-policy replay observation is missing its frozen causal feature")
        return self._physical_action_from_feature(feature)
