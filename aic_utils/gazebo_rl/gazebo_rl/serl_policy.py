from __future__ import annotations

import base64
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file


LEROBOT_AIC_PACKAGE_DIR = Path(__file__).resolve().parents[2] / "lerobot_robot_aic"
if LEROBOT_AIC_PACKAGE_DIR.exists() and str(LEROBOT_AIC_PACKAGE_DIR) in sys.path:
    sys.path.remove(str(LEROBOT_AIC_PACKAGE_DIR))
if LEROBOT_AIC_PACKAGE_DIR.exists():
    sys.path.insert(0, str(LEROBOT_AIC_PACKAGE_DIR))

ACT_CAMERA_KEYS = [
    "observation.images.center_camera",
    "observation.images.left_camera",
    "observation.images.right_camera",
]

from lerobot_robot_aic.runtime_features import AICRuntimeFeatureAssembler, base_state_from_gazebo_observation


def task_vector_from_context(
    *,
    task_family: str | None = None,
    target_port_index: int | None = None,
    target_card_index: int | None = None,
    target_card_valid: int | None = None,
    task_context_json: str | dict[str, Any] | None = None,
) -> np.ndarray | None:
    if task_context_json is None and task_family is None:
        return None
    try:
        from lerobot_robot_aic.task_encoding import encode_task_vector, encode_task_vector_from_metadata
    except ModuleNotFoundError:
        from aic_utils.lerobot_robot_aic.lerobot_robot_aic.task_encoding import (
            encode_task_vector,
            encode_task_vector_from_metadata,
        )

    if task_context_json is not None:
        import json

        data = json.loads(task_context_json) if isinstance(task_context_json, str) else task_context_json
        if not isinstance(data, dict):
            raise ValueError("task_context_json must decode to an object")
        if task_family is None:
            return encode_task_vector_from_metadata(data)
    if task_family is None or target_port_index is None or target_card_index is None:
        raise ValueError(
            "Task context requires task_family, target_port_index, and target_card_index "
            "unless task_context_json supplies them."
        )
    return encode_task_vector(
        task_family=task_family,
        target_port_index=int(target_port_index),
        target_card_index=int(target_card_index),
        target_card_valid=target_card_valid,
    )


def lowdim_state_from_gazebo_observation(obs: dict[str, Any]) -> np.ndarray:
    return base_state_from_gazebo_observation(obs)


class OfflineSERLGazeboPolicy:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        task_vector: np.ndarray | None = None,
    ):
        try:
            from lerobot_robot_aic.offline_serl import GaussianActor
        except ModuleNotFoundError:
            from aic_utils.lerobot_robot_aic.lerobot_robot_aic.offline_serl import GaussianActor

        self.device = torch.device(device)
        ckpt = torch.load(checkpoint, map_location=self.device)
        cfg = ckpt.get("offline_serl_config") or {}
        self.task_vector = None if task_vector is None else torch.as_tensor(
            task_vector, dtype=torch.float32, device=self.device
        )
        obs_dim = int(cfg.get("obs_dim", 0))
        self.feature_assembler = AICRuntimeFeatureAssembler(obs_dim, task_vector=task_vector)
        if self.feature_assembler.uses_task_vector and task_vector is None:
            raise ValueError(
                f"Gazebo SERL policy got obs_dim={cfg.get('obs_dim')}; pass task context "
                "when loading a task/contact-conditioned checkpoint."
            )
        self.action_horizon = int(cfg.get("action_horizon", 1))
        self.single_action_dim = int((ckpt.get("dataset_schema") or {}).get("action_shape", [6])[0])
        hidden = tuple(int(cfg.get("hidden_dim", 256)) for _ in range(int(cfg.get("num_layers", 2))))
        self.actor = GaussianActor(
            obs_dim=int(cfg["obs_dim"]),
            action_dim=int(cfg["action_dim"]),
            hidden_dims=hidden,
        ).to(self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()
        stats = ckpt.get("normalization_stats") or {}
        self.obs_mean = torch.as_tensor(stats["obs_mean"], dtype=torch.float32, device=self.device)
        obs_std = torch.as_tensor(stats["obs_std"], dtype=torch.float32, device=self.device)
        self.obs_std = torch.where(torch.abs(obs_std) < 1.0e-8, torch.ones_like(obs_std), obs_std)
        self.action_mean = torch.as_tensor(stats["action_mean"], dtype=torch.float32, device=self.device)
        self.action_std = torch.as_tensor(stats["action_std"], dtype=torch.float32, device=self.device)

    def act(self, obs: dict[str, Any], *, explore: bool = False) -> list[float]:
        del explore
        lowdim = torch.as_tensor(self.feature_assembler.assemble_gazebo(obs), device=self.device).unsqueeze(0)
        if lowdim.shape[-1] != self.obs_mean.shape[-1]:
            raise ValueError(f"Expected runtime state dim {self.obs_mean.shape[-1]}, got {lowdim.shape[-1]}")
        lowdim = (lowdim - self.obs_mean) / self.obs_std
        with torch.no_grad():
            normalized_action = self.actor.mean_action(lowdim).squeeze(0)
        action = normalized_action * self.action_std + self.action_mean
        return action[: self.single_action_dim].detach().cpu().numpy().astype(float).tolist()


def _activation(name: str) -> nn.Module:
    normalized = name.strip().lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name!r}")


class FourierStateEncoding(nn.Module):
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


def _torchscript_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"ACT TorchScript metadata not found: {metadata_path}")
    import json

    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _resolve_act_checkpoint_dir(path: Path) -> Path:
    metadata = _torchscript_metadata(path)
    checkpoint_dir = Path(str(metadata.get("checkpoint_dir", "")))
    candidates = [checkpoint_dir] if checkpoint_dir.is_absolute() else [Path.cwd() / checkpoint_dir, path.parent / checkpoint_dir]
    for candidate in candidates:
        if (candidate / "policy_preprocessor_step_3_normalizer_processor.safetensors").exists():
            return candidate
    raise FileNotFoundError(
        "Could not resolve ACT checkpoint normalizer directory from "
        f"{path.with_suffix('.json')}: tried {', '.join(str(c) for c in candidates)}"
    )


class ACTRuntimeNormalizer(nn.Module):
    """LeRobot ACT runtime scaling around the exported TorchScript model."""

    def __init__(self, act_torchscript_path: Path, *, state_dim: int, action_dim: int):
        super().__init__()
        checkpoint_dir = _resolve_act_checkpoint_dir(act_torchscript_path)
        stats = load_file(str(checkpoint_dir / "policy_preprocessor_step_3_normalizer_processor.safetensors"))

        def stat(key: str, shape: tuple[int, ...]) -> torch.Tensor:
            if key not in stats:
                raise KeyError(f"ACT normalizer stats are missing {key!r}")
            tensor = stats[key].float().view(*shape)
            if key.endswith(".std"):
                tensor = torch.where(torch.abs(tensor) < 1.0e-8, torch.ones_like(tensor), tensor)
            return tensor

        state_mean = stat("observation.state.mean", (1, -1))[:, :state_dim]
        state_std = stat("observation.state.std", (1, -1))[:, :state_dim]
        if state_dim in (42, 82):
            state_mean[:, -10:] = 0.0
            state_std[:, -10:] = 1.0

        self.register_buffer("state_mean", state_mean, persistent=False)
        self.register_buffer("state_std", state_std, persistent=False)
        self.register_buffer("action_mean", stat("action.mean", (1, -1))[:, :action_dim], persistent=False)
        self.register_buffer("action_std", stat("action.std", (1, -1))[:, :action_dim], persistent=False)
        for key in ACT_CAMERA_KEYS:
            safe_key = key.replace(".", "__")
            self.register_buffer(f"{safe_key}_mean", stat(f"{key}.mean", (1, 3, 1, 1)), persistent=False)
            self.register_buffer(f"{safe_key}_std", stat(f"{key}.std", (1, 3, 1, 1)), persistent=False)

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_mean.to(device=state.device, dtype=state.dtype)) / self.state_std.to(
            device=state.device, dtype=state.dtype
        )

    def normalize_image(self, key: str, image: torch.Tensor) -> torch.Tensor:
        safe_key = key.replace(".", "__")
        mean = getattr(self, f"{safe_key}_mean").to(device=image.device, dtype=image.dtype)
        std = getattr(self, f"{safe_key}_std").to(device=image.device, dtype=image.dtype)
        return (image - mean) / std

    def unnormalize_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        return normalized_action * self.action_std.to(
            device=normalized_action.device, dtype=normalized_action.dtype
        ) + self.action_mean.to(device=normalized_action.device, dtype=normalized_action.dtype)


class IdentityACTRuntimeNormalizer(nn.Module):
    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def normalize_image(self, key: str, image: torch.Tensor) -> torch.Tensor:
        del key
        return image

    def unnormalize_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        return normalized_action


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


def _infer_adapter_shape(actor_state: dict[str, torch.Tensor]) -> tuple[int, int]:
    linear_weights = [
        (key, value)
        for key, value in actor_state.items()
        if key.startswith("adapter.") and key.endswith(".weight")
    ]
    if not linear_weights:
        raise ValueError("Checkpoint actor state does not contain adapter.*.weight tensors")
    linear_weights.sort(key=lambda item: int(item[0].split(".")[1]))
    hidden_dim = int(linear_weights[0][1].shape[0])
    num_layers = len(linear_weights) - 1
    return hidden_dim, num_layers


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
        online_gazebo_cfg = checkpoint.get("online_gazebo_serl_config") or {}
        source = online_gazebo_cfg.get("checkpoint") or {}
        offline_cfg = source.get("vision_offline_serl_config")
    if offline_cfg is None:
        raise KeyError(
            "Checkpoint does not contain vision_offline_serl_config and does not look like "
            "an online SERL checkpoint with online_serl_config.checkpoint.vision_offline_serl_config "
            "or online_gazebo_serl_config.checkpoint.vision_offline_serl_config."
        )
    return (
        offline_cfg,
        source.get("dataset_summary") or {},
        source.get("warmstart_report") or {},
    )


def _adapter_scale_from_checkpoint(checkpoint: dict[str, Any], warmstart: dict[str, Any]) -> float:
    if "adapter_scale" in warmstart:
        return float(warmstart["adapter_scale"])
    online_cfg = checkpoint.get("online_serl_config") or {}
    args = online_cfg.get("args") or {}
    if "adapter_scale" in args:
        return float(args["adapter_scale"])
    online_gazebo_cfg = checkpoint.get("online_gazebo_serl_config") or {}
    args = online_gazebo_cfg.get("args") or {}
    if "adapter_scale" in args:
        return float(args["adapter_scale"])
    return 1.0


class TorchScriptACTAdapterActor(nn.Module):
    """TorchScript ACT base plus a trainable SERL adapter loaded for inference."""

    def __init__(
        self,
        *,
        act_base: torch.jit.ScriptModule,
        act_torchscript_path: Path | None = None,
        state_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int,
        num_layers: int,
        adapter_scale: float,
        adapter_delta_clip: float | None,
        action_clip: float | None,
        adapter_activation: str = "relu",
        state_encoding: str = "none",
        state_encoding_indices: list[int] | tuple[int, ...] = (),
        state_encoding_num_bands: int = 4,
        state_encoding_max_freq: float = 8.0,
        state_encoding_scale: float = 1.0,
        adapter_arch: str = "mlp",
        adapter_layer_norm: bool = False,
    ):
        super().__init__()
        self.act_base = act_base
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.single_action_dim = self.action_dim // self.action_horizon
        self.act_normalizer = (
            ACTRuntimeNormalizer(
                act_torchscript_path=Path(act_torchscript_path),
                state_dim=self.state_dim,
                action_dim=self.single_action_dim,
            )
            if act_torchscript_path is not None
            else IdentityACTRuntimeNormalizer()
        )
        self.adapter_scale = float(adapter_scale)
        self.adapter_delta_clip = None if adapter_delta_clip is None else float(adapter_delta_clip)
        self.action_clip = None if action_clip is None else float(action_clip)
        self.state_encoder, encoded_state_dim = _make_state_encoder(
            state_dim=self.state_dim,
            state_encoding=state_encoding,
            state_encoding_indices=state_encoding_indices,
            state_encoding_num_bands=state_encoding_num_bands,
            state_encoding_max_freq=state_encoding_max_freq,
            state_encoding_scale=state_encoding_scale,
        )
        adapter_input_dim = encoded_state_dim + self.action_dim
        if adapter_arch == "mlp":
            self.adapter = _mlp(
                adapter_input_dim,
                hidden_dim,
                num_layers,
                self.action_dim,
                layer_norm=adapter_layer_norm,
                zero_final=True,
                activation=adapter_activation,
            )
        elif adapter_arch == "gated":
            self.adapter = GatedActionAdapter(
                input_dim=adapter_input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=self.action_dim,
                layer_norm=adapter_layer_norm,
                activation=adapter_activation,
            )
        else:
            raise ValueError(f"Unsupported adapter_arch: {adapter_arch!r}")
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -2.0))

    def action_components(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        normalized_state = self.act_normalizer.normalize_state(obs["state"])
        normalized_images = {
            key: self.act_normalizer.normalize_image(key, value)
            for key, value in obs["images"].items()
        }
        chunk = self.act_base(
            normalized_state,
            normalized_images["observation.images.center_camera"],
            normalized_images["observation.images.left_camera"],
            normalized_images["observation.images.right_camera"],
        )
        base_action = self.act_normalizer.unnormalize_action(chunk)
        base_action = base_action[:, : self.action_horizon, :].reshape(obs["state"].shape[0], -1)
        encoded_state = self.state_encoder(obs["state"])
        raw_delta_action = self.adapter(torch.cat([encoded_state, base_action], dim=-1))
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
) -> TorchScriptACTAdapterActor:
    actor_state = checkpoint["actor"]
    hidden_dim, num_layers = _infer_adapter_shape(actor_state)
    offline_cfg, _, warmstart = _checkpoint_training_context(checkpoint)
    act_base = torch.jit.load(str(act_torchscript), map_location=device).eval()
    for param in act_base.parameters():
        param.requires_grad = False
    actor = TorchScriptACTAdapterActor(
        act_base=act_base,
        act_torchscript_path=act_torchscript if Path(act_torchscript).with_suffix(".json").exists() else None,
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        adapter_scale=_adapter_scale_from_checkpoint(checkpoint, warmstart),
        adapter_delta_clip=adapter_delta_clip,
        action_clip=action_clip,
        adapter_activation=str(offline_cfg.get("adapter_activation", "relu")),
        state_encoding=str(offline_cfg.get("state_encoding", "none")),
        state_encoding_indices=tuple(int(i) for i in offline_cfg.get("state_encoding_indices", ())),
        state_encoding_num_bands=int(offline_cfg.get("state_encoding_num_bands", 4)),
        state_encoding_max_freq=float(offline_cfg.get("state_encoding_max_freq", 8.0)),
        state_encoding_scale=float(offline_cfg.get("state_encoding_scale", 1.0)),
        adapter_arch=str(offline_cfg.get("adapter_arch", "mlp")),
        adapter_layer_norm=bool(offline_cfg.get("adapter_layer_norm", False)),
    ).to(device)
    own_state = actor.state_dict()
    compatible = {
        key: value
        for key, value in actor_state.items()
        if key in own_state and tuple(value.shape) == tuple(own_state[key].shape)
    }
    own_state.update(compatible)
    actor.load_state_dict(own_state, strict=True)
    actor.eval()
    return actor


def _as_image_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, dict) and "data_b64" in value:
        height = int(value.get("height", 0))
        width = int(value.get("width", 0))
        encoding = str(value.get("encoding", "rgb8")).lower()
        raw = base64.b64decode(str(value["data_b64"]))
        if encoding in {"jpeg_rgb8", "jpg_rgb8", "jpeg", "jpg"}:
            from io import BytesIO

            from PIL import Image

            with Image.open(BytesIO(raw)) as decoded:
                image = torch.from_numpy(np.asarray(decoded.convert("RGB")).copy())
        else:
            channels = 4 if encoding in {"rgba8", "bgra8"} else 1 if encoding in {"mono8", "8uc1"} else 3
            image = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            expected = height * width * channels
            if image.numel() < expected:
                raise ValueError(
                    f"Encoded image has {image.numel()} values, expected at least {expected} "
                    f"for {height}x{width} {encoding}"
                )
            image = image[:expected].reshape(height, width, channels)
            if encoding in {"bgr8", "bgra8"}:
                image = image[..., [2, 1, 0]]
            elif encoding in {"rgba8", "rgb8"}:
                image = image[..., :3]
            elif encoding in {"mono8", "8uc1"}:
                image = image.repeat(1, 1, 3)
        value = image
    image = torch.as_tensor(value, dtype=torch.float32, device=device)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected image tensor/list with 3 or 4 dims, got shape {tuple(image.shape)}")
    if image.shape[-1] in (3, 4):
        image = image[..., :3].permute(0, 3, 1, 2).contiguous()
    elif image.shape[1] != 3:
        raise ValueError(f"Expected image channel dimension of 3/4, got shape {tuple(image.shape)}")
    if image.max() > 2.0:
        image = image / 255.0
    return F.interpolate(image, size=(256, 288), mode="bilinear", align_corners=False)


def _gazebo_images_from_observation(
    obs: dict[str, Any],
    *,
    device: torch.device,
    allow_zero_images: bool,
) -> dict[str, torch.Tensor]:
    images = obs.get("images") or obs.get("camera_images") or {}
    out: dict[str, torch.Tensor] = {}
    aliases = {
        "observation.images.center_camera": ["observation.images.center_camera", "center_camera", "center", "center_rgb"],
        "observation.images.left_camera": ["observation.images.left_camera", "left_camera", "left", "left_rgb"],
        "observation.images.right_camera": ["observation.images.right_camera", "right_camera", "right", "right_rgb"],
    }
    missing: list[str] = []
    for key, candidates in aliases.items():
        value = next((images[candidate] for candidate in candidates if candidate in images), None)
        if value is None:
            missing.append(key)
            continue
        out[key] = _as_image_tensor(value, device=device)
    if missing:
        if not allow_zero_images:
            raise RuntimeError(
                "Gazebo observation does not include ACT camera images: "
                f"{missing}. Current GazeboRLBridgePolicy IPC is lowdim-only; pass "
                "allow_zero_images=True only for adapter interface validation, not for real transfer scoring."
            )
        for key in missing:
            out[key] = torch.zeros((1, 3, 256, 288), dtype=torch.float32, device=device)
    return out


class ACTAdapterSERLGazeboPolicy:
    """Gazebo inference wrapper for offline/online ACT-adapter SERL checkpoints."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        act_torchscript: str | Path,
        device: str = "cpu",
        allow_zero_images: bool = False,
        adapter_delta_clip: float | None = None,
        action_clip: float | None = None,
        task_vector: np.ndarray | None = None,
    ):
        self.device = torch.device(device)
        self.allow_zero_images = bool(allow_zero_images)
        self.checkpoint_path = Path(checkpoint)
        self.act_torchscript = Path(act_torchscript)
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        cfg, self.dataset_summary, self.warmstart_report = _checkpoint_training_context(ckpt)
        online_adapter_cfg = ((ckpt.get("online_serl_config") or {}).get("isaac_adapter") or {})
        if adapter_delta_clip is None:
            adapter_delta_clip = online_adapter_cfg.get("adapter_delta_clip")
            if adapter_delta_clip is None:
                adapter_delta_clip = cfg.get("adapter_delta_clip", self.warmstart_report.get("adapter_delta_clip"))
        if action_clip is None:
            action_clip = online_adapter_cfg.get("action_clip") if "action_clip" in online_adapter_cfg else cfg.get("action_clip")
        self.state_dim = int(cfg["state_dim"])
        self.action_horizon = int(cfg["action_horizon"])
        self.action_dim = int(cfg["action_dim"])
        if self.action_dim % self.action_horizon != 0:
            raise ValueError(
                f"action_dim={self.action_dim} is not divisible by action_horizon={self.action_horizon}"
            )
        self.single_action_dim = self.action_dim // self.action_horizon
        self.actor = _load_adapter_actor(
            ckpt,
            act_torchscript=self.act_torchscript,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            device=self.device,
            adapter_delta_clip=adapter_delta_clip,
            action_clip=action_clip,
        )
        self.last_action_components: dict[str, float] = {}
        self.task_vector = None if task_vector is None else torch.as_tensor(
            task_vector, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        self.feature_assembler = AICRuntimeFeatureAssembler(self.state_dim, task_vector=task_vector)

    def _obs_to_actor(self, obs: dict[str, Any]) -> dict[str, Any]:
        lowdim = torch.as_tensor(self.feature_assembler.assemble_gazebo(obs), device=self.device).unsqueeze(0)
        if lowdim.shape[1] < self.state_dim:
            raise ValueError(
                f"Gazebo lowdim state has {lowdim.shape[1]} dims, checkpoint expects {self.state_dim}. "
                "If the checkpoint was trained with task/contact features, pass task context."
            )
        return {
            "state": lowdim[:, : self.state_dim],
            "images": _gazebo_images_from_observation(
                obs,
                device=self.device,
                allow_zero_images=self.allow_zero_images,
            ),
        }

    def act(self, obs: dict[str, Any], *, explore: bool = False) -> list[float]:
        del explore
        action = self.act_chunk(obs, n_action_steps=1)[0]
        return action.astype(float).tolist()

    def act_chunk(self, obs: dict[str, Any], *, n_action_steps: int) -> np.ndarray:
        if n_action_steps < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {n_action_steps}")
        if n_action_steps > self.action_horizon:
            raise ValueError(f"n_action_steps={n_action_steps} exceeds action_horizon={self.action_horizon}")
        actor_obs = self._obs_to_actor(obs)
        with torch.no_grad():
            components = self.actor.action_components(actor_obs)
            action = components["final_action"].reshape(1, self.action_horizon, self.single_action_dim)[
                0, :n_action_steps
            ]
            self.last_action_components = {
                "base_action_norm": float(components["base_action"].norm(dim=-1).mean().detach().cpu()),
                "delta_action_norm": float(components["delta_action"].norm(dim=-1).mean().detach().cpu()),
                "raw_delta_action_norm": float(
                    components.get("raw_delta_action", components["delta_action"]).norm(dim=-1).mean().detach().cpu()
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
        return action.detach().cpu().numpy().astype(np.float32)
