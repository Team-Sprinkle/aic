from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ACT_CAMERA_KEYS = [
    "observation.images.center_camera",
    "observation.images.left_camera",
    "observation.images.right_camera",
]


def lowdim_state_from_gazebo_observation(obs: dict[str, Any]) -> np.ndarray:
    controller = obs.get("controller") or {}
    tcp_pose = controller.get("current_tcp_pose") or {}
    tcp_velocity = controller.get("tcp_velocity") or {}
    joints = obs.get("joints") or {}
    wrench = obs.get("wrist_wrench") or {}

    values: list[float] = []
    values.extend((tcp_pose.get("position") or [0.0, 0.0, 0.0])[:3])
    values.extend((tcp_pose.get("orientation_xyzw") or [0.0, 0.0, 0.0, 1.0])[:4])
    values.extend(((tcp_velocity or {}).get("linear") or [0.0, 0.0, 0.0])[:3])
    values.extend(((tcp_velocity or {}).get("angular") or [0.0, 0.0, 0.0])[:3])
    tcp_error = list(controller.get("tcp_error") or [])
    values.extend((tcp_error + [0.0] * 6)[:6])
    joint_positions = list(joints.get("position") or [])
    values.extend((joint_positions + [0.0] * 7)[:7])
    values.extend(((wrench or {}).get("force") or [0.0, 0.0, 0.0])[:3])
    values.extend(((wrench or {}).get("torque") or [0.0, 0.0, 0.0])[:3])
    return np.asarray(values[:32], dtype=np.float32)


class OfflineSERLGazeboPolicy:
    def __init__(self, checkpoint: str | Path, *, device: str = "cpu"):
        from lerobot_robot_aic.offline_serl import GaussianActor

        self.device = torch.device(device)
        ckpt = torch.load(checkpoint, map_location=self.device)
        cfg = ckpt.get("offline_serl_config") or {}
        if int(cfg.get("obs_dim", 0)) != 32:
            raise ValueError(f"Gazebo SERL policy expects obs_dim=32, got {cfg.get('obs_dim')}")
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
        self.obs_std = torch.as_tensor(stats["obs_std"], dtype=torch.float32, device=self.device)
        self.action_mean = torch.as_tensor(stats["action_mean"], dtype=torch.float32, device=self.device)
        self.action_std = torch.as_tensor(stats["action_std"], dtype=torch.float32, device=self.device)

    def act(self, obs: dict[str, Any], *, explore: bool = False) -> list[float]:
        del explore
        lowdim = torch.as_tensor(lowdim_state_from_gazebo_observation(obs), device=self.device).unsqueeze(0)
        lowdim = (lowdim - self.obs_mean) / self.obs_std
        with torch.no_grad():
            normalized_action = self.actor.mean_action(lowdim).squeeze(0)
        action = normalized_action * self.action_std + self.action_mean
        return action[: self.single_action_dim].detach().cpu().numpy().astype(float).tolist()


def _mlp(input_dim: int, hidden_dim: int, num_layers: int, output_dim: int) -> nn.Sequential:
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
        self.single_action_dim = self.action_dim // self.action_horizon
        self.adapter_scale = float(adapter_scale)
        self.adapter_delta_clip = None if adapter_delta_clip is None else float(adapter_delta_clip)
        self.action_clip = None if action_clip is None else float(action_clip)
        self.adapter = _mlp(self.state_dim + self.action_dim, hidden_dim, num_layers, self.action_dim)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -2.0))

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
    _, _, warmstart = _checkpoint_training_context(checkpoint)
    act_base = torch.jit.load(str(act_torchscript), map_location=device).eval()
    for param in act_base.parameters():
        param.requires_grad = False
    actor = TorchScriptACTAdapterActor(
        act_base=act_base,
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        adapter_scale=_adapter_scale_from_checkpoint(checkpoint, warmstart),
        adapter_delta_clip=adapter_delta_clip,
        action_clip=action_clip,
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
            import cv2

            decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError("Could not decode JPEG image payload from Gazebo observation")
            image = torch.from_numpy(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
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
        adapter_delta_clip: float | None = 0.05,
        action_clip: float | None = 0.05,
    ):
        self.device = torch.device(device)
        self.allow_zero_images = bool(allow_zero_images)
        self.checkpoint_path = Path(checkpoint)
        self.act_torchscript = Path(act_torchscript)
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        cfg, self.dataset_summary, self.warmstart_report = _checkpoint_training_context(ckpt)
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

    def _obs_to_actor(self, obs: dict[str, Any]) -> dict[str, Any]:
        lowdim = torch.as_tensor(lowdim_state_from_gazebo_observation(obs), device=self.device).unsqueeze(0)
        if lowdim.shape[1] < self.state_dim:
            raise ValueError(f"Gazebo lowdim state has {lowdim.shape[1]} dims, checkpoint expects {self.state_dim}")
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
        actor_obs = self._obs_to_actor(obs)
        with torch.no_grad():
            components = self.actor.action_components(actor_obs)
            action = components["final_action"].squeeze(0)
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
        return action[: self.single_action_dim].detach().cpu().numpy().astype(float).tolist()
