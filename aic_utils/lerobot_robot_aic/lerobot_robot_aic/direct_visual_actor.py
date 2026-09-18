"""Direct image/state policy shared by offline learning, Isaac, and Gazebo.

The visual backbone produces features, never a proposed action to correct.
Checkpoint buffers contain observation statistics and physical action limits.
This module deliberately has no LeRobot, ROS, or simulator dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
BACKBONES = ("small_conv", "resnet18", "resnet18_imagenet", "dinov2_vits14")


@dataclass
class DirectVisualActorConfig:
    state_dim: int
    camera_keys: list[str]
    single_action_dim: int = 6
    action_horizon: int = 1
    backbone: str = "resnet18"
    image_size: int = 224
    spatial_grid: int = 2
    hidden_dim: int = 256
    per_camera_dim: int = 128
    freeze_backbone: bool = False
    action_limits: tuple[float, ...] = (0.02, 0.02, 0.02, 0.2, 0.2, 0.2)
    dinov2_revision: str = DINOV2_REVISION


class ObservationNormalizer(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.register_buffer("state_mean", torch.zeros(1, state_dim))
        self.register_buffer("state_std", torch.ones(1, state_dim))
        self.register_buffer("action_mean", torch.zeros(1, action_dim))
        self.register_buffer("action_std", torch.ones(1, action_dim))
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize_state(self, state):
        if state.ndim != 2 or state.shape[-1] != self.state_mean.shape[-1]:
            raise ValueError(f"State shape {tuple(state.shape)} does not match {self.state_mean.shape[-1]} features")
        return ((state - self.state_mean) / self.state_std).clamp(-10, 10)

    def normalize_image(self, key, image):
        del key
        return (image - self.image_mean) / self.image_std


class SpatialBackbone(nn.Module):
    """Keep a small spatial grid rather than discarding all image position."""

    def __init__(self, name: str, grid: int, *, initialize: bool = True, dinov2_revision: str = DINOV2_REVISION):
        super().__init__()
        self.name, self.grid = name, grid
        if name == "small_conv":
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),
            )
            channels = 64
        elif name in {"resnet18", "resnet18_imagenet"}:
            from torchvision.models import resnet18, ResNet18_Weights
            from torchvision.models._utils import IntermediateLayerGetter
            from torchvision.ops.misc import FrozenBatchNorm2d

            # Frozen statistics work with tiny robotics batches; convolutional
            # weights still receive gradients unless explicitly frozen.
            model = resnet18(weights=None, norm_layer=FrozenBatchNorm2d)
            if name == "resnet18_imagenet" and initialize:
                weights = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=False, check_hash=True)
                model.load_state_dict({k: v for k, v in weights.items() if not k.endswith("num_batches_tracked")})
            self.model = IntermediateLayerGetter(model, {"layer4": "features"})
            channels = 512
        elif name == "dinov2_vits14":
            self.model = torch.hub.load(
                f"facebookresearch/dinov2:{dinov2_revision}", "dinov2_vits14",
                pretrained=initialize, trust_repo=True, verbose=False,
            )
            channels = int(self.model.embed_dim)
        else:
            raise ValueError(f"Unknown visual backbone: {name}")
        self.output_dim = channels * grid * grid

    def forward(self, images):
        if self.name == "dinov2_vits14":
            tokens = self.model.forward_features(images)["x_norm_patchtokens"]
            height, width = images.shape[-2] // 14, images.shape[-1] // 14
            features = tokens.transpose(1, 2).reshape(images.shape[0], -1, height, width)
        else:
            features = self.model(images)
            if isinstance(features, dict):
                features = features["features"]
        return F.adaptive_avg_pool2d(features, (self.grid, self.grid)).flatten(1)


class DirectVisualActor(nn.Module):
    actor_mode = "direct_visual"

    def __init__(self, config: DirectVisualActorConfig, *, initialize_backbone: bool = True):
        super().__init__()
        self.config = config
        if config.state_dim < 1 or not config.camera_keys or len(set(config.camera_keys)) != len(config.camera_keys):
            raise ValueError("Positive state width and unique camera keys are required")
        if config.action_horizon < 1 or config.image_size < 28 or config.spatial_grid < 1:
            raise ValueError("Invalid action horizon, image size, or spatial grid")
        if config.backbone == "dinov2_vits14" and config.image_size % 14:
            raise ValueError("DINOv2 image size must be a multiple of 14")
        limits = torch.tensor(config.action_limits, dtype=torch.float32)
        if limits.numel() != config.single_action_dim or not torch.isfinite(limits).all() or not (limits > 0).all():
            raise ValueError("One finite positive physical limit is required per action coordinate")
        self.state_dim = config.state_dim
        self.action_horizon = config.action_horizon
        self.action_dim = config.single_action_dim * config.action_horizon
        self.normalizer = ObservationNormalizer(config.state_dim, config.single_action_dim)
        self.register_buffer("action_limits", limits.repeat(config.action_horizon).unsqueeze(0))
        self.backbone = SpatialBackbone(config.backbone, config.spatial_grid,
                                       initialize=initialize_backbone, dinov2_revision=config.dinov2_revision)
        self.image_projection = nn.Sequential(nn.Linear(self.backbone.output_dim, config.per_camera_dim),
                                              nn.LayerNorm(config.per_camera_dim), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(config.state_dim + len(config.camera_keys) * config.per_camera_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim), nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.GELU(),
            nn.Linear(config.hidden_dim, self.action_dim),
        )
        nn.init.normal_(self.head[-1].weight, std=0.001)
        nn.init.zeros_(self.head[-1].bias)
        # Compatibility with existing deterministic trainer diagnostics. This
        # is not a learned distribution or an entropy-regularized SAC actor.
        self.register_buffer("log_std", torch.full((self.action_dim,), -2.0))
        self.backbone.requires_grad_(not config.freeze_backbone)
        self.train()

    @property
    def act_normalizer(self):
        """Compatibility for existing Isaac observation diagnostics."""
        return self.normalizer

    @property
    def act_torchscript_path(self):
        return None

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_backbone:
            self.backbone.eval()
        return self

    @torch.no_grad()
    def fit_normalization(self, states, actions):
        for name, data, width, floor in (("state", states, self.state_dim, 1e-3),
                                          ("action", actions, self.config.single_action_dim, 1e-4)):
            values = torch.as_tensor(data, dtype=torch.float32, device=self.action_limits.device)
            if values.ndim != 2 or values.shape[-1] != width or not len(values) or not torch.isfinite(values).all():
                raise ValueError(f"Invalid {name} normalization samples")
            getattr(self.normalizer, name + "_mean").copy_(values.mean(0, keepdim=True))
            getattr(self.normalizer, name + "_std").copy_(values.std(0, unbiased=False, keepdim=True).clamp_min(floor))

    def mean_action(self, obs: dict[str, Any]):
        state = self.normalizer.normalize_state(obs["state"])
        if "actor_state" in obs and obs["actor_state"].shape != obs["state"].shape:
            raise ValueError("Direct visual actor expects one state frame; history requires a new schema")
        images = []
        for key in self.config.camera_keys:
            image = obs["images"][key]
            if image.ndim != 4 or image.shape[:2] != (state.shape[0], 3) or not image.is_floating_point():
                raise ValueError(f"{key} must be float RGB [batch,3,height,width] in [0,1]")
            image = F.interpolate(image, size=(self.config.image_size, self.config.image_size),
                                  mode="bilinear", align_corners=False, antialias=True)
            images.append(self.normalizer.normalize_image(key, image))
        # Camera order is part of the checkpoint contract. Share backbone weights.
        features = self.image_projection(self.backbone(torch.cat(images, dim=0)))
        features = torch.cat(features.chunk(len(images), dim=0), dim=-1)
        return torch.tanh(self.head(torch.cat([state, features], dim=-1))) * self.action_limits

    def action_components(self, obs):
        action = self.mean_action(obs)
        # Legacy log/loss fields are neutral. There is no base-policy invocation,
        # residual, correction clip, or preservation loss in this actor.
        zero = torch.zeros_like(action)
        return {"final_action": action, "base_action": action.detach(), "delta_action": zero,
                "raw_delta_action": zero, "unclipped_final_action": action,
                "translation_clipped_action": action, "rotation_clipped_action": action}

    def forward(self, obs):
        return self.action_components(obs)

    def bc_loss(self, prediction, target):
        scale = self.normalizer.action_std.repeat(1, self.action_horizon)
        return F.mse_loss(prediction / scale, target / scale)

    def architecture_config(self):
        return asdict(self.config)

    def initialize_act_backbone(self, checkpoint: Path):
        """Copy only ACT's ResNet, never its action decoder/head or normalizer."""
        from safetensors.torch import load_file
        if self.config.backbone != "resnet18":
            raise ValueError("ACT backbone initialization requires backbone=resnet18")
        checkpoint = Path(checkpoint)
        path = checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
        weights = load_file(str(path), device="cpu")
        prefix = "model.backbone."
        backbone_weights = {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}
        self.backbone.model.load_state_dict(backbone_weights, strict=True)
        return {"source": str(path.resolve()), "loaded_tensors": len(backbone_weights), "action_head_loaded": False}


def direct_config_from_checkpoint(checkpoint):
    config = checkpoint.get("vision_offline_serl_config")
    if config is None:
        context = checkpoint.get("online_serl_config") or checkpoint.get("online_gazebo_serl_config") or {}
        config = (context.get("checkpoint") or {}).get("vision_offline_serl_config") or {}
    if config.get("actor_mode") != "direct_visual":
        raise ValueError("Checkpoint is not a direct_visual policy; residual weights cannot initialize its head")
    return DirectVisualActorConfig(**config["direct_visual_actor"])


def load_direct_visual_actor(checkpoint, *, device="cpu"):
    config = direct_config_from_checkpoint(checkpoint)
    actor = DirectVisualActor(config, initialize_backbone=False)
    actor.load_state_dict(checkpoint["actor"], strict=True)
    return actor.to(device)
