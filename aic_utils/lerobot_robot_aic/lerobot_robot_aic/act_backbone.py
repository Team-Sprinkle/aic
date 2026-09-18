"""Persist and reconstruct ACT spatial feature geometry independently of weights.

Convolution stride is absent from a state dict. Every AIC checkpoint consumer
must use ``load_act_policy`` (or explicitly reject custom geometry), otherwise
a stride-16 ResNet18 silently becomes stride 32 when its weights are reloaded.
Legacy checkpoints without this sidecar retain their torchvision configuration.
"""
from __future__ import annotations

import json
from pathlib import Path


BACKBONE_CONFIG_NAME = "aic_backbone_config.json"


def _value(config, name):
    if isinstance(config, dict):
        return config.get(name, False if name == "replace_final_stride_with_dilation" else None)
    return getattr(config, name)


def backbone_geometry(config, output_stride=None):
    """Canonical schema; only ResNet18 supports the optional stride override."""
    backbone = _value(config, "vision_backbone")
    dilated = bool(_value(config, "replace_final_stride_with_dilation"))
    default_stride = 16 if dilated else 32
    if backbone not in {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}:
        raise ValueError("ACT backbone geometry requires a torchvision ResNet")
    if backbone == "resnet18" and dilated:
        raise ValueError("ResNet18 BasicBlock does not support final-stage dilation")
    mode = "torchvision"
    stride = default_stride if output_stride is None else output_stride
    if type(stride) is not int or stride not in {16, 32}:
        raise ValueError("ACT backbone output stride must be 16 or 32")
    if output_stride is not None:
        if backbone != "resnet18" or dilated:
            raise ValueError("Explicit ACT output-stride overrides require undilated ResNet18")
        mode = "resnet18_layer4_stride1" if stride == 16 else "torchvision"
    return {"version": 1, "vision_backbone": backbone, "feature_layer": "layer4",
            "output_stride": stride, "mode": mode}


def validate_backbone_geometry(geometry, config):
    if not isinstance(geometry, dict):
        raise ValueError("ACT backbone geometry must be a versioned mapping")
    if type(geometry.get("version")) is not int or type(geometry.get("output_stride")) is not int:
        raise ValueError("ACT backbone geometry version and output_stride must be integers")
    custom = geometry.get("mode") == "resnet18_layer4_stride1"
    expected = backbone_geometry(config, 16 if custom else None)
    if geometry != expected:
        raise ValueError(f"Unsupported or inconsistent ACT backbone geometry: {geometry}")
    return dict(expected)


def read_backbone_geometry(checkpoint, config=None):
    checkpoint = Path(checkpoint)
    if config is None:
        config = json.loads((checkpoint / "config.json").read_text())
    sidecar = checkpoint / BACKBONE_CONFIG_NAME
    action_path = checkpoint / "aic_action_config.json"
    action = json.loads(action_path.read_text()) if action_path.is_file() else {}
    recorded = json.loads(sidecar.read_text()) if sidecar.is_file() else action.get("backbone_geometry")
    if not sidecar.is_file() and "backbone_geometry" not in action:
        return backbone_geometry(config)
    if sidecar.is_file() and action.get("backbone_geometry", recorded) != recorded:
        raise ValueError("Checkpoint backbone sidecar and action contract disagree")
    return validate_backbone_geometry(recorded, config)


def apply_backbone_geometry(policy, geometry=None):
    geometry = validate_backbone_geometry(
        backbone_geometry(policy.config) if geometry is None else geometry, policy.config)
    if geometry["vision_backbone"] == "resnet18":
        block = policy.model.backbone["layer4"][0]
        if (type(block).__name__ != "BasicBlock" or block.downsample is None
                or tuple(block.conv1.dilation) != (1, 1) or tuple(block.conv2.dilation) != (1, 1)):
            raise ValueError("Unexpected ResNet18 final-stage structure")
        stride = 1 if geometry["output_stride"] == 16 else 2
        block.conv1.stride = (stride, stride)
        block.downsample[0].stride = (stride, stride)
        block.stride = stride
    policy._aic_backbone_geometry = geometry
    return policy


def save_backbone_geometry(policy, checkpoint):
    geometry = validate_backbone_geometry(getattr(policy, "_aic_backbone_geometry",
                                                   backbone_geometry(policy.config)), policy.config)
    if geometry["vision_backbone"] == "resnet18":
        block = policy.model.backbone["layer4"][0]
        stride = (1, 1) if geometry["output_stride"] == 16 else (2, 2)
        if tuple(block.conv1.stride) != stride or tuple(block.downsample[0].stride) != stride:
            raise ValueError("ACT backbone strides disagree with the persisted geometry")
    path = Path(checkpoint) / BACKBONE_CONFIG_NAME
    path.write_text(json.dumps(geometry, indent=2) + "\n")
    return geometry


def load_act_policy(checkpoint, **kwargs):
    """Load weights and spatial geometry together; legacy checkpoints are valid."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    config = kwargs.get("config")
    geometry = read_backbone_geometry(checkpoint, config)
    policy = ACTPolicy.from_pretrained(checkpoint, **kwargs)
    return apply_backbone_geometry(policy, geometry)


def require_default_backbone_geometry(checkpoint):
    """Fail closed in historical runners that only construct torchvision ACT."""
    geometry = read_backbone_geometry(checkpoint)
    if geometry["mode"] != "torchvision":
        raise ValueError("This ACT checkpoint requires the AIC geometry-aware loader or RunACTTorchScript")
