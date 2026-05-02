"""ACT checkpoint inspection and conservative offline-SERL warm-start helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def resolve_act_checkpoint_dir(path: Path) -> Path:
    """Return the LeRobot ACT `pretrained_model` directory for a file or directory."""
    path = Path(path)
    if path.is_file():
        return path.parent
    if (path / "model.safetensors").exists():
        return path
    nested = path / "pretrained_model"
    if (nested / "model.safetensors").exists():
        return nested
    raise FileNotFoundError(
        f"Could not find model.safetensors in ACT checkpoint path: {path}"
    )


def inspect_act_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint_dir = resolve_act_checkpoint_dir(path)
    config_path = checkpoint_dir / "config.json"
    train_config_path = checkpoint_dir / "train_config.json"
    model_path = checkpoint_dir / "model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing ACT config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    train_config = (
        json.loads(train_config_path.read_text(encoding="utf-8"))
        if train_config_path.exists()
        else {}
    )
    output_features = config.get("output_features") or {}
    action_feature = output_features.get("action") or {}
    input_features = config.get("input_features") or {}
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "model_path": str(model_path),
        "config_path": str(config_path),
        "train_config_path": str(train_config_path) if train_config_path.exists() else None,
        "type": config.get("type"),
        "chunk_size": config.get("chunk_size"),
        "n_action_steps": config.get("n_action_steps"),
        "n_obs_steps": config.get("n_obs_steps"),
        "action_shape": action_feature.get("shape"),
        "state_shape": (input_features.get("observation.state") or {}).get("shape"),
        "camera_keys": sorted(
            key for key, spec in input_features.items() if (spec or {}).get("type") == "VISUAL"
        ),
        "dataset_root": ((train_config.get("dataset") or {}).get("root")),
    }


def load_act_action_head_bias(path: Path) -> torch.Tensor:
    checkpoint_dir = resolve_act_checkpoint_dir(path)
    model_path = checkpoint_dir / "model.safetensors"
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - dependency is present in pixi env
        raise RuntimeError("Loading ACT checkpoints requires safetensors.") from exc
    tensors = load_file(str(model_path))
    key = "model.action_head.bias"
    if key not in tensors:
        raise KeyError(f"ACT checkpoint does not contain {key!r}: {model_path}")
    return tensors[key].detach().cpu().float()
