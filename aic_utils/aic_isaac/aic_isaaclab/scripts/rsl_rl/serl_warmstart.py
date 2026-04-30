"""Conservative offline-SERL to RSL-RL actor initialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _last_actor_bias_key(state_dict: dict[str, torch.Tensor]) -> str:
    candidates = [
        key
        for key, value in state_dict.items()
        if key.startswith("actor.") and key.endswith(".bias") and value.ndim == 1
    ]
    if not candidates:
        raise KeyError("Could not find an RSL-RL actor output bias in model_state_dict.")
    return sorted(candidates, key=lambda key: int(key.split(".")[1]))[-1]


def _first_action_stats(
    stats: dict[str, Any],
    *,
    source_action_dim: int,
    target_action_dim: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    mean = stats.get("action_mean")
    std = stats.get("action_std")
    if mean is None or std is None:
        return None, None
    mean_t = torch.as_tensor(mean, dtype=torch.float32).flatten()
    std_t = torch.as_tensor(std, dtype=torch.float32).flatten()
    if mean_t.numel() < target_action_dim or std_t.numel() < target_action_dim:
        return None, None
    if source_action_dim < target_action_dim:
        return None, None
    return mean_t[:target_action_dim], std_t[:target_action_dim]


def build_warmstarted_rsl_state_dict(
    rsl_state_dict: dict[str, torch.Tensor],
    serl_checkpoint: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return an RSL-RL state dict initialized from compatible SERL tensors.

    Full hidden-layer transfer is only performed for exact tensor shape matches.
    For the current AIC lowdim SERL -> camera PPO case, the architectures do not
    match, so the bridge transfers a safe action prior into the PPO actor output
    bias and `std` from SERL normalization stats when available.
    """
    updated = {key: value.detach().clone() for key, value in rsl_state_dict.items()}
    actor = serl_checkpoint.get("actor") or {}
    cfg = serl_checkpoint.get("offline_serl_config") or {}
    stats = serl_checkpoint.get("normalization_stats") or {}
    report: dict[str, Any] = {
        "source": "offline_serl",
        "mode": "partial_action_prior",
        "copied_exact_tensors": [],
        "copied_action_prior": False,
        "warnings": [],
        "source_obs_dim": cfg.get("obs_dim"),
        "source_action_dim": cfg.get("action_dim"),
        "source_action_horizon": cfg.get("action_horizon"),
    }

    for source_key, target_key in (
        ("backbone.0.weight", "actor.0.weight"),
        ("backbone.0.bias", "actor.0.bias"),
        ("mean_head.weight", "actor.6.weight"),
        ("mean_head.bias", "actor.6.bias"),
    ):
        if source_key not in actor or target_key not in updated:
            continue
        source_tensor = actor[source_key].detach().cpu()
        if tuple(source_tensor.shape) == tuple(updated[target_key].shape):
            updated[target_key] = source_tensor.to(updated[target_key].dtype)
            report["copied_exact_tensors"].append(f"actor.{source_key}->{target_key}")

    out_bias_key = _last_actor_bias_key(updated)
    target_action_dim = int(updated[out_bias_key].numel())
    source_action_dim = int(cfg.get("action_dim") or 0)
    action_mean, action_std = _first_action_stats(
        stats,
        source_action_dim=source_action_dim,
        target_action_dim=target_action_dim,
    )
    if action_mean is not None:
        updated[out_bias_key] = action_mean.to(updated[out_bias_key].dtype)
        report["copied_action_prior"] = True
        report["action_prior_key"] = out_bias_key
    else:
        report["warnings"].append("SERL action normalization stats were unavailable or incompatible.")

    if "std" in updated and action_std is not None and updated["std"].numel() == target_action_dim:
        updated["std"] = action_std.clamp_min(1e-4).to(updated["std"].dtype)
        report["copied_std_prior"] = True
    else:
        report["copied_std_prior"] = False

    report["target_action_dim"] = target_action_dim
    return updated, report


def apply_offline_serl_warmstart(actor_critic: Any, checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    serl_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "offline_serl_config" not in serl_checkpoint:
        raise ValueError(f"Checkpoint is not an offline SERL checkpoint: {checkpoint_path}")
    current = actor_critic.state_dict()
    updated, report = build_warmstarted_rsl_state_dict(current, serl_checkpoint)
    actor_critic.load_state_dict(updated, strict=True)
    report["checkpoint"] = str(checkpoint_path)
    return report


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
