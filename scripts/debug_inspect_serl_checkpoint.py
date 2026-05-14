#!/usr/bin/env python3
"""Inspect ACT and ACT-adapter SERL artifacts for runtime compatibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


CAMERA_KEYS = [
    "observation.images.center_camera",
    "observation.images.left_camera",
    "observation.images.right_camera",
]


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float().cpu()
    finite = torch.isfinite(value)
    out: dict[str, Any] = {
        "shape": list(value.shape),
        "finite": bool(finite.all()),
        "nan_count": int(torch.isnan(value).sum()),
        "inf_count": int(torch.isinf(value).sum()),
    }
    if value.numel():
        finite_value = value[finite]
        if finite_value.numel():
            out.update(
                {
                    "mean": float(finite_value.mean()),
                    "std": float(finite_value.std(unbiased=False)) if finite_value.numel() > 1 else 0.0,
                    "min": float(finite_value.min()),
                    "max": float(finite_value.max()),
                    "abs_max": float(finite_value.abs().max()),
                    "norm": float(value.norm()),
                }
            )
    return out


def normalizer_report(checkpoint_dir: Path, state_dim: int) -> dict[str, Any]:
    path = checkpoint_dir / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    if not path.exists():
        return {"exists": False, "path": str(path)}
    stats = load_file(str(path))
    state_std = stats["observation.state.std"].float().reshape(-1)
    state_mean = stats["observation.state.mean"].float().reshape(-1)
    sliced_std = state_std[:state_dim]
    tiny = torch.nonzero(sliced_std.abs() < 1.0e-8, as_tuple=False).flatten().tolist()
    task_slice = slice(max(0, state_dim - 10), state_dim) if state_dim in (42, 82) else slice(0, 0)
    camera = {}
    for key in CAMERA_KEYS:
        mean_key = f"{key}.mean"
        std_key = f"{key}.std"
        camera[key] = {
            "mean": tensor_stats(stats[mean_key]) if mean_key in stats else None,
            "std": tensor_stats(stats[std_key]) if std_key in stats else None,
        }
    return {
        "exists": True,
        "path": str(path),
        "state_mean": tensor_stats(state_mean[:state_dim]),
        "state_std": tensor_stats(sliced_std),
        "tiny_std_indices_lt_1e-8": tiny,
        "tiny_std_count": len(tiny),
        "task_vector_identity_required": state_dim in (42, 82),
        "task_vector_saved_mean": state_mean[task_slice].tolist() if state_dim in (42, 82) else None,
        "task_vector_saved_std": state_std[task_slice].tolist() if state_dim in (42, 82) else None,
        "runtime_expected_clamp": "std abs < 1e-8 -> 1.0; task vector mean=0/std=1 for 42D/82D",
        "camera_stats": camera,
    }


def inspect_serl(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return {"path": str(checkpoint_path), "exists": False}
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("vision_offline_serl_config") or (ckpt.get("online_serl_config") or {}).get("checkpoint", {}).get(
        "vision_offline_serl_config", {}
    )
    actor = ckpt.get("actor") or {}
    adapter_tensors = {k: v for k, v in actor.items() if k.startswith("adapter.")}
    act_tensors = {k: v for k, v in actor.items() if k.startswith("act_base.")}
    raw_adapter = torch.cat([v.detach().float().reshape(-1) for v in adapter_tensors.values()]) if adapter_tensors else torch.empty(0)
    periodic = sorted(str(path) for path in checkpoint_path.parent.glob("checkpoint_*.pt"))
    return {
        "path": str(checkpoint_path),
        "exists": True,
        "top_level_keys": sorted(ckpt.keys()),
        "step": ckpt.get("step"),
        "state_dim": cfg.get("state_dim"),
        "action_dim": cfg.get("action_dim"),
        "action_horizon": cfg.get("action_horizon"),
        "single_action_dim": (int(cfg["action_dim"]) // int(cfg["action_horizon"])) if cfg.get("action_dim") and cfg.get("action_horizon") else None,
        "actor_mode": cfg.get("actor_mode"),
        "freeze_act": cfg.get("freeze_act"),
        "adapter_delta_clip": cfg.get("adapter_delta_clip"),
        "action_clip": cfg.get("action_clip"),
        "adapter_arch": cfg.get("adapter_arch"),
        "adapter_activation": cfg.get("adapter_activation"),
        "adapter_tensor_count": len(adapter_tensors),
        "act_base_tensor_count": len(act_tensors),
        "act_frozen_in_config": bool(cfg.get("freeze_act", False)),
        "adapter_stats": tensor_stats(raw_adapter),
        "dataset_summary": ckpt.get("dataset_summary"),
        "warmstart_report": ckpt.get("warmstart_report"),
        "periodic_checkpoints_in_dir": periodic,
        "has_checkpoint_best_val": (checkpoint_path.parent / "checkpoint_best_val.pt").exists(),
        "has_checkpoint_latest": (checkpoint_path.parent / "checkpoint_latest.pt").exists(),
    }


def inspect_act(torchscript_path: Path, checkpoint_dir: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {"torchscript_path": str(torchscript_path), "exists": torchscript_path.exists()}
    meta_path = torchscript_path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        out["metadata_path"] = str(meta_path)
        out["metadata"] = meta
        if checkpoint_dir is None and meta.get("checkpoint_dir"):
            checkpoint_dir = Path(str(meta["checkpoint_dir"]))
    else:
        out["metadata_path"] = None
        out["metadata"] = {}
    if checkpoint_dir is not None:
        out["checkpoint_dir"] = str(checkpoint_dir)
        state_shape = out["metadata"].get("state_shape") or [82]
        out["normalizer"] = normalizer_report(checkpoint_dir, int(state_shape[0]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serl-checkpoint", type=Path, required=True)
    parser.add_argument("--act-torchscript", type=Path, required=True)
    parser.add_argument("--act-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "act": inspect_act(args.act_torchscript, args.act_checkpoint_dir),
        "serl": inspect_serl(args.serl_checkpoint),
        "runtime_normalizer_consistency": {
            "gazebo_RunACTTorchScript": "clamps std abs < 1e-8 and task-vector dims to mean=0/std=1",
            "gazebo_ACTAdapterSERL": "uses gazebo_rl.ACTRuntimeNormalizer with same clamp/task-vector identity",
            "isaac_online_SERL": "uses local ACTRuntimeNormalizer with same clamp/task-vector identity",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
