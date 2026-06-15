#!/usr/bin/env python3
"""Filter and merge Isaac SERL replay buffers by diagnostic geometry."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch


def _finite(value: Any, default: float = float("nan")) -> float:
    if isinstance(value, list):
        value = value[0] if value else default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _by_env(data: dict[str, Any], key: str, env_index: int) -> float:
    values = data.get(f"{key}_by_env") if isinstance(data, dict) else None
    if isinstance(values, list) and 0 <= env_index < len(values):
        return _finite(values[env_index])
    if env_index == 0 and isinstance(data, dict):
        return _finite(data.get(f"{key}_env0", data.get(f"{key}_mean")))
    if isinstance(data, dict):
        return _finite(data.get(f"{key}_mean"))
    return float("nan")


def _geometry(transition: dict[str, Any]) -> dict[str, float]:
    metadata = transition.get("metadata") or {}
    env_index = int(metadata.get("env_index", 0) or 0)
    geom = metadata.get("post_step_insertion_geometry") or {}
    all_body = metadata.get("post_step_all_body_insertion_geometry") or {}
    module = all_body.get("sfp_module_link") if isinstance(all_body, dict) else {}
    module = module if isinstance(module, dict) else {}
    return {
        "s": _by_env(geom, "signed_depth_m", env_index),
        "r": _by_env(geom, "lateral_error_m", env_index),
        "theta": _by_env(geom, "orientation_error_rad", env_index),
        "module_r": _by_env(module, "lateral_error_m", env_index),
    }


def _matches(g: dict[str, float], args: argparse.Namespace) -> bool:
    checks = (
        g["s"] >= args.min_s_m,
        g["s"] <= args.max_s_m,
        g["r"] <= args.max_r_m,
        g["theta"] >= args.min_theta_rad,
        g["theta"] <= args.max_theta_rad,
    )
    if not all(checks):
        return False
    if math.isfinite(args.max_module_r_m) and g["module_r"] > args.max_module_r_m:
        return False
    return True


def _load_transitions(path: Path) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("transitions"), list):
        return list(payload["transitions"])
    if isinstance(payload, list):
        return list(payload)
    raise ValueError(f"unsupported replay payload in {path}: {type(payload).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-s-m", type=float, default=-float("inf"))
    parser.add_argument("--max-s-m", type=float, default=float("inf"))
    parser.add_argument("--max-r-m", type=float, default=float("inf"))
    parser.add_argument("--min-theta-rad", type=float, default=-float("inf"))
    parser.add_argument("--max-theta-rad", type=float, default=float("inf"))
    parser.add_argument("--max-module-r-m", type=float, default=float("inf"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    kept: list[dict[str, Any]] = []
    source_count = 0
    for path in args.input:
        transitions = _load_transitions(path)
        source_count += len(transitions)
        for transition in transitions:
            if _matches(_geometry(transition), args):
                kept.append(transition)

    if args.limit > 0:
        kept = kept[-args.limit :]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "capacity": max(len(kept), 1),
        "size": source_count,
        "filter": vars(args) | {"input": [str(p) for p in args.input], "output": str(args.output)},
        "saved_size": len(kept),
        "transitions": kept,
    }
    torch.save(payload, args.output)
    print({"source_size": source_count, "saved_size": len(kept), "output": str(args.output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
