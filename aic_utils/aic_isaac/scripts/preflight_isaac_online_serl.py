#!/usr/bin/env python3
"""Preflight checks for ACT-adapter Isaac online SERL inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml


def _load_episodes(root: Path) -> list[dict[str, Any]]:
    episodes_dir = root if root.name == "episodes" else root / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Episode config dir has no episodes directory: {root}")
    episodes = []
    for path in sorted(episodes_dir.glob("episode_*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Episode YAML must be a mapping: {path}")
        data["_path"] = str(path)
        episodes.append(data)
    if not episodes:
        raise ValueError(f"No episode_*.yaml files found in {episodes_dir}")
    return episodes


def _task_vector(context: dict[str, Any]) -> list[float]:
    family_name = str(context["task_family"])
    port_index = int(context["target_port_index"])
    card_index = int(context["target_card_index"])
    card_valid = int(context["target_card_valid"])
    family = [1.0, 0.0] if family_name == "sfp_to_nic" else [0.0, 1.0]
    port = [1.0, 0.0] if port_index == 0 else [0.0, 1.0]
    card = [0.0] * 5
    if card_valid:
        card[card_index] = 1.0
    return family + port + card + [float(card_valid)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--act-torchscript", required=True, type=Path)
    parser.add_argument("--episode-config-dir", required=True, type=Path)
    parser.add_argument("--expected-state-dim", type=int, default=82)
    parser.add_argument("--expected-action-horizon", type=int, default=4)
    parser.add_argument("--expected-action-dim", type=int, default=24)
    args = parser.parse_args()

    report: dict[str, Any] = {"ok": True, "errors": []}
    if not args.checkpoint.exists():
        report["ok"] = False
        report["errors"].append(f"Missing checkpoint: {args.checkpoint}")
    else:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        cfg = ckpt.get("vision_offline_serl_config") or (ckpt.get("online_serl_config") or {}).get("checkpoint", {}).get("vision_offline_serl_config") or {}
        report["checkpoint"] = {
            "path": str(args.checkpoint),
            "step": ckpt.get("step"),
            "has_actor": "actor" in ckpt,
            "has_critic1": "critic1" in ckpt,
            "has_critic2": "critic2" in ckpt,
            "state_dim": cfg.get("state_dim"),
            "action_dim": cfg.get("action_dim"),
            "action_horizon": cfg.get("action_horizon"),
            "actor_mode": cfg.get("actor_mode"),
            "freeze_act": cfg.get("freeze_act"),
        }
        for key, expected in (
            ("state_dim", args.expected_state_dim),
            ("action_dim", args.expected_action_dim),
            ("action_horizon", args.expected_action_horizon),
        ):
            if int(cfg.get(key, -1)) != expected:
                report["ok"] = False
                report["errors"].append(f"Checkpoint {key}={cfg.get(key)} does not match expected {expected}")
        if not ("actor" in ckpt and "critic1" in ckpt and "critic2" in ckpt):
            report["ok"] = False
            report["errors"].append("Checkpoint must contain actor, critic1, and critic2")

    if not args.act_torchscript.exists():
        report["ok"] = False
        report["errors"].append(f"Missing ACT TorchScript: {args.act_torchscript}")
    else:
        report["act_torchscript"] = {"path": str(args.act_torchscript), "size": args.act_torchscript.stat().st_size}

    episodes = _load_episodes(args.episode_config_dir)
    episode_errors = []
    families = set()
    for episode in episodes:
        context = episode.get("task_context")
        scene = episode.get("scene") or {}
        target = scene.get("target") or {}
        if not isinstance(context, dict):
            episode_errors.append(f"{episode['_path']}: missing task_context")
            continue
        families.add(context.get("task_family"))
        try:
            vector = _task_vector(context)
        except Exception as exc:
            episode_errors.append(f"{episode['_path']}: invalid task vector context: {exc}")
            continue
        if len(vector) != 10:
            episode_errors.append(f"{episode['_path']}: task vector length {len(vector)}")
        if not target.get("target_pose_world"):
            episode_errors.append(f"{episode['_path']}: missing scene.target.target_pose_world")
        if not episode.get("isaac_randomization"):
            episode_errors.append(f"{episode['_path']}: missing isaac_randomization")
    if episode_errors:
        report["ok"] = False
        report["errors"].extend(episode_errors)
    report["episodes"] = {
        "dir": str(args.episode_config_dir),
        "count": len(episodes),
        "families": sorted(str(f) for f in families),
        "first": {k: v for k, v in episodes[0].items() if k != "_path"} if episodes else None,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
