#!/usr/bin/env python3
"""Launch the Isaac online ACT-adapter SERL/SAC Stage 5 path."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
ISAAC_SERL_TRAIN = (
    REPO_ROOT / "aic_utils" / "aic_isaac" / "aic_isaaclab" / "scripts" / "serl" / "train.py"
)


@dataclass
class ReplayBufferConfig:
    capacity: int = 100_000
    batch_size: int = 256
    warmup_steps: int = 1_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="AIC-Task-v0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--act-torchscript", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train/isaac_stage5_serl"))
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--max-wall-time-minutes", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--act-lr", type=float, default=1e-5)
    parser.add_argument("--adapter-penalty-weight", type=float, default=1e-3)
    parser.add_argument("--act-preservation-weight", type=float, default=1e-2)
    parser.add_argument("--adapter-delta-clip", type=float, default=0.05)
    parser.add_argument("--action-clip", type=float, default=0.05)
    parser.add_argument("--freeze-act", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--isaaclab",
        default=os.environ.get("ISAACLAB_LAUNCHER", "isaaclab"),
        help="Isaac Lab launcher command or path. Defaults to ISAACLAB_LAUNCHER or 'isaaclab'.",
    )
    parser.add_argument("--run-name", default="stage5_online_serl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[])
    return parser.parse_args()


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    return {
        "path": str(path),
        "step": checkpoint.get("step"),
        "vision_offline_serl_config": checkpoint.get("vision_offline_serl_config"),
        "dataset_summary": checkpoint.get("dataset_summary"),
        "warmstart_report": checkpoint.get("warmstart_report"),
        "has_actor": "actor" in checkpoint,
        "has_critics": "critic1" in checkpoint and "critic2" in checkpoint,
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    replay = ReplayBufferConfig(
        capacity=args.replay_capacity,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
    )
    checkpoint = inspect_checkpoint(args.checkpoint)
    return {
        "status": "implemented_short_run_capable",
        "note": (
            "This validates the ACT-adapter SERL checkpoint and records the intended "
            "off-policy Isaac SERL/SAC configuration. Without --dry-run, this wrapper "
            "launches the Isaac Lab online SERL trainer."
        ),
        "task": args.task,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "device": args.device,
        "headless": args.headless,
        "output_dir": str(args.output_dir),
        "steps": args.steps,
        "updates": args.updates,
        "max_wall_time_minutes": args.max_wall_time_minutes,
        "freeze_act": args.freeze_act,
        "act_torchscript": str(args.act_torchscript),
        "gamma": args.gamma,
        "tau": args.tau,
        "adapter_lr": args.adapter_lr,
        "critic_lr": args.critic_lr,
        "act_lr": args.act_lr,
        "adapter_penalty_weight": args.adapter_penalty_weight,
        "act_preservation_weight": args.act_preservation_weight,
        "adapter_delta_clip": args.adapter_delta_clip,
        "action_clip": args.action_clip,
        "replay_buffer": asdict(replay),
        "checkpoint": checkpoint,
        "implemented": [
            "Launch Isaac Lab AIC-Task-v0 as an off-policy env.",
            "Restore ACTAdapterSERLActor and twin critics from checkpoint.",
            "Collect transitions into replay buffer.",
            "Update critics and adapter actor with ACT frozen by default.",
            "Save online SERL checkpoints in the same ACT-adapter format.",
        ],
        "known_limitations": [
            "The trainer disables PPO-specific ResNet observation terms but keeps Isaac camera sensors enabled.",
            "Raw Isaac camera RGB tensors are resized to the LeRobot ACT image shape before ACT TorchScript inference.",
        ],
    }


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    cmd = [
        args.isaaclab,
        "-p",
        str(ISAAC_SERL_TRAIN),
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--checkpoint",
        str(args.checkpoint),
        "--act_torchscript",
        str(args.act_torchscript),
        "--output_dir",
        str(args.output_dir),
        "--run_name",
        args.run_name,
        "--steps",
        str(args.steps),
        "--updates",
        str(args.updates),
        "--batch_size",
        str(args.batch_size),
        "--replay_capacity",
        str(args.replay_capacity),
        "--max_wall_time_minutes",
        str(args.max_wall_time_minutes),
        "--log_every",
        str(args.log_every),
        "--gamma",
        str(args.gamma),
        "--tau",
        str(args.tau),
        "--adapter_lr",
        str(args.adapter_lr),
        "--critic_lr",
        str(args.critic_lr),
        "--act_lr",
        str(args.act_lr),
        "--adapter_penalty_weight",
        str(args.adapter_penalty_weight),
        "--act_preservation_weight",
        str(args.act_preservation_weight),
        "--adapter_delta_clip",
        str(args.adapter_delta_clip),
        "--action_clip",
        str(args.action_clip),
    ]
    cmd.append("--freeze_act" if args.freeze_act else "--no-freeze_act")
    if args.headless:
        cmd.append("--headless")
    if args.device:
        cmd.extend(["--device", args.device])
    cmd.extend(args.extra_arg)
    env = os.environ.copy()
    env["AIC_ISAAC_DISABLE_CAMERAS"] = "0"
    return cmd, env


def main() -> int:
    args = parse_args()
    plan = build_plan(args)
    rendered = json.dumps(plan, indent=2, sort_keys=True)
    if args.dry_run:
        cmd, _ = build_command(args)
        print(rendered)
        print("COMMAND:", " ".join(cmd))
        return 0
    cmd, env = build_command(args)
    launcher = shutil.which(cmd[0]) if os.path.basename(cmd[0]) == cmd[0] else cmd[0]
    if launcher is None or not Path(launcher).exists():
        raise FileNotFoundError(
            f"Isaac Lab launcher '{cmd[0]}' was not found. Run inside the Isaac Lab container, "
            "add 'isaaclab' to PATH, pass --isaaclab /path/to/isaaclab.sh, or set ISAACLAB_LAUNCHER."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "online_serl_plan.json").write_text(rendered + "\n", encoding="utf-8")
    return subprocess.run(cmd, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
