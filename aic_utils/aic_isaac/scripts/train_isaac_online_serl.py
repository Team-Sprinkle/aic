#!/usr/bin/env python3
"""Launch Isaac online ACT-adapter SERL/SAC training."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from isaac_episode_configs import _split_filenames, materialize_episode_configs, materialize_many_episode_configs

ISAAC_SERL_TRAIN = (
    REPO_ROOT / "aic_utils" / "aic_isaac" / "aic_isaaclab" / "scripts" / "serl" / "train.py"
)


@dataclass
class ReplayBufferConfig:
    capacity: int = 100_000
    batch_size: int = 256
    warmup_steps: int = 1_000
    actor_update_start_steps: int = 0
    actor_update_end_steps: int = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="AIC-Task-v0")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument(
        "--rendering-mode",
        choices=["quality", "balanced", "performance"],
        default="performance",
        help="Isaac/RTX rendering preset. ACT-backed SERL requires cameras; performance is the default smoke/training preset.",
    )
    parser.add_argument(
        "--kit-args",
        default=None,
        help="Optional raw Omniverse Kit args forwarded to IsaacLab, for driver/runtime debugging.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--act-only",
        action="store_true",
        help="Start from ACT TorchScript only with zero adapter and fresh critics.",
    )
    parser.add_argument("--act-only-state-dim", type=int, default=82)
    parser.add_argument(
        "--act-only-action-horizon",
        type=int,
        default=0,
        help="ACT output chunk size for ACT-only mode. 0 lets trainer read TorchScript chunk_size metadata.",
    )
    parser.add_argument("--act-only-single-action-dim", type=int, default=6)
    parser.add_argument("--n-action-steps", type=int, default=4)
    parser.add_argument("--act-only-adapter-hidden-dim", type=int, default=256)
    parser.add_argument("--act-only-adapter-num-layers", type=int, default=2)
    parser.add_argument("--act-only-adapter-activation", choices=["relu", "gelu", "tanh"], default="gelu")
    parser.add_argument(
        "--act-only-actor-mode",
        choices=["act_adapter", "act_direct"],
        default="act_adapter",
        help=(
            "act_adapter learns a residual added to ACT. act_direct initializes to ACT "
            "but trains an unconstrained full-action override before final TCP caps."
        ),
    )
    parser.add_argument("--act-torchscript", type=Path, required=True)
    parser.add_argument(
        "--act-torchscript-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for frozen ACT TorchScript inference. Auto uses CUDA only for CUDA-exported TorchScript.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train/isaac_online_serl"))
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument(
        "--update-every-steps",
        type=int,
        default=1,
        help="Run one gradient update every N Isaac environment steps after warmup.",
    )
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument(
        "--actor-update-start-steps",
        type=int,
        default=0,
        help="Delay adapter actor updates until this many Isaac environment steps have been collected.",
    )
    parser.add_argument(
        "--actor-update-end-steps",
        type=int,
        default=0,
        help="If >0, freeze actor updates after this Isaac environment step while continuing rollouts.",
    )
    parser.add_argument(
        "--episode-length-s",
        type=float,
        default=0.0,
        help="Optional Isaac episode timeout in seconds. 0 keeps the task default.",
    )
    parser.add_argument(
        "--policy-hz",
        type=float,
        default=20.0,
        help="Isaac policy/control rate in Hz. Defaults to 20 Hz to match Gazebo expert datasets.",
    )
    parser.add_argument("--max-wall-time-minutes", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=0,
        help="Save periodic online SERL checkpoints every N Isaac environment steps. 0 disables periodic saves.",
    )
    parser.add_argument(
        "--save-latest-every-steps",
        type=int,
        default=0,
        help="Overwrite checkpoint_latest.pt every N Isaac environment steps. 0 only writes final latest.",
    )
    parser.add_argument(
        "--ram-watchdog-min-available-gb",
        type=float,
        default=0.0,
        help="If >0, trainer saves/exits when host MemAvailable drops below this many GiB.",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--act-lr", type=float, default=1e-5)
    parser.add_argument(
        "--actor-q-weight",
        type=float,
        default=1.0,
        help="Scale the actor -Q term. Use 0 during guide-imitation warm starts when the critic is not trustworthy yet.",
    )
    parser.add_argument(
        "--bc-weight",
        type=float,
        default=0.0,
        help="Legacy expert BC fallback weight. Defaults to zero so BC is strictly opt-in.",
    )
    parser.add_argument("--expert-dataset-root", type=Path, default=None)
    parser.add_argument("--expert-bc-weight", type=float, default=None)
    parser.add_argument("--expert-bc-max-samples", type=int, default=8192)
    parser.add_argument("--expert-bc-neighbor-chunk", type=int, default=8192)
    parser.add_argument(
        "--expert-bc-every",
        type=int,
        default=4,
        help="Compute expert BC regularization every N actor updates. Use 1 for every update.",
    )
    parser.add_argument("--adapter-penalty-weight", type=float, default=1e-3)
    parser.add_argument("--act-preservation-weight", type=float, default=1e-2)
    parser.add_argument("--target-action-guide-weight", type=float, default=0.0)
    parser.add_argument(
        "--target-action-guide-mode",
        choices=["axis", "cheatcode_transform"],
        default="axis",
        help=(
            "Guide-action geometry. 'axis' keeps the conservative lateral/axial guide; "
            "'cheatcode_transform' mirrors the ROS CheatCode rigid transform."
        ),
    )
    parser.add_argument("--target-action-guide-step-size", type=float, default=0.001)
    parser.add_argument("--target-action-guide-rotation-step-size", type=float, default=0.0)
    parser.add_argument("--target-action-guide-axial-step-size", type=float, default=0.0)
    parser.add_argument("--target-action-guide-lateral-switch-m", type=float, default=0.002)
    parser.add_argument("--target-action-guide-axial-blend-lateral-m", type=float, default=0.006)
    parser.add_argument("--target-action-guide-collect-blend", type=float, default=0.0)
    parser.add_argument("--target-action-guide-collect-steps", type=int, default=0)
    parser.add_argument("--target-action-guide-collect-decay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-action-guide-prefix-decay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-action-guide-train-executed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adapter-delta-clip", type=float, default=0.05)
    parser.add_argument(
        "--tcp-translation-action-clip",
        type=float,
        default=0.0,
        help="Optional per-step TCP translation norm cap in meters for each action in the predicted chunk.",
    )
    parser.add_argument(
        "--tcp-rotation-action-clip",
        type=float,
        default=0.0,
        help="Optional per-step TCP rotation-vector norm cap in radians for each action in the predicted chunk.",
    )
    parser.add_argument("--insertion-action-guard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--insertion-action-guard-lateral-threshold-m", type=float, default=0.002)
    parser.add_argument("--insertion-action-guard-lateral-step-m", type=float, default=0.0005)
    parser.add_argument(
        "--insertion-action-guard-adaptive-lateral-sign",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--insertion-action-guard-adaptive-lateral-flip-margin-m", type=float, default=0.00005)
    parser.add_argument("--insertion-action-guard-centered-axial-step-m", type=float, default=0.0)
    parser.add_argument("--insertion-action-guard-retention", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--insertion-action-guard-retention-entry-depth-m", type=float, default=0.0)
    parser.add_argument("--insertion-action-guard-retention-lateral-threshold-m", type=float, default=0.0015)
    parser.add_argument(
        "--insertion-action-guard-retention-ignore-lateral-threshold",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--insertion-action-guard-retention-min-axial-step-m", type=float, default=0.0)
    parser.add_argument("--insertion-action-guard-retention-lateral-scale", type=float, default=1.0)
    parser.add_argument(
        "--action-clip",
        type=float,
        default=0.0,
        help="Optional full-action clamp. Defaults disabled because it can distort the ACT base action.",
    )
    parser.add_argument(
        "--isaac-action-scale",
        type=float,
        default=1.0,
        help="Isaac IK action scale. Keep 1.0 for ACT/SERL physical TCP-delta actions.",
    )
    parser.add_argument(
        "--act-normalized-state-clip",
        type=float,
        default=0.0,
        help="If >0, clamp ACT-normalized state before TorchScript inference inside Isaac.",
    )
    parser.add_argument("--freeze-act", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--randomization-profile",
        choices=["none", "light", "heavy"],
        default="none",
        help=(
            "Isaac scene/randomization profile. Use 'none' to match fixed Gazebo "
            "card0/port0 smoke data as closely as the current Isaac scene allows."
        ),
    )
    parser.add_argument("--insertion-progress-weight", type=float, default=0.25)
    parser.add_argument("--insertion-progress-scale", type=float, default=0.003)
    parser.add_argument("--insertion-distance-weight", type=float, default=0.25)
    parser.add_argument("--insertion-close-weight", type=float, default=0.35)
    parser.add_argument("--insertion-orientation-weight", type=float, default=0.10)
    parser.add_argument("--insertion-orientation-std", type=float, default=0.03)
    parser.add_argument("--insertion-orientation-gate-sigma", type=float, default=0.012)
    parser.add_argument("--insertion-reaching-weight", type=float, default=0.0)
    parser.add_argument("--insertion-terminal-weight", type=float, default=1.0)
    parser.add_argument("--insertion-lateral-weight", type=float, default=-0.05)
    parser.add_argument("--insertion-lateral-gate-sigma", type=float, default=0.012)
    parser.add_argument("--insertion-lateral-error-scale", type=float, default=0.006)
    parser.add_argument("--insertion-motion-projection-weight", type=float, default=0.0)
    parser.add_argument("--insertion-motion-projection-scale", type=float, default=0.001)
    parser.add_argument("--insertion-lateral-progress-weight", type=float, default=0.0)
    parser.add_argument("--insertion-lateral-progress-scale", type=float, default=0.001)
    parser.add_argument("--insertion-axial-progress-weight", type=float, default=0.0)
    parser.add_argument("--insertion-axial-progress-scale", type=float, default=0.001)
    parser.add_argument("--insertion-corridor-weight", type=float, default=0.0)
    parser.add_argument("--insertion-corridor-sigma", type=float, default=0.0025)
    parser.add_argument("--insertion-bypass-penalty-scale", type=float, default=1.0)
    parser.add_argument("--insertion-axis", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument(
        "--reward-preset",
        choices=["default", "near_gate_corridor_v1"],
        default="default",
        help="Apply a named online SERL reward-shaping preset before launch.",
    )
    parser.add_argument("--force-delta-penalty-weight", type=float, default=0.3)
    parser.add_argument("--force-delta-threshold", type=float, default=10.0)
    parser.add_argument("--force-delta-reference", type=float, default=20.0)
    parser.add_argument("--force-delta-saturation", type=float, default=30.0)
    parser.add_argument("--force-delta-knee-penalty-fraction", type=float, default=0.1)
    parser.add_argument("--isaac-force-observation-clip-n", type=float, default=35.0)
    parser.add_argument("--target-reward-body", default="sfp_tip_link")
    parser.add_argument("--target-reward-distance-std", type=float, default=0.02)
    parser.add_argument("--target-reward-close-sigma", type=float, default=0.006)
    parser.add_argument("--target-reward-reaching-threshold", type=float, default=0.01)
    parser.add_argument("--terminate-on-target-success", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--target-success-termination-threshold",
        type=float,
        default=0.0005,
        help=(
            "Strict target-distance threshold for terminating successful episodes. "
            "This is intentionally separate from --target-reward-reaching-threshold."
        ),
    )
    parser.add_argument("--target-success-axial-threshold", type=float, default=None)
    parser.add_argument("--target-success-lateral-threshold", type=float, default=None)
    parser.add_argument("--target-reward-position-offset", type=float, nargs=3, default=None)
    parser.add_argument("--target-reward-body-position-offset", type=float, nargs=3, default=None)
    parser.add_argument("--state-source", choices=["lerobot_compatible", "policy_prefix"], default="lerobot_compatible")
    parser.add_argument("--task-family", choices=["sfp_to_nic", "sc_to_sc"], default="sfp_to_nic")
    parser.add_argument("--target-port-index", type=int, default=0)
    parser.add_argument("--target-card-index", type=int, default=0)
    parser.add_argument("--target-card-valid", type=int, default=1)
    parser.add_argument(
        "--task-distribution-yaml",
        type=Path,
        default=None,
        help=(
            "Optional expert-generation-style YAML for sampling task conditioning "
            "inside Isaac online SERL."
        ),
    )
    parser.add_argument(
        "--isaac-user-config-yaml",
        type=Path,
        default=None,
        help=(
            "Minimal user YAML, matching the expert data-generation request format. "
            "The wrapper materializes per-episode Isaac YAMLs and passes them to the trainer."
        ),
    )
    parser.add_argument(
        "--isaac-user-config-dir",
        type=Path,
        default=None,
        help="Directory containing multiple minimal expert-generation-style YAMLs.",
    )
    parser.add_argument(
        "--isaac-user-config-filenames",
        default="",
        help=(
            "Whitespace- or comma-separated filenames inside --isaac-user-config-dir. "
            "When omitted, all *.yaml/*.yml files in the directory are used."
        ),
    )
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=1,
        help="Maximum GPU shards to materialize for curriculum episode configs.",
    )
    parser.add_argument(
        "--episode-config-dir",
        type=Path,
        default=None,
        help="Optional pre-materialized Isaac episode config directory.",
    )
    parser.add_argument(
        "--episode-config-count",
        type=int,
        default=None,
        help="Number of per-episode configs to generate from --isaac-user-config-yaml.",
    )
    parser.add_argument("--gripper-joint-position", type=float, default=0.0035405)
    parser.add_argument(
        "--initial-arm-joint-pos",
        default=None,
        help=(
            "Optional comma-separated six-joint Isaac arm reset pose. Use this for "
            "near-port curriculum starts; omit it to keep the task default."
        ),
    )
    parser.add_argument(
        "--isaaclab",
        default=os.environ.get("ISAACLAB_LAUNCHER", "isaaclab"),
        help="Isaac Lab launcher command or path. Defaults to ISAACLAB_LAUNCHER or 'isaaclab'.",
    )
    parser.add_argument("--run-name", default="isaac_online_serl")
    parser.add_argument("--debug-timing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-fabric", action="store_true", default=False)
    parser.add_argument("--save-step-images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--debug-visual-overlays",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay insertion-frame markers and body metrics on saved debug images.",
    )
    parser.add_argument("--image-log-every", type=int, default=1)
    parser.add_argument("--max-logged-image-steps", type=int, default=200)
    parser.add_argument(
        "--swap-rgb-channels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reverse Isaac camera RGB channel order before ACT/SERL inference and debug image logging.",
    )
    parser.add_argument(
        "--force-camera-render-before-read",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force an Isaac render before reading camera tensors if freshness diagnostics show stale visual observations.",
    )
    parser.add_argument(
        "--disable-ppo-resnet-observation-terms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable Isaac policy ResNet image observation terms; raw camera tensors are still read by SERL.",
    )
    parser.add_argument("--debug-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diagnostics-every", type=int, default=100)
    parser.add_argument("--debug-audit-steps", type=int, default=0)
    parser.add_argument("--audit-act-only", action="store_true", default=False)
    parser.add_argument("--audit-zero-adapter", action="store_true", default=False)
    parser.add_argument(
        "--treat-time-limit-truncation-as-terminal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forwarded to trainer. Defaults false so TD bootstraps through time-limit truncation.",
    )
    parser.add_argument(
        "--tcp-action-frame",
        choices=["gripper_tcp", "wrist_3_link", "root"],
        default="gripper_tcp",
        help="Frame convention for ACT/SERL TCP delta actions before passing them to Isaac IK.",
    )
    parser.add_argument("--debug-audit-axis-magnitude", type=float, default=0.0)
    parser.add_argument("--debug-audit-constant-action", type=float, nargs=6, default=None)
    parser.add_argument(
        "--enable-contact-sensor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep Isaac contact sensor enabled so 72D/82D states populate Gazebo-compatible force/contact slots.",
    )
    parser.add_argument(
        "--fix-isaac-ik-xy-sign",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flip Isaac IK root-frame x/y translation commands to match realized TCP motion direction.",
    )
    parser.add_argument(
        "--isaac-ik-xy-sign-by-target-card",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply Isaac IK x/y sign correction per SFP-to-NIC target card parity.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[])
    return parser.parse_args(argv)


def inspect_checkpoint(path: Path, *, required: bool) -> dict[str, Any]:
    if path is None:
        if required:
            raise ValueError("--checkpoint is required unless --act-only is set")
        return {
            "path": None,
            "exists": False,
            "inspect_skipped": True,
        }
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return {
            "path": str(path),
            "exists": False,
            "inspect_skipped": True,
        }
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    return {
        "path": str(path),
        "exists": True,
        "step": checkpoint.get("step"),
        "vision_offline_serl_config": checkpoint.get("vision_offline_serl_config"),
        "dataset_summary": checkpoint.get("dataset_summary"),
        "warmstart_report": checkpoint.get("warmstart_report"),
        "has_actor": "actor" in checkpoint,
        "has_critics": "critic1" in checkpoint and "critic2" in checkpoint,
    }


def build_plan(args: argparse.Namespace, *, inspect_required: bool = True) -> dict[str, Any]:
    normalize_reward_sign_args(args)
    replay = ReplayBufferConfig(
        capacity=args.replay_capacity,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        actor_update_start_steps=args.actor_update_start_steps,
    )
    act_only = bool(getattr(args, "act_only", False))
    act_only_state_dim = int(getattr(args, "act_only_state_dim", 82))
    act_only_action_horizon = int(getattr(args, "act_only_action_horizon", 0))
    act_only_single_action_dim = int(getattr(args, "act_only_single_action_dim", 6))
    act_only_adapter_hidden_dim = int(getattr(args, "act_only_adapter_hidden_dim", 256))
    act_only_adapter_num_layers = int(getattr(args, "act_only_adapter_num_layers", 2))
    act_only_adapter_activation = str(getattr(args, "act_only_adapter_activation", "gelu"))
    checkpoint = (
        {
            "path": None,
            "exists": False,
            "act_only": True,
            "vision_offline_serl_config": {
                "state_dim": act_only_state_dim,
                "action_dim": act_only_action_horizon * act_only_single_action_dim,
                "action_horizon": act_only_action_horizon,
            },
        }
        if act_only
        else inspect_checkpoint(args.checkpoint, required=inspect_required)
    )
    insertion_progress_weight = float(getattr(args, "insertion_progress_weight", 0.25))
    insertion_progress_scale = float(getattr(args, "insertion_progress_scale", 0.003))
    insertion_orientation_std = float(getattr(args, "insertion_orientation_std", 0.03))
    insertion_orientation_gate_sigma = float(getattr(args, "insertion_orientation_gate_sigma", 0.012))
    insertion_terminal_weight = float(getattr(args, "insertion_terminal_weight", 1.0))
    bc_weight = float(getattr(args, "bc_weight", 0.0))
    expert_dataset_root = getattr(args, "expert_dataset_root", None)
    expert_bc_weight = getattr(args, "expert_bc_weight", None)
    expert_bc_max_samples = int(getattr(args, "expert_bc_max_samples", 8192))
    expert_bc_neighbor_chunk = int(getattr(args, "expert_bc_neighbor_chunk", 8192))
    expert_bc_every = int(getattr(args, "expert_bc_every", 4))
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
        "rendering_mode": args.rendering_mode,
        "kit_args": args.kit_args,
        "output_dir": str(args.output_dir),
        "steps": args.steps,
        "updates": args.updates,
        "update_every_steps": int(getattr(args, "update_every_steps", 1)),
        "warmup_steps": args.warmup_steps,
        "actor_update_start_steps": args.actor_update_start_steps,
        "actor_update_end_steps": args.actor_update_end_steps,
        "episode_length_s": float(getattr(args, "episode_length_s", 0.0)),
        "policy_hz": float(getattr(args, "policy_hz", 20.0)),
        "max_wall_time_minutes": args.max_wall_time_minutes,
        "save_every_steps": getattr(args, "save_every_steps", 0),
        "save_latest_every_steps": getattr(args, "save_latest_every_steps", 0),
        "ram_watchdog_min_available_gb": getattr(args, "ram_watchdog_min_available_gb", 0.0),
        "freeze_act": args.freeze_act,
        "act_only": act_only,
        "act_only_state_dim": act_only_state_dim,
        "act_only_action_horizon": act_only_action_horizon,
        "act_only_single_action_dim": act_only_single_action_dim,
        "n_action_steps": int(getattr(args, "n_action_steps", 4)),
        "act_only_adapter_hidden_dim": act_only_adapter_hidden_dim,
        "act_only_adapter_num_layers": act_only_adapter_num_layers,
        "act_only_adapter_activation": act_only_adapter_activation,
        "act_only_actor_mode": str(getattr(args, "act_only_actor_mode", "act_adapter")),
        "act_torchscript": str(args.act_torchscript),
        "act_torchscript_device": args.act_torchscript_device,
        "gamma": args.gamma,
        "tau": args.tau,
        "adapter_lr": args.adapter_lr,
        "critic_lr": args.critic_lr,
        "act_lr": args.act_lr,
        "actor_q_weight": args.actor_q_weight,
        "bc_weight": bc_weight,
        "expert_dataset_root": str(expert_dataset_root) if expert_dataset_root else None,
        "expert_bc_weight": expert_bc_weight,
        "expert_bc_max_samples": expert_bc_max_samples,
        "expert_bc_neighbor_chunk": expert_bc_neighbor_chunk,
        "expert_bc_every": expert_bc_every,
        "adapter_penalty_weight": args.adapter_penalty_weight,
        "act_preservation_weight": args.act_preservation_weight,
        "target_action_guide_weight": getattr(args, "target_action_guide_weight", 0.0),
        "target_action_guide_mode": getattr(args, "target_action_guide_mode", "axis"),
        "target_action_guide_step_size": getattr(args, "target_action_guide_step_size", 0.001),
        "target_action_guide_rotation_step_size": getattr(args, "target_action_guide_rotation_step_size", 0.0),
        "target_action_guide_axial_step_size": getattr(args, "target_action_guide_axial_step_size", 0.0),
        "target_action_guide_lateral_switch_m": getattr(args, "target_action_guide_lateral_switch_m", 0.002),
        "target_action_guide_axial_blend_lateral_m": getattr(
            args, "target_action_guide_axial_blend_lateral_m", 0.006
        ),
        "target_action_guide_collect_blend": getattr(args, "target_action_guide_collect_blend", 0.0),
        "target_action_guide_collect_steps": getattr(args, "target_action_guide_collect_steps", 0),
        "target_action_guide_collect_decay": bool(getattr(args, "target_action_guide_collect_decay", False)),
        "target_action_guide_prefix_decay": bool(getattr(args, "target_action_guide_prefix_decay", False)),
        "target_action_guide_train_executed": bool(getattr(args, "target_action_guide_train_executed", False)),
        "adapter_delta_clip": args.adapter_delta_clip,
        "tcp_translation_action_clip": args.tcp_translation_action_clip,
        "tcp_rotation_action_clip": args.tcp_rotation_action_clip,
        "action_clip": args.action_clip,
        "isaac_action_scale": args.isaac_action_scale,
        "act_normalized_state_clip": args.act_normalized_state_clip,
        "randomization_profile": args.randomization_profile,
        "insertion_distance_weight": args.insertion_distance_weight,
        "insertion_close_weight": args.insertion_close_weight,
        "insertion_progress_weight": insertion_progress_weight,
        "insertion_progress_scale": insertion_progress_scale,
        "insertion_orientation_weight": args.insertion_orientation_weight,
        "insertion_orientation_std": insertion_orientation_std,
        "insertion_orientation_gate_sigma": insertion_orientation_gate_sigma,
        "insertion_reaching_weight": args.insertion_reaching_weight,
        "insertion_terminal_weight": insertion_terminal_weight,
        "insertion_lateral_weight": args.insertion_lateral_weight,
        "insertion_lateral_gate_sigma": args.insertion_lateral_gate_sigma,
        "insertion_lateral_error_scale": args.insertion_lateral_error_scale,
        "insertion_motion_projection_weight": getattr(args, "insertion_motion_projection_weight", 0.0),
        "insertion_motion_projection_scale": getattr(args, "insertion_motion_projection_scale", 0.001),
        "insertion_lateral_progress_weight": getattr(args, "insertion_lateral_progress_weight", 0.0),
        "insertion_lateral_progress_scale": getattr(args, "insertion_lateral_progress_scale", 0.001),
        "insertion_axial_progress_weight": getattr(args, "insertion_axial_progress_weight", 0.0),
        "insertion_axial_progress_scale": getattr(args, "insertion_axial_progress_scale", 0.001),
        "insertion_corridor_weight": getattr(args, "insertion_corridor_weight", 0.0),
        "insertion_corridor_sigma": getattr(args, "insertion_corridor_sigma", 0.0025),
        "insertion_bypass_penalty_scale": getattr(args, "insertion_bypass_penalty_scale", 1.0),
        "insertion_axis": args.insertion_axis,
        "force_delta_penalty_weight": args.force_delta_penalty_weight,
        "force_delta_threshold": args.force_delta_threshold,
        "force_delta_reference": args.force_delta_reference,
        "force_delta_saturation": args.force_delta_saturation,
        "force_delta_knee_penalty_fraction": args.force_delta_knee_penalty_fraction,
        "isaac_force_observation_clip_n": args.isaac_force_observation_clip_n,
        "target_reward_body": args.target_reward_body,
        "target_reward_distance_std": args.target_reward_distance_std,
        "target_reward_close_sigma": args.target_reward_close_sigma,
        "target_reward_reaching_threshold": args.target_reward_reaching_threshold,
        "terminate_on_target_success": bool(getattr(args, "terminate_on_target_success", False)),
        "target_success_termination_threshold": args.target_success_termination_threshold,
        "target_success_axial_threshold": getattr(args, "target_success_axial_threshold", None),
        "target_success_lateral_threshold": getattr(args, "target_success_lateral_threshold", None),
        "target_reward_position_offset": args.target_reward_position_offset,
        "target_reward_body_position_offset": args.target_reward_body_position_offset,
        "initial_arm_joint_pos": args.initial_arm_joint_pos,
        "state_source": args.state_source,
        "task_context": {
            "task_family": args.task_family,
            "target_port_index": args.target_port_index,
            "target_card_index": args.target_card_index,
            "target_card_valid": args.target_card_valid,
        },
        "task_distribution_yaml": str(args.task_distribution_yaml) if args.task_distribution_yaml else None,
        "isaac_user_config_yaml": str(args.isaac_user_config_yaml) if args.isaac_user_config_yaml else None,
        "isaac_user_config_dir": str(args.isaac_user_config_dir) if args.isaac_user_config_dir else None,
        "isaac_user_config_filenames": args.isaac_user_config_filenames,
        "episode_config_dir": str(args.episode_config_dir) if args.episode_config_dir else None,
        "episode_config_count": args.episode_config_count,
        "max_gpus": args.max_gpus,
        "gripper_joint_position": args.gripper_joint_position,
        "debug_timing": args.debug_timing,
        "disable_fabric": args.disable_fabric,
        "save_step_images": args.save_step_images,
        "image_log_every": args.image_log_every,
        "max_logged_image_steps": args.max_logged_image_steps,
        "critic_action_representation": "first_executed_6d",
        "debug_diagnostics": bool(getattr(args, "debug_diagnostics", False)),
        "diagnostics_every": int(getattr(args, "diagnostics_every", 100)),
        "debug_audit_steps": int(getattr(args, "debug_audit_steps", 0)),
        "audit_act_only": bool(getattr(args, "audit_act_only", False)),
        "audit_zero_adapter": bool(getattr(args, "audit_zero_adapter", False)),
        "treat_time_limit_truncation_as_terminal": bool(
            getattr(args, "treat_time_limit_truncation_as_terminal", False)
        ),
        "tcp_action_frame": str(getattr(args, "tcp_action_frame", "gripper_tcp")),
        "debug_audit_axis_magnitude": float(getattr(args, "debug_audit_axis_magnitude", 0.0)),
        "debug_audit_constant_action": getattr(args, "debug_audit_constant_action", None),
        "enable_contact_sensor": bool(getattr(args, "enable_contact_sensor", True)),
        "fix_isaac_ik_xy_sign": bool(getattr(args, "fix_isaac_ik_xy_sign", True)),
        "isaac_ik_xy_sign_by_target_card": bool(
            getattr(args, "isaac_ik_xy_sign_by_target_card", False)
        ),
        "force_camera_render_before_read": bool(getattr(args, "force_camera_render_before_read", False)),
        "disable_ppo_resnet_observation_terms": bool(getattr(args, "disable_ppo_resnet_observation_terms", True)),
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
    normalize_reward_sign_args(args)
    insertion_progress_weight = float(getattr(args, "insertion_progress_weight", 0.25))
    insertion_progress_scale = float(getattr(args, "insertion_progress_scale", 0.003))
    insertion_orientation_std = float(getattr(args, "insertion_orientation_std", 0.03))
    insertion_orientation_gate_sigma = float(getattr(args, "insertion_orientation_gate_sigma", 0.012))
    insertion_terminal_weight = float(getattr(args, "insertion_terminal_weight", 1.0))
    bc_weight = float(getattr(args, "bc_weight", 0.0))
    expert_dataset_root = getattr(args, "expert_dataset_root", None)
    expert_bc_weight = getattr(args, "expert_bc_weight", None)
    expert_bc_max_samples = int(getattr(args, "expert_bc_max_samples", 8192))
    expert_bc_neighbor_chunk = int(getattr(args, "expert_bc_neighbor_chunk", 8192))
    expert_bc_every = int(getattr(args, "expert_bc_every", 4))
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
        "--act_torchscript",
        str(args.act_torchscript),
        "--act_torchscript_device",
        args.act_torchscript_device,
        "--output_dir",
        str(args.output_dir),
        "--run_name",
        args.run_name,
        "--steps",
        str(args.steps),
        "--updates",
        str(args.updates),
        "--update_every_steps",
        str(int(getattr(args, "update_every_steps", 1))),
        "--batch_size",
        str(args.batch_size),
        "--replay_capacity",
        str(args.replay_capacity),
        "--warmup_steps",
        str(args.warmup_steps),
        "--actor_update_start_steps",
        str(args.actor_update_start_steps),
        "--actor_update_end_steps",
        str(args.actor_update_end_steps),
        "--n_action_steps",
        str(int(getattr(args, "n_action_steps", 4))),
        "--episode_length_s",
        str(float(getattr(args, "episode_length_s", 0.0))),
        "--policy_hz",
        str(float(getattr(args, "policy_hz", 20.0))),
        "--max_wall_time_minutes",
        str(args.max_wall_time_minutes),
        "--log_every",
        str(args.log_every),
        "--save_every_steps",
        str(getattr(args, "save_every_steps", 0)),
        "--save_latest_every_steps",
        str(getattr(args, "save_latest_every_steps", 0)),
        "--ram_watchdog_min_available_gb",
        str(getattr(args, "ram_watchdog_min_available_gb", 0.0)),
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
        "--actor_q_weight",
        str(args.actor_q_weight),
        "--bc_weight",
        str(bc_weight),
        "--adapter_penalty_weight",
        str(args.adapter_penalty_weight),
        "--act_preservation_weight",
        str(args.act_preservation_weight),
        "--target_action_guide_weight",
        str(getattr(args, "target_action_guide_weight", 0.0)),
        "--target_action_guide_mode",
        str(getattr(args, "target_action_guide_mode", "axis")),
        "--target_action_guide_step_size",
        str(getattr(args, "target_action_guide_step_size", 0.001)),
        "--target_action_guide_rotation_step_size",
        str(getattr(args, "target_action_guide_rotation_step_size", 0.0)),
        "--target_action_guide_axial_step_size",
        str(getattr(args, "target_action_guide_axial_step_size", 0.0)),
        "--target_action_guide_lateral_switch_m",
        str(getattr(args, "target_action_guide_lateral_switch_m", 0.002)),
        "--target_action_guide_axial_blend_lateral_m",
        str(getattr(args, "target_action_guide_axial_blend_lateral_m", 0.006)),
        "--target_action_guide_collect_blend",
        str(getattr(args, "target_action_guide_collect_blend", 0.0)),
        "--target_action_guide_collect_steps",
        str(getattr(args, "target_action_guide_collect_steps", 0)),
        "--target_action_guide_collect_decay" if getattr(args, "target_action_guide_collect_decay", False) else "--no-target_action_guide_collect_decay",
        "--target_action_guide_prefix_decay" if getattr(args, "target_action_guide_prefix_decay", False) else "--no-target_action_guide_prefix_decay",
        "--target_action_guide_train_executed" if getattr(args, "target_action_guide_train_executed", False) else "--no-target_action_guide_train_executed",
        "--adapter_delta_clip",
        str(args.adapter_delta_clip),
        "--tcp_translation_action_clip",
        str(args.tcp_translation_action_clip),
        "--tcp_rotation_action_clip",
        str(args.tcp_rotation_action_clip),
        "--insertion_action_guard" if getattr(args, "insertion_action_guard", False) else "--no-insertion_action_guard",
        "--insertion_action_guard_lateral_threshold_m",
        str(getattr(args, "insertion_action_guard_lateral_threshold_m", 0.002)),
        "--insertion_action_guard_lateral_step_m",
        str(getattr(args, "insertion_action_guard_lateral_step_m", 0.0005)),
        "--insertion_action_guard_adaptive_lateral_sign"
        if getattr(args, "insertion_action_guard_adaptive_lateral_sign", False)
        else "--no-insertion_action_guard_adaptive_lateral_sign",
        "--insertion_action_guard_adaptive_lateral_flip_margin_m",
        str(getattr(args, "insertion_action_guard_adaptive_lateral_flip_margin_m", 0.00005)),
        "--insertion_action_guard_centered_axial_step_m",
        str(getattr(args, "insertion_action_guard_centered_axial_step_m", 0.0)),
        "--insertion_action_guard_retention" if getattr(args, "insertion_action_guard_retention", False) else "--no-insertion_action_guard_retention",
        "--insertion_action_guard_retention_entry_depth_m",
        str(getattr(args, "insertion_action_guard_retention_entry_depth_m", 0.0)),
        "--insertion_action_guard_retention_lateral_threshold_m",
        str(getattr(args, "insertion_action_guard_retention_lateral_threshold_m", 0.0015)),
        (
            "--insertion_action_guard_retention_ignore_lateral_threshold"
            if getattr(args, "insertion_action_guard_retention_ignore_lateral_threshold", False)
            else "--no-insertion_action_guard_retention_ignore_lateral_threshold"
        ),
        "--insertion_action_guard_retention_min_axial_step_m",
        str(getattr(args, "insertion_action_guard_retention_min_axial_step_m", 0.0)),
        "--insertion_action_guard_retention_lateral_scale",
        str(getattr(args, "insertion_action_guard_retention_lateral_scale", 1.0)),
        "--action_clip",
        str(args.action_clip),
        "--isaac_action_scale",
        str(args.isaac_action_scale),
        "--act_normalized_state_clip",
        str(args.act_normalized_state_clip),
        "--state_source",
        args.state_source,
        "--task_family",
        args.task_family,
        "--target_port_index",
        str(args.target_port_index),
        "--target_card_index",
        str(args.target_card_index),
        "--target_card_valid",
        str(args.target_card_valid),
        "--gripper_joint_position",
        str(args.gripper_joint_position),
        "--target_reward_distance_weight",
        str(args.insertion_distance_weight),
        "--target_reward_close_weight",
        str(args.insertion_close_weight),
        "--target_reward_progress_weight",
        str(insertion_progress_weight),
        "--target_reward_progress_scale",
        str(insertion_progress_scale),
        "--target_reward_orientation_weight",
        str(args.insertion_orientation_weight),
        "--target_reward_orientation_std",
        str(insertion_orientation_std),
        "--target_reward_orientation_gate_sigma",
        str(insertion_orientation_gate_sigma),
        "--target_reward_reaching_weight",
        str(args.insertion_reaching_weight),
        "--target_reward_terminal_weight",
        str(insertion_terminal_weight),
        "--target_reward_lateral_weight",
        str(args.insertion_lateral_weight),
        "--target_reward_lateral_gate_sigma",
        str(args.insertion_lateral_gate_sigma),
        "--target_reward_lateral_error_scale",
        str(args.insertion_lateral_error_scale),
        "--target_reward_motion_projection_weight",
        str(getattr(args, "insertion_motion_projection_weight", 0.0)),
        "--target_reward_motion_projection_scale",
        str(getattr(args, "insertion_motion_projection_scale", 0.001)),
        "--target_reward_lateral_progress_weight",
        str(getattr(args, "insertion_lateral_progress_weight", 0.0)),
        "--target_reward_lateral_progress_scale",
        str(getattr(args, "insertion_lateral_progress_scale", 0.001)),
        "--target_reward_axial_progress_weight",
        str(getattr(args, "insertion_axial_progress_weight", 0.0)),
        "--target_reward_axial_progress_scale",
        str(getattr(args, "insertion_axial_progress_scale", 0.001)),
        "--target_reward_insertion_corridor_weight",
        str(getattr(args, "insertion_corridor_weight", 0.0)),
        "--target_reward_insertion_corridor_sigma",
        str(getattr(args, "insertion_corridor_sigma", 0.0025)),
        "--target_reward_insertion_bypass_penalty_scale",
        str(getattr(args, "insertion_bypass_penalty_scale", 1.0)),
        "--target_reward_insertion_axis",
        str(args.insertion_axis),
        "--reward_preset",
        str(getattr(args, "reward_preset", "default")),
        "--force_delta_penalty_weight",
        str(args.force_delta_penalty_weight),
        "--force_delta_threshold",
        str(args.force_delta_threshold),
        "--force_delta_reference",
        str(args.force_delta_reference),
        "--force_delta_saturation",
        str(args.force_delta_saturation),
        "--force_delta_knee_penalty_fraction",
        str(args.force_delta_knee_penalty_fraction),
        "--isaac_force_observation_clip_n",
        str(args.isaac_force_observation_clip_n),
        "--target_reward_body",
        args.target_reward_body,
        "--target_reward_distance_std",
        str(args.target_reward_distance_std),
        "--target_reward_close_sigma",
        str(args.target_reward_close_sigma),
        "--target_reward_reaching_threshold",
        str(args.target_reward_reaching_threshold),
        "--terminate_on_target_success" if getattr(args, "terminate_on_target_success", False) else "--no-terminate_on_target_success",
        "--target_success_termination_threshold",
        str(args.target_success_termination_threshold),
    ]
    if getattr(args, "target_success_axial_threshold", None) is not None:
        cmd.extend(["--target_success_axial_threshold", str(args.target_success_axial_threshold)])
    if getattr(args, "target_success_lateral_threshold", None) is not None:
        cmd.extend(["--target_success_lateral_threshold", str(args.target_success_lateral_threshold)])
    act_only = bool(getattr(args, "act_only", False))
    if act_only:
        cmd.extend(
            [
                "--act_only",
                "--act_only_state_dim",
                str(getattr(args, "act_only_state_dim", 82)),
                "--act_only_action_horizon",
                str(getattr(args, "act_only_action_horizon", 0)),
                "--act_only_single_action_dim",
                str(getattr(args, "act_only_single_action_dim", 6)),
                "--act_only_adapter_hidden_dim",
                str(getattr(args, "act_only_adapter_hidden_dim", 256)),
                "--act_only_adapter_num_layers",
                str(getattr(args, "act_only_adapter_num_layers", 2)),
                "--act_only_adapter_activation",
                str(getattr(args, "act_only_adapter_activation", "gelu")),
                "--act_only_actor_mode",
                str(getattr(args, "act_only_actor_mode", "act_adapter")),
            ]
        )
        if args.checkpoint is not None:
            cmd.extend(["--checkpoint", str(args.checkpoint)])
    else:
        cmd.extend(["--checkpoint", str(args.checkpoint)])
    if expert_dataset_root is not None:
        cmd.extend(["--expert_dataset_root", str(expert_dataset_root)])
    if expert_bc_weight is not None:
        cmd.extend(["--expert_bc_weight", str(expert_bc_weight)])
    cmd.extend(
        [
            "--expert_bc_max_samples",
            str(expert_bc_max_samples),
            "--expert_bc_neighbor_chunk",
            str(expert_bc_neighbor_chunk),
            "--expert_bc_every",
            str(expert_bc_every),
        ]
    )
    if args.task_distribution_yaml is not None:
        cmd.extend(["--task_distribution_yaml", str(args.task_distribution_yaml)])
    if args.episode_config_dir is not None:
        cmd.extend(["--episode_config_dir", str(args.episode_config_dir)])
    if args.target_reward_position_offset is not None:
        cmd.extend(["--target_reward_position_offset", *(str(v) for v in args.target_reward_position_offset)])
    if args.target_reward_body_position_offset is not None:
        cmd.extend(["--target_reward_body_position_offset", *(str(v) for v in args.target_reward_body_position_offset)])
    cmd.append("--freeze_act" if args.freeze_act else "--no-freeze_act")
    cmd.append("--debug_timing" if args.debug_timing else "--no-debug_timing")
    cmd.append("--save_step_images" if args.save_step_images else "--no-save_step_images")
    cmd.append(
        "--debug_visual_overlays"
        if bool(getattr(args, "debug_visual_overlays", True))
        else "--no-debug_visual_overlays"
    )
    cmd.extend(["--image_log_every", str(args.image_log_every)])
    cmd.extend(["--max_logged_image_steps", str(args.max_logged_image_steps)])
    cmd.append("--swap_rgb_channels" if bool(getattr(args, "swap_rgb_channels", False)) else "--no-swap_rgb_channels")
    cmd.append(
        "--force_camera_render_before_read"
        if bool(getattr(args, "force_camera_render_before_read", False))
        else "--no-force_camera_render_before_read"
    )
    cmd.append(
        "--disable_ppo_resnet_observation_terms"
        if bool(getattr(args, "disable_ppo_resnet_observation_terms", True))
        else "--no-disable_ppo_resnet_observation_terms"
    )
    cmd.append("--debug_diagnostics" if bool(getattr(args, "debug_diagnostics", False)) else "--no-debug_diagnostics")
    cmd.extend(["--diagnostics_every", str(getattr(args, "diagnostics_every", 100))])
    cmd.extend(["--debug_audit_steps", str(getattr(args, "debug_audit_steps", 0))])
    if getattr(args, "audit_act_only", False):
        cmd.append("--audit_act_only")
    if getattr(args, "audit_zero_adapter", False):
        cmd.append("--audit_zero_adapter")
    cmd.append(
        "--treat_time_limit_truncation_as_terminal"
        if bool(getattr(args, "treat_time_limit_truncation_as_terminal", False))
        else "--no-treat_time_limit_truncation_as_terminal"
    )
    cmd.extend(["--tcp_action_frame", str(getattr(args, "tcp_action_frame", "gripper_tcp"))])
    cmd.extend(["--debug_audit_axis_magnitude", str(getattr(args, "debug_audit_axis_magnitude", 0.0))])
    if getattr(args, "debug_audit_constant_action", None) is not None:
        cmd.extend(["--debug_audit_constant_action", *(str(v) for v in args.debug_audit_constant_action)])
    cmd.append("--enable_contact_sensor" if bool(getattr(args, "enable_contact_sensor", True)) else "--no-enable_contact_sensor")
    cmd.append("--fix_isaac_ik_xy_sign" if bool(getattr(args, "fix_isaac_ik_xy_sign", True)) else "--no-fix_isaac_ik_xy_sign")
    cmd.append(
        "--isaac_ik_xy_sign_by_target_card"
        if bool(getattr(args, "isaac_ik_xy_sign_by_target_card", False))
        else "--no-isaac_ik_xy_sign_by_target_card"
    )
    if args.disable_fabric:
        cmd.append("--disable_fabric")
    if args.headless:
        cmd.append("--headless")
    if args.device:
        cmd.extend(["--device", args.device])
    if args.rendering_mode:
        cmd.extend(["--rendering_mode", args.rendering_mode])
    if args.kit_args:
        cmd.extend(["--kit_args", args.kit_args])
    cmd.extend(args.extra_arg)
    env = os.environ.copy()
    env["AIC_ISAAC_DISABLE_CAMERAS"] = "0"
    env["AIC_ISAAC_POLICY_HZ"] = str(float(getattr(args, "policy_hz", 20.0)))
    env["AIC_ISAAC_RANDOMIZATION_PROFILE"] = args.randomization_profile
    env["AIC_ISAAC_INSERTION_DISTANCE_WEIGHT"] = str(args.insertion_distance_weight)
    env["AIC_ISAAC_INSERTION_CLOSE_WEIGHT"] = str(args.insertion_close_weight)
    env["AIC_ISAAC_INSERTION_PROGRESS_WEIGHT"] = str(insertion_progress_weight)
    env["AIC_ISAAC_INSERTION_ORIENTATION_WEIGHT"] = str(args.insertion_orientation_weight)
    env["AIC_ISAAC_INSERTION_ORIENTATION_GATED_WEIGHT"] = str(args.insertion_orientation_weight)
    env["AIC_ISAAC_INSERTION_REACHING_WEIGHT"] = str(args.insertion_reaching_weight)
    env["AIC_ISAAC_INSERTION_TERMINAL_WEIGHT"] = str(insertion_terminal_weight)
    env["AIC_ISAAC_INSERTION_LATERAL_WEIGHT"] = str(args.insertion_lateral_weight)
    env["AIC_ISAAC_INSERTION_AXIAL_PROGRESS_WEIGHT"] = str(getattr(args, "insertion_axial_progress_weight", 0.0))
    env["AIC_ISAAC_FORCE_DELTA_PENALTY_WEIGHT"] = str(args.force_delta_penalty_weight)
    if args.initial_arm_joint_pos:
        env["AIC_ISAAC_INITIAL_ARM_JOINT_POS"] = args.initial_arm_joint_pos
    if args.episode_config_dir is not None:
        env["AIC_ISAAC_EPISODE_CONFIG_DIR"] = str(args.episode_config_dir)
    return cmd, env


def validate_launch_inputs(args: argparse.Namespace) -> None:
    if not args.act_torchscript.exists():
        raise FileNotFoundError(f"ACT TorchScript checkpoint does not exist: {args.act_torchscript}")
    if bool(getattr(args, "act_only", False)):
        if args.checkpoint is not None and not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required unless --act-only is set")
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if int(getattr(args, "actor_update_start_steps", 0)) < 0:
        raise ValueError("--actor-update-start-steps must be non-negative")
    if int(getattr(args, "actor_update_end_steps", 0)) < 0:
        raise ValueError("--actor-update-end-steps must be non-negative")
    if float(getattr(args, "actor_q_weight", 1.0)) < 0.0:
        raise ValueError("--actor-q-weight must be non-negative")
    if float(getattr(args, "tcp_translation_action_clip", 0.0)) < 0.0:
        raise ValueError("--tcp-translation-action-clip must be non-negative")
    if float(getattr(args, "tcp_rotation_action_clip", 0.0)) < 0.0:
        raise ValueError("--tcp-rotation-action-clip must be non-negative")
    if float(getattr(args, "target_action_guide_lateral_switch_m", 0.002)) < 0.0:
        raise ValueError("--target-action-guide-lateral-switch-m must be non-negative")
    if float(getattr(args, "target_action_guide_rotation_step_size", 0.0)) < 0.0:
        raise ValueError("--target-action-guide-rotation-step-size must be non-negative")
    if float(getattr(args, "target_action_guide_axial_step_size", 0.0)) < 0.0:
        raise ValueError("--target-action-guide-axial-step-size must be non-negative")
    if float(getattr(args, "target_action_guide_axial_blend_lateral_m", 0.006)) < 0.0:
        raise ValueError("--target-action-guide-axial-blend-lateral-m must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.replay_capacity < args.batch_size:
        raise ValueError("--replay-capacity must be at least --batch-size")
    if args.isaac_action_scale <= 0.0:
        raise ValueError("--isaac-action-scale must be positive")
    if int(getattr(args, "n_action_steps", 4)) < 1:
        raise ValueError("--n-action-steps must be >= 1")
    if int(getattr(args, "act_only_action_horizon", 0)) > 0 and int(getattr(args, "n_action_steps", 4)) > int(
        args.act_only_action_horizon
    ):
        raise ValueError("--n-action-steps must be <= --act-only-action-horizon when the latter is set")
    if float(getattr(args, "policy_hz", 20.0)) <= 0.0:
        raise ValueError("--policy-hz must be positive")
    if float(getattr(args, "target_success_termination_threshold", 0.0)) < 0.0:
        raise ValueError("--target-success-termination-threshold must be non-negative")
    if args.target_success_axial_threshold is not None and float(args.target_success_axial_threshold) < 0.0:
        raise ValueError("--target-success-axial-threshold must be non-negative")
    if args.target_success_lateral_threshold is not None and float(args.target_success_lateral_threshold) < 0.0:
        raise ValueError("--target-success-lateral-threshold must be non-negative")
    if bool(getattr(args, "terminate_on_target_success", False)) and float(
        getattr(args, "target_success_termination_threshold", 0.0)
    ) <= 0.0:
        raise ValueError("--target-success-termination-threshold must be positive when success termination is enabled")
    if args.target_reward_position_offset is not None and len(args.target_reward_position_offset) != 3:
        raise ValueError("--target-reward-position-offset must contain exactly three values")
    if args.target_reward_body_position_offset is not None and len(args.target_reward_body_position_offset) != 3:
        raise ValueError("--target-reward-body-position-offset must contain exactly three values")
    if args.task_distribution_yaml is not None and not args.task_distribution_yaml.exists():
        raise FileNotFoundError(f"--task-distribution-yaml does not exist: {args.task_distribution_yaml}")
    if args.isaac_user_config_yaml is not None and not args.isaac_user_config_yaml.exists():
        raise FileNotFoundError(f"--isaac-user-config-yaml does not exist: {args.isaac_user_config_yaml}")
    if args.isaac_user_config_dir is not None and not args.isaac_user_config_dir.exists():
        raise FileNotFoundError(f"--isaac-user-config-dir does not exist: {args.isaac_user_config_dir}")
    if args.episode_config_dir is not None and not args.episode_config_dir.exists():
        raise FileNotFoundError(f"--episode-config-dir does not exist: {args.episode_config_dir}")
    if args.episode_config_count is not None and args.episode_config_count <= 0:
        raise ValueError("--episode-config-count must be positive")
    is_audit = int(getattr(args, "debug_audit_steps", 0)) > 0
    if not is_audit and args.warmup_steps > args.steps * args.num_envs:
        raise ValueError(
            "--warmup-steps is larger than the maximum transitions collected by "
            "--steps * --num-envs, so no online updates would run"
        )
    if getattr(args, "save_every_steps", 0) < 0:
        raise ValueError("--save-every-steps must be non-negative")
    if getattr(args, "save_latest_every_steps", 0) < 0:
        raise ValueError("--save-latest-every-steps must be non-negative")
    if getattr(args, "ram_watchdog_min_available_gb", 0.0) < 0.0:
        raise ValueError("--ram-watchdog-min-available-gb must be non-negative")


def prepare_user_episode_configs(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.isaac_user_config_yaml is None and args.isaac_user_config_dir is None:
        return None
    output_dir = args.episode_config_dir or (args.output_dir / "isaac_episode_configs")
    if args.isaac_user_config_dir is not None:
        input_yamls = [args.isaac_user_config_yaml] if args.isaac_user_config_yaml is not None else []
        summary = materialize_many_episode_configs(
            input_dir=args.isaac_user_config_dir,
            output_dir=output_dir,
            filenames=_split_filenames(args.isaac_user_config_filenames),
            input_yamls=input_yamls,
            max_gpus=args.max_gpus,
            episode_count_override=args.episode_config_count,
        )
    else:
        summary = materialize_episode_configs(
            args.isaac_user_config_yaml,
            output_dir,
            episode_count=args.episode_config_count,
        )
    args.episode_config_dir = Path(summary["episodes_dir"])
    if args.task_distribution_yaml is None:
        args.task_distribution_yaml = Path(summary["task_distribution_yaml"])
    return summary


def normalize_reward_sign_args(args: argparse.Namespace) -> None:
    """Keep CLI penalty magnitudes from accidentally becoming rewards."""
    lateral_weight = float(getattr(args, "insertion_lateral_weight", 0.0))
    if lateral_weight > 0.0:
        corrected = -lateral_weight
        print(
            "[AIC SERL][warning] --insertion-lateral-weight is a lateral-error "
            f"penalty; interpreting positive magnitude {lateral_weight:g} as {corrected:g}.",
            flush=True,
        )
        args.insertion_lateral_weight = corrected


def apply_reward_preset(args: argparse.Namespace) -> None:
    if getattr(args, "reward_preset", "default") == "default":
        return
    if args.reward_preset != "near_gate_corridor_v1":
        raise ValueError(f"Unsupported reward preset: {args.reward_preset}")
    args.insertion_distance_weight = 0.02
    args.insertion_close_weight = 0.05
    args.insertion_progress_weight = 0.0
    args.insertion_lateral_weight = -0.10
    args.insertion_lateral_error_scale = 0.006
    args.insertion_lateral_progress_weight = 0.25
    args.insertion_lateral_progress_scale = 0.001
    args.insertion_axial_progress_weight = 0.25
    args.insertion_axial_progress_scale = 0.001
    args.insertion_lateral_gate_sigma = 0.004
    args.insertion_corridor_weight = 0.50
    args.insertion_corridor_sigma = 0.0025
    args.insertion_bypass_penalty_scale = 2.0
    args.insertion_orientation_weight = 0.05
    args.insertion_orientation_std = 0.10
    args.insertion_orientation_gate_sigma = 0.010
    args.force_delta_penalty_weight = 0.05
    args.terminate_on_target_success = True


def main() -> int:
    args = parse_args()
    apply_reward_preset(args)
    normalize_reward_sign_args(args)
    user_config_summary = prepare_user_episode_configs(args)
    plan = build_plan(args, inspect_required=not args.dry_run)
    if user_config_summary is not None:
        plan["isaac_user_config_materialization"] = user_config_summary
    rendered = json.dumps(plan, indent=2, sort_keys=True)
    if args.dry_run:
        cmd, _ = build_command(args)
        print(rendered)
        print("COMMAND:", shlex.join(cmd))
        return 0
    validate_launch_inputs(args)
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
