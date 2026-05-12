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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="AIC-Task-v0")
    parser.add_argument("--num-envs", type=int, default=4)
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--act-torchscript", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train/isaac_online_serl"))
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
    parser.add_argument("--insertion-distance-weight", type=float, default=0.5)
    parser.add_argument("--insertion-close-weight", type=float, default=0.3)
    parser.add_argument("--insertion-orientation-weight", type=float, default=0.0)
    parser.add_argument("--insertion-reaching-weight", type=float, default=1.0)
    parser.add_argument("--insertion-lateral-weight", type=float, default=0.0)
    parser.add_argument("--force-delta-penalty-weight", type=float, default=0.0)
    parser.add_argument("--force-delta-threshold", type=float, default=3.0)
    parser.add_argument("--force-delta-reference", type=float, default=20.0)
    parser.add_argument("--target-reward-body", default="sfp_tip_link")
    parser.add_argument("--target-reward-distance-std", type=float, default=0.02)
    parser.add_argument("--target-reward-close-sigma", type=float, default=0.01)
    parser.add_argument("--target-reward-reaching-threshold", type=float, default=0.01)
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
    parser.add_argument("--image-log-every", type=int, default=1)
    parser.add_argument("--max-logged-image-steps", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[])
    return parser.parse_args(argv)


def inspect_checkpoint(path: Path, *, required: bool) -> dict[str, Any]:
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
    replay = ReplayBufferConfig(
        capacity=args.replay_capacity,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
    )
    checkpoint = inspect_checkpoint(args.checkpoint, required=inspect_required)
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
        "warmup_steps": args.warmup_steps,
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
        "isaac_action_scale": args.isaac_action_scale,
        "randomization_profile": args.randomization_profile,
        "insertion_distance_weight": args.insertion_distance_weight,
        "insertion_close_weight": args.insertion_close_weight,
        "insertion_orientation_weight": args.insertion_orientation_weight,
        "insertion_reaching_weight": args.insertion_reaching_weight,
        "insertion_lateral_weight": args.insertion_lateral_weight,
        "force_delta_penalty_weight": args.force_delta_penalty_weight,
        "force_delta_threshold": args.force_delta_threshold,
        "force_delta_reference": args.force_delta_reference,
        "target_reward_body": args.target_reward_body,
        "target_reward_distance_std": args.target_reward_distance_std,
        "target_reward_close_sigma": args.target_reward_close_sigma,
        "target_reward_reaching_threshold": args.target_reward_reaching_threshold,
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
        "--warmup_steps",
        str(args.warmup_steps),
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
        "--isaac_action_scale",
        str(args.isaac_action_scale),
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
        "--target_reward_orientation_weight",
        str(args.insertion_orientation_weight),
        "--target_reward_reaching_weight",
        str(args.insertion_reaching_weight),
        "--target_reward_lateral_weight",
        str(args.insertion_lateral_weight),
        "--force_delta_penalty_weight",
        str(args.force_delta_penalty_weight),
        "--force_delta_threshold",
        str(args.force_delta_threshold),
        "--force_delta_reference",
        str(args.force_delta_reference),
        "--target_reward_body",
        args.target_reward_body,
        "--target_reward_distance_std",
        str(args.target_reward_distance_std),
        "--target_reward_close_sigma",
        str(args.target_reward_close_sigma),
        "--target_reward_reaching_threshold",
        str(args.target_reward_reaching_threshold),
    ]
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
    cmd.extend(["--image_log_every", str(args.image_log_every)])
    cmd.extend(["--max_logged_image_steps", str(args.max_logged_image_steps)])
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
    env["AIC_ISAAC_RANDOMIZATION_PROFILE"] = args.randomization_profile
    env["AIC_ISAAC_INSERTION_DISTANCE_WEIGHT"] = str(args.insertion_distance_weight)
    env["AIC_ISAAC_INSERTION_CLOSE_WEIGHT"] = str(args.insertion_close_weight)
    env["AIC_ISAAC_INSERTION_ORIENTATION_WEIGHT"] = str(args.insertion_orientation_weight)
    env["AIC_ISAAC_INSERTION_REACHING_WEIGHT"] = str(args.insertion_reaching_weight)
    env["AIC_ISAAC_INSERTION_LATERAL_WEIGHT"] = str(args.insertion_lateral_weight)
    env["AIC_ISAAC_FORCE_DELTA_PENALTY_WEIGHT"] = str(args.force_delta_penalty_weight)
    if args.initial_arm_joint_pos:
        env["AIC_ISAAC_INITIAL_ARM_JOINT_POS"] = args.initial_arm_joint_pos
    if args.episode_config_dir is not None:
        env["AIC_ISAAC_EPISODE_CONFIG_DIR"] = str(args.episode_config_dir)
    return cmd, env


def validate_launch_inputs(args: argparse.Namespace) -> None:
    if not args.act_torchscript.exists():
        raise FileNotFoundError(f"ACT TorchScript checkpoint does not exist: {args.act_torchscript}")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.replay_capacity < args.batch_size:
        raise ValueError("--replay-capacity must be at least --batch-size")
    if args.isaac_action_scale <= 0.0:
        raise ValueError("--isaac-action-scale must be positive")
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
    if args.warmup_steps > args.steps * args.num_envs:
        raise ValueError(
            "--warmup-steps is larger than the maximum transitions collected by "
            "--steps * --num-envs, so no online updates would run"
        )


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


def main() -> int:
    args = parse_args()
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
