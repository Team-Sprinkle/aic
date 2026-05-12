#!/usr/bin/env python3
"""Online ACT-adapter SERL/SAC-style training loop for AIC Isaac Lab."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
import copy
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="AIC-Task-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--act_torchscript", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="outputs/train/isaac_online_serl")
parser.add_argument("--run_name", default="isaac_online_serl")
parser.add_argument("--steps", type=int, default=64)
parser.add_argument("--updates", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--replay_capacity", type=int, default=10000)
parser.add_argument("--warmup_steps", type=int, default=0)
parser.add_argument("--max_wall_time_minutes", type=float, default=0.0)
parser.add_argument("--log_every", type=int, default=10)
parser.add_argument("--debug_timing", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--adapter_lr", type=float, default=1e-4)
parser.add_argument("--critic_lr", type=float, default=1e-4)
parser.add_argument("--act_lr", type=float, default=1e-5)
parser.add_argument("--freeze_act", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--adapter_penalty_weight", type=float, default=1e-3)
parser.add_argument("--act_preservation_weight", type=float, default=1e-2)
parser.add_argument("--adapter_delta_clip", type=float, default=0.05)
parser.add_argument(
    "--action_clip",
    type=float,
    default=0.0,
    help="Optional full-action clamp. Defaults disabled because it can distort the ACT base action.",
)
parser.add_argument(
    "--isaac_action_scale",
    type=float,
    default=1.0,
    help="Scale applied by Isaac's IK action term. ACT/SERL actions are already physical TCP deltas.",
)
parser.add_argument("--bc_weight", type=float, default=0.0)
parser.add_argument("--cql_weight", type=float, default=0.0)
parser.add_argument("--state_source", choices=["lerobot_compatible", "policy_prefix"], default="lerobot_compatible")
parser.add_argument("--task_family", choices=["sfp_to_nic", "sc_to_sc"], default="sfp_to_nic")
parser.add_argument("--target_port_index", type=int, default=0)
parser.add_argument("--target_card_index", type=int, default=0)
parser.add_argument("--target_card_valid", type=int, default=1)
parser.add_argument(
    "--task_distribution_yaml",
    type=str,
    default=None,
    help=(
        "Optional expert-generation-style YAML used to sample canonical 10D task vectors. "
        "The current Isaac scene still uses the configured randomization profile for physical "
        "part placement; this YAML controls task-family/card/port conditioning."
    ),
)
parser.add_argument(
    "--episode_config_dir",
    type=str,
    default=None,
    help="Directory containing generated Isaac per-episode YAML configs.",
)
parser.add_argument("--target_reward_body", default="sfp_tip_link")
parser.add_argument("--target_reward_distance_weight", type=float, default=0.5)
parser.add_argument("--target_reward_close_weight", type=float, default=0.3)
parser.add_argument("--target_reward_orientation_weight", type=float, default=0.0)
parser.add_argument("--target_reward_reaching_weight", type=float, default=1.0)
parser.add_argument("--target_reward_lateral_weight", type=float, default=0.0)
parser.add_argument("--force_delta_penalty_weight", type=float, default=0.0)
parser.add_argument("--force_delta_threshold", type=float, default=3.0)
parser.add_argument("--force_delta_reference", type=float, default=20.0)
parser.add_argument("--target_reward_distance_std", type=float, default=0.02)
parser.add_argument("--target_reward_close_sigma", type=float, default=0.01)
parser.add_argument("--target_reward_reaching_threshold", type=float, default=0.01)
parser.add_argument(
    "--target_reward_position_offset",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help=(
        "Target-local XYZ offset from the Isaac target rigid-object root to the insertion reward point. "
        "Defaults to the Gazebo SFP port entrance for sfp_to_nic/nic_card and the calibrated SC-port offset for sc_to_sc."
    ),
)
parser.add_argument(
    "--target_reward_body_position_offset",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help=(
        "Body-local XYZ offset from target_reward_body origin to the point measured against the target. "
        "Defaults to the SFP module tip for sfp_to_nic and body origin for sc_to_sc."
    ),
)
parser.add_argument("--disable_command_pose_rewards", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--gripper_joint_position", type=float, default=0.0035405)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--save_step_images", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--image_log_every", type=int, default=1)
parser.add_argument("--max_logged_image_steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
os.environ["AIC_ISAAC_TASK_FAMILY"] = args_cli.task_family
if args_cli.episode_config_dir:
    os.environ["AIC_ISAAC_EPISODE_CONFIG_DIR"] = args_cli.episode_config_dir
args_cli.enable_cameras = True
PROCESS_START_TIME = time.monotonic()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils import math as math_utils

import aic_task.tasks  # noqa: F401


from torch import nn
from torch.nn import functional as F

LEROBOT_AIC_PACKAGE_DIR = Path(__file__).resolve().parents[4] / "lerobot_robot_aic"
LEROBOT_AIC_MODULE_DIR = LEROBOT_AIC_PACKAGE_DIR / "lerobot_robot_aic"
if LEROBOT_AIC_MODULE_DIR.exists() and str(LEROBOT_AIC_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(LEROBOT_AIC_MODULE_DIR))

from contact_recovery_features import CONTACT_RECOVERY_FEATURE_DIM, ContactRecoveryFeatureComputer


SFP_PORT_LOCAL = {
    0: (0.01295, -0.031572, 0.00501),
    1: (-0.01025, -0.031572, 0.00501),
}
SFP_PORT_RPY = (4.69895, 0.0, 0.0)
SFP_PORT_ENTRANCE_LOCAL = (0.0, 0.0, -0.0458)
SFP_PORT_SEATED_TARGET_ROOT_LOCAL = {
    0: (0.01059, -0.07594, 0.01540),
    1: (-0.01261, -0.07594, 0.01540),
}
SFP_TIP_LOCAL = (0.0, -0.02365, 0.0)
SFP_TIP_RPY = (1.5708, 0.0, 0.0)
CONTROLLED_TCP_BODY = "gripper_tcp"


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> tuple[tuple[float, float, float], ...]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _matvec(mat: tuple[tuple[float, float, float], ...], vec: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(sum(mat[row][col] * vec[col] for col in range(3)) for row in range(3))


def _offset_from_port_frame(port_index: int, offset: tuple[float, float, float]) -> tuple[float, float, float]:
    port = SFP_PORT_LOCAL[int(port_index)]
    rotated = _matvec(_rpy_matrix(*SFP_PORT_RPY), offset)
    return tuple(port[i] + rotated[i] for i in range(3))


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        cy * cr * cp + sy * sr * sp,
        cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp,
        sy * cr * cp - cy * sr * sp,
    )


def _activation(name: str) -> nn.Module:
    normalized = name.strip().lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name!r}")


class FourierStateEncoding(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        indices: list[int] | tuple[int, ...],
        num_bands: int,
        max_freq: float,
        scale: float,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.indices = tuple(int(i) for i in indices)
        if any(i < 0 or i >= self.state_dim for i in self.indices):
            raise ValueError(f"State encoding indices must be within [0, {self.state_dim}); got {self.indices}")
        self.num_bands = int(num_bands)
        if self.num_bands < 1:
            raise ValueError("num_bands must be >= 1")
        self.max_freq = float(max_freq)
        self.scale = float(scale)
        freqs = torch.logspace(0.0, torch.log10(torch.tensor(self.max_freq)), self.num_bands)
        self.register_buffer("freqs", freqs.float(), persistent=False)
        self.output_dim = self.state_dim + len(self.indices) * self.num_bands * 2

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        selected = state[:, self.indices] * self.scale
        angles = selected.unsqueeze(-1) * self.freqs.to(device=state.device, dtype=state.dtype).view(1, 1, -1)
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(start_dim=1)
        return torch.cat([state, encoded], dim=-1)


def _make_state_encoder(
    *,
    state_dim: int,
    state_encoding: str,
    state_encoding_indices: list[int] | tuple[int, ...],
    state_encoding_num_bands: int,
    state_encoding_max_freq: float,
    state_encoding_scale: float,
) -> tuple[nn.Module, int]:
    if state_encoding == "none":
        return nn.Identity(), int(state_dim)
    if state_encoding == "fourier":
        encoder = FourierStateEncoding(
            state_dim=state_dim,
            indices=state_encoding_indices,
            num_bands=state_encoding_num_bands,
            max_freq=state_encoding_max_freq,
            scale=state_encoding_scale,
        )
        return encoder, int(encoder.output_dim)
    raise ValueError(f"Unsupported state_encoding: {state_encoding!r}")


CAMERA_KEYS = [
    "observation.images.center_camera",
    "observation.images.left_camera",
    "observation.images.right_camera",
]

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def _load_task_distribution(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    distribution_path = Path(path)
    if not distribution_path.exists():
        raise FileNotFoundError(f"task_distribution_yaml does not exist: {distribution_path}")
    data = yaml.safe_load(distribution_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("task_distribution_yaml must contain a mapping")
    return data


TASK_DISTRIBUTION = _load_task_distribution(args_cli.task_distribution_yaml)


def _load_episode_configs(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    root = Path(path)
    episodes_dir = root if root.name == "episodes" else root / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"episode_config_dir does not contain an episodes directory: {root}")
    episodes: list[dict[str, Any]] = []
    for yaml_path in sorted(episodes_dir.glob("episode_*.yaml")):
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Episode config must contain a mapping: {yaml_path}")
        episodes.append(data)
    if not episodes:
        raise ValueError(f"No episode_*.yaml files found in {episodes_dir}")
    return episodes


EPISODE_CONFIGS = _load_episode_configs(args_cli.episode_config_dir)


def _choices(value: Any, default: list[Any]) -> list[Any]:
    if value is None or value == "auto":
        return list(default)
    if isinstance(value, list):
        return value
    return [value]


def _sample_int_choice(value: Any, default: list[int], rng: random.Random) -> int:
    return int(rng.choice([int(v) for v in _choices(value, default)]))


def _port_index(value: Any) -> int:
    if isinstance(value, str):
        if value.startswith("sfp_port_"):
            return int(value.removeprefix("sfp_port_"))
        if value.startswith("sc_port_"):
            return int(value.removeprefix("sc_port_"))
    return int(value)


def _sample_task_context_from_distribution(rng: random.Random) -> tuple[str, int, int, int]:
    if EPISODE_CONFIGS:
        idx = int(getattr(_sample_task_context_from_distribution, "_calls", 0))
        setattr(_sample_task_context_from_distribution, "_calls", idx + 1)
        context = EPISODE_CONFIGS[idx % len(EPISODE_CONFIGS)].get("task_context") or {}
        return (
            str(context["task_family"]),
            int(context["target_port_index"]),
            int(context["target_card_index"]),
            int(context["target_card_valid"]),
        )
    cfg = TASK_DISTRIBUTION
    if not cfg:
        return (
            args_cli.task_family,
            int(args_cli.target_port_index),
            int(args_cli.target_card_index),
            int(args_cli.target_card_valid),
        )
    episodes = cfg.get("episodes")
    if isinstance(episodes, list) and episodes:
        idx = int(getattr(_sample_task_context_from_distribution, "_distribution_episode_calls", 0))
        setattr(_sample_task_context_from_distribution, "_distribution_episode_calls", idx + 1)
        context = episodes[idx % len(episodes)]
        return (
            str(context["task_family"]),
            int(context["target_port_index"]),
            int(context["target_card_index"]),
            int(context["target_card_valid"]),
        )
    scene = cfg.get("scene") or {}
    family = str(rng.choice(_choices(cfg.get("task_family"), [args_cli.task_family])))
    if family == "sfp_to_nic":
        nic_cfg = scene.get("nic_cards") or {}
        nic_count = _sample_int_choice(nic_cfg.get("count"), [1, 2, 3, 4, 5], rng)
        nic_count = max(1, min(5, nic_count))
        target_card_raw = nic_cfg.get("target_card", "auto")
        target_card = rng.randrange(nic_count) if target_card_raw == "auto" else _sample_int_choice(target_card_raw, list(range(nic_count)), rng)
        target_port = _port_index(rng.choice(_choices(nic_cfg.get("target_port"), [0, 1])))
        if target_port not in {0, 1}:
            raise ValueError(f"sfp_to_nic target_port must resolve to 0 or 1, got {target_port}")
        return family, target_port, target_card, 1
    if family == "sc_to_sc":
        sc_cfg = scene.get("sc_ports") or {}
        sc_count = _sample_int_choice(sc_cfg.get("count"), [1, 2], rng)
        sc_count = max(1, min(2, sc_count))
        target_port_raw = sc_cfg.get("target_port", "auto")
        target_port = rng.randrange(sc_count) if target_port_raw == "auto" else _port_index(rng.choice(_choices(target_port_raw, list(range(sc_count)))))
        if target_port not in {0, 1}:
            raise ValueError(f"sc_to_sc target_port must resolve to 0 or 1, got {target_port}")
        return family, target_port, -1, 0
    raise ValueError(f"Unsupported task_family in task_distribution_yaml: {family!r}")


def _current_episode_by_env(env) -> dict[int, dict[str, Any]]:
    return dict(getattr(env.unwrapped, "_aic_current_episode_by_env", {}) or {})


def _episode_context_tuple(episode: dict[str, Any]) -> tuple[str, int, int, int]:
    context = episode.get("task_context") or {}
    return (
        str(context["task_family"]),
        int(context["target_port_index"]),
        int(context["target_card_index"]),
        int(context["target_card_valid"]),
    )


def _task_vector_from_contexts(contexts: list[tuple[str, int, int, int]], *, device: torch.device) -> torch.Tensor:
    rows: list[list[float]] = []
    for family_name, port_index, card_index, card_valid in contexts:
        family = [1.0, 0.0] if family_name == "sfp_to_nic" else [0.0, 1.0]
        port = [1.0, 0.0] if port_index == 0 else [0.0, 1.0]
        card = [0.0] * 5
        if card_valid:
            card[card_index] = 1.0
        rows.append(family + port + card + [float(card_valid)])
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _canonical_task_vector(args: argparse.Namespace, *, device: torch.device, batch_size: int) -> torch.Tensor:
    if TASK_DISTRIBUTION is not None or EPISODE_CONFIGS:
        rng = random.Random(int(args.seed) + int(getattr(_canonical_task_vector, "_calls", 0)))
        setattr(_canonical_task_vector, "_calls", int(getattr(_canonical_task_vector, "_calls", 0)) + 1)
        rows: list[list[float]] = []
        for _ in range(batch_size):
            family_name, port_index, card_index, card_valid = _sample_task_context_from_distribution(rng)
            family = [1.0, 0.0] if family_name == "sfp_to_nic" else [0.0, 1.0]
            port = [1.0, 0.0] if port_index == 0 else [0.0, 1.0]
            card = [0.0] * 5
            if card_valid:
                card[card_index] = 1.0
            rows.append(family + port + card + [float(card_valid)])
        return torch.tensor(rows, dtype=torch.float32, device=device)

    if args.task_family == "sfp_to_nic":
        family = [1.0, 0.0]
        if args.target_card_valid != 1:
            raise ValueError("sfp_to_nic Isaac task context requires target_card_valid=1")
        if args.target_card_index < 0 or args.target_card_index >= 5:
            raise ValueError("sfp_to_nic target_card_index must be in 0..4")
    elif args.task_family == "sc_to_sc":
        family = [0.0, 1.0]
        if args.target_card_valid != 0 or args.target_card_index != -1:
            raise ValueError("sc_to_sc Isaac task context requires target_card_index=-1 and target_card_valid=0")
    else:
        raise ValueError(f"Unsupported task_family: {args.task_family}")
    if args.target_port_index not in {0, 1}:
        raise ValueError("target_port_index must be 0 or 1")
    port = [1.0, 0.0] if args.target_port_index == 0 else [0.0, 1.0]
    card = [0.0] * 5
    if args.target_card_valid:
        card[args.target_card_index] = 1.0
    vector = torch.tensor(
        family + port + card + [float(args.target_card_valid)],
        dtype=torch.float32,
        device=device,
    )
    return vector.unsqueeze(0).expand(batch_size, -1)


def _isaac_target_scene_name(args: argparse.Namespace) -> str:
    if args.task_family == "sfp_to_nic":
        return "nic_card"
    if args.task_family != "sc_to_sc":
        raise ValueError(f"Unsupported task_family for Isaac target reward: {args.task_family}")
    if args.target_port_index == 0:
        return "sc_port"
    if args.target_port_index == 1:
        return "sc_port_2"
    raise ValueError("target_port_index must be 0 or 1")


def _target_reward_position_offset(args: argparse.Namespace) -> tuple[float, float, float]:
    if args.target_reward_position_offset is not None:
        return tuple(float(v) for v in args.target_reward_position_offset)
    if args.task_family == "sfp_to_nic":
        return SFP_PORT_SEATED_TARGET_ROOT_LOCAL[int(args.target_port_index)]
    return (0.093, 0.140, 0.020)


def _target_reward_orientation_offset(args: argparse.Namespace) -> tuple[float, float, float, float] | None:
    if args.task_family == "sfp_to_nic":
        return _quat_from_rpy(*SFP_PORT_RPY)
    return None


def _target_reward_body_position_offset(args: argparse.Namespace) -> tuple[float, float, float]:
    if args.target_reward_body_position_offset is not None:
        return tuple(float(v) for v in args.target_reward_body_position_offset)
    if args.task_family == "sfp_to_nic":
        if args.target_reward_body == "sfp_tip_link":
            return (0.0, 0.0, 0.0)
        return SFP_TIP_LOCAL
    return (0.0, 0.0, 0.0)


def _target_reward_body_orientation_offset(args: argparse.Namespace) -> tuple[float, float, float, float] | None:
    if args.task_family == "sfp_to_nic":
        if args.target_reward_body == "sfp_tip_link":
            return _quat_from_rpy(0.0, math.pi, 0.0)
        return _quat_from_rpy(*SFP_TIP_RPY)
    return None


def _configure_task_geometry_rewards(env_cfg: Any, args: argparse.Namespace) -> dict[str, Any]:
    rewards = env_cfg.rewards
    target_scene_name = _isaac_target_scene_name(args)
    target_body = [str(args.target_reward_body)]
    target_position_offset = _target_reward_position_offset(args)
    body_position_offset = _target_reward_body_position_offset(args)
    target_orientation_offset = _target_reward_orientation_offset(args)
    body_orientation_offset = _target_reward_body_orientation_offset(args)
    for name in (
        "target_distance_tanh",
        "target_distance_exp",
        "target_orientation_tanh",
        "target_reaching_bonus",
        "target_lateral_error",
    ):
        term = getattr(rewards, name)
        term.params["target_cfg"].name = target_scene_name
        term.params["body_cfg"].body_names = target_body
        if "target_position_offset" in term.params:
            term.params["target_position_offset"] = target_position_offset
        if "body_position_offset" in term.params:
            term.params["body_position_offset"] = body_position_offset
        if "target_orientation_offset" in term.params:
            term.params["target_orientation_offset"] = target_orientation_offset
        if "body_orientation_offset" in term.params:
            term.params["body_orientation_offset"] = body_orientation_offset
    rewards.target_distance_tanh.params["std"] = float(args.target_reward_distance_std)
    rewards.target_distance_exp.params["sigma"] = float(args.target_reward_close_sigma)
    rewards.target_reaching_bonus.params["threshold"] = float(args.target_reward_reaching_threshold)

    rewards.target_distance_tanh.weight = float(args.target_reward_distance_weight)
    rewards.target_distance_exp.weight = float(args.target_reward_close_weight)
    rewards.target_orientation_tanh.weight = float(args.target_reward_orientation_weight)
    rewards.target_reaching_bonus.weight = float(args.target_reward_reaching_weight)
    rewards.target_lateral_error.weight = float(args.target_reward_lateral_weight)
    if hasattr(rewards, "force_delta_penalty"):
        rewards.force_delta_penalty.weight = float(args.force_delta_penalty_weight)
        rewards.force_delta_penalty.params["threshold"] = float(args.force_delta_threshold)
        rewards.force_delta_penalty.params["reference"] = float(args.force_delta_reference)

    if args.disable_command_pose_rewards:
        for name in (
            "end_effector_position_tracking",
            "end_effector_position_tracking_fine_grained",
            "end_effector_position_tracking_exp",
            "end_effector_orientation_tracking",
            "end_effector_orientation_tracking_fine_grained",
            "reaching_bonus",
        ):
            getattr(rewards, name).weight = 0.0

    return {
        "target_scene_name": target_scene_name,
        "target_body": target_body[0],
        "distance_weight": float(rewards.target_distance_tanh.weight),
        "close_weight": float(rewards.target_distance_exp.weight),
        "orientation_weight": float(rewards.target_orientation_tanh.weight),
        "reaching_weight": float(rewards.target_reaching_bonus.weight),
        "lateral_weight": float(rewards.target_lateral_error.weight),
        "force_delta_penalty_weight": float(rewards.force_delta_penalty.weight) if hasattr(rewards, "force_delta_penalty") else 0.0,
        "force_delta_threshold": float(args.force_delta_threshold),
        "force_delta_reference": float(args.force_delta_reference),
        "target_position_offset": [float(v) for v in target_position_offset],
        "body_position_offset": [float(v) for v in body_position_offset],
        "target_orientation_offset": None if target_orientation_offset is None else [float(v) for v in target_orientation_offset],
        "body_orientation_offset": None if body_orientation_offset is None else [float(v) for v in body_orientation_offset],
        "distance_std": float(args.target_reward_distance_std),
        "close_sigma": float(args.target_reward_close_sigma),
        "reaching_threshold": float(args.target_reward_reaching_threshold),
        "command_pose_rewards_disabled": bool(args.disable_command_pose_rewards),
    }


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data = deque(maxlen=capacity)

    def append(self, transition: dict[str, Any]) -> None:
        self.data.append({key: self._detach(value) for key, value in transition.items()})

    def _detach(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: self._detach(v) for k, v in value.items()}
        return value

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, Any]:
        indices = torch.randint(len(self.data), (batch_size,)).tolist()
        items = [self.data[i] for i in indices]
        return {
            "obs": self._stack_obs([item["obs"] for item in items], device),
            "next_obs": self._stack_obs([item["next_obs"] for item in items], device),
            "action": torch.stack([item["action"] for item in items]).to(device),
            "reward": torch.stack([item["reward"] for item in items]).to(device),
            "done": torch.stack([item["done"] for item in items]).to(device),
        }

    def _stack_obs(self, obs_items: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
        return {
            "state": torch.stack([item["state"] for item in obs_items]).to(device),
            "images": {
                key: torch.stack([item["images"][key] for item in obs_items]).to(device)
                for key in CAMERA_KEYS
            },
        }


def _policy_tensor(obs: Any) -> torch.Tensor:
    if isinstance(obs, dict):
        obs = obs.get("policy", obs)
    if not isinstance(obs, torch.Tensor):
        raise TypeError(f"Expected Isaac policy observation tensor, got {type(obs)}")
    return obs


def _camera_tensor(env, sensor_name: str, *, device: torch.device) -> torch.Tensor:
    sensor = env.unwrapped.scene.sensors[sensor_name]
    output = sensor.data.output
    if "rgb" not in output:
        raise RuntimeError(f"Camera sensor {sensor_name!r} does not expose rgb output keys: {sorted(output)}")
    image = output["rgb"].to(device)
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    else:
        image = image.float()
        if image.max() > 2.0:
            image = image / 255.0
    if image.ndim != 4:
        raise RuntimeError(f"Camera sensor {sensor_name!r} rgb output has unexpected shape: {tuple(image.shape)}")
    if image.shape[-1] in (3, 4):
        image = image[..., :3].permute(0, 3, 1, 2).contiguous()
    elif image.shape[1] != 3:
        raise RuntimeError(f"Camera sensor {sensor_name!r} rgb output has unexpected shape: {tuple(image.shape)}")
    return F.interpolate(image, size=(256, 288), mode="bilinear", align_corners=False)


def _raw_camera_images(env, *, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "observation.images.center_camera": _camera_tensor(env, "center_camera", device=device),
        "observation.images.left_camera": _camera_tensor(env, "left_camera", device=device),
        "observation.images.right_camera": _camera_tensor(env, "right_camera", device=device),
    }


def _to_act_obs(policy_obs: torch.Tensor, images: dict[str, torch.Tensor], *, state_dim: int) -> dict[str, Any]:
    state = policy_obs[:, :state_dim]
    return {"state": state, "images": images}


def _named_index(names: list[str], target: str) -> int:
    try:
        return names.index(target)
    except ValueError as exc:
        raise RuntimeError(f"Isaac robot does not expose {target!r}; available names: {names}") from exc


def _isaac_lerobot_state(env, args: argparse.Namespace, *, device: torch.device, state_dim: int) -> torch.Tensor:
    robot = env.unwrapped.scene["robot"]
    body_names = list(getattr(robot, "body_names", []))
    joint_names = list(getattr(robot, "joint_names", []))
    wrist_index = _named_index(body_names, CONTROLLED_TCP_BODY)
    joint_indices = [_named_index(joint_names, name) for name in ARM_JOINT_NAMES]
    data = robot.data
    batch_size = int(data.body_pos_w.shape[0])

    root_pos_w = data.root_pos_w
    root_quat_w = data.root_quat_w
    tcp_pos_w = data.body_pos_w[:, wrist_index]
    tcp_quat_w = data.body_quat_w[:, wrist_index]
    tcp_pos, tcp_quat = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, tcp_pos_w, tcp_quat_w)
    # Isaac Lab returns quaternions as wxyz; the LeRobot/Gazebo observation
    # contract stores orientation fields as xyzw.
    tcp_quat_xyzw = torch.cat([tcp_quat[:, 1:4], tcp_quat[:, 0:1]], dim=-1)
    tcp_lin_vel = getattr(data, "body_lin_vel_w", torch.zeros(batch_size, len(body_names), 3, device=device))[
        :, wrist_index
    ]
    tcp_ang_vel = getattr(data, "body_ang_vel_w", torch.zeros(batch_size, len(body_names), 3, device=device))[
        :, wrist_index
    ]
    tcp_lin_vel = math_utils.quat_apply_inverse(root_quat_w, tcp_lin_vel)
    tcp_ang_vel = math_utils.quat_apply_inverse(root_quat_w, tcp_ang_vel)
    tcp_error = torch.zeros(batch_size, 6, dtype=torch.float32, device=device)
    joint_pos = data.joint_pos[:, joint_indices]
    gripper = torch.full(
        (batch_size, 1),
        float(args.gripper_joint_position),
        dtype=torch.float32,
        device=device,
    )
    incoming_wrench = getattr(data, "body_incoming_wrench_w", None)
    if incoming_wrench is None:
        incoming_wrench = getattr(data, "body_incoming_wrench_b", None)
    if incoming_wrench is None:
        wrench = torch.zeros(batch_size, 6, dtype=torch.float32, device=device)
    else:
        wrench = incoming_wrench[:, wrist_index, :6].to(device=device, dtype=torch.float32)
    base_state = torch.cat(
        [tcp_pos, tcp_quat_xyzw, tcp_lin_vel, tcp_ang_vel, tcp_error, joint_pos, gripper, wrench],
        dim=-1,
    )
    if base_state.shape[1] != 32:
        raise RuntimeError(f"Expected Isaac LeRobot-compatible base state dim 32, got {base_state.shape[1]}")
    episode_by_env = _current_episode_by_env(env)
    if episode_by_env:
        contexts: list[tuple[str, int, int, int]] = []
        for env_id in range(batch_size):
            episode = episode_by_env.get(env_id)
            contexts.append(
                _episode_context_tuple(episode)
                if episode is not None
                else _sample_task_context_from_distribution(random.Random(int(args.seed) + env_id))
            )
        task_vector = _task_vector_from_contexts(contexts, device=device)
    else:
        task_vector = _canonical_task_vector(args, device=device, batch_size=batch_size)
    if state_dim in {32, 42}:
        state = torch.cat([base_state, task_vector], dim=-1)
    elif state_dim in {72, 82}:
        contact_features = _isaac_contact_recovery_features(base_state, env=env, device=device)
        state = torch.cat([base_state, contact_features, task_vector], dim=-1)
    else:
        raise RuntimeError(f"Unsupported Isaac LeRobot-compatible state dim {state_dim}")
    if state.shape[1] < state_dim:
        raise RuntimeError(f"Isaac LeRobot-compatible state has {state.shape[1]} dims, checkpoint expects {state_dim}")
    return state[:, :state_dim]


def _isaac_contact_recovery_features(base_state: torch.Tensor, *, env, device: torch.device) -> torch.Tensor:
    batch_size = int(base_state.shape[0])
    computers = getattr(_isaac_contact_recovery_features, "_computers", None)
    if computers is None or len(computers) != batch_size:
        computers = [ContactRecoveryFeatureComputer() for _ in range(batch_size)]
        setattr(_isaac_contact_recovery_features, "_computers", computers)
        setattr(_isaac_contact_recovery_features, "_step_count", 0)
    step_dt = float(getattr(env.unwrapped, "step_dt", 1.0 / 20.0))
    step_count = int(getattr(_isaac_contact_recovery_features, "_step_count", 0))
    time_sec = step_count * step_dt
    force = base_state[:, 26:29].detach().cpu().numpy()
    torque = base_state[:, 29:32].detach().cpu().numpy()
    pos = base_state[:, 0:3].detach().cpu().numpy()
    quat = base_state[:, 3:7].detach().cpu().numpy()
    features = [
        computers[idx].update(
            time_sec=time_sec,
            tcp_position_base=pos[idx],
            tcp_orientation_xyzw=quat[idx],
            force=force[idx],
            torque=torque[idx],
        )
        for idx in range(batch_size)
    ]
    setattr(_isaac_contact_recovery_features, "_step_count", step_count + 1)
    out = torch.as_tensor(np.stack(features, axis=0), dtype=torch.float32, device=device)
    if out.shape[1] != CONTACT_RECOVERY_FEATURE_DIM:
        raise RuntimeError(f"Expected {CONTACT_RECOVERY_FEATURE_DIM} contact features, got {out.shape[1]}")
    return out


def _force_delta_metrics(env, *, device: torch.device) -> dict[str, torch.Tensor]:
    robot = env.unwrapped.scene["robot"]
    data = robot.data
    body_names = list(getattr(robot, "body_names", []))
    wrist_index = _named_index(body_names, CONTROLLED_TCP_BODY)
    incoming_wrench = getattr(data, "body_incoming_wrench_w", None)
    if incoming_wrench is None:
        incoming_wrench = getattr(data, "body_incoming_wrench_b", None)
    if incoming_wrench is None:
        force = torch.zeros(env.unwrapped.num_envs, 3, device=device)
    else:
        force = incoming_wrench[:, wrist_index, :3].to(device=device, dtype=torch.float32)
    prev = getattr(_force_delta_metrics, "_previous_force", None)
    if prev is None or prev.shape != force.shape:
        prev = force.detach().clone()
    delta_norm = torch.norm(force - prev.to(device), dim=1)
    setattr(_force_delta_metrics, "_previous_force", force.detach().clone())
    normalized = ((delta_norm - float(args_cli.force_delta_threshold)) / max(float(args_cli.force_delta_reference), 1e-6)).clamp(min=0.0, max=1.0)
    return {
        "force_norm": torch.norm(force, dim=1),
        "force_delta_norm": delta_norm,
        "force_delta_penalty": -float(args_cli.force_delta_penalty_weight) * normalized,
    }


def _episode_metadata(env, env_index: int) -> dict[str, Any]:
    episode = _current_episode_by_env(env).get(env_index) or {}
    context = episode.get("task_context") or {}
    curriculum = episode.get("curriculum") or {}
    return {
        "episode_id": episode.get("episode_id"),
        "task_family": context.get("task_family"),
        "target_port_index": context.get("target_port_index"),
        "target_card_index": context.get("target_card_index"),
        "target_card_valid": context.get("target_card_valid"),
        "source_request": (episode.get("source_request") or {}).get("name"),
        "global_episode_index": curriculum.get("global_episode_index"),
        "gpu_id": curriculum.get("gpu_id"),
        "start_near_gate": bool((episode.get("scene") or {}).get("start_near_gate")),
    }


def _save_images(images: dict[str, torch.Tensor], *, run_dir: Path, step: int, max_steps: int, every: int) -> None:
    if max_steps >= 0 and step > max_steps:
        return
    if every <= 0 or step % every != 0:
        return
    from torchvision.utils import save_image

    image_dir = run_dir / "step_images" / f"step_{step:06d}"
    image_dir.mkdir(parents=True, exist_ok=True)
    for key, tensor in images.items():
        camera = key.rsplit(".", 1)[-1]
        for env_idx in range(tensor.shape[0]):
            save_image(tensor[env_idx].detach().cpu().clamp(0.0, 1.0), image_dir / f"env_{env_idx:04d}_{camera}.png")


def _tcp_delta_action_to_isaac_base_action(env, tcp_action: torch.Tensor) -> torch.Tensor:
    """Convert Gazebo/LeRobot gripper-tcp delta actions to Isaac IK root-frame deltas."""
    robot = env.unwrapped.scene["robot"]
    body_names = list(getattr(robot, "body_names", []))
    wrist_index = _named_index(body_names, CONTROLLED_TCP_BODY)
    data = robot.data
    _, tcp_quat_b = math_utils.subtract_frame_transforms(
        data.root_pos_w,
        data.root_quat_w,
        data.body_pos_w[:, wrist_index],
        data.body_quat_w[:, wrist_index],
    )
    delta_pos_b = math_utils.quat_apply(tcp_quat_b, tcp_action[:, :3])
    delta_rot_b = math_utils.quat_apply(tcp_quat_b, tcp_action[:, 3:6])
    return torch.cat([delta_pos_b, delta_rot_b], dim=-1)


def _act_obs_from_env(
    env,
    policy_obs: torch.Tensor,
    images: dict[str, torch.Tensor],
    args: argparse.Namespace,
    *,
    device: torch.device,
    state_dim: int,
) -> dict[str, Any]:
    if args.state_source == "policy_prefix":
        return _to_act_obs(policy_obs, images, state_dim=state_dim)
    return {"state": _isaac_lerobot_state(env, args, device=device, state_dim=state_dim), "images": images}


def _repeat_first_action(action: torch.Tensor, *, action_horizon: int, single_action_dim: int) -> torch.Tensor:
    first = action[:, :single_action_dim]
    return first.repeat(1, action_horizon)


def _timing_log(enabled: bool, label: str, start_time: float) -> None:
    if enabled:
        print(f"[AIC SERL][timing] {label}: {time.monotonic() - start_time:.3f}s", flush=True)


def _mlp(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    output_dim: int,
    *,
    layer_norm: bool = False,
    zero_final: bool = False,
    activation: str = "relu",
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(_activation(activation))
        dim = hidden_dim
    final = nn.Linear(dim, output_dim)
    if zero_final:
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


class IsaacACTAdapterActor(nn.Module):
    """TorchScript ACT base plus the offline-trained adapter."""

    def __init__(
        self,
        *,
        act_base: torch.jit.ScriptModule,
        act_torchscript_path: Path,
        state_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int,
        num_layers: int,
        adapter_scale: float,
        adapter_delta_clip: float | None,
        action_clip: float | None,
        adapter_activation: str = "relu",
        state_encoding: str = "none",
        state_encoding_indices: list[int] | tuple[int, ...] = (),
        state_encoding_num_bands: int = 4,
        state_encoding_max_freq: float = 8.0,
        state_encoding_scale: float = 1.0,
    ):
        super().__init__()
        self.act_base = act_base
        self.act_torchscript_path = Path(act_torchscript_path)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.adapter_scale = float(adapter_scale)
        self.adapter_delta_clip = None if adapter_delta_clip is None else float(adapter_delta_clip)
        self.action_clip = None if action_clip is None else float(action_clip)
        self.state_encoder, encoded_state_dim = _make_state_encoder(
            state_dim=self.state_dim,
            state_encoding=state_encoding,
            state_encoding_indices=state_encoding_indices,
            state_encoding_num_bands=state_encoding_num_bands,
            state_encoding_max_freq=state_encoding_max_freq,
            state_encoding_scale=state_encoding_scale,
        )
        self.adapter = _mlp(
            encoded_state_dim + action_dim,
            hidden_dim,
            num_layers,
            action_dim,
            activation=adapter_activation,
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -2.0))

    def action_components(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        output_device = obs["state"].device
        chunk = self.act_base(
            obs["state"].to("cpu"),
            obs["images"]["observation.images.center_camera"].to("cpu"),
            obs["images"]["observation.images.left_camera"].to("cpu"),
            obs["images"]["observation.images.right_camera"].to("cpu"),
        )
        base_action = chunk[:, : self.action_horizon, :].reshape(obs["state"].shape[0], -1).to(output_device)
        encoded_state = self.state_encoder(obs["state"])
        raw_delta_action = self.adapter(torch.cat([encoded_state, base_action], dim=-1))
        if self.adapter_delta_clip is not None and self.adapter_delta_clip > 0.0:
            delta_action = raw_delta_action.clamp(-self.adapter_delta_clip, self.adapter_delta_clip)
        else:
            delta_action = raw_delta_action
        unclipped_final_action = base_action + self.adapter_scale * delta_action
        if self.action_clip is not None and self.action_clip > 0.0:
            final_action = unclipped_final_action.clamp(-self.action_clip, self.action_clip)
        else:
            final_action = unclipped_final_action
        return {
            "base_action": base_action,
            "raw_delta_action": raw_delta_action,
            "delta_action": delta_action,
            "unclipped_final_action": unclipped_final_action,
            "final_action": final_action,
        }

    def mean_action(self, obs: dict[str, Any]) -> torch.Tensor:
        return self.action_components(obs)["final_action"]


class ImageStateEncoder(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        camera_keys: list[str],
        feature_dim: int = 256,
        image_encoder: str = "small_conv",
        per_camera_dim: int = 64,
        layer_norm: bool = False,
        activation: str = "relu",
        state_encoding: str = "none",
        state_encoding_indices: list[int] | tuple[int, ...] = (),
        state_encoding_num_bands: int = 4,
        state_encoding_max_freq: float = 8.0,
        state_encoding_scale: float = 1.0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.camera_keys = list(camera_keys)
        self.image_encoder_name = image_encoder
        self.activation = str(activation)
        self.state_encoding = str(state_encoding)
        self.state_encoding_indices = tuple(int(i) for i in state_encoding_indices)
        self.state_encoding_num_bands = int(state_encoding_num_bands)
        self.state_encoding_max_freq = float(state_encoding_max_freq)
        self.state_encoding_scale = float(state_encoding_scale)
        self.state_encoder, encoded_state_dim = _make_state_encoder(
            state_dim=self.state_dim,
            state_encoding=self.state_encoding,
            state_encoding_indices=self.state_encoding_indices,
            state_encoding_num_bands=self.state_encoding_num_bands,
            state_encoding_max_freq=self.state_encoding_max_freq,
            state_encoding_scale=self.state_encoding_scale,
        )
        if image_encoder == "small_conv":
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=4, padding=2),
                _activation(activation),
                nn.Conv2d(16, 32, kernel_size=3, stride=4, padding=1),
                _activation(activation),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(32, per_camera_dim),
                _activation(activation),
            )
        elif image_encoder in {"resnet18", "resnet18_imagenet"}:
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.IMAGENET1K_V1 if image_encoder == "resnet18_imagenet" else None
            backbone = resnet18(weights=weights)
            backbone.fc = nn.Identity()
            self.image_encoder = nn.Sequential(backbone, nn.Linear(512, per_camera_dim), _activation(activation))
        elif image_encoder in {"convnext_tiny", "convnext_tiny_imagenet"}:
            from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

            weights = (
                ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if image_encoder == "convnext_tiny_imagenet" else None
            )
            backbone = convnext_tiny(weights=weights)
            backbone.classifier = nn.Identity()
            self.image_encoder = nn.Sequential(backbone, nn.Flatten(), nn.Linear(768, per_camera_dim), _activation(activation))
        else:
            raise ValueError(f"Unsupported critic image encoder: {image_encoder!r}")
        proj_layers: list[nn.Module] = [nn.Linear(encoded_state_dim + per_camera_dim * len(self.camera_keys), feature_dim)]
        if layer_norm:
            proj_layers.append(nn.LayerNorm(feature_dim))
        proj_layers.append(_activation(activation))
        self.proj = nn.Sequential(*proj_layers)

    def forward(self, obs: dict[str, Any]) -> torch.Tensor:
        image_features = [self.image_encoder(obs["images"][key]) for key in self.camera_keys]
        return self.proj(torch.cat([self.state_encoder(obs["state"]), *image_features], dim=-1))


class VisionCritic(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        camera_keys: list[str],
        action_dim: int,
        feature_dim: int = 256,
        image_encoder: str = "small_conv",
        arch: str = "concat",
        hidden_dim: int = 256,
        num_layers: int = 2,
        per_camera_dim: int = 64,
        layer_norm: bool = False,
        activation: str = "relu",
        state_encoding: str = "none",
        state_encoding_indices: list[int] | tuple[int, ...] = (),
        state_encoding_num_bands: int = 4,
        state_encoding_max_freq: float = 8.0,
        state_encoding_scale: float = 1.0,
    ):
        super().__init__()
        self.arch = arch
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.per_camera_dim = int(per_camera_dim)
        self.layer_norm = bool(layer_norm)
        self.activation = str(activation)
        self.state_encoding = str(state_encoding)
        self.state_encoding_indices = tuple(int(i) for i in state_encoding_indices)
        self.state_encoding_num_bands = int(state_encoding_num_bands)
        self.state_encoding_max_freq = float(state_encoding_max_freq)
        self.state_encoding_scale = float(state_encoding_scale)
        self.encoder = ImageStateEncoder(
            state_dim=state_dim,
            camera_keys=camera_keys,
            feature_dim=feature_dim,
            image_encoder=image_encoder,
            per_camera_dim=per_camera_dim,
            layer_norm=layer_norm,
            activation=activation,
            state_encoding=state_encoding,
            state_encoding_indices=state_encoding_indices,
            state_encoding_num_bands=state_encoding_num_bands,
            state_encoding_max_freq=state_encoding_max_freq,
            state_encoding_scale=state_encoding_scale,
        )
        if arch == "concat":
            self.q = _mlp(
                feature_dim + action_dim,
                hidden_dim,
                num_layers,
                1,
                layer_norm=layer_norm,
                zero_final=False,
                activation=activation,
            )
        elif arch == "multiplicative":
            self.action_proj = nn.Sequential(nn.Linear(action_dim, feature_dim), _activation(activation))
            self.q = _mlp(
                feature_dim * 3,
                hidden_dim,
                num_layers,
                1,
                layer_norm=layer_norm,
                zero_final=False,
                activation=activation,
            )
        elif arch == "value_advantage":
            self.value = _mlp(
                feature_dim,
                hidden_dim,
                num_layers,
                1,
                layer_norm=layer_norm,
                zero_final=False,
                activation=activation,
            )
            self.advantage = _mlp(
                feature_dim + action_dim,
                hidden_dim,
                num_layers,
                1,
                layer_norm=layer_norm,
                zero_final=False,
                activation=activation,
            )
        else:
            raise ValueError(f"Unsupported critic architecture: {arch!r}")

    def forward(self, obs: dict[str, Any], action: torch.Tensor) -> torch.Tensor:
        obs_feature = self.encoder(obs)
        if self.arch == "concat":
            return self.q(torch.cat([obs_feature, action], dim=-1))
        if self.arch == "multiplicative":
            action_feature = self.action_proj(action)
            return self.q(torch.cat([obs_feature, action_feature, obs_feature * action_feature], dim=-1))
        if self.arch == "value_advantage":
            return self.value(obs_feature) + self.advantage(torch.cat([obs_feature, action], dim=-1))
        raise RuntimeError(f"Unsupported critic architecture at forward: {self.arch!r}")


class OnlineSERLTrainer:
    def __init__(
        self,
        *,
        actor: IsaacACTAdapterActor,
        critic1: VisionCritic,
        critic2: VisionCritic,
        gamma: float,
        tau: float,
        adapter_lr: float,
        critic_lr: float,
        adapter_penalty_weight: float,
        act_preservation_weight: float,
        device: torch.device,
    ):
        self.actor = actor.to(device)
        # TorchScript ACT exports are traced on CPU and include CPU-created latent tensors.
        # Keep the frozen ACT base on CPU while training the adapter/critics on Isaac's device.
        self.actor.act_base = torch.jit.load(str(self.actor.act_torchscript_path), map_location="cpu").eval()
        for param in self.actor.act_base.parameters():
            param.requires_grad = False
        self.critic1 = critic1.to(device)
        self.critic2 = critic2.to(device)
        self.target_critic1 = copy.deepcopy(self.critic1).to(device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.gamma = gamma
        self.tau = tau
        self.adapter_penalty_weight = adapter_penalty_weight
        self.act_preservation_weight = act_preservation_weight
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=adapter_lr)
        self.critic_opt = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr)

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        obs, next_obs = batch["obs"], batch["next_obs"]
        action, reward, done = batch["action"], batch["reward"], batch["done"]
        with torch.no_grad():
            next_action = self.actor.mean_action(next_obs)
            target_q = torch.minimum(self.target_critic1(next_obs, next_action), self.target_critic2(next_obs, next_action))
            td_target = reward + self.gamma * (1.0 - done) * target_q
        q1 = self.critic1(obs, action)
        q2 = self.critic2(obs, action)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        components = self.actor.action_components(obs)
        actor_action = components["final_action"]
        base_action = components["base_action"]
        delta_action = components["delta_action"]
        actor_q = torch.minimum(self.critic1(obs, actor_action), self.critic2(obs, actor_action))
        adapter_penalty = delta_action.square().mean()
        act_preservation_loss = F.mse_loss(actor_action, base_action)
        actor_loss = -actor_q.mean() + self.adapter_penalty_weight * adapter_penalty + self.act_preservation_weight * act_preservation_loss
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        self._soft_update()
        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "critic_loss": float(critic_loss.detach().cpu()),
            "q_mean": float(torch.minimum(q1, q2).mean().detach().cpu()),
            "adapter_delta_norm": float(delta_action.norm(dim=-1).mean().detach().cpu()),
            "raw_adapter_delta_norm": float(
                components.get("raw_delta_action", delta_action).norm(dim=-1).mean().detach().cpu()
            ),
            "adapter_penalty": float(adapter_penalty.detach().cpu()),
            "act_preservation_loss": float(act_preservation_loss.detach().cpu()),
            "final_minus_act_norm": float((actor_action - base_action).norm(dim=-1).mean().detach().cpu()),
            "unclipped_final_minus_act_norm": float(
                (components.get("unclipped_final_action", actor_action) - base_action)
                .norm(dim=-1)
                .mean()
                .detach()
                .cpu()
            ),
            "log_std_mean": float(self.actor.log_std.mean().detach().cpu()),
        }

    def _soft_update(self) -> None:
        for target, source in ((self.target_critic1, self.critic1), (self.target_critic2, self.critic2)):
            for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * source_param.data)


def _infer_adapter_shape(actor_state: dict[str, torch.Tensor]) -> tuple[int, int]:
    linear_weights = [(key, value) for key, value in actor_state.items() if key.startswith("adapter.") and key.endswith(".weight")]
    linear_weights.sort(key=lambda item: int(item[0].split(".")[1]))
    hidden_dim = int(linear_weights[0][1].shape[0])
    num_layers = len(linear_weights) - 1
    return hidden_dim, num_layers


def _load_adapter_actor(
    checkpoint: dict[str, Any],
    *,
    act_torchscript: Path,
    state_dim: int,
    action_dim: int,
    action_horizon: int,
    device: torch.device,
    adapter_delta_clip: float | None,
    action_clip: float | None,
) -> IsaacACTAdapterActor:
    actor_state = checkpoint["actor"]
    hidden_dim, num_layers = _infer_adapter_shape(actor_state)
    offline_cfg, _, warmstart = _checkpoint_training_context(checkpoint)
    adapter_scale = float(warmstart.get("adapter_scale", 1.0))
    act_base = torch.jit.load(str(act_torchscript), map_location="cpu").eval()
    for param in act_base.parameters():
        param.requires_grad = False
    actor = IsaacACTAdapterActor(
        act_base=act_base,
        act_torchscript_path=act_torchscript,
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        adapter_scale=adapter_scale,
        adapter_delta_clip=adapter_delta_clip,
        action_clip=action_clip,
        adapter_activation=str(offline_cfg.get("adapter_activation", "relu")),
        state_encoding=str(offline_cfg.get("state_encoding", "none")),
        state_encoding_indices=tuple(int(i) for i in offline_cfg.get("state_encoding_indices", ())),
        state_encoding_num_bands=int(offline_cfg.get("state_encoding_num_bands", 4)),
        state_encoding_max_freq=float(offline_cfg.get("state_encoding_max_freq", 8.0)),
        state_encoding_scale=float(offline_cfg.get("state_encoding_scale", 1.0)),
    ).to(device)
    actor.act_base = torch.jit.load(str(actor.act_torchscript_path), map_location="cpu").eval()
    for param in actor.act_base.parameters():
        param.requires_grad = False
    own_state = actor.state_dict()
    compatible = {key: value for key, value in actor_state.items() if key in own_state and tuple(value.shape) == tuple(own_state[key].shape)}
    own_state.update(compatible)
    actor.load_state_dict(own_state, strict=True)
    actor.act_base = torch.jit.load(str(actor.act_torchscript_path), map_location="cpu").eval()
    for param in actor.act_base.parameters():
        param.requires_grad = False
    return actor


def _save_checkpoint(path: Path, trainer: OnlineSERLTrainer, train_config: dict[str, Any], step: int) -> None:
    torch.save(
        {
            "actor": trainer.actor.state_dict(),
            "critic1": trainer.critic1.state_dict(),
            "critic2": trainer.critic2.state_dict(),
            "target_critic1": trainer.target_critic1.state_dict(),
            "target_critic2": trainer.target_critic2.state_dict(),
            "actor_optimizer": trainer.actor_opt.state_dict(),
            "critic_optimizer": trainer.critic_opt.state_dict(),
            "online_serl_config": train_config,
            "step": step,
        },
        path,
    )


def _checkpoint_training_context(checkpoint: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if "vision_offline_serl_config" in checkpoint:
        return (
            checkpoint["vision_offline_serl_config"],
            checkpoint.get("dataset_summary") or {},
            checkpoint.get("warmstart_report") or {},
        )
    online_cfg = checkpoint.get("online_serl_config") or {}
    source = online_cfg.get("checkpoint") or {}
    offline_cfg = source.get("vision_offline_serl_config")
    if offline_cfg is None:
        raise KeyError(
            "Checkpoint does not contain vision_offline_serl_config and does not look like "
            "an online SERL checkpoint with online_serl_config.checkpoint.vision_offline_serl_config."
        )
    return (
        offline_cfg,
        source.get("dataset_summary") or {},
        source.get("warmstart_report") or {},
    )


def main() -> None:
    torch.manual_seed(args_cli.seed)
    checkpoint_path = Path(args_cli.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    offline_cfg, dataset_summary, warmstart = _checkpoint_training_context(checkpoint)
    state_dim = int(offline_cfg["state_dim"])
    action_horizon = int(offline_cfg["action_horizon"])
    single_action_dim = int(offline_cfg["action_dim"] // action_horizon)

    device = torch.device(args_cli.device)
    actor = _load_adapter_actor(
        checkpoint,
        act_torchscript=Path(args_cli.act_torchscript),
        state_dim=state_dim,
        action_dim=int(offline_cfg["action_dim"]),
        action_horizon=action_horizon,
        device=device,
        adapter_delta_clip=args_cli.adapter_delta_clip,
        action_clip=args_cli.action_clip,
    )
    critic_image_encoder = str(offline_cfg.get("critic_image_encoder", "small_conv"))
    critic_arch = str(offline_cfg.get("critic_arch", "concat"))
    critic_feature_dim = int(offline_cfg.get("critic_feature_dim", 256))
    critic_hidden_dim = int(offline_cfg.get("critic_hidden_dim", 256))
    critic_num_layers = int(offline_cfg.get("critic_num_layers", 2))
    critic_per_camera_dim = int(offline_cfg.get("critic_per_camera_dim", 64))
    critic_layer_norm = bool(offline_cfg.get("critic_layer_norm", False))
    critic_activation = str(offline_cfg.get("critic_activation", "relu"))
    state_encoding = str(offline_cfg.get("state_encoding", "none"))
    state_encoding_indices = tuple(int(i) for i in offline_cfg.get("state_encoding_indices", ()))
    state_encoding_num_bands = int(offline_cfg.get("state_encoding_num_bands", 4))
    state_encoding_max_freq = float(offline_cfg.get("state_encoding_max_freq", 8.0))
    state_encoding_scale = float(offline_cfg.get("state_encoding_scale", 1.0))
    critic1 = VisionCritic(
        state_dim=state_dim,
        camera_keys=CAMERA_KEYS,
        action_dim=int(offline_cfg["action_dim"]),
        feature_dim=critic_feature_dim,
        image_encoder=critic_image_encoder,
        arch=critic_arch,
        hidden_dim=critic_hidden_dim,
        num_layers=critic_num_layers,
        per_camera_dim=critic_per_camera_dim,
        layer_norm=critic_layer_norm,
        activation=critic_activation,
        state_encoding=state_encoding,
        state_encoding_indices=state_encoding_indices,
        state_encoding_num_bands=state_encoding_num_bands,
        state_encoding_max_freq=state_encoding_max_freq,
        state_encoding_scale=state_encoding_scale,
    )
    critic2 = VisionCritic(
        state_dim=state_dim,
        camera_keys=CAMERA_KEYS,
        action_dim=int(offline_cfg["action_dim"]),
        feature_dim=critic_feature_dim,
        image_encoder=critic_image_encoder,
        arch=critic_arch,
        hidden_dim=critic_hidden_dim,
        num_layers=critic_num_layers,
        per_camera_dim=critic_per_camera_dim,
        layer_norm=critic_layer_norm,
        activation=critic_activation,
        state_encoding=state_encoding,
        state_encoding_indices=state_encoding_indices,
        state_encoding_num_bands=state_encoding_num_bands,
        state_encoding_max_freq=state_encoding_max_freq,
        state_encoding_scale=state_encoding_scale,
    )
    critic1.load_state_dict(checkpoint["critic1"], strict=True)
    critic2.load_state_dict(checkpoint["critic2"], strict=True)
    trainer = OnlineSERLTrainer(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        gamma=args_cli.gamma,
        tau=args_cli.tau,
        adapter_lr=args_cli.adapter_lr,
        critic_lr=args_cli.critic_lr,
        adapter_penalty_weight=args_cli.adapter_penalty_weight,
        act_preservation_weight=args_cli.act_preservation_weight,
        device=device,
    )

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    # The SERL actor consumes raw camera tensors directly from the sensors. Avoid
    # computing the PPO-specific ResNet feature observation terms, but keep the
    # camera sensors enabled and rendered.
    env_cfg.observations.policy.center_rgb = None
    env_cfg.observations.policy.left_rgb = None
    env_cfg.observations.policy.right_rgb = None
    env_cfg.actions.arm_action.scale = args_cli.isaac_action_scale
    task_geometry_reward_config = _configure_task_geometry_rewards(env_cfg, args_cli)
    print("[AIC SERL] Creating Isaac env", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[AIC SERL] Isaac env created", flush=True)
    replay = ReplayBuffer(args_cli.replay_capacity)

    run_dir = Path(args_cli.output_dir) / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args_cli.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    train_config = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint": {
            "vision_offline_serl_config": offline_cfg,
            "dataset_summary": dataset_summary,
            "warmstart_report": warmstart,
        },
        "isaac_adapter": {
            "state_source": args_cli.state_source,
            "state_contract": (
                "LeRobot-compatible base32 plus optional contact40 plus optional canonical task10"
                if args_cli.state_source == "lerobot_compatible"
                else f"first_{state_dim}_dims"
            ),
            "image_source": "raw_isaac_camera_sensor_rgb_resized_to_3x256x288",
            "act_torchscript": str(args_cli.act_torchscript),
            "action_executed": "first_action_from_flattened_chunk",
            "action_frame_contract": (
                "actor outputs LeRobot/Gazebo gripper-tcp deltas; Isaac execution converts them "
                "to robot-base deltas for DifferentialInverseKinematicsAction"
            ),
            "isaac_action_scale": args_cli.isaac_action_scale,
            "adapter_delta_clip": args_cli.adapter_delta_clip,
            "action_clip": args_cli.action_clip,
            "ppo_resnet_observation_terms_disabled": True,
            "camera_sensors_enabled": True,
            "critic_image_encoder": critic_image_encoder,
            "task_context": {
                "task_family": args_cli.task_family,
                "target_port_index": args_cli.target_port_index,
                "target_card_index": args_cli.target_card_index,
                "target_card_valid": args_cli.target_card_valid,
            },
            "task_distribution_yaml": args_cli.task_distribution_yaml,
            "task_distribution": TASK_DISTRIBUTION,
            "episode_config_dir": args_cli.episode_config_dir,
            "episode_config_count": len(EPISODE_CONFIGS),
            "first_episode_config": EPISODE_CONFIGS[0] if EPISODE_CONFIGS else None,
            "task_distribution_scope": (
                "When episode configs are provided, the Isaac reset event stores the current child YAML "
                "per env and the policy task vector/replay metadata read that same assignment. Without "
                "episode configs, the task vector falls back to task_distribution_yaml or CLI task fields."
            ),
            "gripper_joint_position": args_cli.gripper_joint_position,
            "task_geometry_reward": task_geometry_reward_config,
            "image_logging": {
                "save_step_images": args_cli.save_step_images,
                "image_log_every": args_cli.image_log_every,
                "max_logged_image_steps": args_cli.max_logged_image_steps,
                "note": "Replay stores image tensors in memory for training; PNG logging is for audit/debug artifacts.",
            },
        },
        "args": vars(args_cli),
    }
    (run_dir / "train_config.json").write_text(json.dumps(train_config, indent=2, sort_keys=True), encoding="utf-8")

    print("[AIC SERL] Resetting Isaac env", flush=True)
    obs, _ = env.reset()
    if hasattr(_isaac_contact_recovery_features, "_computers"):
        delattr(_isaac_contact_recovery_features, "_computers")
    setattr(_isaac_contact_recovery_features, "_step_count", 0)
    print("[AIC SERL] Isaac env reset complete", flush=True)
    policy_obs = _policy_tensor(obs).to(device)
    print(f"[AIC SERL] Policy obs shape: {tuple(policy_obs.shape)}", flush=True)
    current_images = _raw_camera_images(env, device=device)
    if args_cli.save_step_images:
        _save_images(
            current_images,
            run_dir=run_dir,
            step=0,
            max_steps=args_cli.max_logged_image_steps,
            every=max(args_cli.image_log_every, 1),
        )
    print("[AIC SERL] Initial raw camera read complete", flush=True)
    updates_done = 0
    last_metrics: dict[str, float] = {}
    stop_reason = "max_steps"
    for step in range(1, args_cli.steps + 1):
        step_start_time = time.monotonic()
        if args_cli.max_wall_time_minutes > 0.0:
            elapsed_minutes = (time.monotonic() - PROCESS_START_TIME) / 60.0
            if elapsed_minutes >= args_cli.max_wall_time_minutes:
                stop_reason = "max_wall_time"
                print(
                    f"[AIC SERL] Wall-time limit reached at step {step - 1}: "
                    f"{elapsed_minutes:.2f} min",
                    flush=True,
                )
                break
        t0 = time.monotonic()
        act_obs = _act_obs_from_env(
            env,
            policy_obs,
            current_images,
            args_cli,
            device=device,
            state_dim=state_dim,
        )
        _timing_log(args_cli.debug_timing and step == 1, "build_act_obs", t0)
        t0 = time.monotonic()
        with torch.no_grad():
            action_chunk = trainer.actor.mean_action(act_obs)
        _timing_log(args_cli.debug_timing and step == 1, "actor_forward", t0)
        policy_tcp_action = action_chunk[:, :single_action_dim]
        t0 = time.monotonic()
        env_action = _tcp_delta_action_to_isaac_base_action(env, policy_tcp_action)
        _timing_log(args_cli.debug_timing and step == 1, "tcp_to_isaac_action", t0)
        t0 = time.monotonic()
        next_obs, reward, terminated, truncated, _ = env.step(env_action)
        force_metrics = _force_delta_metrics(env, device=device)
        _timing_log(args_cli.debug_timing and step == 1, "env_step", t0)
        next_policy_obs = _policy_tensor(next_obs).to(device)
        t0 = time.monotonic()
        next_images = _raw_camera_images(env, device=device)
        if args_cli.save_step_images:
            _save_images(
                next_images,
                run_dir=run_dir,
                step=step,
                max_steps=args_cli.max_logged_image_steps,
                every=args_cli.image_log_every,
            )
        _timing_log(args_cli.debug_timing and step == 1, "next_camera_read", t0)
        t0 = time.monotonic()
        next_act_obs = _act_obs_from_env(
            env,
            next_policy_obs,
            next_images,
            args_cli,
            device=device,
            state_dim=state_dim,
        )
        _timing_log(args_cli.debug_timing and step == 1, "build_next_act_obs", t0)
        done = torch.logical_or(terminated, truncated).float().reshape(-1, 1).to(device)
        reward = reward.reshape(-1, 1).to(device)
        action_for_critic = _repeat_first_action(
            policy_tcp_action,
            action_horizon=action_horizon,
            single_action_dim=single_action_dim,
        )
        for env_index in range(policy_obs.shape[0]):
            replay.append(
                {
                    "obs": {"state": act_obs["state"][env_index], "images": {k: v[env_index] for k, v in act_obs["images"].items()}},
                    "next_obs": {
                        "state": next_act_obs["state"][env_index],
                        "images": {k: v[env_index] for k, v in next_act_obs["images"].items()},
                    },
                    "action": action_for_critic[env_index],
                    "reward": reward[env_index],
                    "done": done[env_index],
                    "metadata": _episode_metadata(env, env_index),
                }
            )
        policy_obs = next_policy_obs
        current_images = next_images

        if (
            len(replay) >= args_cli.batch_size
            and len(replay) >= args_cli.warmup_steps
            and updates_done < args_cli.updates
        ):
            t0 = time.monotonic()
            batch = replay.sample(args_cli.batch_size, device)
            _timing_log(args_cli.debug_timing and step == 1, "sample_replay", t0)
            t0 = time.monotonic()
            last_metrics = trainer.train_step(batch)
            _timing_log(args_cli.debug_timing and step == 1, "train_step", t0)
            updates_done += 1

        row = {
            "step": step,
            "updates_done": updates_done,
            "replay_size": len(replay),
            "reward_mean": float(reward.mean().detach().cpu()),
            "force_norm_mean": float(force_metrics["force_norm"].mean().detach().cpu()),
            "force_delta_norm_mean": float(force_metrics["force_delta_norm"].mean().detach().cpu()),
            "force_delta_penalty_mean": float(force_metrics["force_delta_penalty"].mean().detach().cpu()),
            "episodes": [_episode_metadata(env, idx) for idx in range(policy_obs.shape[0])],
            **last_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        if args_cli.log_every > 0 and (step == 1 or step % args_cli.log_every == 0):
            print(
                f"[AIC SERL] step={step} updates={updates_done} replay={len(replay)} "
                f"reward={row['reward_mean']:.6f} step_wall={time.monotonic() - step_start_time:.3f}s",
                flush=True,
            )
        if updates_done >= args_cli.updates:
            stop_reason = "target_updates"
            break

    train_config["result"] = {
        "stop_reason": stop_reason,
        "steps_completed": step if stop_reason != "max_wall_time" else max(step - 1, 0),
        "updates_done": updates_done,
        "elapsed_minutes": (time.monotonic() - PROCESS_START_TIME) / 60.0,
    }
    (run_dir / "train_config.json").write_text(json.dumps(train_config, indent=2, sort_keys=True), encoding="utf-8")
    _save_checkpoint(run_dir / "checkpoint_latest.pt", trainer, train_config, train_config["result"]["steps_completed"])
    print(f"Wrote online SERL checkpoint: {run_dir / 'checkpoint_latest.pt'}")
    print(f"Wrote metrics: {metrics_path}")
    env.close()


if __name__ == "__main__":
    status = 0
    try:
        main()
    except BaseException:
        status = 1
        traceback.print_exc()
    finally:
        simulation_app.close()
    raise SystemExit(status)
