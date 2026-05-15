#!/usr/bin/env python3
"""Online ACT-adapter SERL/SAC-style training loop for AIC Isaac Lab."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
import traceback
import copy
import random
import zlib
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="AIC-Task-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument(
    "--act_only",
    action="store_true",
    help="Initialize from ACT TorchScript only with a zero adapter and fresh critics.",
)
parser.add_argument("--act_only_state_dim", type=int, default=82)
parser.add_argument(
    "--act_only_action_horizon",
    type=int,
    default=0,
    help="ACT output chunk size for ACT-only online SERL. 0 reads chunk_size from TorchScript metadata.",
)
parser.add_argument("--act_only_single_action_dim", type=int, default=6)
parser.add_argument(
    "--n_action_steps",
    type=int,
    default=4,
    help="Number of predicted chunk actions to execute before the next actor inference.",
)
parser.add_argument("--act_only_adapter_hidden_dim", type=int, default=256)
parser.add_argument("--act_only_adapter_num_layers", type=int, default=2)
parser.add_argument("--act_only_adapter_activation", choices=["relu", "gelu", "tanh"], default="gelu")
parser.add_argument(
    "--act_only_actor_mode",
    choices=["act_adapter", "act_direct"],
    default="act_adapter",
    help=(
        "act_adapter learns a clipped residual added to ACT. act_direct initializes "
        "from ACT but trains the full TCP action override, with only final TCP caps applied."
    ),
)
parser.add_argument("--act_only_state_encoding", choices=["none", "fourier"], default="fourier")
parser.add_argument("--act_only_state_encoding_indices", type=int, nargs="*", default=[0, 1, 2, 13, 14, 15])
parser.add_argument("--act_only_state_encoding_num_bands", type=int, default=4)
parser.add_argument("--act_only_state_encoding_max_freq", type=float, default=8.0)
parser.add_argument("--act_only_state_encoding_scale", type=float, default=10.0)
parser.add_argument("--act_torchscript", type=str, required=True)
parser.add_argument(
    "--act_torchscript_device",
    choices=["auto", "cpu", "cuda"],
    default="auto",
    help="Device for frozen ACT TorchScript inference. Auto uses CUDA only for CUDA-exported TorchScript.",
)
parser.add_argument("--output_dir", type=str, default="outputs/train/isaac_online_serl")
parser.add_argument("--run_name", default="isaac_online_serl")
parser.add_argument("--steps", type=int, default=64)
parser.add_argument("--updates", type=int, default=8)
parser.add_argument(
    "--update_every_steps",
    type=int,
    default=1,
    help="Run one gradient update every N environment steps after warmup.",
)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--replay_capacity", type=int, default=10000)
parser.add_argument("--warmup_steps", type=int, default=0)
parser.add_argument(
    "--actor_update_start_steps",
    type=int,
    default=0,
    help="Do not update the adapter actor until this many environment steps have been collected.",
)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=0.0,
    help="Optional Isaac episode timeout in seconds. 0 keeps the task default.",
)
parser.add_argument("--max_wall_time_minutes", type=float, default=0.0)
parser.add_argument("--log_every", type=int, default=10)
parser.add_argument(
    "--save_every_steps",
    type=int,
    default=0,
    help="Save periodic online SERL checkpoints every N environment steps. 0 disables periodic saves.",
)
parser.add_argument(
    "--save_latest_every_steps",
    type=int,
    default=0,
    help="Overwrite checkpoint_latest.pt every N environment steps. 0 only writes final latest.",
)
parser.add_argument(
    "--ram_watchdog_min_available_gb",
    type=float,
    default=0.0,
    help="If >0, save and stop when MemAvailable drops below this many GiB.",
)
parser.add_argument("--debug_timing", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--adapter_lr", type=float, default=1e-4)
parser.add_argument("--critic_lr", type=float, default=1e-4)
parser.add_argument("--act_lr", type=float, default=1e-5)
parser.add_argument(
    "--actor_q_weight",
    type=float,
    default=1.0,
    help="Scale the actor's -Q term. Lower values let guide/BC losses dominate early near-gate debugging.",
)
parser.add_argument(
    "--actor_update_end_steps",
    type=int,
    default=0,
    help="If >0, freeze actor updates after this environment step while continuing critic updates and rollouts.",
)
parser.add_argument("--target_action_guide_weight", type=float, default=0.0)
parser.add_argument("--target_action_guide_step_size", type=float, default=0.001)
parser.add_argument(
    "--target_action_guide_axial_step_size",
    type=float,
    default=0.0,
    help=(
        "Optional insertion-axis guide step size. If <=0, uses --target_action_guide_step_size. "
        "This lets near-gate runs center laterally with a small step, then push harder along the port axis."
    ),
)
parser.add_argument(
    "--target_action_guide_lateral_switch_m",
    type=float,
    default=0.002,
    help="Use pure axial guide motion at or below this lateral error.",
)
parser.add_argument(
    "--target_action_guide_axial_blend_lateral_m",
    type=float,
    default=0.006,
    help="Start blending axial guide motion once lateral error is below this distance.",
)
parser.add_argument("--target_action_guide_collect_blend", type=float, default=0.0)
parser.add_argument("--target_action_guide_collect_steps", type=int, default=0)
parser.add_argument(
    "--target_action_guide_collect_decay",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "If enabled with --target_action_guide_collect_steps > 0, linearly decay the "
        "guide collection blend to zero instead of stopping abruptly."
    ),
)
parser.add_argument(
    "--target_action_guide_prefix_decay",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "If enabled, the guide target for later actions in the executed chunk decays linearly. "
        "This prevents a near-gate policy from repeating the same corrective move too long."
    ),
)
parser.add_argument("--freeze_act", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--adapter_penalty_weight", type=float, default=1e-3)
parser.add_argument("--act_preservation_weight", type=float, default=1e-2)
parser.add_argument("--adapter_delta_clip", type=float, default=0.05)
parser.add_argument(
    "--tcp_translation_action_clip",
    type=float,
    default=0.0,
    help=(
        "Optional per-step TCP translation norm cap in meters, applied to every 3D "
        "translation inside the predicted action chunk after ACT+adapter. 0 disables it."
    ),
)
parser.add_argument(
    "--tcp_rotation_action_clip",
    type=float,
    default=0.0,
    help=(
        "Optional per-step TCP rotation-vector norm cap in radians, applied to every 3D "
        "rotation inside the predicted action chunk after ACT+adapter. 0 disables it."
    ),
)
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
parser.add_argument(
    "--act_normalized_state_clip",
    type=float,
    default=0.0,
    help=(
        "If >0, clamp the ACT-normalized state before TorchScript inference. "
        "This is useful for Isaac runtime outliers while leaving raw replay state unchanged."
    ),
)
parser.add_argument("--bc_weight", type=float, default=0.0)
parser.add_argument(
    "--expert_dataset_root",
    type=str,
    default=None,
    help="Optional clean LeRobot dataset root used for a small online BC regularizer.",
)
parser.add_argument("--expert_bc_weight", type=float, default=None)
parser.add_argument("--expert_bc_max_samples", type=int, default=8192)
parser.add_argument("--expert_bc_neighbor_chunk", type=int, default=8192)
parser.add_argument(
    "--expert_bc_every",
    type=int,
    default=4,
    help="Compute expert BC actor regularization every N actor updates. Use 1 for every update.",
)
parser.add_argument(
    "--expert_bc_state_indices",
    type=int,
    nargs="*",
    default=[0, 1, 2, 13, 14, 15, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81],
    help=(
        "State dimensions used for nearest expert lookup. Defaults to TCP position, "
        "TCP error position, and the 10D task vector."
    ),
)
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
parser.add_argument("--target_reward_progress_weight", type=float, default=0.25)
parser.add_argument("--target_reward_progress_scale", type=float, default=0.003)
parser.add_argument("--target_reward_distance_weight", type=float, default=0.25)
parser.add_argument("--target_reward_close_weight", type=float, default=0.35)
parser.add_argument("--target_reward_orientation_weight", type=float, default=0.10)
parser.add_argument("--target_reward_orientation_std", type=float, default=0.03)
parser.add_argument("--target_reward_orientation_gate_sigma", type=float, default=0.012)
parser.add_argument("--target_reward_reaching_weight", type=float, default=0.0)
parser.add_argument("--target_reward_terminal_weight", type=float, default=1.0)
parser.add_argument("--target_reward_lateral_weight", type=float, default=-0.05)
parser.add_argument("--target_reward_lateral_gate_sigma", type=float, default=0.012)
parser.add_argument("--target_reward_lateral_error_scale", type=float, default=0.006)
parser.add_argument("--target_reward_motion_projection_weight", type=float, default=0.0)
parser.add_argument("--target_reward_motion_projection_scale", type=float, default=0.001)
parser.add_argument("--target_reward_lateral_progress_weight", type=float, default=0.0)
parser.add_argument("--target_reward_lateral_progress_scale", type=float, default=0.001)
parser.add_argument("--target_reward_axial_progress_weight", type=float, default=0.0)
parser.add_argument("--target_reward_axial_progress_scale", type=float, default=0.001)
parser.add_argument("--target_reward_insertion_corridor_weight", type=float, default=0.0)
parser.add_argument("--target_reward_insertion_corridor_sigma", type=float, default=0.0025)
parser.add_argument("--target_reward_insertion_bypass_penalty_scale", type=float, default=1.0)
parser.add_argument("--target_reward_insertion_axis", type=int, choices=[0, 1, 2], default=0)
parser.add_argument("--force_delta_penalty_weight", type=float, default=0.3)
parser.add_argument("--force_delta_threshold", type=float, default=10.0)
parser.add_argument("--force_delta_reference", type=float, default=20.0)
parser.add_argument("--force_delta_saturation", type=float, default=30.0)
parser.add_argument("--force_delta_knee_penalty_fraction", type=float, default=0.1)
parser.add_argument(
    "--isaac_force_observation_clip_n",
    type=float,
    default=35.0,
    help=(
        "Clamp Isaac contact-proxy force observation magnitude before filling Gazebo-compatible "
        "state force slots. This prevents raw contact sensor spikes from dominating ACT inputs."
    ),
)
parser.add_argument("--target_reward_distance_std", type=float, default=0.02)
parser.add_argument("--target_reward_close_sigma", type=float, default=0.006)
parser.add_argument("--target_reward_reaching_threshold", type=float, default=0.01)
parser.add_argument("--terminate_on_target_success", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument(
    "--target_success_termination_threshold",
    type=float,
    default=0.0035,
    help=(
        "Strict distance threshold used only for episode termination when --terminate_on_target_success is enabled. "
        "Keep this tighter than the reward reaching threshold so near-gate resets do not count as completed insertions."
    ),
)
parser.add_argument(
    "--policy_hz",
    type=float,
    default=20.0,
    help="Isaac policy/control rate in Hz. Defaults to 20 Hz to match Gazebo expert datasets.",
)
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
parser.add_argument("--debug_diagnostics", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--diagnostics_every", type=int, default=100)
parser.add_argument("--debug_audit_steps", type=int, default=0)
parser.add_argument("--audit_act_only", action="store_true", default=False)
parser.add_argument("--audit_zero_adapter", action="store_true", default=False)
parser.add_argument(
    "--treat_time_limit_truncation_as_terminal",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "If true, store terminated OR truncated as replay done. Defaults false so TD bootstraps "
        "through time-limit truncation."
    ),
)
parser.add_argument(
    "--tcp_action_frame",
    choices=["gripper_tcp", "wrist_3_link", "root"],
    default="gripper_tcp",
    help="Frame convention for ACT/SERL TCP delta actions before passing them to Isaac IK.",
)
parser.add_argument(
    "--debug_audit_axis_magnitude",
    type=float,
    default=0.0,
    help="If positive in audit mode, bypass actor output and execute +/- x/y/z pure translation actions.",
)
parser.add_argument(
    "--debug_audit_constant_action",
    type=float,
    nargs=6,
    default=None,
    metavar=("DX", "DY", "DZ", "DRX", "DRY", "DRZ"),
    help="If set in audit mode, bypass actor output and execute this constant 6D TCP action every step.",
)
parser.add_argument(
    "--fix_isaac_ik_xy_sign",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Flip Isaac IK root-frame x/y translation commands to match realized TCP motion direction.",
)
parser.add_argument(
    "--enable_contact_sensor",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Enable Isaac ContactSensor diagnostics and Gazebo-compatible force observation fallback. "
        "Keep enabled for 72D/82D contact-feature checkpoints."
    ),
)
parser.add_argument(
    "--swap_rgb_channels",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Reverse Isaac camera RGB channel order before ACT/SERL inference and debug image logging.",
)
parser.add_argument(
    "--force_camera_render_before_read",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Force an Isaac render before reading camera tensors. Useful only if camera freshness diagnostics show stale frames.",
)
parser.add_argument(
    "--disable_ppo_resnet_observation_terms",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Disable policy observation ResNet image terms. This is faster, but camera freshness diagnostics "
        "should be checked because some Isaac sensor paths only update when observation terms are active."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if float(args_cli.target_reward_lateral_weight) > 0.0:
    corrected_lateral_weight = -float(args_cli.target_reward_lateral_weight)
    print(
        "[AIC SERL][warning] --target_reward_lateral_weight penalizes lateral "
        f"error; interpreting positive magnitude {args_cli.target_reward_lateral_weight:g} "
        f"as {corrected_lateral_weight:g}.",
        flush=True,
    )
    args_cli.target_reward_lateral_weight = corrected_lateral_weight
os.environ["AIC_ISAAC_TASK_FAMILY"] = args_cli.task_family
if args_cli.episode_config_dir:
    os.environ["AIC_ISAAC_EPISODE_CONFIG_DIR"] = args_cli.episode_config_dir
os.environ["AIC_ISAAC_POLICY_HZ"] = str(float(args_cli.policy_hz))
if args_cli.enable_contact_sensor:
    os.environ["AIC_ISAAC_ENABLE_CONTACT_SENSOR"] = "1"
args_cli.enable_cameras = True
PROCESS_START_TIME = time.monotonic()
STOP_REQUESTED: str | None = None


def _request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = f"signal_{signum}"
    print(f"[AIC SERL] Stop requested by signal {signum}; will save checkpoint after current step.", flush=True)


signal.signal(signal.SIGTERM, _request_stop)
signal.signal(signal.SIGINT, _request_stop)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import yaml
from safetensors.torch import load_file

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils import math as math_utils

import aic_task.tasks  # noqa: F401


from torch import nn
from torch.nn import functional as F

LEROBOT_AIC_PACKAGE_DIR = next(
    (
        parent / "lerobot_robot_aic"
        for parent in Path(__file__).resolve().parents
        if (parent / "lerobot_robot_aic" / "lerobot_robot_aic" / "__init__.py").exists()
    ),
    Path(__file__).resolve().parents[4] / "lerobot_robot_aic",
)
LEROBOT_AIC_MODULE_DIR = LEROBOT_AIC_PACKAGE_DIR / "lerobot_robot_aic"
if LEROBOT_AIC_PACKAGE_DIR.exists() and str(LEROBOT_AIC_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(LEROBOT_AIC_PACKAGE_DIR))
if LEROBOT_AIC_MODULE_DIR.exists() and str(LEROBOT_AIC_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(LEROBOT_AIC_MODULE_DIR))

from contact_recovery_features import (
    CONTACT_RECOVERY_FEATURE_DIM,
    CONTACT_RECOVERY_FEATURE_NAMES,
    ContactRecoveryFeatureComputer,
)


def _stack_vector_column(series: pd.Series, key: str) -> np.ndarray:
    values = [np.asarray(value, dtype=np.float32).reshape(-1) for value in series]
    if not values:
        raise ValueError(f"Column {key!r} is empty")
    dim = int(values[0].shape[0])
    if any(int(value.shape[0]) != dim for value in values):
        raise ValueError(f"Column {key!r} has inconsistent vector sizes")
    return np.stack(values, axis=0).astype(np.float32)


def _action_chunks(action: np.ndarray, episode: np.ndarray, horizon: int) -> np.ndarray:
    horizon = int(horizon)
    if horizon <= 1:
        return action.astype(np.float32)
    chunks = np.empty((action.shape[0], action.shape[1] * horizon), dtype=np.float32)
    for idx in range(action.shape[0]):
        episode_id = episode[idx]
        values = []
        last_valid = action[idx]
        for offset in range(horizon):
            src = idx + offset
            if src < action.shape[0] and episode[src] == episode_id:
                last_valid = action[src]
            values.append(last_valid)
        chunks[idx] = np.concatenate(values, axis=0)
    return chunks


def _load_lerobot_expert_arrays(
    dataset_root: Path,
    action_horizon: int,
    *,
    max_samples: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data_dir = dataset_root / "data"
    data_files = sorted(data_dir.rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files under {data_dir}")
    required = {"observation.state", "action", "episode_index", "frame_index"}
    df = pd.concat([pd.read_parquet(path, columns=sorted(required)) for path in data_files], ignore_index=True)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Expert dataset missing columns: {sorted(missing)}")
    sort_keys = ["episode_index", "frame_index"]
    if "index" in df.columns:
        sort_keys.append("index")
    df = df.sort_values(sort_keys).reset_index(drop=True)
    full_count = int(len(df))
    episode = df["episode_index"].to_numpy(dtype=np.int64)
    terminal_repeated_slots = 0
    terminal_total_slots = 0
    if max_samples > 0 and full_count > int(max_samples):
        row_indices = np.linspace(0, full_count - 1, num=int(max_samples), dtype=np.int64)
        obs = _stack_vector_column(df["observation.state"].iloc[row_indices], "observation.state")
        action_series = df["action"]
        first_action = np.asarray(action_series.iloc[int(row_indices[0])], dtype=np.float32).reshape(-1)
        action = np.empty((row_indices.shape[0], first_action.shape[0] * int(action_horizon)), dtype=np.float32)
        for out_idx, row_idx in enumerate(row_indices):
            values = []
            last_valid = np.asarray(action_series.iloc[int(row_idx)], dtype=np.float32).reshape(-1)
            for offset in range(int(action_horizon)):
                src = int(row_idx) + offset
                terminal_total_slots += 1
                if src < full_count and episode[src] == episode[int(row_idx)]:
                    last_valid = np.asarray(action_series.iloc[src], dtype=np.float32).reshape(-1)
                else:
                    terminal_repeated_slots += 1
                values.append(last_valid)
            action[out_idx] = np.concatenate(values, axis=0)
        single_action_dim = int(first_action.shape[0])
    else:
        obs = _stack_vector_column(df["observation.state"], "observation.state")
        single_action = _stack_vector_column(df["action"], "action")
        action = _action_chunks(single_action, episode, int(action_horizon))
        single_action_dim = int(single_action.shape[1])
        for row_idx in range(full_count):
            for offset in range(int(action_horizon)):
                terminal_total_slots += 1
                src = row_idx + offset
                if src >= full_count or episode[src] != episode[row_idx]:
                    terminal_repeated_slots += 1
    schema = {
        "dataset_root": str(dataset_root),
        "num_frames": full_count,
        "sampled_frames": int(obs.shape[0]),
        "num_episodes": int(df["episode_index"].nunique()),
        "obs_dim": int(obs.shape[1]),
        "single_action_dim": single_action_dim,
        "action_horizon": int(action_horizon),
        "action_dim": int(action.shape[1]),
        "terminal_repeated_action_slots": int(terminal_repeated_slots),
        "terminal_total_action_slots": int(terminal_total_slots),
        "terminal_repeated_action_slot_fraction": float(terminal_repeated_slots / max(terminal_total_slots, 1)),
    }
    return obs, action, schema


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
FORCE_WRENCH_BODY = "wrist_3_link"
FORCE_CONTACT_PROXY_BODIES = ("sfp_tip_link", "sfp_module_link", "gripper_tcp", "wrist_3_link")


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


def _torchscript_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"ACT TorchScript metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _resolve_act_checkpoint_dir(path: Path) -> Path:
    metadata = _torchscript_metadata(path)
    checkpoint_dir = Path(str(metadata.get("checkpoint_dir", "")))
    candidates: list[Path]
    if checkpoint_dir.is_absolute():
        candidates = [checkpoint_dir]
        host_prefix = "/data1/chmin/yj/ws_aic/src/aic"
        container_prefix = "/workspace/isaaclab/aic"
        checkpoint_str = str(checkpoint_dir)
        if checkpoint_str.startswith(host_prefix):
            candidates.append(Path(container_prefix + checkpoint_str[len(host_prefix) :]))
    else:
        candidates = [Path.cwd() / checkpoint_dir, Path.cwd() / "aic" / checkpoint_dir, path.parent / checkpoint_dir]
    for candidate in candidates:
        if (candidate / "policy_preprocessor_step_3_normalizer_processor.safetensors").exists():
            return candidate
    raise FileNotFoundError(
        "Could not resolve ACT checkpoint normalizer directory from "
        f"{path.with_suffix('.json')}: tried {', '.join(str(c) for c in candidates)}"
    )


class ACTRuntimeNormalizer(nn.Module):
    """LeRobot ACT runtime scaling around the exported TorchScript model."""

    def __init__(self, act_torchscript_path: Path, *, state_dim: int, action_dim: int):
        super().__init__()
        checkpoint_dir = _resolve_act_checkpoint_dir(act_torchscript_path)
        stats = load_file(str(checkpoint_dir / "policy_preprocessor_step_3_normalizer_processor.safetensors"))

        def stat(key: str, shape: tuple[int, ...]) -> torch.Tensor:
            if key not in stats:
                raise KeyError(f"ACT normalizer stats are missing {key!r}")
            tensor = stats[key].float().view(*shape)
            if key.endswith(".std"):
                tensor = torch.where(torch.abs(tensor) < 1.0e-8, torch.ones_like(tensor), tensor)
            return tensor

        state_mean = stat("observation.state.mean", (1, -1))[:, :state_dim]
        state_std = stat("observation.state.std", (1, -1))[:, :state_dim]
        if state_dim in (42, 82):
            state_mean[:, -10:] = 0.0
            state_std[:, -10:] = 1.0

        self.register_buffer("state_mean", state_mean, persistent=False)
        self.register_buffer("state_std", state_std, persistent=False)
        self.register_buffer("action_mean", stat("action.mean", (1, -1))[:, :action_dim], persistent=False)
        self.register_buffer("action_std", stat("action.std", (1, -1))[:, :action_dim], persistent=False)
        for key in CAMERA_KEYS:
            safe_key = key.replace(".", "__")
            self.register_buffer(f"{safe_key}_mean", stat(f"{key}.mean", (1, 3, 1, 1)), persistent=False)
            self.register_buffer(f"{safe_key}_std", stat(f"{key}.std", (1, 3, 1, 1)), persistent=False)

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_mean.to(device=state.device, dtype=state.dtype)) / self.state_std.to(
            device=state.device, dtype=state.dtype
        )

    def normalize_image(self, key: str, image: torch.Tensor) -> torch.Tensor:
        safe_key = key.replace(".", "__")
        mean = getattr(self, f"{safe_key}_mean").to(device=image.device, dtype=image.dtype)
        std = getattr(self, f"{safe_key}_std").to(device=image.device, dtype=image.dtype)
        return (image - mean) / std

    def unnormalize_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        return normalized_action * self.action_std.to(
            device=normalized_action.device, dtype=normalized_action.dtype
        ) + self.action_mean.to(device=normalized_action.device, dtype=normalized_action.dtype)


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
        "target_distance_progress",
        "target_orientation_tanh",
        "target_orientation_gated_exp",
        "target_reaching_bonus",
        "target_success_once_bonus",
        "target_lateral_error",
        "target_motion_projection",
        "target_lateral_progress",
        "target_axial_progress",
        "target_insertion_corridor",
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
    rewards.target_distance_progress.params["scale"] = float(args.target_reward_progress_scale)
    for name in ("target_distance_tanh", "target_distance_exp", "target_distance_progress"):
        term = getattr(rewards, name)
        term.params["insertion_axis"] = int(args.target_reward_insertion_axis)
        term.params["lateral_gate_sigma"] = float(args.target_reward_lateral_gate_sigma)
    rewards.target_lateral_error.params["insertion_axis"] = int(args.target_reward_insertion_axis)
    rewards.target_lateral_error.params["scale"] = float(args.target_reward_lateral_error_scale)
    rewards.target_motion_projection.params["scale"] = float(args.target_reward_motion_projection_scale)
    rewards.target_motion_projection.params["insertion_axis"] = int(args.target_reward_insertion_axis)
    rewards.target_motion_projection.params["lateral_gate_sigma"] = float(args.target_reward_lateral_gate_sigma)
    rewards.target_lateral_progress.params["scale"] = float(args.target_reward_lateral_progress_scale)
    rewards.target_lateral_progress.params["insertion_axis"] = int(args.target_reward_insertion_axis)
    rewards.target_axial_progress.params["scale"] = float(args.target_reward_axial_progress_scale)
    rewards.target_axial_progress.params["insertion_axis"] = int(args.target_reward_insertion_axis)
    rewards.target_axial_progress.params["lateral_gate_sigma"] = float(args.target_reward_lateral_gate_sigma)
    rewards.target_insertion_corridor.params["insertion_axis"] = int(args.target_reward_insertion_axis)
    rewards.target_insertion_corridor.params["lateral_gate_sigma"] = float(
        args.target_reward_insertion_corridor_sigma
    )
    rewards.target_insertion_corridor.params["bypass_penalty_scale"] = float(
        args.target_reward_insertion_bypass_penalty_scale
    )
    rewards.target_reaching_bonus.params["threshold"] = float(args.target_reward_reaching_threshold)
    rewards.target_success_once_bonus.params["threshold"] = float(args.target_reward_reaching_threshold)
    rewards.target_success_once_bonus.params["insertion_axis"] = int(args.target_reward_insertion_axis)
    rewards.target_success_once_bonus.params["axial_threshold"] = float(args.target_success_termination_threshold)
    rewards.target_success_once_bonus.params["lateral_threshold"] = float(args.target_success_termination_threshold)
    rewards.target_success_once_bonus.params["target_orientation_offset"] = target_orientation_offset
    rewards.target_orientation_gated_exp.params["std"] = float(args.target_reward_orientation_std)
    rewards.target_orientation_gated_exp.params["gate_sigma"] = float(args.target_reward_orientation_gate_sigma)

    env_step_dt = float(getattr(env_cfg, "sim").dt) * float(getattr(env_cfg, "decimation"))
    reward_weight_multiplier = 1.0 / max(env_step_dt, 1.0e-9)
    rewards.target_distance_tanh.weight = float(args.target_reward_distance_weight) * reward_weight_multiplier
    rewards.target_distance_exp.weight = float(args.target_reward_close_weight) * reward_weight_multiplier
    rewards.target_distance_progress.weight = float(args.target_reward_progress_weight) * reward_weight_multiplier
    rewards.target_orientation_tanh.weight = 0.0
    rewards.target_orientation_gated_exp.weight = float(args.target_reward_orientation_weight) * reward_weight_multiplier
    rewards.target_reaching_bonus.weight = float(args.target_reward_reaching_weight) * reward_weight_multiplier
    rewards.target_success_once_bonus.weight = float(args.target_reward_terminal_weight) * reward_weight_multiplier
    rewards.target_lateral_error.weight = float(args.target_reward_lateral_weight) * reward_weight_multiplier
    rewards.target_motion_projection.weight = float(args.target_reward_motion_projection_weight) * reward_weight_multiplier
    rewards.target_lateral_progress.weight = float(args.target_reward_lateral_progress_weight) * reward_weight_multiplier
    rewards.target_axial_progress.weight = float(args.target_reward_axial_progress_weight) * reward_weight_multiplier
    rewards.target_insertion_corridor.weight = (
        float(args.target_reward_insertion_corridor_weight) * reward_weight_multiplier
    )
    if hasattr(rewards, "force_delta_penalty"):
        rewards.force_delta_penalty.weight = float(args.force_delta_penalty_weight) * reward_weight_multiplier
        rewards.force_delta_penalty.params["threshold"] = float(args.force_delta_threshold)
        rewards.force_delta_penalty.params["reference"] = float(args.force_delta_reference)
        rewards.force_delta_penalty.params["saturation"] = float(args.force_delta_saturation)
        rewards.force_delta_penalty.params["knee_penalty_fraction"] = float(args.force_delta_knee_penalty_fraction)
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is not None and hasattr(terminations, "target_success"):
        target_success = terminations.target_success
        target_success.params["target_cfg"].name = target_scene_name
        target_success.params["body_cfg"].name = "robot"
        target_success.params["body_cfg"].body_names = target_body
        target_success.params["target_position_offset"] = target_position_offset
        target_success.params["body_position_offset"] = body_position_offset
        target_success.params["target_orientation_offset"] = target_orientation_offset
        target_success.params["insertion_axis"] = int(args.target_reward_insertion_axis)
        target_success.params["axial_threshold"] = (
            float(args.target_success_termination_threshold) if bool(args.terminate_on_target_success) else None
        )
        target_success.params["lateral_threshold"] = (
            float(args.target_success_termination_threshold) if bool(args.terminate_on_target_success) else None
        )
        target_success.params["threshold"] = (
            float(args.target_success_termination_threshold) if bool(args.terminate_on_target_success) else 0.0
        )

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
        "isaac_env_step_dt": env_step_dt,
        "isaac_reward_weight_multiplier": reward_weight_multiplier,
        "distance_weight": float(args.target_reward_distance_weight),
        "close_weight": float(args.target_reward_close_weight),
        "progress_weight": float(args.target_reward_progress_weight),
        "progress_scale": float(args.target_reward_progress_scale),
        "orientation_weight": float(args.target_reward_orientation_weight),
        "orientation_std": float(args.target_reward_orientation_std),
        "orientation_gate_sigma": float(args.target_reward_orientation_gate_sigma),
        "reaching_weight": float(args.target_reward_reaching_weight),
        "terminal_weight": float(args.target_reward_terminal_weight),
        "lateral_weight": float(args.target_reward_lateral_weight),
        "lateral_gate_sigma": float(args.target_reward_lateral_gate_sigma),
        "lateral_error_scale": float(args.target_reward_lateral_error_scale),
        "motion_projection_weight": float(args.target_reward_motion_projection_weight),
        "motion_projection_scale": float(args.target_reward_motion_projection_scale),
        "lateral_progress_weight": float(args.target_reward_lateral_progress_weight),
        "lateral_progress_scale": float(args.target_reward_lateral_progress_scale),
        "axial_progress_weight": float(args.target_reward_axial_progress_weight),
        "axial_progress_scale": float(args.target_reward_axial_progress_scale),
        "insertion_corridor_weight": float(args.target_reward_insertion_corridor_weight),
        "insertion_corridor_sigma": float(args.target_reward_insertion_corridor_sigma),
        "insertion_bypass_penalty_scale": float(args.target_reward_insertion_bypass_penalty_scale),
        "insertion_axis": int(args.target_reward_insertion_axis),
        "force_delta_penalty_weight": float(args.force_delta_penalty_weight) if hasattr(rewards, "force_delta_penalty") else 0.0,
        "force_delta_threshold": float(args.force_delta_threshold),
        "force_delta_reference": float(args.force_delta_reference),
        "force_delta_saturation": float(args.force_delta_saturation),
        "force_delta_knee_penalty_fraction": float(args.force_delta_knee_penalty_fraction),
        "isaac_force_observation_clip_n": float(args.isaac_force_observation_clip_n),
        "target_position_offset": [float(v) for v in target_position_offset],
        "body_position_offset": [float(v) for v in body_position_offset],
        "target_orientation_offset": None if target_orientation_offset is None else [float(v) for v in target_orientation_offset],
        "body_orientation_offset": None if body_orientation_offset is None else [float(v) for v in body_orientation_offset],
        "distance_std": float(args.target_reward_distance_std),
        "close_sigma": float(args.target_reward_close_sigma),
        "reaching_threshold": float(args.target_reward_reaching_threshold),
        "terminate_on_target_success": bool(args.terminate_on_target_success),
        "success_termination_threshold": (
            float(args.target_success_termination_threshold) if bool(args.terminate_on_target_success) else 0.0
        ),
        "success_bonus_geometry": {
            "axial_threshold": float(args.target_success_termination_threshold),
            "lateral_threshold": float(args.target_success_termination_threshold),
            "insertion_axis": int(args.target_reward_insertion_axis),
        },
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
        batch = {
            "obs": self._stack_obs([item["obs"] for item in items], device),
            "next_obs": self._stack_obs([item["next_obs"] for item in items], device),
            "action": torch.stack([item["action"] for item in items]).to(device),
            "reward": torch.stack([item["reward"] for item in items]).to(device),
            "done": torch.stack([item["done"] for item in items]).to(device),
        }
        if all("guide_action" in item for item in items):
            batch["guide_action"] = torch.stack([item["guide_action"] for item in items]).to(device)
        return batch

    def diagnostic_snapshot(self) -> dict[str, Any]:
        if not self.data:
            return {"size": 0}
        latest = self.data[-1]
        out: dict[str, Any] = {
            "size": len(self.data),
            "latest_action": _tensor_stats(latest["action"]),
            "latest_reward": _jsonable(latest["reward"]),
            "latest_done_for_bootstrap": _jsonable(latest["done"]),
            "latest_metadata": latest.get("metadata"),
        }
        if len(self.data) > 1:
            sample = self.data[len(self.data) // 2]
            out["middle_sample_action"] = _tensor_stats(sample["action"])
            out["middle_sample_reward"] = _jsonable(sample["reward"])
            out["middle_sample_done_for_bootstrap"] = _jsonable(sample["done"])
            out["middle_sample_metadata"] = sample.get("metadata")
        return out

    def age_diagnostics(self, current_step: int | None = None) -> dict[str, Any]:
        if not self.data:
            return {"size": 0}
        rows = []
        for label, index in (("oldest", 0), ("middle", len(self.data) // 2), ("latest", len(self.data) - 1)):
            item = self.data[index]
            metadata = item.get("metadata") or {}
            inserted_step = metadata.get("inserted_env_step")
            age = None
            if current_step is not None and inserted_step is not None:
                age = int(current_step) - int(inserted_step)
            state = item.get("obs", {}).get("state")
            target_distance = None
            if isinstance(state, torch.Tensor) and state.numel() >= 3:
                target_distance = metadata.get("distance_to_target_after")
            rows.append(
                {
                    "label": label,
                    "buffer_index": int(index),
                    "age_steps": age,
                    "reward": _jsonable(item.get("reward")),
                    "done_for_bootstrap": _jsonable(item.get("done")),
                    "episode_id": metadata.get("episode_id"),
                    "target_distance_after": target_distance,
                }
            )
        return {"size": len(self.data), "samples": rows}

    def _stack_obs(self, obs_items: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
        return {
            "state": torch.stack([item["state"] for item in obs_items]).to(device),
            "images": {
                key: _unpack_replay_images([item["images"][key] for item in obs_items], device)
                for key in CAMERA_KEYS
            },
        }


class ExpertActionPrior:
    """Nearest-neighbor expert action prior for a small online BC regularizer."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        action_horizon: int,
        state_dim: int,
        action_dim: int,
        state_indices: list[int] | tuple[int, ...],
        max_samples: int,
        neighbor_chunk: int,
        device: torch.device,
    ):
        obs_all, action_all, schema = _load_lerobot_expert_arrays(
            dataset_root,
            action_horizon,
            max_samples=max_samples,
        )
        if obs_all.shape[1] != state_dim:
            raise ValueError(
                f"Expert dataset state dim {obs_all.shape[1]} does not match online state dim {state_dim}"
            )
        if action_all.shape[1] != action_dim:
            raise ValueError(
                f"Expert dataset action dim {action_all.shape[1]} does not match online action dim {action_dim}"
            )
        count = int(schema.get("num_frames", obs_all.shape[0]))
        indices = np.arange(obs_all.shape[0], dtype=np.int64)
        selected_indices = [int(i) for i in state_indices if 0 <= int(i) < state_dim]
        if not selected_indices:
            raise ValueError("expert BC state index set is empty")
        obs = obs_all[indices][:, selected_indices].astype(np.float32)
        mean = obs.mean(axis=0, keepdims=True)
        std = obs.std(axis=0, keepdims=True) + 1.0e-6
        self.dataset_root = Path(dataset_root)
        self.schema = schema
        self.count = int(count)
        self.sampled_count = int(indices.shape[0])
        self.state_indices = selected_indices
        self.neighbor_chunk = int(max(1, neighbor_chunk))
        self.raw_obs_selected = torch.as_tensor(obs, dtype=torch.float32, device=device)
        self.obs = torch.as_tensor((obs - mean) / std, dtype=torch.float32, device=device)
        self.action = torch.as_tensor(action_all[indices], dtype=torch.float32, device=device)
        self.mean = torch.as_tensor(mean.reshape(-1), dtype=torch.float32, device=device)
        self.std = torch.as_tensor(std.reshape(-1), dtype=torch.float32, device=device)

    def nearest_actions(self, state: torch.Tensor) -> torch.Tensor:
        query = (state[:, self.state_indices] - self.mean) / self.std
        best_dist: torch.Tensor | None = None
        best_index: torch.Tensor | None = None
        for start in range(0, self.obs.shape[0], self.neighbor_chunk):
            ref = self.obs[start : start + self.neighbor_chunk]
            dist = torch.cdist(query, ref, p=2.0)
            chunk_dist, chunk_index = dist.min(dim=1)
            if best_dist is None:
                best_dist = chunk_dist
                best_index = chunk_index + start
            else:
                mask = chunk_dist < best_dist
                best_dist = torch.where(mask, chunk_dist, best_dist)
                best_index = torch.where(mask, chunk_index + start, best_index)
        if best_index is None:
            raise RuntimeError("Expert prior has no actions")
        return self.action[best_index]

    def nearest_debug(self, state: torch.Tensor) -> dict[str, Any]:
        if state.numel() == 0:
            return {"available": False, "reason": "empty_state"}
        query_raw = state[:1, self.state_indices]
        query = (query_raw - self.mean) / self.std
        best_dist: torch.Tensor | None = None
        best_index: torch.Tensor | None = None
        for start in range(0, self.obs.shape[0], self.neighbor_chunk):
            ref = self.obs[start : start + self.neighbor_chunk]
            dist = torch.cdist(query, ref, p=2.0)
            chunk_dist, chunk_index = dist.min(dim=1)
            if best_dist is None or bool((chunk_dist < best_dist).item()):
                best_dist = chunk_dist
                best_index = chunk_index + start
        if best_index is None or best_dist is None:
            return {"available": False, "reason": "empty_prior"}
        idx = int(best_index[0].detach().cpu())
        return {
            "available": True,
            "dataset_root": str(self.dataset_root),
            "state_indices": self.state_indices,
            "nearest_index": idx,
            "nn_distance_normalized": float(best_dist[0].detach().cpu()),
            "query_selected_raw": _sample_vector(query_raw, limit=min(24, len(self.state_indices))),
            "query_selected_normalized": _sample_vector(query, limit=min(24, len(self.state_indices))),
            "nearest_expert_selected_raw": _sample_vector(self.raw_obs_selected[idx : idx + 1], limit=min(24, len(self.state_indices))),
            "nearest_expert_action_first12": _sample_vector(self.action[idx : idx + 1], limit=12),
            "schema": self.schema,
        }

    def report(self) -> dict[str, Any]:
        return {
            "dataset_root": str(self.dataset_root),
            "num_transitions": self.count,
            "sampled_transitions": self.sampled_count,
            "state_indices": self.state_indices,
            "neighbor_chunk": self.neighbor_chunk,
            "dataset_schema": self.schema,
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
    if bool(getattr(args_cli, "swap_rgb_channels", False)):
        image = image.flip(1)
    return F.interpolate(image, size=(256, 288), mode="bilinear", align_corners=False)


def _raw_camera_images(env, *, device: torch.device) -> dict[str, torch.Tensor]:
    if bool(getattr(args_cli, "force_camera_render_before_read", True)):
        sim = getattr(env.unwrapped, "sim", None)
        if sim is not None and hasattr(sim, "render"):
            sim.render()
        step_dt = float(getattr(env.unwrapped, "step_dt", 0.0) or 0.0)
        for sensor_name in ("center_camera", "left_camera", "right_camera"):
            sensor = env.unwrapped.scene.sensors.get(sensor_name)
            if sensor is not None and hasattr(sensor, "update"):
                try:
                    sensor.update(step_dt, force_recompute=True)
                except TypeError:
                    sensor.update(step_dt)
    return {
        "observation.images.center_camera": _camera_tensor(env, "center_camera", device=device),
        "observation.images.left_camera": _camera_tensor(env, "left_camera", device=device),
        "observation.images.right_camera": _camera_tensor(env, "right_camera", device=device),
    }


def _tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    finite = torch.isfinite(detached)
    if detached.numel() == 0:
        return {"shape": list(detached.shape), "numel": 0}
    finite_values = detached[finite]
    out: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "finite": bool(finite.all().detach().cpu()),
        "nan_count": int(torch.isnan(detached).sum().detach().cpu()),
        "inf_count": int(torch.isinf(detached).sum().detach().cpu()),
    }
    if finite_values.numel() > 0:
        out.update(
            {
                "mean": float(finite_values.mean().detach().cpu()),
                "std": float(finite_values.std(unbiased=False).detach().cpu()) if finite_values.numel() > 1 else 0.0,
                "min": float(finite_values.min().detach().cpu()),
                "max": float(finite_values.max().detach().cpu()),
                "abs_max": float(finite_values.abs().max().detach().cpu()),
            }
        )
    return out


def _sample_vector(value: torch.Tensor, *, row: int = 0, limit: int = 12) -> list[float]:
    if value.ndim == 0:
        return [float(value.detach().cpu())]
    sample = value.detach().flatten(start_dim=1)[row, :limit] if value.ndim > 1 else value.detach().flatten()[:limit]
    return [float(v) for v in sample.cpu().tolist()]


BASE_STATE_FEATURE_NAMES = [
    "tcp_position.x",
    "tcp_position.y",
    "tcp_position.z",
    "tcp_orientation_xyzw.x",
    "tcp_orientation_xyzw.y",
    "tcp_orientation_xyzw.z",
    "tcp_orientation_xyzw.w",
    "tcp_linear_velocity.x",
    "tcp_linear_velocity.y",
    "tcp_linear_velocity.z",
    "tcp_angular_velocity.x",
    "tcp_angular_velocity.y",
    "tcp_angular_velocity.z",
    "tcp_error.x",
    "tcp_error.y",
    "tcp_error.z",
    "tcp_error.rx",
    "tcp_error.ry",
    "tcp_error.rz",
    "joint_position.shoulder_pan",
    "joint_position.shoulder_lift",
    "joint_position.elbow",
    "joint_position.wrist_1",
    "joint_position.wrist_2",
    "joint_position.wrist_3",
    "joint_position.gripper",
    "wrist_force.x",
    "wrist_force.y",
    "wrist_force.z",
    "wrist_torque.x",
    "wrist_torque.y",
    "wrist_torque.z",
]

TASK_VECTOR_FEATURE_NAMES = [
    "task.family_sfp_to_nic",
    "task.family_sc_to_sc",
    "task.target_card_index_norm",
    "task.target_card_valid",
    "task.target_port_index_norm",
    "task.sc_to_sc_source_port_norm",
    "task.sc_to_sc_target_port_norm",
    "task.num_ports_norm",
    "task.num_nic_cards_norm",
    "task.reserved",
]


def _state_feature_names(state_dim: int) -> list[str]:
    if state_dim in {32, 42}:
        names = BASE_STATE_FEATURE_NAMES + TASK_VECTOR_FEATURE_NAMES
    elif state_dim in {72, 82}:
        names = BASE_STATE_FEATURE_NAMES + list(CONTACT_RECOVERY_FEATURE_NAMES) + TASK_VECTOR_FEATURE_NAMES
    else:
        names = [f"state.{idx}" for idx in range(state_dim)]
    return names[:state_dim]


def _state_schema_ranges(state_dim: int) -> list[dict[str, Any]]:
    ranges = [{"name": "base32", "start": 0, "end_exclusive": min(32, state_dim)}]
    if state_dim in {72, 82}:
        ranges.append({"name": "contact_recovery40", "start": 32, "end_exclusive": min(72, state_dim)})
    if state_dim in {42, 82}:
        ranges.append({"name": "task_vector10", "start": state_dim - 10, "end_exclusive": state_dim})
    return ranges


def _state_schema_diagnostics(
    state: torch.Tensor,
    actor: "IsaacACTAdapterActor",
    *,
    env=None,
    expert_prior: "ExpertActionPrior | None" = None,
) -> dict[str, Any]:
    state_dim = int(state.shape[-1])
    normalized = actor.act_normalizer.normalize_state(state)
    mean = actor.act_normalizer.state_mean[:, :state_dim].detach().to(state.device)
    std = actor.act_normalizer.state_std[:, :state_dim].detach().to(state.device)
    names = _state_feature_names(state_dim)
    row_raw = state[0].detach()
    row_norm = normalized[0].detach()
    row_mean = mean[0].detach()
    row_std = std[0].detach()
    dim_rows = []
    bad_norm = []
    for idx, name in enumerate(names):
        raw = float(row_raw[idx].cpu())
        norm = float(row_norm[idx].cpu())
        item = {
            "index": idx,
            "name": name,
            "raw": raw,
            "mean": float(row_mean[idx].cpu()),
            "std": float(row_std[idx].cpu()),
            "normalized": norm,
        }
        dim_rows.append(item)
        if abs(norm) > 10.0:
            bad_norm.append(item)
    top_abs_norm = sorted(dim_rows, key=lambda item: abs(float(item["normalized"])), reverse=True)[:12]
    frame = {
        "tcp_error_raw_env0": _sample_vector(state[:, 13:19], limit=6) if state_dim >= 19 else None,
        "tcp_error_norm_env0": _sample_vector(normalized[:, 13:19], limit=6) if state_dim >= 19 else None,
        "wrist_force_raw_env0": _sample_vector(state[:, 26:29], limit=3) if state_dim >= 29 else None,
        "wrist_torque_raw_env0": _sample_vector(state[:, 29:32], limit=3) if state_dim >= 32 else None,
        "contact_recovery_raw_env0": _sample_vector(state[:, 32:72], limit=40) if state_dim >= 72 else None,
        "contact_recovery_norm_env0": _sample_vector(normalized[:, 32:72], limit=40) if state_dim >= 72 else None,
        "task_vector_raw_env0": _sample_vector(state[:, -10:], limit=10) if state_dim >= 10 else None,
        "task_vector_norm_env0": _sample_vector(normalized[:, -10:], limit=10) if state_dim >= 10 else None,
    }
    env_origin_rows = []
    if env is not None:
        try:
            origins = env.unwrapped.scene.env_origins.detach().cpu()
            root = env.unwrapped.scene["robot"].data.root_pos_w.detach().cpu()
            for env_id in range(min(int(origins.shape[0]), 8)):
                env_origin_rows.append(
                    {
                        "env_id": env_id,
                        "env_origin_world": [float(v) for v in origins[env_id].tolist()],
                        "robot_root_pos_world": [float(v) for v in root[env_id].tolist()],
                        "state_tcp_position_root_frame": _sample_vector(state[env_id : env_id + 1, 0:3], limit=3),
                        "root_minus_env_origin": [float(v) for v in (root[env_id] - origins[env_id]).tolist()],
                    }
                )
        except Exception as exc:  # noqa: BLE001 - diagnostics should not break training.
            env_origin_rows.append({"error": str(exc)})
    warnings = []
    if state_dim >= 19 and float(state[:, 13:19].abs().max().detach().cpu()) == 0.0:
        warnings.append(
            "tcp_error features are all zero in this Isaac sample. Expert Gazebo datasets record controller tcp_error; "
            "verify this is intentional or replace with a compatible Isaac signal."
        )
    if bad_norm:
        warnings.append("Some normalized state dimensions exceed abs(value)>10; inspect top_abs_normalized_dims.")
    return {
        "state_dim": state_dim,
        "schema_ranges": _state_schema_ranges(state_dim),
        "isaac_wrench_source": str(getattr(_isaac_lerobot_state, "_last_wrench_source", "unknown")),
        "feature_names": [{"index": idx, "name": name} for idx, name in enumerate(names)],
        "env0_by_dim": dim_rows,
        "top_abs_normalized_dims": top_abs_norm,
        "abs_normalized_gt_10": bad_norm[:24],
        "frame_blocks": frame,
        "env_origin_audit": env_origin_rows,
        "expert_nearest_neighbor": None if expert_prior is None else expert_prior.nearest_debug(state),
        "warnings": warnings,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().flatten()[:32].tolist()
    if isinstance(value, np.ndarray):
        return value.flatten()[:32].tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _ids_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, slice):
        return str(value)
    try:
        return [int(v) for v in value]
    except TypeError:
        return str(value)


def _norm_stats(value: torch.Tensor) -> dict[str, float]:
    if value.ndim <= 1:
        norm = value.detach().reshape(1, -1).norm(dim=-1)
    else:
        norm = value.detach().flatten(start_dim=1).norm(dim=-1)
    return {
        "norm_mean": float(norm.mean().detach().cpu()),
        "norm_max": float(norm.max().detach().cpu()),
        "abs_max": float(value.detach().abs().max().detach().cpu()) if value.numel() else 0.0,
        "mean": float(value.detach().mean().detach().cpu()) if value.numel() else 0.0,
        "std": float(value.detach().std(unbiased=False).detach().cpu()) if value.numel() > 1 else 0.0,
    }


def _grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total += float(torch.sum(grad * grad).detach().cpu())
    return math.sqrt(total)


def _executed_critic_action(action: torch.Tensor, *, single_action_dim: int) -> torch.Tensor:
    """Return the 6D action actually executed by Isaac for critic training."""
    return action[:, :single_action_dim]


def _to_act_obs(policy_obs: torch.Tensor, images: dict[str, torch.Tensor], *, state_dim: int) -> dict[str, Any]:
    state = policy_obs[:, :state_dim]
    return {"state": state, "images": images}


def _named_index(names: list[str], target: str) -> int:
    try:
        return names.index(target)
    except ValueError as exc:
        raise RuntimeError(f"Isaac robot does not expose {target!r}; available names: {names}") from exc


def _isaac_wrench_observation(env, *, device: torch.device) -> tuple[torch.Tensor, str]:
    """Return Gazebo-layout wrist wrench [fx, fy, fz, tx, ty, tz] for Isaac obs.

    Isaac may not expose a wrist force/torque sensor tensor. In that case, use
    the contact sensor net force on the same physical wrist/tip proxy bodies and
    leave torque at zero. The state layout remains identical to Gazebo runtime:
    base slots 26:29 are force xyz and 29:32 are torque xyz.
    """
    robot = env.unwrapped.scene["robot"]
    data = robot.data
    body_names = list(getattr(robot, "body_names", []))
    batch_size = int(data.body_pos_w.shape[0])
    force_body_index = _named_index(body_names, FORCE_WRENCH_BODY)
    contact_force, contact_source = _isaac_contact_proxy_force(env, device=device, batch_size=batch_size)
    for attr in ("body_incoming_wrench_w", "body_incoming_wrench_b", "body_incoming_joint_wrench_b"):
        incoming_wrench = getattr(data, attr, None)
        if incoming_wrench is not None:
            wrench = incoming_wrench[:, force_body_index, :6].to(device=device, dtype=torch.float32)
            if contact_force is not None and torch.all(torch.norm(wrench[:, :3], dim=1) <= 1.0e-6):
                out = torch.zeros(batch_size, 6, dtype=torch.float32, device=device)
                out[:, :3] = _clip_force_observation(contact_force)
                return out, f"{contact_source}_because_{attr}_zero"
            wrench = wrench.clone()
            wrench[:, :3] = _clip_force_observation(wrench[:, :3])
            return wrench, attr

    wrench = torch.zeros(batch_size, 6, dtype=torch.float32, device=device)
    if contact_force is not None:
        wrench[:, :3] = _clip_force_observation(contact_force)
        return wrench, contact_source
    return wrench, "zeros_no_wrench_or_contact_sensor"


def _isaac_contact_proxy_force(env, *, device: torch.device, batch_size: int) -> tuple[torch.Tensor | None, str]:
    sensors = getattr(env.unwrapped.scene, "sensors", {})
    force_rows = []
    sensor_sources = []
    for sensor_name in ("contact_forces",):
        sensor = sensors.get(sensor_name)
        if sensor is None:
            continue
        net = getattr(sensor.data, "net_forces_w", None)
        if net is None:
            continue
        sensor_names = list(getattr(sensor, "body_names", []) or [])
        ids = [idx for idx, name in enumerate(sensor_names) if name in FORCE_CONTACT_PROXY_BODIES]
        if not ids:
            continue
        force_rows.append(net[:, ids, :3].sum(dim=1).to(device=device, dtype=torch.float32))
        sensor_sources.append(f"{sensor_name}.net_forces_w")
    if force_rows:
        del batch_size
        return torch.stack(force_rows, dim=0).sum(dim=0), "+".join(sensor_sources)
    del batch_size
    return None, "no_contact_sensor_force"


def _clip_force_observation(force: torch.Tensor) -> torch.Tensor:
    max_norm = float(getattr(args_cli, "isaac_force_observation_clip_n", 0.0))
    if max_norm <= 0.0:
        return force
    norm = torch.norm(force, dim=1, keepdim=True).clamp(min=1.0e-9)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return force * scale


def _isaac_lerobot_state(env, args: argparse.Namespace, *, device: torch.device, state_dim: int) -> torch.Tensor:
    robot = env.unwrapped.scene["robot"]
    body_names = list(getattr(robot, "body_names", []))
    joint_names = list(getattr(robot, "joint_names", []))
    tcp_index = _named_index(body_names, CONTROLLED_TCP_BODY)
    joint_indices = [_named_index(joint_names, name) for name in ARM_JOINT_NAMES]
    data = robot.data
    batch_size = int(data.body_pos_w.shape[0])

    root_pos_w = data.root_pos_w
    root_quat_w = data.root_quat_w
    tcp_pos_w = data.body_pos_w[:, tcp_index]
    tcp_quat_w = data.body_quat_w[:, tcp_index]
    tcp_pos, tcp_quat = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, tcp_pos_w, tcp_quat_w)
    # Isaac Lab returns quaternions as wxyz; the LeRobot/Gazebo observation
    # contract stores orientation fields as xyzw.
    tcp_quat_xyzw = torch.cat([tcp_quat[:, 1:4], tcp_quat[:, 0:1]], dim=-1)
    # Quaternions q and -q represent the same pose, but the ACT normalizer was
    # trained on a consistent positive-w convention. Canonicalize here so Isaac
    # does not feed the frozen ACT base a sign-flipped, hundreds-of-sigma state.
    tcp_quat_xyzw = torch.where(tcp_quat_xyzw[:, 3:4] < 0.0, -tcp_quat_xyzw, tcp_quat_xyzw)
    tcp_lin_vel = getattr(data, "body_lin_vel_w", torch.zeros(batch_size, len(body_names), 3, device=device))[
        :, tcp_index
    ]
    tcp_ang_vel = getattr(data, "body_ang_vel_w", torch.zeros(batch_size, len(body_names), 3, device=device))[
        :, tcp_index
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
    wrench, wrench_source = _isaac_wrench_observation(env, device=device)
    setattr(_isaac_lerobot_state, "_last_wrench_source", wrench_source)
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
    step_dt = float(getattr(env.unwrapped, "step_dt", 1.0 / 20.0))
    step_count = int(getattr(env.unwrapped, "common_step_counter", 0))
    time_sec = step_count * step_dt
    reset_mask = getattr(env.unwrapped, "episode_length_buf", None)
    if reset_mask is not None:
        for idx in range(batch_size):
            if int(reset_mask[idx].detach().cpu()) <= 1:
                computers[idx].reset()
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
    out = torch.as_tensor(np.stack(features, axis=0), dtype=torch.float32, device=device)
    if out.shape[1] != CONTACT_RECOVERY_FEATURE_DIM:
        raise RuntimeError(f"Expected {CONTACT_RECOVERY_FEATURE_DIM} contact features, got {out.shape[1]}")
    return out


def _force_delta_metrics(env, *, device: torch.device) -> dict[str, torch.Tensor]:
    wrench, source = _isaac_wrench_observation(env, device=device)
    force = wrench[:, :3]
    prev = getattr(_force_delta_metrics, "_previous_force", None)
    reset_mask = getattr(env.unwrapped, "episode_length_buf", None)
    if prev is None or prev.shape != force.shape:
        prev = force.detach().clone()
    elif reset_mask is not None:
        prev = prev.to(device)
        prev = torch.where(reset_mask.to(device).reshape(-1, 1) <= 1, force.detach(), prev)
    delta_norm = torch.norm(force - prev.to(device), dim=1)
    setattr(_force_delta_metrics, "_previous_force", force.detach().clone())
    threshold = float(args_cli.force_delta_threshold)
    reference = max(float(args_cli.force_delta_reference), threshold + 1.0e-6)
    saturation = max(float(args_cli.force_delta_saturation), reference + 1.0e-6)
    knee_fraction = min(max(float(args_cli.force_delta_knee_penalty_fraction), 0.0), 1.0)
    below_knee = ((delta_norm - threshold) / (reference - threshold)).clamp(min=0.0, max=1.0)
    low_penalty = knee_fraction * torch.square(below_knee)
    above_knee = ((delta_norm - reference) / (saturation - reference)).clamp(min=0.0, max=1.0)
    smooth = torch.square(above_knee) * (3.0 - 2.0 * above_knee)
    normalized = torch.where(delta_norm <= reference, low_penalty, knee_fraction + (1.0 - knee_fraction) * smooth)
    return {
        "force_norm": torch.norm(force, dim=1),
        "force_delta_norm": delta_norm,
        "force_delta_penalty": -float(args_cli.force_delta_penalty_weight) * normalized,
        "force_source": source,
        "force_body": FORCE_WRENCH_BODY,
    }


def _reward_term_metrics(env, *, device: torch.device) -> dict[str, Any]:
    """Return reward-manager term contributions without re-evaluating reward funcs."""
    manager = getattr(env.unwrapped, "reward_manager", None)
    if manager is None:
        return {}
    names = list(getattr(manager, "active_terms", []))
    step_reward = getattr(manager, "_step_reward", None)
    episode_sums = getattr(manager, "_episode_sums", None)
    if step_reward is None:
        return {}
    dt = float(getattr(env.unwrapped, "step_dt", 1.0))
    step_reward = step_reward.to(device=device, dtype=torch.float32)
    step_contrib = step_reward * dt
    out: dict[str, Any] = {
        "reward_step_dt": dt,
        "reward_terms_mean": {
            name: float(step_contrib[:, idx].mean().detach().cpu())
            for idx, name in enumerate(names)
        },
        "reward_terms_rate_mean": {
            name: float(step_reward[:, idx].mean().detach().cpu())
            for idx, name in enumerate(names)
        },
        "reward_terms_abs_mean": {
            name: float(step_contrib[:, idx].abs().mean().detach().cpu())
            for idx, name in enumerate(names)
        },
    }
    if isinstance(episode_sums, dict):
        out["reward_terms_accum_mean"] = {
            name: float(episode_sums[name].to(device=device, dtype=torch.float32).mean().detach().cpu())
            for name in names
            if name in episode_sums
        }
        out["reward_total_accum_mean"] = float(
            sum(
                episode_sums[name].to(device=device, dtype=torch.float32).mean()
                for name in names
                if name in episode_sums
            ).detach().cpu()
        )
    out["reward_terms_total_from_terms_mean"] = float(step_contrib.sum(dim=1).mean().detach().cpu())
    return out


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
        "tcp_reset": (getattr(env.unwrapped, "_aic_tcp_reset_report_by_env", {}) or {}).get(env_index),
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


def _camera_freshness_diagnostics(
    previous: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(set(previous) | set(current)):
        before = previous.get(key)
        after = current.get(key)
        if before is None or after is None or before.shape != after.shape:
            out[key] = {
                "previous_shape": None if before is None else list(before.shape),
                "current_shape": None if after is None else list(after.shape),
                "warning": "missing camera tensor or shape changed",
            }
            continue
        diff = (after - before).detach()
        out[key] = {
            "mean_abs_frame_delta": float(diff.abs().mean().cpu()),
            "max_abs_frame_delta": float(diff.abs().max().cpu()),
            "unchanged_fraction": float((diff.abs() < 1.0e-6).float().mean().cpu()),
            "likely_stale": bool(float(diff.abs().max().cpu()) < 1.0e-6),
        }
    return out


def _selected_body_orientations(env) -> dict[str, torch.Tensor | None]:
    return {
        body_name: _body_orientation_by_name(env, body_name)
        for body_name in ("wrist_3_link", "gripper_tcp", "sfp_tip_link", CONTROLLED_TCP_BODY)
    }


def _realized_rotation_diagnostics(
    *,
    before: dict[str, torch.Tensor | None],
    after: dict[str, torch.Tensor | None],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for body_name, before_quat in before.items():
        after_quat = after.get(body_name)
        if before_quat is None or after_quat is None:
            out[body_name] = None
            continue
        error = math_utils.quat_error_magnitude(after_quat, before_quat)
        out[body_name] = {
            "before_quat_wxyz_env0": _sample_vector(before_quat, limit=4),
            "after_quat_wxyz_env0": _sample_vector(after_quat, limit=4),
            "realized_orientation_delta_rad_env0": float(error[0].detach().cpu()),
            "realized_orientation_delta_rad_mean": float(error.mean().detach().cpu()),
        }
    return out


def _tcp_delta_action_to_isaac_base_action(
    env,
    tcp_action: torch.Tensor,
    *,
    action_frame: str = "gripper_tcp",
    apply_ik_sign_fix: bool = True,
) -> torch.Tensor:
    """Convert Gazebo/LeRobot gripper-tcp delta actions to Isaac IK root-frame deltas."""
    if action_frame == "root":
        action = tcp_action.clone()
    else:
        robot = env.unwrapped.scene["robot"]
        body_names = list(getattr(robot, "body_names", []))
        frame_index = _named_index(body_names, action_frame)
        data = robot.data
        _, frame_quat_b = math_utils.subtract_frame_transforms(
            data.root_pos_w,
            data.root_quat_w,
            data.body_pos_w[:, frame_index],
            data.body_quat_w[:, frame_index],
        )
        delta_pos_b = math_utils.quat_apply(frame_quat_b, tcp_action[:, :3])
        delta_rot_b = math_utils.quat_apply(frame_quat_b, tcp_action[:, 3:6])
        action = torch.cat([delta_pos_b, delta_rot_b], dim=-1)
    if bool(apply_ik_sign_fix) and bool(args_cli.fix_isaac_ik_xy_sign):
        action = action.clone()
        action[:, 0:2] *= -1.0
    return action


def _target_guided_policy_action(
    env,
    task_geometry_reward_config: dict[str, Any],
    *,
    step_size: float,
    axial_step_size: float | None,
    lateral_switch_m: float,
    axial_blend_lateral_m: float,
    action_frame: str,
    device: torch.device,
) -> torch.Tensor:
    """Small policy-space action that moves the reward body toward the target point."""
    target_pos_w = _target_position_from_reward_config(env, task_geometry_reward_config)
    body_name = str(task_geometry_reward_config.get("target_body") or "sfp_tip_link")
    body_pos_w = _body_position_by_name(env, body_name)
    robot = env.unwrapped.scene["robot"]
    batch_size = int(robot.data.root_pos_w.shape[0])
    guide = torch.zeros((batch_size, 6), dtype=torch.float32, device=device)
    if target_pos_w is None or body_pos_w is None or float(step_size) <= 0.0:
        return guide
    delta_w = target_pos_w.to(device) - body_pos_w.to(device)
    axis_w = _episode_insertion_axis_from_yaml(env)
    if axis_w is not None:
        axis_w = axis_w.to(device=device, dtype=delta_w.dtype)
        axial_delta = torch.sum(delta_w * axis_w, dim=1, keepdim=True) * axis_w
        lateral_delta = delta_w - axial_delta
        lateral_distance = torch.linalg.norm(lateral_delta, dim=1, keepdim=True)
        axial_distance = torch.linalg.norm(axial_delta, dim=1, keepdim=True)
        lateral_direction = lateral_delta / lateral_distance.clamp(min=1.0e-9)
        axial_direction = axial_delta / axial_distance.clamp(min=1.0e-9)
        lateral_switch = max(float(lateral_switch_m), 0.0)
        # Keep the guide insertion-safe: first remove lateral error, then insert.
        # Blending axial motion while off-axis let the policy reduce Euclidean
        # distance by descending beside the port instead of entering it.
        use_lateral_only = lateral_distance > lateral_switch
        move_direction = torch.where(use_lateral_only, lateral_direction, axial_direction)
        remaining = torch.where(use_lateral_only, lateral_distance, axial_distance)
        lateral_step = max(float(step_size), 0.0)
        axial_step = lateral_step if axial_step_size is None or float(axial_step_size) <= 0.0 else float(axial_step_size)
        step_limit = torch.where(
            use_lateral_only,
            torch.full_like(remaining, lateral_step),
            torch.full_like(remaining, max(axial_step, 0.0)),
        )
        move_w = move_direction * torch.minimum(remaining, step_limit)
    else:
        distance = torch.linalg.norm(delta_w, dim=1, keepdim=True).clamp(min=1.0e-9)
        move_w = delta_w / distance * torch.minimum(distance, torch.full_like(distance, float(step_size)))
    root_quat_w = robot.data.root_quat_w.to(device)
    desired_root_delta = math_utils.quat_apply_inverse(root_quat_w, move_w)
    pre_sign_root_delta = desired_root_delta.clone()
    if bool(args_cli.fix_isaac_ik_xy_sign):
        pre_sign_root_delta[:, 0:2] *= -1.0
    if action_frame == "root":
        guide[:, :3] = pre_sign_root_delta
        return guide
    body_names = list(getattr(robot, "body_names", []))
    frame_index = _named_index(body_names, action_frame)
    _, frame_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w.to(device),
        robot.data.root_quat_w.to(device),
        robot.data.body_pos_w[:, frame_index].to(device),
        robot.data.body_quat_w[:, frame_index].to(device),
    )
    guide[:, :3] = math_utils.quat_apply_inverse(frame_quat_b, pre_sign_root_delta)
    return guide


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


def _offset_position_w(pos_w: torch.Tensor, quat_w: torch.Tensor, offset: Any) -> torch.Tensor:
    if offset is None:
        return pos_w
    offset_tensor = torch.tensor(offset, dtype=pos_w.dtype, device=pos_w.device).view(1, 3)
    return pos_w + math_utils.quat_apply(quat_w, offset_tensor.expand(pos_w.shape[0], -1))


def _target_position_from_reward_config(env, reward_config: dict[str, Any]) -> torch.Tensor | None:
    episode_positions = _episode_target_position_from_yaml(env)
    if episode_positions is not None:
        return episode_positions
    scene_name = reward_config.get("target_scene_name")
    if not scene_name:
        return None
    try:
        target = env.unwrapped.scene[scene_name]
    except Exception:
        return None
    return _offset_position_w(
        target.data.root_pos_w,
        target.data.root_quat_w,
        reward_config.get("target_position_offset"),
    )


def _episode_target_position_from_yaml(env) -> torch.Tensor | None:
    episodes = _current_episode_by_env(env)
    if not episodes:
        return None
    rows: list[torch.Tensor] = []
    origins = env.unwrapped.scene.env_origins
    for env_id in range(env.unwrapped.num_envs):
        episode = episodes.get(env_id)
        target = (((episode or {}).get("scene") or {}).get("target") or {})
        pose = target.get("target_pose_world") or {}
        position = pose.get("position")
        if position is None:
            return None
        position_tensor = torch.tensor(position, dtype=origins.dtype, device=origins.device)
        entrance = (target.get("entrance_pose_world") or {}).get("position")
        axis = target.get("insertion_axis_world")
        if entrance is not None and axis is not None:
            entrance_tensor = torch.tensor(entrance, dtype=origins.dtype, device=origins.device)
            axis_tensor = torch.tensor(axis, dtype=origins.dtype, device=origins.device)
            axis_tensor = axis_tensor / torch.linalg.norm(axis_tensor).clamp(min=1.0e-9)
            seated_depth = torch.sum((position_tensor - entrance_tensor) * axis_tensor).clamp(min=0.0)
            position_tensor = entrance_tensor + seated_depth * axis_tensor
        rows.append(position_tensor + origins[env_id])
    return torch.stack(rows, dim=0)


def _episode_insertion_axis_from_yaml(env) -> torch.Tensor | None:
    episodes = _current_episode_by_env(env)
    if not episodes:
        return None
    origins = env.unwrapped.scene.env_origins
    rows: list[torch.Tensor] = []
    for env_id in range(env.unwrapped.num_envs):
        episode = episodes.get(env_id)
        target = ((episode or {}).get("scene") or {}).get("target") or {}
        axis = target.get("insertion_axis_world")
        if axis is None:
            return None
        axis_tensor = torch.tensor(axis, dtype=origins.dtype, device=origins.device)
        rows.append(axis_tensor / torch.linalg.norm(axis_tensor).clamp(min=1.0e-9))
    return torch.stack(rows, dim=0)


def _episode_target_orientation_from_yaml(env) -> torch.Tensor | None:
    episodes = _current_episode_by_env(env)
    if not episodes:
        return None
    origins = env.unwrapped.scene.env_origins
    rows: list[torch.Tensor] = []
    for env_id in range(env.unwrapped.num_envs):
        episode = episodes.get(env_id)
        target = (((episode or {}).get("scene") or {}).get("target") or {}).get("target_pose_world") or {}
        orientation = target.get("orientation_wxyz")
        if orientation is None:
            return None
        quat = torch.tensor(orientation, dtype=origins.dtype, device=origins.device)
        rows.append(quat / torch.linalg.norm(quat).clamp(min=1.0e-9))
    return torch.stack(rows, dim=0)


def _offset_quat_w(quat_w: torch.Tensor, offset: Any) -> torch.Tensor:
    if offset is None:
        return quat_w
    offset_tensor = torch.tensor(offset, dtype=quat_w.dtype, device=quat_w.device).view(1, 4)
    return math_utils.quat_mul(quat_w, offset_tensor.expand(quat_w.shape[0], -1))


def _target_orientation_from_reward_config(env, reward_config: dict[str, Any]) -> torch.Tensor | None:
    episode_orientations = _episode_target_orientation_from_yaml(env)
    if episode_orientations is not None:
        return episode_orientations
    scene_name = reward_config.get("target_scene_name")
    if not scene_name:
        return None
    try:
        target = env.unwrapped.scene[scene_name]
    except Exception:
        return None
    return _offset_quat_w(target.data.root_quat_w, reward_config.get("target_orientation_offset"))


def _body_position_by_name(env, body_name: str, body_offset: Any = None) -> torch.Tensor | None:
    robot = env.unwrapped.scene["robot"]
    body_names = list(getattr(robot, "body_names", []))
    if body_name not in body_names:
        return None
    body_idx = body_names.index(body_name)
    return _offset_position_w(
        robot.data.body_pos_w[:, body_idx],
        robot.data.body_quat_w[:, body_idx],
        body_offset,
    )


def _body_orientation_by_name(env, body_name: str, body_offset: Any = None) -> torch.Tensor | None:
    robot = env.unwrapped.scene["robot"]
    body_names = list(getattr(robot, "body_names", []))
    if body_name not in body_names:
        return None
    body_idx = body_names.index(body_name)
    return _offset_quat_w(robot.data.body_quat_w[:, body_idx], body_offset)


def _resolved_reward_terms(env) -> dict[str, Any]:
    manager = getattr(env.unwrapped, "reward_manager", None)
    names = list(getattr(manager, "_term_names", None) or getattr(manager, "active_terms", []) or [])
    cfgs = list(getattr(manager, "_term_cfgs", None) or getattr(manager, "term_cfgs", None) or [])
    out: dict[str, Any] = {}
    for idx, cfg in enumerate(cfgs):
        name = names[idx] if idx < len(names) else getattr(cfg, "name", None)
        params = getattr(cfg, "params", {}) or {}
        if not name:
            continue
        body_cfg = params.get("body_cfg")
        target_cfg = params.get("target_cfg")
        asset_cfg = params.get("asset_cfg")
        out[str(name)] = {
            "weight": float(getattr(cfg, "weight", 0.0)),
            "asset_cfg_name": None if asset_cfg is None else getattr(asset_cfg, "name", None),
            "asset_body_names": None if asset_cfg is None else list(getattr(asset_cfg, "body_names", []) or []),
            "asset_body_ids": None if asset_cfg is None else _ids_jsonable(getattr(asset_cfg, "body_ids", None)),
            "body_cfg_name": None if body_cfg is None else getattr(body_cfg, "name", None),
            "body_names": None if body_cfg is None else list(getattr(body_cfg, "body_names", []) or []),
            "body_ids": None if body_cfg is None else _ids_jsonable(getattr(body_cfg, "body_ids", None)),
            "target_cfg_name": None if target_cfg is None else getattr(target_cfg, "name", None),
            "target_body_names": None if target_cfg is None else list(getattr(target_cfg, "body_names", []) or []),
            "target_body_ids": None if target_cfg is None else _ids_jsonable(getattr(target_cfg, "body_ids", None)),
            "target_position_offset": params.get("target_position_offset"),
            "body_position_offset": params.get("body_position_offset"),
            "target_orientation_offset": params.get("target_orientation_offset"),
            "body_orientation_offset": params.get("body_orientation_offset"),
        }
    return out


def _reward_weight_diagnostics(env) -> dict[str, Any]:
    manager = getattr(env.unwrapped, "reward_manager", None)
    names = list(getattr(manager, "_term_names", None) or getattr(manager, "active_terms", []) or [])
    cfgs = list(getattr(manager, "_term_cfgs", None) or getattr(manager, "term_cfgs", None) or [])
    return {
        str(names[idx] if idx < len(names) else getattr(cfg, "name", f"term_{idx}")): float(getattr(cfg, "weight", 0.0))
        for idx, cfg in enumerate(cfgs)
    }


def _action_manager_diagnostics(env) -> dict[str, Any]:
    manager = getattr(env.unwrapped, "action_manager", None)
    out: dict[str, Any] = {
        "manager_type": None if manager is None else type(manager).__name__,
        "configured_body": None,
        "configured_scale": None,
        "terms": {},
    }
    if manager is None:
        return out
    names = list(getattr(manager, "_term_names", None) or getattr(manager, "active_terms", []) or [])
    cfgs = list(getattr(manager, "_term_cfgs", None) or getattr(manager, "term_cfgs", None) or [])
    for idx, cfg in enumerate(cfgs):
        name = str(names[idx] if idx < len(names) else getattr(cfg, "name", "unknown"))
        out["terms"][name] = {
            "body_name": getattr(cfg, "body_name", None),
            "scale": getattr(cfg, "scale", None),
            "asset_name": getattr(cfg, "asset_name", None),
        }
        if name == "arm_action":
            out["configured_body"] = getattr(cfg, "body_name", None)
            out["configured_scale"] = getattr(cfg, "scale", None)
    terms = getattr(manager, "_terms", None) or {}
    if isinstance(terms, dict) and "arm_action" in terms:
        term = terms["arm_action"]
        cfg = getattr(term, "cfg", None)
        out["runtime_arm_action"] = {
            "type": type(term).__name__,
            "body_name": getattr(cfg, "body_name", None) if cfg is not None else None,
            "scale": getattr(cfg, "scale", None) if cfg is not None else None,
        }
    return out


def _pose_reward_diagnostics(env, reward_config: dict[str, Any]) -> dict[str, Any]:
    target_pos = _target_position_from_reward_config(env, reward_config)
    target_quat = _target_orientation_from_reward_config(env, reward_config)
    body_offset = reward_config.get("body_position_offset")
    body_orientation_offset = reward_config.get("body_orientation_offset")
    out: dict[str, Any] = {
        "target_scene_name": reward_config.get("target_scene_name"),
        "target_reward_body_arg": reward_config.get("target_body"),
        "target_position_offset": reward_config.get("target_position_offset"),
        "body_position_offset": body_offset,
        "target_orientation_offset": reward_config.get("target_orientation_offset"),
        "body_orientation_offset": body_orientation_offset,
        "target_position_world_env0": None if target_pos is None else _sample_vector(target_pos, limit=3),
        "target_orientation_wxyz_env0": None if target_quat is None else _sample_vector(target_quat, limit=4),
        "target_asset_root_world_env0": None,
        "target_asset_root_quat_wxyz_env0": None,
        "body_positions_world_env0": {},
        "body_quats_wxyz_env0": {},
        "body_distances_to_target_env0": {},
        "body_orientation_error_rad_env0": {},
    }
    scene_name = reward_config.get("target_scene_name")
    if scene_name:
        try:
            target_asset = env.unwrapped.scene[scene_name]
            root_pos = target_asset.data.root_pos_w
            out["target_asset_root_world_env0"] = _sample_vector(root_pos, limit=3)
            out["target_asset_root_quat_wxyz_env0"] = _sample_vector(target_asset.data.root_quat_w, limit=4)
        except Exception:
            out["target_asset_root_world_env0"] = None
    for body_name in ("wrist_3_link", "gripper_tcp", "sfp_tip_link", CONTROLLED_TCP_BODY):
        body_pos = _body_position_by_name(env, body_name, body_offset if body_name == reward_config.get("target_body") else None)
        body_quat = _body_orientation_by_name(
            env,
            body_name,
            body_orientation_offset if body_name == reward_config.get("target_body") else None,
        )
        if body_pos is None:
            out["body_positions_world_env0"][body_name] = None
            out["body_distances_to_target_env0"][body_name] = None
            out["body_quats_wxyz_env0"][body_name] = None
            out["body_orientation_error_rad_env0"][body_name] = None
            continue
        out["body_positions_world_env0"][body_name] = _sample_vector(body_pos, limit=3)
        out["body_quats_wxyz_env0"][body_name] = None if body_quat is None else _sample_vector(body_quat, limit=4)
        if target_pos is not None:
            out["body_distances_to_target_env0"][body_name] = float(
                torch.norm(body_pos[0] - target_pos[0]).detach().cpu()
            )
        if target_quat is not None and body_quat is not None:
            out["body_orientation_error_rad_env0"][body_name] = float(
                math_utils.quat_error_magnitude(body_quat[0:1], target_quat[0:1]).detach().cpu().reshape(-1)[0]
            )
    if reward_config.get("target_body") not in ("wrist_3_link", "gripper_tcp", "sfp_tip_link"):
        out["warnings"] = [f"target_reward_body {reward_config.get('target_body')!r} is not one of common audit bodies"]
    return out


def _insertion_geometry_diagnostics(env, reward_config: dict[str, Any]) -> dict[str, Any]:
    target_pos = _target_position_from_reward_config(env, reward_config)
    body_name = str(reward_config.get("target_body") or "sfp_tip_link")
    body_pos = _body_position_by_name(env, body_name, reward_config.get("body_position_offset"))
    axis_w = _episode_insertion_axis_from_yaml(env)
    out: dict[str, Any] = {
        "body_name": body_name,
        "has_target": target_pos is not None,
        "has_body": body_pos is not None,
        "has_episode_axis": axis_w is not None,
        "target_world_env0": None if target_pos is None else _sample_vector(target_pos, limit=3),
        "body_world_env0": None if body_pos is None else _sample_vector(body_pos, limit=3),
    }
    if target_pos is None or body_pos is None or axis_w is None:
        return out
    episodes = _current_episode_by_env(env)
    origins = env.unwrapped.scene.env_origins
    entrance_rows: list[torch.Tensor] = []
    for env_id in range(env.unwrapped.num_envs):
        target = (((episodes.get(env_id) or {}).get("scene") or {}).get("target") or {})
        entrance = (target.get("entrance_pose_world") or {}).get("position")
        if entrance is None:
            out["has_entrance"] = False
            return out
        entrance_rows.append(
            torch.tensor(entrance, dtype=body_pos.dtype, device=body_pos.device)
            + origins[env_id].to(device=body_pos.device, dtype=body_pos.dtype)
        )
    entrance_w = torch.stack(entrance_rows, dim=0)
    axis_w = axis_w.to(device=body_pos.device, dtype=body_pos.dtype)
    delta_from_entrance = body_pos - entrance_w
    signed_depth = torch.sum(delta_from_entrance * axis_w, dim=1)
    target_depth = torch.sum((target_pos - entrance_w) * axis_w, dim=1).clamp(min=1.0e-9)
    axial_component = signed_depth.unsqueeze(1) * axis_w
    lateral_error = torch.linalg.norm(delta_from_entrance - axial_component, dim=1)
    corridor_sigma = float(reward_config.get("insertion_corridor_sigma", 0.0025))
    lateral_gate = torch.exp(-torch.square(lateral_error / max(corridor_sigma, 1.0e-9)))
    depth_fraction = (signed_depth / target_depth).clamp(min=0.0, max=1.0)
    out.update(
        {
            "has_entrance": True,
            "entrance_world_env0": _sample_vector(entrance_w, limit=3),
            "axis_world_env0": _sample_vector(axis_w, limit=3),
            "signed_depth_m_mean": float(signed_depth.mean().detach().cpu()),
            "signed_depth_m_env0": float(signed_depth[0].detach().cpu()),
            "target_depth_m_mean": float(target_depth.mean().detach().cpu()),
            "target_depth_m_env0": float(target_depth[0].detach().cpu()),
            "depth_fraction_mean": float(depth_fraction.mean().detach().cpu()),
            "depth_fraction_env0": float(depth_fraction[0].detach().cpu()),
            "lateral_error_m_mean": float(lateral_error.mean().detach().cpu()),
            "lateral_error_m_env0": float(lateral_error[0].detach().cpu()),
            "lateral_gate_mean": float(lateral_gate.mean().detach().cpu()),
            "lateral_gate_env0": float(lateral_gate[0].detach().cpu()),
            "inside_2mm_fraction": float((lateral_error <= 0.002).float().mean().detach().cpu()),
            "inside_3mm_fraction": float((lateral_error <= 0.003).float().mean().detach().cpu()),
            "bypass_risk_fraction": float(
                torch.logical_and(signed_depth > 0.0, lateral_error > corridor_sigma).float().mean().detach().cpu()
            ),
        }
    )
    return out


def _scene_asset_pose_diagnostics(env, reward_config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("task_board", "nic_card", "sc_port", "sc_port_2"):
        try:
            asset = env.unwrapped.scene[name]
        except Exception:
            out[name] = None
            continue
        out[name] = {
            "root_position_world_env0": _sample_vector(asset.data.root_pos_w, limit=3),
            "root_quat_wxyz_env0": _sample_vector(asset.data.root_quat_w, limit=4),
            "target_position_with_current_offset_env0": None,
        }
        if name in ("nic_card", "sc_port", "sc_port_2"):
            out[name]["target_position_with_current_offset_env0"] = _sample_vector(
                _offset_position_w(
                    asset.data.root_pos_w,
                    asset.data.root_quat_w,
                    reward_config.get("target_position_offset"),
                ),
                limit=3,
            )
    return out


def _episode_scene_diagnostics(env, reward_config: dict[str, Any], *, device: torch.device) -> dict[str, Any]:
    episodes = _current_episode_by_env(env)
    rows: list[dict[str, Any]] = []
    target_pos = _target_position_from_reward_config(env, reward_config)
    target_quat = _target_orientation_from_reward_config(env, reward_config)
    for env_id in range(min(env.unwrapped.num_envs, 8)):
        episode = episodes.get(env_id)
        context = (episode or {}).get("task_context") or {}
        scene = (episode or {}).get("scene") or {}
        target = (scene.get("target") or {}).get("target_pose_world") or {}
        if context:
            vector = _task_vector_from_contexts([_episode_context_tuple(episode)], device=device)[0].detach().cpu().tolist()
        else:
            vector = _task_vector_from_contexts(
                [(
                    str(args_cli.task_family),
                    int(args_cli.target_port_index),
                    int(args_cli.target_card_index),
                    int(args_cli.target_card_valid),
                )],
                device=device,
            )[0].detach().cpu().tolist()
        rows.append(
            {
                "env_id": env_id,
                "episode_id": None if episode is None else episode.get("episode_id"),
                "uses_current_episode_assignment": episode is not None,
                "decoded_context": context or None,
                "task_vector": [float(v) for v in vector],
                "episode_target_position_local": target.get("position"),
                "episode_target_orientation_wxyz": target.get("orientation_wxyz"),
                "reward_target_position_world": None if target_pos is None else _sample_vector(target_pos[env_id : env_id + 1], limit=3),
                "reward_target_orientation_wxyz": None if target_quat is None else _sample_vector(target_quat[env_id : env_id + 1], limit=4),
                "start_near_gate": scene.get("start_near_gate"),
                "tcp_reset_report": (getattr(env.unwrapped, "_aic_tcp_reset_report_by_env", {}) or {}).get(env_id),
            }
        )
    return {
        "episode_config_dir": args_cli.episode_config_dir,
        "episode_count_loaded": len(EPISODE_CONFIGS),
        "rows": rows,
    }


def _reset_control_body_diagnostics(env, reward_config: dict[str, Any]) -> dict[str, Any]:
    action_diag = _action_manager_diagnostics(env)
    reset_reports = getattr(env.unwrapped, "_aic_tcp_reset_report_by_env", {}) or {}
    reset_body = None
    if reset_reports:
        first_report = next(iter(reset_reports.values()))
        if isinstance(first_report, dict):
            reset_body = first_report.get("body_name")
    body_positions = {
        name: _body_position_by_name(env, name)
        for name in ("wrist_3_link", "gripper_tcp", "sfp_tip_link")
    }
    distances: dict[str, Any] = {}
    names = list(body_positions)
    for i, lhs in enumerate(names):
        for rhs in names[i + 1 :]:
            a, b = body_positions[lhs], body_positions[rhs]
            distances[f"{lhs}__{rhs}"] = None if a is None or b is None else float(torch.norm(a[0] - b[0]).detach().cpu())
    return {
        "reset_reports_by_env": reset_reports,
        "reset_event_order_tail": getattr(env.unwrapped, "_aic_reset_event_order", []) or [],
        "near_gate_reset_body": reset_body or "gripper_tcp",
        "controlled_ik_body": action_diag.get("configured_body") or (action_diag.get("runtime_arm_action") or {}).get("body_name"),
        "act_tcp_action_frame": args_cli.tcp_action_frame,
        "reward_body": reward_config.get("target_body"),
        "body_distances_at_reset_env0": distances,
    }


def _quaternion_convention_diagnostics(env, reward_config: dict[str, Any]) -> dict[str, Any]:
    out = {
        "robot_root_quat_wxyz_env0": None,
        "target_frame_quat_wxyz_env0": None,
        "assets": {},
        "warnings": [],
    }
    try:
        robot = env.unwrapped.scene["robot"]
        out["robot_root_quat_wxyz_env0"] = _sample_vector(robot.data.root_quat_w, limit=4)
    except Exception:
        pass
    target_quat = _target_orientation_from_reward_config(env, reward_config)
    if target_quat is not None:
        out["target_frame_quat_wxyz_env0"] = _sample_vector(target_quat, limit=4)
    for name in ("task_board", "nic_card", "sc_port", "sc_port_2"):
        try:
            asset = env.unwrapped.scene[name]
        except Exception:
            continue
        quat = asset.data.root_quat_w
        sample = _sample_vector(quat, limit=4)
        out["assets"][name] = sample
        if len(sample) == 4 and abs(sample[3] - 1.0) < 1.0e-3 and abs(sample[0]) < 1.0e-3:
            out["warnings"].append(
                f"{name} quaternion looks like xyzw identity [0,0,0,1] in Isaac wxyz storage"
            )
    return out


def _force_wrench_diagnostics(env) -> dict[str, Any]:
    robot = env.unwrapped.scene["robot"]
    body_names = list(getattr(robot, "body_names", []))
    data = robot.data
    wrench = getattr(data, "body_incoming_wrench_w", None)
    source = "body_incoming_wrench_w"
    if wrench is None:
        wrench = getattr(data, "body_incoming_wrench_b", None)
        source = "body_incoming_wrench_b" if wrench is not None else None
    if wrench is None:
        wrench = getattr(data, "body_incoming_joint_wrench_b", None)
        source = "body_incoming_joint_wrench_b" if wrench is not None else None
    out: dict[str, Any] = {
        "body_names": body_names,
        "wrench_source": source,
        "selected_reward_force_body_default": "all bodies unless reward asset_cfg body_names is set",
        "norms_env0": {},
        "contact_sensor": None,
        "warnings": [],
    }
    contact_sensors = {}
    for sensor_name in ("contact_forces",):
        sensor = getattr(env.unwrapped.scene, "sensors", {}).get(sensor_name)
        if sensor is None:
            continue
        sensor_names = list(getattr(sensor, "body_names", []) or [])
        sensor_net = getattr(sensor.data, "net_forces_w", None)
        contact_info: dict[str, Any] = {
            "name": sensor_name,
            "body_names": sensor_names,
            "net_forces_shape": None if sensor_net is None else list(sensor_net.shape),
            "net_force_norms_env0": {},
        }
        if sensor_net is not None and sensor_net.numel() > 0:
            norms = torch.norm(sensor_net[0, :, :3], dim=-1).detach().cpu().tolist()
            for idx, name in enumerate(sensor_names):
                contact_info["net_force_norms_env0"][name] = float(norms[idx])
        contact_sensors[sensor_name] = contact_info
    out["contact_sensor"] = contact_sensors or None
    if wrench is None:
        out["warnings"].append("robot data exposes no body incoming wrench tensor; falling back to contact_forces sensor if present")
        return out
    norms = torch.norm(wrench[0, :, :3], dim=-1).detach().cpu().tolist()
    for idx, name in enumerate(body_names):
        out["norms_env0"][name] = float(norms[idx])
    for required in ("wrist_3_link", "gripper_tcp", "sfp_tip_link"):
        if required not in body_names:
            out["warnings"].append(f"{required} body missing from robot")
    if "gripper_tcp" in body_names and out["norms_env0"].get("gripper_tcp", 0.0) == 0.0:
        out["warnings"].append("gripper_tcp wrench norm is zero at this instant; this is normal without contact")
    return out


def _target_source_diagnostics(env, reward_config: dict[str, Any]) -> dict[str, Any]:
    episodes = _current_episode_by_env(env)
    episode = episodes.get(0)
    episode_target = (((episode or {}).get("scene") or {}).get("target") or {}).get("target_pose_world") or {}
    uses_episode_position = episode is not None and episode_target.get("position") is not None
    uses_episode_orientation = episode is not None and episode_target.get("orientation_wxyz") is not None
    warnings: list[str] = []
    notes: list[str] = []
    if uses_episode_position and not uses_episode_orientation and reward_config.get("orientation_weight", 0.0) > 0.0:
        warnings.append(
            "target position comes from episode YAML, but orientation reward uses target asset root plus offset"
        )
    if uses_episode_position and uses_episode_orientation:
        notes.append("target position and orientation both come from episode YAML")
    if float(reward_config.get("lateral_weight", 0.0)) > 0.0:
        warnings.append("target_lateral_error is normally used with a negative penalty weight")
    return {
        "target_position_source": "episode_yaml" if uses_episode_position else "target_asset_root_plus_offset",
        "target_orientation_source": "episode_yaml" if uses_episode_orientation else "target_asset_root_plus_offset",
        "episode_target_pose_world_env0": episode_target or None,
        "notes": notes,
        "warnings": warnings,
    }


def _observation_diagnostics(
    *,
    obs: Any,
    policy_obs: torch.Tensor,
    act_obs: dict[str, Any],
    actor: IsaacACTAdapterActor,
    images: dict[str, torch.Tensor],
    env=None,
    expert_prior: ExpertActionPrior | None = None,
) -> dict[str, Any]:
    state = act_obs["state"]
    normalized_state = actor.act_normalizer.normalize_state(state)
    image_stats: dict[str, Any] = {}
    for key, image in images.items():
        normalized = actor.act_normalizer.normalize_image(key, image)
        image_stats[key] = {
            "raw": _tensor_stats(image),
            "normalized": _tensor_stats(normalized),
        }
    raw_task = state[:, -10:] if state.shape[-1] >= 10 else state[:, 0:0]
    normalized_task = normalized_state[:, -10:] if normalized_state.shape[-1] >= 10 else normalized_state[:, 0:0]
    return {
        "obs_type": type(obs).__name__,
        "obs_keys": list(obs.keys()) if isinstance(obs, dict) else None,
        "policy_obs": _tensor_stats(policy_obs),
        "act_state": _tensor_stats(state),
        "normalized_state": _tensor_stats(normalized_state),
        "raw_task_vector_env0": _sample_vector(raw_task, limit=10),
        "normalized_task_vector_env0": _sample_vector(normalized_task, limit=10),
        "task_vector_normalizer_mean_tail": _sample_vector(actor.act_normalizer.state_mean[:, -10:], limit=10),
        "task_vector_normalizer_std_tail": _sample_vector(actor.act_normalizer.state_std[:, -10:], limit=10),
        "state_dim": int(state.shape[-1]),
        "actor_state_dim": int(actor.state_dim),
        "actor_action_dim": int(actor.action_dim),
        "actor_action_horizon": int(actor.action_horizon),
        "image_keys": list(images.keys()),
        "images": image_stats,
        "swap_rgb_channels": bool(getattr(args_cli, "swap_rgb_channels", False)),
        "state_schema": _state_schema_diagnostics(state, actor, env=env, expert_prior=expert_prior),
    }


def _previous_action_diagnostics(
    *,
    state: torch.Tensor,
    policy_tcp_action: torch.Tensor,
    env_action: torch.Tensor,
    env,
) -> dict[str, Any]:
    manager = getattr(env.unwrapped, "action_manager", None)
    manager_action = None
    manager_prev_action = None
    if manager is not None:
        manager_action = getattr(manager, "action", None)
        manager_prev_action = getattr(manager, "prev_action", None)
    return {
        "state_contract": (
            "82D state is base32 + contact_recovery40 + task10. It has force-delta memory features, "
            "but no previous-action feature block."
        ),
        "state_dim": int(state.shape[-1]),
        "state_contact_feature_slice_32_72_env0": _sample_vector(state[:, 32:72], limit=16) if state.shape[-1] >= 72 else None,
        "state_task_slice_tail10_env0": _sample_vector(state[:, -10:], limit=10) if state.shape[-1] >= 10 else None,
        "actual_previous_policy_tcp_action_first6_env0": _sample_vector(policy_tcp_action, limit=6),
        "actual_env_action_after_frame_conversion_first6_env0": _sample_vector(env_action, limit=6),
        "env_action_manager_action": None if manager_action is None else _tensor_stats(manager_action),
        "env_action_manager_action_env0": None if manager_action is None else _sample_vector(manager_action, limit=12),
        "env_action_manager_prev_action": None if manager_prev_action is None else _tensor_stats(manager_prev_action),
        "env_action_manager_prev_action_env0": None if manager_prev_action is None else _sample_vector(manager_prev_action, limit=12),
        "action_scale_note": "policy_tcp_action is before Isaac frame conversion/IK action scale; env_action is what env.step receives.",
    }


def _frequency_diagnostics(env_cfg: Any, env: Any, *, action_horizon: int, single_action_dim: int) -> dict[str, Any]:
    sim_dt = float(getattr(getattr(env_cfg, "sim", None), "dt", float("nan")))
    decimation = int(getattr(env_cfg, "decimation", 0))
    policy_dt = float(getattr(env.unwrapped, "step_dt", sim_dt * decimation if decimation else float("nan")))
    expert_fps = 20.0
    return {
        "sim_dt": sim_dt,
        "decimation": decimation,
        "policy_dt": policy_dt,
        "policy_hz": None if policy_dt <= 0.0 else 1.0 / policy_dt,
        "expert_dataset_fps": expert_fps,
        "expert_action_dt": 1.0 / expert_fps,
        "action_horizon": int(action_horizon),
        "single_action_dim": int(single_action_dim),
        "physical_chunk_duration_isaac_s": policy_dt * int(action_horizon),
        "physical_chunk_duration_expert_s": int(action_horizon) / expert_fps,
        "warning": (
            None
            if abs(policy_dt - 1.0 / expert_fps) < 1.0e-6
            else "Isaac policy step duration differs from 20 Hz expert dataset action duration."
        ),
    }


def _act_chunk_inference_diagnostics(offline_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "gazebo_RunACT_path": "aic_example_policies/aic_example_policies/ros/RunACT.py",
        "gazebo_runtime": "RunACT overrides ACTPolicy.config.n_action_steps from AIC_ACT_N_ACTION_STEPS, default 4.",
        "gazebo_torchscript_runtime": "RunACTTorchScript queues the first AIC_ACT_N_ACTION_STEPS actions from each predicted chunk, default 4.",
        "isaac_online_runtime": "Isaac online SERL queues the first --n_action_steps actions from each predicted chunk, default 4.",
        "offline_cfg_chunk_size_or_action_horizon": offline_cfg.get("action_horizon"),
        "offline_cfg_action_dim": offline_cfg.get("action_dim"),
        "warning": (
            "All current ACT/SERL runtime paths should use chunk queue execution; confirm n_action_steps in logs."
        ),
    }


def _checkpoint_compatibility_diagnostics(
    *,
    actor: IsaacACTAdapterActor,
    offline_cfg: dict[str, Any],
    dataset_summary: dict[str, Any],
    state_dim: int,
    action_horizon: int,
    single_action_dim: int,
) -> dict[str, Any]:
    normalizer = actor.act_normalizer
    out = {
        "runtime_state_dim": int(state_dim),
        "runtime_action_dim": int(actor.action_dim),
        "runtime_action_horizon": int(action_horizon),
        "runtime_single_action_dim": int(single_action_dim),
        "actor_state_dim": int(actor.state_dim),
        "actor_action_dim": int(actor.action_dim),
        "actor_action_horizon": int(actor.action_horizon),
        "act_torchscript": str(actor.act_torchscript_path),
        "normalizer_state_dim": int(normalizer.state_mean.shape[-1]),
        "normalizer_single_action_dim": int(normalizer.action_mean.shape[-1]),
        "camera_keys_runtime_order": list(CAMERA_KEYS),
        "camera_keys_offline_cfg": offline_cfg.get("camera_keys"),
        "camera_keys_dataset_summary": dataset_summary.get("camera_keys"),
        "state_encoding": offline_cfg.get("state_encoding"),
        "state_encoding_indices": offline_cfg.get("state_encoding_indices"),
        "task_vector_layout": "last 10 dims, canonical task_encoding.py order",
        "warnings": [],
    }
    if int(normalizer.state_mean.shape[-1]) != int(state_dim):
        out["warnings"].append("ACT normalizer state dim does not match online runtime state dim")
    if int(normalizer.action_mean.shape[-1]) != int(single_action_dim):
        out["warnings"].append("ACT normalizer single action dim does not match online executed action dim")
    if list(offline_cfg.get("camera_keys") or CAMERA_KEYS) != list(CAMERA_KEYS):
        out["warnings"].append("Offline camera key order differs from Isaac runtime camera key order")
    return out


def _act_freeze_diagnostics(trainer: OnlineSERLTrainer) -> dict[str, Any]:
    act_params = list(trainer.actor.act_base.parameters())
    adapter_params = list(trainer.actor.adapter.parameters())
    optimizer_param_ids = {id(param) for group in trainer.actor_opt.param_groups for param in group["params"]}
    act_in_optimizer = sum(1 for param in act_params if id(param) in optimizer_param_ids)
    return {
        "actor_training_mode": bool(trainer.actor.training),
        "act_base_training_mode": bool(trainer.actor.act_base.training),
        "act_base_param_count": sum(param.numel() for param in act_params),
        "act_base_requires_grad_count": sum(param.numel() for param in act_params if param.requires_grad),
        "act_base_params_in_optimizer": int(act_in_optimizer),
        "adapter_param_count": sum(param.numel() for param in adapter_params),
        "adapter_requires_grad_count": sum(param.numel() for param in adapter_params if param.requires_grad),
        "freeze_act_arg": bool(args_cli.freeze_act),
        "warning": (
            "ACT parameters are present in actor.parameters() but require_grad=False; Adam will not update them."
            if act_in_optimizer > 0
            else None
        ),
    }


def _gripper_cable_state_diagnostics(env) -> dict[str, Any]:
    robot = env.unwrapped.scene["robot"]
    joint_names = list(getattr(robot, "joint_names", []))
    joint_pos = getattr(robot.data, "joint_pos", None)
    joints: dict[str, Any] = {}
    if joint_pos is not None:
        for name in joint_names:
            if "finger" in name or "gripper" in name:
                idx = joint_names.index(name)
                joints[name] = float(joint_pos[0, idx].detach().cpu())
    return {
        "gripper_joint_positions_env0": joints,
        "configured_gripper_joint_position": float(args_cli.gripper_joint_position),
        "body_pose": {
            name: {
                "position": None if _body_position_by_name(env, name) is None else _sample_vector(_body_position_by_name(env, name), limit=3),
                "quat_wxyz": None if _body_orientation_by_name(env, name) is None else _sample_vector(_body_orientation_by_name(env, name), limit=4),
            }
            for name in ("gripper_tcp", "sfp_tip_link", "wrist_3_link")
        },
    }


def _randomization_diagnostics(env_cfg: Any) -> dict[str, Any]:
    events = getattr(env_cfg, "events", None)
    randomize_parts = getattr(events, "randomize_board_and_parts", None) if events is not None else None
    randomize_light = getattr(events, "randomize_light", None) if events is not None else None
    reset_joints = getattr(events, "reset_robot_joints", None) if events is not None else None
    return {
        "AIC_ISAAC_RANDOMIZATION_PROFILE": os.environ.get("AIC_ISAAC_RANDOMIZATION_PROFILE", "light"),
        "board_and_parts_params": None if randomize_parts is None else _jsonable(getattr(randomize_parts, "params", {})),
        "light_params": None if randomize_light is None else _jsonable(getattr(randomize_light, "params", {})),
        "reset_joints_params": None if reset_joints is None else _jsonable(getattr(reset_joints, "params", {})),
        "fixed_no_randomization_audit_hint": "Set AIC_ISAAC_RANDOMIZATION_PROFILE=none for easiest fixed-scene diagnostics.",
    }


def _action_components_diagnostics(
    components: dict[str, torch.Tensor],
    *,
    single_action_dim: int,
) -> dict[str, Any]:
    base = components["base_action"]
    raw = components["raw_delta_action"]
    clipped = components["delta_action"]
    final = components["final_action"]
    eps = 1.0e-9
    clipped_fraction = float((raw.sub(clipped).abs() > eps).float().mean().detach().cpu())
    return {
        "base_action_first6": _norm_stats(base[:, :single_action_dim]),
        "raw_adapter_delta_first6": _norm_stats(raw[:, :single_action_dim]),
        "clipped_adapter_delta_first6": _norm_stats(clipped[:, :single_action_dim]),
        "final_action_first6": _norm_stats(final[:, :single_action_dim]),
        "base_action_env0_first12": _sample_vector(base, limit=12),
        "raw_adapter_delta_env0_first12": _sample_vector(raw, limit=12),
        "clipped_adapter_delta_env0_first12": _sample_vector(clipped, limit=12),
        "final_action_env0_first12": _sample_vector(final, limit=12),
        "adapter_clipped_fraction": clipped_fraction,
        "final_minus_base_norm": float((final - base).norm(dim=-1).mean().detach().cpu()),
        "adapter_delta_clip": None if args_cli.adapter_delta_clip is None else float(args_cli.adapter_delta_clip),
        "adapter_scale": float(getattr(args_cli, "adapter_scale", 1.0)) if hasattr(args_cli, "adapter_scale") else None,
    }


def _realized_delta_diagnostics(
    *,
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    requested_tcp_action: torch.Tensor,
    desired_root_action: torch.Tensor | None = None,
    env_action: torch.Tensor,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "requested_tcp_delta_env0": _sample_vector(requested_tcp_action[:, :3], limit=3),
        "requested_tcp_rot_env0": _sample_vector(requested_tcp_action[:, 3:6], limit=3),
        "desired_root_delta_env0": None
        if desired_root_action is None
        else _sample_vector(desired_root_action[:, :3], limit=3),
        "env_action_first6_env0": _sample_vector(env_action[:, :6], limit=6),
        "bodies": {},
    }
    for body_name, before_pos in before.items():
        after_pos = after.get(body_name)
        if before_pos is None or after_pos is None:
            out["bodies"][body_name] = None
            continue
        realized = after_pos - before_pos
        requested = requested_tcp_action[:, :3]
        desired_root = desired_root_action[:, :3] if desired_root_action is not None else None
        ratio = realized / torch.where(requested.abs() < 1.0e-9, torch.full_like(requested, float("nan")), requested)
        root_ratio = (
            None
            if desired_root is None
            else realized / torch.where(desired_root.abs() < 1.0e-9, torch.full_like(desired_root, float("nan")), desired_root)
        )
        out["bodies"][body_name] = {
            "before_env0": _sample_vector(before_pos, limit=3),
            "after_env0": _sample_vector(after_pos, limit=3),
            "realized_delta_env0": _sample_vector(realized, limit=3),
            "realized_delta_norm_mean": float(realized.norm(dim=-1).mean().detach().cpu()),
            "requested_delta_norm_mean": float(requested.norm(dim=-1).mean().detach().cpu()),
            "realized_over_requested_xyz_env0": _sample_vector(ratio, limit=3),
            "realized_over_desired_root_xyz_env0": None if root_ratio is None else _sample_vector(root_ratio, limit=3),
        }
    return out


def _selected_body_positions(env) -> dict[str, torch.Tensor | None]:
    return {
        body_name: _body_position_by_name(env, body_name)
        for body_name in ("wrist_3_link", "gripper_tcp", "sfp_tip_link", CONTROLLED_TCP_BODY)
    }


def _debug_axis_action(
    *,
    step: int,
    batch_size: int,
    action_dim: int,
    single_action_dim: int,
    magnitude: float,
    device: torch.device,
) -> torch.Tensor:
    action = torch.zeros((batch_size, action_dim), dtype=torch.float32, device=device)
    axis_index = (int(step) - 1) % 6
    sign = 1.0 if axis_index < 3 else -1.0
    coord = axis_index % 3
    action[:, coord] = sign * float(magnitude)
    if action_dim > single_action_dim:
        action[:, single_action_dim:] = 0.0
    return action


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


def _straight_through_clamp(x: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    """Clamp in the forward pass while keeping identity gradients for actor training."""
    clipped = x.clamp(min_value, max_value)
    return x + (clipped - x).detach()


def _clip_tcp_translation_norm(action: torch.Tensor, *, single_action_dim: int, max_norm: float) -> torch.Tensor:
    if max_norm <= 0.0:
        return action
    if action.shape[-1] % single_action_dim != 0:
        raise ValueError(
            f"Action dim {action.shape[-1]} is not divisible by single_action_dim={single_action_dim}"
        )
    steps = action.reshape(action.shape[0], action.shape[-1] // single_action_dim, single_action_dim)
    translation = steps[:, :, :3]
    norm = translation.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
    scale = torch.clamp(float(max_norm) / norm, max=1.0)
    clipped_translation = translation * scale
    clipped = torch.cat([clipped_translation, steps[:, :, 3:]], dim=-1).reshape_as(action)
    return clipped


def _clip_tcp_rotation_norm(action: torch.Tensor, *, single_action_dim: int, max_norm: float) -> torch.Tensor:
    if max_norm <= 0.0:
        return action
    if single_action_dim < 6:
        raise ValueError(f"single_action_dim={single_action_dim} does not contain a 3D rotation component")
    if action.shape[-1] % single_action_dim != 0:
        raise ValueError(
            f"Action dim {action.shape[-1]} is not divisible by single_action_dim={single_action_dim}"
        )
    steps = action.reshape(action.shape[0], action.shape[-1] // single_action_dim, single_action_dim)
    rotation = steps[:, :, 3:6]
    norm = rotation.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
    scale = torch.clamp(float(max_norm) / norm, max=1.0)
    clipped_rotation = rotation * scale
    clipped = torch.cat([steps[:, :, :3], clipped_rotation, steps[:, :, 6:]], dim=-1).reshape_as(action)
    return clipped


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
        tcp_translation_action_clip: float | None,
        tcp_rotation_action_clip: float | None,
        action_clip: float | None,
        normalized_state_clip: float | None = None,
        adapter_activation: str = "relu",
        actor_mode: str = "act_adapter",
        state_encoding: str = "none",
        state_encoding_indices: list[int] | tuple[int, ...] = (),
        state_encoding_num_bands: int = 4,
        state_encoding_max_freq: float = 8.0,
        state_encoding_scale: float = 1.0,
    ):
        super().__init__()
        self.act_base = act_base
        self.act_torchscript_path = Path(act_torchscript_path)
        self.act_base_device = torch.device("cpu")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.act_normalizer = ACTRuntimeNormalizer(
            self.act_torchscript_path,
            state_dim=self.state_dim,
            action_dim=self.action_dim // self.action_horizon,
        )
        self.adapter_scale = float(adapter_scale)
        self.adapter_delta_clip = None if adapter_delta_clip is None else float(adapter_delta_clip)
        self.tcp_translation_action_clip = (
            None if tcp_translation_action_clip is None else float(tcp_translation_action_clip)
        )
        self.tcp_rotation_action_clip = None if tcp_rotation_action_clip is None else float(tcp_rotation_action_clip)
        self.action_clip = None if action_clip is None else float(action_clip)
        self.normalized_state_clip = None if normalized_state_clip is None else float(normalized_state_clip)
        if actor_mode not in {"act_adapter", "act_direct"}:
            raise ValueError(f"Unsupported ACT-backed actor mode: {actor_mode}")
        self.actor_mode = actor_mode
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
            zero_final=True,
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -2.0))

    def action_components(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        output_device = obs["state"].device
        act_device = self.act_base_device
        normalized_state = self.act_normalizer.normalize_state(obs["state"])
        if self.normalized_state_clip is not None and self.normalized_state_clip > 0.0:
            clip = float(self.normalized_state_clip)
            normalized_state = normalized_state.clamp(-clip, clip)
        normalized_state = normalized_state.to(act_device)
        normalized_images = {
            key: self.act_normalizer.normalize_image(key, value).to(act_device)
            for key, value in obs["images"].items()
        }
        chunk = self.act_base(
            normalized_state,
            normalized_images["observation.images.center_camera"],
            normalized_images["observation.images.left_camera"],
            normalized_images["observation.images.right_camera"],
        )
        base_action = self.act_normalizer.unnormalize_action(chunk).to(output_device)
        base_action = base_action[:, : self.action_horizon, :].reshape(obs["state"].shape[0], -1)
        encoded_state = self.state_encoder(obs["state"])
        raw_delta_action = self.adapter(torch.cat([encoded_state, base_action], dim=-1))
        if self.actor_mode == "act_adapter" and self.adapter_delta_clip is not None and self.adapter_delta_clip > 0.0:
            delta_action = _straight_through_clamp(
                raw_delta_action,
                -self.adapter_delta_clip,
                self.adapter_delta_clip,
            )
        else:
            delta_action = raw_delta_action
        unclipped_final_action = base_action + self.adapter_scale * delta_action
        translation_clipped_action = (
            _clip_tcp_translation_norm(
                unclipped_final_action,
                single_action_dim=self.action_dim // self.action_horizon,
                max_norm=self.tcp_translation_action_clip,
            )
            if self.tcp_translation_action_clip is not None and self.tcp_translation_action_clip > 0.0
            else unclipped_final_action
        )
        rotation_clipped_action = (
            _clip_tcp_rotation_norm(
                translation_clipped_action,
                single_action_dim=self.action_dim // self.action_horizon,
                max_norm=self.tcp_rotation_action_clip,
            )
            if self.tcp_rotation_action_clip is not None and self.tcp_rotation_action_clip > 0.0
            else translation_clipped_action
        )
        if self.action_clip is not None and self.action_clip > 0.0:
            final_action = rotation_clipped_action.clamp(-self.action_clip, self.action_clip)
        else:
            final_action = rotation_clipped_action
        return {
            "base_action": base_action,
            "raw_delta_action": raw_delta_action,
            "delta_action": delta_action,
            "unclipped_final_action": unclipped_final_action,
            "translation_clipped_action": translation_clipped_action,
            "rotation_clipped_action": rotation_clipped_action,
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


def _torchscript_export_device(path: Path) -> str | None:
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = data.get("torchscript_export_device")
    return str(value) if value is not None else None


def _resolve_act_torchscript_device(path: Path, requested: str, train_device: torch.device) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if train_device.type != "cuda":
            raise ValueError("--act_torchscript_device=cuda requires a CUDA training device")
        return train_device
    exported = _torchscript_export_device(path)
    if exported and exported.startswith("cuda") and train_device.type == "cuda":
        return train_device
    return torch.device("cpu")


def _load_act_base(path: Path, act_device: torch.device) -> torch.jit.ScriptModule:
    module = torch.jit.load(str(path), map_location=act_device).eval()
    for param in module.parameters():
        param.requires_grad = False
    return module


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
        actor_q_weight: float,
        target_action_guide_weight: float,
        target_action_guide_prefix_decay: bool,
        bc_weight: float,
        expert_prior: ExpertActionPrior | None,
        expert_bc_every: int,
        single_action_dim: int,
        actor_update_action_steps: int,
        debug_diagnostics: bool,
        act_torchscript_device: torch.device,
        device: torch.device,
    ):
        self.actor = actor.to(device)
        self.actor.act_base = _load_act_base(self.actor.act_torchscript_path, act_torchscript_device)
        self.actor.act_base_device = act_torchscript_device
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
        self.actor_q_weight = float(actor_q_weight)
        self.target_action_guide_weight = float(target_action_guide_weight)
        self.target_action_guide_prefix_decay = bool(target_action_guide_prefix_decay)
        self.bc_weight = float(bc_weight)
        self.expert_prior = expert_prior
        self.expert_bc_every = max(1, int(expert_bc_every))
        self.single_action_dim = int(single_action_dim)
        self.actor_update_action_steps = max(1, int(actor_update_action_steps))
        self.debug_diagnostics = bool(debug_diagnostics)
        self.update_count = 0
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=adapter_lr)
        self.critic_opt = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr)

    def train_step(self, batch: dict[str, Any], *, update_actor: bool = True) -> dict[str, float]:
        self.update_count += 1
        obs, next_obs = batch["obs"], batch["next_obs"]
        action, reward, done = batch["action"], batch["reward"], batch["done"]
        with torch.no_grad():
            next_action_full = self.actor.mean_action(next_obs)
            next_action = _executed_critic_action(next_action_full, single_action_dim=self.single_action_dim)
            target_q = torch.minimum(self.target_critic1(next_obs, next_action), self.target_critic2(next_obs, next_action))
            td_target = reward + self.gamma * (1.0 - done) * target_q
        q1 = self.critic1(obs, action)
        q2 = self.critic2(obs, action)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = _grad_norm(list(self.critic1.parameters()) + list(self.critic2.parameters()))
        self.critic_opt.step()

        components = self.actor.action_components(obs)
        actor_action_full = components["final_action"]
        actor_action = _executed_critic_action(actor_action_full, single_action_dim=self.single_action_dim)
        base_action = components["base_action"]
        delta_action = components["delta_action"]
        raw_delta_action = components.get("raw_delta_action", delta_action)
        q_actions = []
        max_action_steps = min(
            self.actor_update_action_steps,
            max(1, actor_action_full.shape[-1] // self.single_action_dim),
        )
        for action_idx in range(max_action_steps):
            start = action_idx * self.single_action_dim
            action_i = actor_action_full[:, start : start + self.single_action_dim]
            q_actions.append(torch.minimum(self.critic1(obs, action_i), self.critic2(obs, action_i)))
        actor_q_all = torch.stack(q_actions, dim=0)
        actor_q = actor_q_all.mean(dim=0)
        adapter_penalty = raw_delta_action.norm(dim=-1).mean()
        clipped_adapter_penalty = delta_action.norm(dim=-1).mean()
        act_preservation_loss = (actor_action_full - base_action).norm(dim=-1).mean()
        compute_bc = (
            self.expert_prior is not None
            and self.bc_weight > 0.0
            and self.update_count % self.expert_bc_every == 0
        )
        if compute_bc:
            bc_t0 = time.monotonic()
            expert_action = self.expert_prior.nearest_actions(obs["state"]).detach()
            bc_loss = F.smooth_l1_loss(actor_action_full, expert_action)
            bc_elapsed_ms = (time.monotonic() - bc_t0) * 1000.0
        else:
            bc_loss = torch.zeros((), dtype=actor_action_full.dtype, device=actor_action_full.device)
            bc_elapsed_ms = 0.0
        guide_action = batch.get("guide_action")
        if guide_action is not None and self.target_action_guide_weight > 0.0:
            guide_prefix_steps = min(
                max_action_steps,
                max(1, actor_action_full.shape[-1] // self.single_action_dim),
            )
            actor_action_prefix = actor_action_full[:, : guide_prefix_steps * self.single_action_dim]
            guide_action_prefix = guide_action.repeat(1, guide_prefix_steps)
            guide_action_prefix_steps = guide_action_prefix.reshape(
                guide_action_prefix.shape[0],
                guide_prefix_steps,
                self.single_action_dim,
            )
            if self.target_action_guide_prefix_decay and guide_prefix_steps > 1:
                decay = torch.linspace(
                    1.0,
                    1.0 / float(guide_prefix_steps),
                    guide_prefix_steps,
                    dtype=guide_action_prefix_steps.dtype,
                    device=guide_action_prefix_steps.device,
                ).view(1, guide_prefix_steps, 1)
                guide_action_prefix_steps = guide_action_prefix_steps * decay
                guide_action_prefix = guide_action_prefix_steps.reshape_as(guide_action_prefix)
            actor_action_prefix_steps = actor_action_prefix.reshape(
                actor_action_prefix.shape[0],
                guide_prefix_steps,
                self.single_action_dim,
            )
            guide_translation_loss = F.l1_loss(actor_action_prefix_steps[:, :, :3], guide_action_prefix_steps[:, :, :3])
            guide_rotation_loss = F.l1_loss(actor_action_prefix_steps[:, :, 3:], guide_action_prefix_steps[:, :, 3:])
            guide_loss = guide_translation_loss + 0.1 * guide_rotation_loss
        else:
            guide_prefix_steps = 0
            actor_action_prefix = None
            guide_action_prefix = None
            guide_translation_loss = None
            guide_rotation_loss = None
            guide_loss = torch.zeros((), dtype=actor_action_full.dtype, device=actor_action_full.device)
        bc_loss_weighted = self.bc_weight * bc_loss
        guide_loss_weighted = self.target_action_guide_weight * guide_loss
        q_actor_loss = -actor_q.mean()
        q_actor_loss_weighted = self.actor_q_weight * q_actor_loss
        adapter_penalty_weighted = self.adapter_penalty_weight * adapter_penalty
        act_preservation_loss_weighted = self.act_preservation_weight * act_preservation_loss
        actor_loss = (
            q_actor_loss_weighted
            + adapter_penalty_weighted
            + act_preservation_loss_weighted
            + bc_loss_weighted
            + guide_loss_weighted
        )
        if update_actor:
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            adapter_grad_norm = _grad_norm(self.actor.adapter.parameters())
            self.actor_opt.step()
        else:
            adapter_grad_norm = 0.0
        self._soft_update()
        raw_delta = raw_delta_action
        clipped_fraction = float((raw_delta.sub(delta_action).abs() > 1.0e-9).float().mean().detach().cpu())
        unclipped_final_action = components.get("unclipped_final_action", actor_action_full)
        pre_translation_steps = unclipped_final_action.reshape(
            unclipped_final_action.shape[0],
            unclipped_final_action.shape[-1] // self.single_action_dim,
            self.single_action_dim,
        )[:, :, :3]
        post_translation_steps = actor_action_full.reshape(
            actor_action_full.shape[0],
            actor_action_full.shape[-1] // self.single_action_dim,
            self.single_action_dim,
        )[:, :, :3]
        translation_action_clipped_fraction = float(
            (pre_translation_steps.sub(post_translation_steps).abs() > 1.0e-9).float().mean().detach().cpu()
        )
        rotation_clipped_action = components.get("rotation_clipped_action", actor_action_full)
        pre_rotation_steps = components.get("translation_clipped_action", actor_action_full).reshape(
            actor_action_full.shape[0],
            actor_action_full.shape[-1] // self.single_action_dim,
            self.single_action_dim,
        )[:, :, 3:6]
        post_rotation_steps = rotation_clipped_action.reshape(
            actor_action_full.shape[0],
            actor_action_full.shape[-1] // self.single_action_dim,
            self.single_action_dim,
        )[:, :, 3:6]
        rotation_action_clipped_fraction = float(
            (pre_rotation_steps.sub(post_rotation_steps).abs() > 1.0e-9).float().mean().detach().cpu()
        )
        metrics = {
            "actor_loss": float(actor_loss.detach().cpu()),
            "actor_q_loss": float(q_actor_loss.detach().cpu()),
            "actor_q_loss_weighted": float(q_actor_loss_weighted.detach().cpu()),
            "actor_q_weight": float(self.actor_q_weight),
            "critic_loss": float(critic_loss.detach().cpu()),
            "batch_reward_mean": float(reward.mean().detach().cpu()),
            "batch_reward_std": float(reward.std(unbiased=False).detach().cpu()) if reward.numel() > 1 else 0.0,
            "batch_reward_min": float(reward.min().detach().cpu()),
            "batch_reward_max": float(reward.max().detach().cpu()),
            "q_mean": float(torch.minimum(q1, q2).mean().detach().cpu()),
            "td_target_mean": float(td_target.mean().detach().cpu()),
            "td_target_std": float(td_target.std(unbiased=False).detach().cpu()) if td_target.numel() > 1 else 0.0,
            "q1_mean": float(q1.mean().detach().cpu()),
            "q1_std": float(q1.std(unbiased=False).detach().cpu()) if q1.numel() > 1 else 0.0,
            "q1_min": float(q1.min().detach().cpu()),
            "q1_max": float(q1.max().detach().cpu()),
            "q2_mean": float(q2.mean().detach().cpu()),
            "q2_std": float(q2.std(unbiased=False).detach().cpu()) if q2.numel() > 1 else 0.0,
            "q2_min": float(q2.min().detach().cpu()),
            "q2_max": float(q2.max().detach().cpu()),
            "actor_q_mean": float(actor_q.mean().detach().cpu()),
            "actor_q_action_steps": float(max_action_steps),
            "critic_grad_norm": float(critic_grad_norm),
            "adapter_grad_norm": float(adapter_grad_norm),
            "base_action_norm": float(base_action.norm(dim=-1).mean().detach().cpu()),
            "base_action_abs_max": float(base_action.abs().max().detach().cpu()),
            "base_action_min": float(base_action.min().detach().cpu()),
            "base_action_max": float(base_action.max().detach().cpu()),
            "final_action_norm": float(actor_action.norm(dim=-1).mean().detach().cpu()),
            "final_action_abs_max": float(actor_action.abs().max().detach().cpu()),
            "adapter_delta_norm": float(delta_action.norm(dim=-1).mean().detach().cpu()),
            "adapter_delta_abs_max": float(delta_action.abs().max().detach().cpu()),
            "clipped_adapter_penalty": float(clipped_adapter_penalty.detach().cpu()),
            "raw_adapter_delta_norm": float(
                components.get("raw_delta_action", delta_action).norm(dim=-1).mean().detach().cpu()
            ),
            "raw_adapter_delta_abs_max": float(
                components.get("raw_delta_action", delta_action).abs().max().detach().cpu()
            ),
            "adapter_clipped_fraction": clipped_fraction,
            "tcp_translation_action_clip": (
                0.0
                if self.actor.tcp_translation_action_clip is None
                else float(self.actor.tcp_translation_action_clip)
            ),
            "tcp_translation_action_clipped_fraction": translation_action_clipped_fraction,
            "unclipped_tcp_translation_norm_mean": float(pre_translation_steps.norm(dim=-1).mean().detach().cpu()),
            "clipped_tcp_translation_norm_mean": float(post_translation_steps.norm(dim=-1).mean().detach().cpu()),
            "tcp_rotation_action_clip": (
                0.0 if self.actor.tcp_rotation_action_clip is None else float(self.actor.tcp_rotation_action_clip)
            ),
            "tcp_rotation_action_clipped_fraction": rotation_action_clipped_fraction,
            "unclipped_tcp_rotation_norm_mean": float(pre_rotation_steps.norm(dim=-1).mean().detach().cpu()),
            "clipped_tcp_rotation_norm_mean": float(post_rotation_steps.norm(dim=-1).mean().detach().cpu()),
            "adapter_penalty": float(adapter_penalty.detach().cpu()),
            "adapter_penalty_weighted": float(adapter_penalty_weighted.detach().cpu()),
            "act_preservation_loss": float(act_preservation_loss.detach().cpu()),
            "act_preservation_loss_weighted": float(act_preservation_loss_weighted.detach().cpu()),
            "bc_loss": float(bc_loss.detach().cpu()),
            "bc_loss_weighted": float(bc_loss_weighted.detach().cpu()),
            "bc_weight": float(self.bc_weight),
            "bc_computed": float(compute_bc),
            "bc_every": float(self.expert_bc_every),
            "bc_lookup_ms": float(bc_elapsed_ms),
            "target_action_guide_loss": float(guide_loss.detach().cpu()),
            "target_action_guide_loss_weighted": float(guide_loss_weighted.detach().cpu()),
            "target_action_guide_weight": float(self.target_action_guide_weight),
            "target_action_guide_prefix_decay": float(self.target_action_guide_prefix_decay),
            "target_action_guide_prefix_steps": float(guide_prefix_steps),
            "target_action_guide_prefix_l1": (
                float(F.l1_loss(actor_action_prefix, guide_action_prefix).detach().cpu())
                if actor_action_prefix is not None and guide_action_prefix is not None
                else 0.0
            ),
            "target_action_guide_translation_loss": (
                float(guide_translation_loss.detach().cpu()) if guide_translation_loss is not None else 0.0
            ),
            "target_action_guide_rotation_loss": (
                float(guide_rotation_loss.detach().cpu()) if guide_rotation_loss is not None else 0.0
            ),
            "actor_update_enabled": float(bool(update_actor)),
            "final_minus_act_norm": float((actor_action_full - base_action).norm(dim=-1).mean().detach().cpu()),
            "unclipped_final_minus_act_norm": float(
                (unclipped_final_action - base_action)
                .norm(dim=-1)
                .mean()
                .detach()
                .cpu()
            ),
            "log_std_mean": float(self.actor.log_std.mean().detach().cpu()),
            "critic_action_representation": "first_executed_6d",
            "loss_scale_summary": {
                "actor_q_loss": float(q_actor_loss.detach().cpu()),
                "actor_q_loss_weighted": float(q_actor_loss_weighted.detach().cpu()),
                "adapter_penalty_weighted": float(adapter_penalty_weighted.detach().cpu()),
                "act_preservation_loss_weighted": float(act_preservation_loss_weighted.detach().cpu()),
                "bc_loss_weighted": float(bc_loss_weighted.detach().cpu()),
                "target_action_guide_loss_weighted": float(guide_loss_weighted.detach().cpu()),
                "critic_loss": float(critic_loss.detach().cpu()),
                "batch_reward_mean": float(reward.mean().detach().cpu()),
            },
        }
        if self.debug_diagnostics:
            metrics["diagnostic_action_representations"] = {
                "full_actor_action_shape": list(actor_action_full.shape),
                "critic_actor_action_shape": list(actor_action.shape),
                "replay_action_shape": list(action.shape),
                "target_next_action_shape": list(next_action.shape),
                "full_actor_action_env0_first12": _sample_vector(actor_action_full, limit=12),
                "critic_actor_action_env0_first12": _sample_vector(actor_action, limit=12),
                "replay_action_env0_first12": _sample_vector(action, limit=12),
                "target_next_action_env0_first12": _sample_vector(next_action, limit=12),
            }
        return metrics

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
    act_torchscript_device: torch.device,
    adapter_delta_clip: float | None,
    tcp_translation_action_clip: float | None,
    tcp_rotation_action_clip: float | None,
    action_clip: float | None,
    normalized_state_clip: float | None,
) -> IsaacACTAdapterActor:
    actor_state = checkpoint["actor"]
    hidden_dim, num_layers = _infer_adapter_shape(actor_state)
    offline_cfg, _, warmstart = _checkpoint_training_context(checkpoint)
    adapter_scale = float(warmstart.get("adapter_scale", 1.0))
    act_base = _load_act_base(act_torchscript, act_torchscript_device)
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
        tcp_translation_action_clip=tcp_translation_action_clip,
        tcp_rotation_action_clip=tcp_rotation_action_clip,
        action_clip=action_clip,
        normalized_state_clip=normalized_state_clip,
        adapter_activation=str(offline_cfg.get("adapter_activation", "relu")),
        actor_mode=str(offline_cfg.get("actor_mode", "act_adapter")),
        state_encoding=str(offline_cfg.get("state_encoding", "none")),
        state_encoding_indices=tuple(int(i) for i in offline_cfg.get("state_encoding_indices", ())),
        state_encoding_num_bands=int(offline_cfg.get("state_encoding_num_bands", 4)),
        state_encoding_max_freq=float(offline_cfg.get("state_encoding_max_freq", 8.0)),
        state_encoding_scale=float(offline_cfg.get("state_encoding_scale", 1.0)),
    ).to(device)
    actor.act_base = _load_act_base(actor.act_torchscript_path, act_torchscript_device)
    actor.act_base_device = act_torchscript_device
    own_state = actor.state_dict()
    compatible = {key: value for key, value in actor_state.items() if key in own_state and tuple(value.shape) == tuple(own_state[key].shape)}
    own_state.update(compatible)
    actor.load_state_dict(own_state, strict=True)
    actor.act_base = _load_act_base(actor.act_torchscript_path, act_torchscript_device)
    actor.act_base_device = act_torchscript_device
    return actor


def _act_only_training_context(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if int(args.act_only_action_horizon) > 0:
        action_horizon = int(args.act_only_action_horizon)
    else:
        metadata = _torchscript_metadata(Path(args.act_torchscript))
        action_horizon = int(metadata.get("chunk_size") or 1)
    single_action_dim = int(args.act_only_single_action_dim)
    action_dim = action_horizon * single_action_dim
    offline_cfg = {
        "state_dim": int(args.act_only_state_dim),
        "action_dim": action_dim,
        "action_horizon": action_horizon,
        "camera_keys": list(CAMERA_KEYS),
        "actor_mode": str(args.act_only_actor_mode),
        "actor_update_mode": "online_q_from_act_only",
        "freeze_act": True,
        "critic_image_encoder": "small_conv",
        "critic_arch": "multiplicative",
        "critic_feature_dim": 256,
        "critic_hidden_dim": 256,
        "critic_num_layers": 2,
        "critic_per_camera_dim": 64,
        "critic_layer_norm": False,
        "critic_activation": "gelu",
        "adapter_arch": "mlp",
        "adapter_layer_norm": False,
        "adapter_activation": str(args.act_only_adapter_activation),
        "state_encoding": str(args.act_only_state_encoding),
        "state_encoding_indices": list(args.act_only_state_encoding_indices),
        "state_encoding_num_bands": int(args.act_only_state_encoding_num_bands),
        "state_encoding_max_freq": float(args.act_only_state_encoding_max_freq),
        "state_encoding_scale": float(args.act_only_state_encoding_scale),
    }
    dataset_summary = {
        "source": "act_only_online",
        "state_dim": offline_cfg["state_dim"],
        "single_action_dim": single_action_dim,
        "action_dim": action_dim,
        "action_horizon": action_horizon,
        "camera_keys": list(CAMERA_KEYS),
    }
    warmstart = {
        "mode": "act_only_zero_adapter" if str(args.act_only_actor_mode) == "act_adapter" else "act_only_direct_init_to_act",
        "act_torchscript": str(args.act_torchscript),
        "adapter_scale": 1.0,
        "adapter_delta_clip": args.adapter_delta_clip,
        "initial_delta_norm": 0.0,
        "initial_final_minus_act_norm": 0.0,
    }
    return offline_cfg, dataset_summary, warmstart


def _init_zero_act_adapter_actor(
    args: argparse.Namespace,
    *,
    act_torchscript: Path,
    state_dim: int,
    action_dim: int,
    action_horizon: int,
    device: torch.device,
    act_torchscript_device: torch.device,
    adapter_delta_clip: float | None,
    tcp_translation_action_clip: float | None,
    tcp_rotation_action_clip: float | None,
    action_clip: float | None,
    normalized_state_clip: float | None,
) -> IsaacACTAdapterActor:
    act_base = _load_act_base(act_torchscript, act_torchscript_device)
    actor = IsaacACTAdapterActor(
        act_base=act_base,
        act_torchscript_path=act_torchscript,
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        hidden_dim=int(args.act_only_adapter_hidden_dim),
        num_layers=int(args.act_only_adapter_num_layers),
        adapter_scale=1.0,
        adapter_delta_clip=adapter_delta_clip,
        tcp_translation_action_clip=tcp_translation_action_clip,
        tcp_rotation_action_clip=tcp_rotation_action_clip,
        action_clip=action_clip,
        normalized_state_clip=normalized_state_clip,
        adapter_activation=str(args.act_only_adapter_activation),
        actor_mode=str(args.act_only_actor_mode),
        state_encoding=str(args.act_only_state_encoding),
        state_encoding_indices=tuple(int(i) for i in args.act_only_state_encoding_indices),
        state_encoding_num_bands=int(args.act_only_state_encoding_num_bands),
        state_encoding_max_freq=float(args.act_only_state_encoding_max_freq),
        state_encoding_scale=float(args.act_only_state_encoding_scale),
    ).to(device)
    with torch.no_grad():
        actor.log_std.fill_(-2.0)
    actor.act_base = _load_act_base(actor.act_torchscript_path, act_torchscript_device)
    actor.act_base_device = act_torchscript_device
    return actor


def _save_checkpoint(path: Path, trainer: OnlineSERLTrainer, train_config: dict[str, Any], step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
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
        tmp,
    )
    os.replace(tmp, path)


def _progress_result(stop_reason: str, step: int, updates_done: int) -> dict[str, Any]:
    return {
        "stop_reason": stop_reason,
        "steps_completed": step,
        "updates_done": updates_done,
        "elapsed_minutes": (time.monotonic() - PROCESS_START_TIME) / 60.0,
    }


def _mem_available_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        return None
    return None


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _pack_replay_image(image: torch.Tensor) -> dict[str, Any]:
    image_cpu = image.detach().cpu().contiguous()
    array = image_cpu.numpy()
    raw = array.tobytes(order="C")
    return {
        "__aic_packed_tensor__": True,
        "shape": tuple(int(dim) for dim in image_cpu.shape),
        "dtype": str(array.dtype),
        "data": zlib.compress(raw, level=1),
    }


def _unpack_replay_images(images: list[Any] | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(images, torch.Tensor):
        return images.to(device)
    tensors: list[torch.Tensor] = []
    for image in images:
        if isinstance(image, dict) and image.get("__aic_packed_tensor__"):
            raw = zlib.decompress(image["data"])
            array = np.frombuffer(raw, dtype=np.dtype(image["dtype"])).copy().reshape(tuple(image["shape"]))
            tensors.append(torch.from_numpy(array))
        else:
            tensors.append(image)
    return torch.stack(tensors).to(device)


def _transition_payload(
    *,
    act_obs: dict[str, Any],
    next_act_obs: dict[str, Any],
    action_for_critic: torch.Tensor,
    guide_action: torch.Tensor | None,
    reward: torch.Tensor,
    done: torch.Tensor,
    env_index: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = {
            "obs": {
                "state": act_obs["state"][env_index],
                "images": {key: _pack_replay_image(value[env_index]) for key, value in act_obs["images"].items()},
            },
            "next_obs": {
                "state": next_act_obs["state"][env_index],
                "images": {key: _pack_replay_image(value[env_index]) for key, value in next_act_obs["images"].items()},
            },
            "action": action_for_critic[env_index],
            "reward": reward[env_index],
            "done": done[env_index],
            "metadata": metadata,
    }
    if guide_action is not None:
        payload["guide_action"] = guide_action[env_index]
    return _cpu_tree(payload)


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
    if args_cli.act_only:
        checkpoint_path = Path(args_cli.checkpoint) if args_cli.checkpoint else None
        if checkpoint_path is None:
            checkpoint: dict[str, Any] | None = None
            offline_cfg, dataset_summary, warmstart = _act_only_training_context(args_cli)
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            offline_cfg, dataset_summary, warmstart = _checkpoint_training_context(checkpoint)
    else:
        if not args_cli.checkpoint:
            raise ValueError("--checkpoint is required unless --act_only is set")
        checkpoint_path = Path(args_cli.checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        offline_cfg, dataset_summary, warmstart = _checkpoint_training_context(checkpoint)
    state_dim = int(offline_cfg["state_dim"])
    action_horizon = int(offline_cfg["action_horizon"])
    single_action_dim = int(offline_cfg["action_dim"] // action_horizon)
    critic_action_dim = single_action_dim
    n_action_steps = int(args_cli.n_action_steps)
    if n_action_steps < 1:
        raise ValueError(f"--n_action_steps must be >= 1, got {n_action_steps}")
    if n_action_steps > action_horizon:
        raise ValueError(f"--n_action_steps={n_action_steps} exceeds action_horizon/chunk_size={action_horizon}")

    device = torch.device(args_cli.device)
    act_torchscript_path = Path(args_cli.act_torchscript)
    act_torchscript_device = _resolve_act_torchscript_device(
        act_torchscript_path,
        args_cli.act_torchscript_device,
        device,
    )
    if checkpoint is not None:
        actor = _load_adapter_actor(
            checkpoint,
            act_torchscript=act_torchscript_path,
            state_dim=state_dim,
            action_dim=int(offline_cfg["action_dim"]),
            action_horizon=action_horizon,
            device=device,
            act_torchscript_device=act_torchscript_device,
            adapter_delta_clip=args_cli.adapter_delta_clip,
            tcp_translation_action_clip=args_cli.tcp_translation_action_clip,
            tcp_rotation_action_clip=args_cli.tcp_rotation_action_clip,
            action_clip=args_cli.action_clip,
            normalized_state_clip=args_cli.act_normalized_state_clip,
        )
        print(f"[AIC SERL] Resumed ACT-adapter actor from checkpoint: {checkpoint_path}", flush=True)
    elif args_cli.act_only:
        actor = _init_zero_act_adapter_actor(
            args_cli,
            act_torchscript=act_torchscript_path,
            state_dim=state_dim,
            action_dim=int(offline_cfg["action_dim"]),
            action_horizon=action_horizon,
            device=device,
            act_torchscript_device=act_torchscript_device,
            adapter_delta_clip=args_cli.adapter_delta_clip,
            tcp_translation_action_clip=args_cli.tcp_translation_action_clip,
            tcp_rotation_action_clip=args_cli.tcp_rotation_action_clip,
            action_clip=args_cli.action_clip,
            normalized_state_clip=args_cli.act_normalized_state_clip,
        )
    else:
        raise RuntimeError("Non-ACT-only training requires a checkpoint.")
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
        action_dim=critic_action_dim,
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
        action_dim=critic_action_dim,
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
    if checkpoint is not None:
        print(
            "[AIC SERL][diagnostic] Initializing fresh online critics. "
            "Online SERL critic always consumes the executed first 6D action, "
            f"while checkpoint critics were trained with action_dim={offline_cfg['action_dim']}.",
            flush=True,
        )
    expert_bc_weight = float(args_cli.expert_bc_weight if args_cli.expert_bc_weight is not None else args_cli.bc_weight)
    expert_prior = None
    if args_cli.expert_dataset_root and expert_bc_weight > 0.0:
        expert_prior = ExpertActionPrior(
            dataset_root=Path(args_cli.expert_dataset_root),
            action_horizon=action_horizon,
            state_dim=state_dim,
            action_dim=int(offline_cfg["action_dim"]),
            state_indices=tuple(args_cli.expert_bc_state_indices),
            max_samples=args_cli.expert_bc_max_samples,
            neighbor_chunk=args_cli.expert_bc_neighbor_chunk,
            device=device,
        )
        print(
            f"[AIC SERL] Loaded expert BC prior: {expert_prior.sampled_count}/{expert_prior.count} "
            f"transitions from {expert_prior.dataset_root}",
            flush=True,
        )
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
        actor_q_weight=args_cli.actor_q_weight,
        target_action_guide_weight=args_cli.target_action_guide_weight,
        target_action_guide_prefix_decay=args_cli.target_action_guide_prefix_decay,
        bc_weight=expert_bc_weight,
        expert_prior=expert_prior,
        expert_bc_every=args_cli.expert_bc_every,
        single_action_dim=single_action_dim,
        actor_update_action_steps=n_action_steps,
        debug_diagnostics=args_cli.debug_diagnostics,
        act_torchscript_device=act_torchscript_device,
        device=device,
    )
    if checkpoint is not None and (
        "actor_optimizer" in checkpoint
        or "critic_optimizer" in checkpoint
        or "target_critic1" in checkpoint
        or "target_critic2" in checkpoint
    ):
        print(
            "[AIC SERL][diagnostic] Preserving actor weights only; optimizer and target critic state are fresh "
            "because online Q uses the executed first 6D action.",
            flush=True,
        )

    checkpoint_compatibility = _checkpoint_compatibility_diagnostics(
        actor=trainer.actor,
        offline_cfg=offline_cfg,
        dataset_summary=dataset_summary,
        state_dim=state_dim,
        action_horizon=action_horizon,
        single_action_dim=single_action_dim,
    )
    print(
        "[AIC SERL][diagnostic] checkpoint_compatibility "
        + json.dumps(_jsonable(checkpoint_compatibility), sort_keys=True),
        flush=True,
    )
    print(
        "[AIC SERL][diagnostic] act_freeze "
        + json.dumps(_jsonable(_act_freeze_diagnostics(trainer)), sort_keys=True),
        flush=True,
    )
    if state_dim >= 72 and not bool(args_cli.enable_contact_sensor):
        args_cli.enable_contact_sensor = True
        os.environ["AIC_ISAAC_ENABLE_CONTACT_SENSOR"] = "1"
        print(
            "[AIC SERL][diagnostic] enabling_contact_sensor_for_state_contract "
            + json.dumps(
                {
                    "state_dim": state_dim,
                    "reason": "72D/82D policy state includes contact/recovery features derived from wrist force slots",
                },
                sort_keys=True,
            ),
            flush=True,
        )

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    if float(args_cli.episode_length_s) > 0.0:
        env_cfg.episode_length_s = float(args_cli.episode_length_s)
    # The SERL actor consumes raw camera tensors directly from the sensors. Avoid
    # computing the PPO-specific ResNet feature observation terms, but keep the
    # camera sensors enabled and rendered.
    if bool(args_cli.disable_ppo_resnet_observation_terms):
        env_cfg.observations.policy.center_rgb = None
        env_cfg.observations.policy.left_rgb = None
        env_cfg.observations.policy.right_rgb = None
    env_cfg_arm_action_scale_before = getattr(env_cfg.actions.arm_action, "scale", None)
    env_cfg.actions.arm_action.scale = args_cli.isaac_action_scale
    task_geometry_reward_config = _configure_task_geometry_rewards(env_cfg, args_cli)
    print(
        "[AIC SERL][diagnostic] action_scale_config "
        + json.dumps(
            {
                "args_cli.isaac_action_scale": float(args_cli.isaac_action_scale),
                "env_cfg.actions.arm_action.scale_before_override": env_cfg_arm_action_scale_before,
                "env_cfg.actions.arm_action.scale_after_override": getattr(env_cfg.actions.arm_action, "scale", None),
                "env_cfg.actions.arm_action.body_name": getattr(env_cfg.actions.arm_action, "body_name", None),
            },
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )
    print(
        "[AIC SERL][diagnostic] target_reward_config "
        + json.dumps(_jsonable(task_geometry_reward_config), sort_keys=True),
        flush=True,
    )
    print("[AIC SERL] Creating Isaac env", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[AIC SERL] Isaac env created", flush=True)
    print(
        "[AIC SERL][diagnostic] action_manager "
        + json.dumps(_jsonable(_action_manager_diagnostics(env)), sort_keys=True),
        flush=True,
    )
    print(
        "[AIC SERL][diagnostic] reward_terms "
        + json.dumps(_jsonable(_resolved_reward_terms(env)), sort_keys=True),
        flush=True,
    )
    replay = ReplayBuffer(args_cli.replay_capacity)

    if int(args_cli.debug_audit_steps) > 0:
        run_dir = Path(args_cli.output_dir) / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args_cli.run_name}"
    else:
        run_dir = Path(args_cli.output_dir) / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args_cli.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    train_config = {
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "act_only": bool(args_cli.act_only),
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
            "force_torque_observation_contract": (
                "Base slots 26:29 are wrist/contact-proxy force xyz and 29:32 are torque xyz. "
                "If Isaac exposes no body incoming wrench, contact_forces.net_forces_w fills force xyz "
                "and torque xyz remains zero; contact40 features are computed from the same slots. "
                "Contact-proxy force magnitude is clipped by --isaac_force_observation_clip_n."
            ),
            "isaac_force_observation_clip_n": float(args_cli.isaac_force_observation_clip_n),
            "state_dim": state_dim,
            "state_schema_ranges": _state_schema_ranges(state_dim),
            "state_feature_names": _state_feature_names(state_dim),
            "image_source": "raw_isaac_camera_sensor_rgb_resized_to_3x256x288",
            "swap_rgb_channels": bool(args_cli.swap_rgb_channels),
            "act_torchscript": str(args_cli.act_torchscript),
            "act_torchscript_device": str(act_torchscript_device),
            "action_executed": "chunk_queue_prefix",
            "action_frame_contract": (
                "actor outputs LeRobot/Gazebo gripper-tcp deltas; Isaac execution converts them "
                "to robot-base deltas for DifferentialInverseKinematicsAction"
            ),
            "isaac_action_scale": args_cli.isaac_action_scale,
            "critic_action_representation": "first_executed_6d",
            "critic_action_dim": critic_action_dim,
            "actor_action_dim": int(offline_cfg["action_dim"]),
            "action_horizon": action_horizon,
            "n_action_steps": n_action_steps,
            "single_action_dim": single_action_dim,
            "adapter_delta_clip": args_cli.adapter_delta_clip,
            "tcp_translation_action_clip": args_cli.tcp_translation_action_clip,
            "tcp_rotation_action_clip": args_cli.tcp_rotation_action_clip,
            "action_clip": args_cli.action_clip,
            "actor_q_weight": args_cli.actor_q_weight,
            "actor_update_end_steps": args_cli.actor_update_end_steps,
            "target_action_guide_prefix_decay": args_cli.target_action_guide_prefix_decay,
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
            "episode_length_s": float(getattr(env_cfg, "episode_length_s", 0.0)),
            "episode_config_count": len(EPISODE_CONFIGS),
            "first_episode_config": EPISODE_CONFIGS[0] if EPISODE_CONFIGS else None,
            "task_distribution_scope": (
                "When episode configs are provided, the Isaac reset event stores the current child YAML "
                "per env and the policy task vector/replay metadata read that same assignment. Without "
                "episode configs, the task vector falls back to task_distribution_yaml or CLI task fields."
            ),
            "gripper_joint_position": args_cli.gripper_joint_position,
            "task_geometry_reward": task_geometry_reward_config,
            "expert_bc": {
                "weight": expert_bc_weight,
                "enabled": expert_prior is not None,
                "prior": None if expert_prior is None else expert_prior.report(),
                "every_updates": args_cli.expert_bc_every,
                "loss": "smooth_l1(actor_action, nearest_expert_action_by_selected_state_indices)",
                "scope": (
                    "Small actor regularizer only. It does not change environment rewards "
                    "and is not computed from cheatcode rollouts during training."
                ),
            },
            "image_logging": {
                "save_step_images": args_cli.save_step_images,
                "image_log_every": args_cli.image_log_every,
                "max_logged_image_steps": args_cli.max_logged_image_steps,
                "note": "Replay stores image tensors in memory for training; PNG logging is for audit/debug artifacts.",
            },
            "terminal_handling": {
                "treat_time_limit_truncation_as_terminal": bool(args_cli.treat_time_limit_truncation_as_terminal),
                "td_bootstrap_done": (
                    "terminated_or_truncated"
                    if args_cli.treat_time_limit_truncation_as_terminal
                    else "terminated_only"
                ),
            },
            "frequency": _frequency_diagnostics(
                env_cfg,
                env,
                action_horizon=action_horizon,
                single_action_dim=single_action_dim,
            ),
            "act_chunk_inference": _act_chunk_inference_diagnostics(offline_cfg),
            "checkpoint_compatibility": checkpoint_compatibility,
            "checkpointing": {
                "save_every_steps": args_cli.save_every_steps,
                "periodic_dir": "checkpoints",
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
    diagnostics_enabled = bool(args_cli.debug_diagnostics or int(args_cli.debug_audit_steps) > 0)
    diagnostics_every = max(1, int(args_cli.diagnostics_every))
    audit_log_path = run_dir / "audit_log.jsonl"
    diagnostics_summary_path = run_dir / "diagnostics_summary.json"
    if diagnostics_enabled:
        initial_act_obs = _act_obs_from_env(
            env,
            policy_obs,
            current_images,
            args_cli,
            device=device,
            state_dim=state_dim,
        )
        initial_summary = {
            "mode": "initial",
            "debug_audit_steps": int(args_cli.debug_audit_steps),
            "audit_act_only": bool(args_cli.audit_act_only),
            "audit_zero_adapter": bool(args_cli.audit_zero_adapter),
            "critic_action_representation": "first_executed_6d",
            "action_scale": {
                "args_cli.isaac_action_scale": float(args_cli.isaac_action_scale),
                "env_cfg_before_override": env_cfg_arm_action_scale_before,
                "env_cfg_after_override": getattr(env_cfg.actions.arm_action, "scale", None),
                "action_manager": _action_manager_diagnostics(env),
            },
            "reward_resolution": {
                "target_reward_arg": task_geometry_reward_config,
                "reward_weights_before_env_creation": _jsonable({
                    name: getattr(getattr(env_cfg.rewards, name), "weight", None)
                    for name in vars(env_cfg.rewards)
                    if not name.startswith("_") and hasattr(getattr(env_cfg.rewards, name), "weight")
                }),
                "resolved_terms": _resolved_reward_terms(env),
                "all_reward_weights_after_env_creation": _reward_weight_diagnostics(env),
                "pose_distances": _pose_reward_diagnostics(env, task_geometry_reward_config),
                "insertion_geometry": _insertion_geometry_diagnostics(env, task_geometry_reward_config),
                "scene_assets": _scene_asset_pose_diagnostics(env, task_geometry_reward_config),
                "target_source": _target_source_diagnostics(env, task_geometry_reward_config),
            },
            "reset_control_body": _reset_control_body_diagnostics(env, task_geometry_reward_config),
            "task_vector_scene_match": _episode_scene_diagnostics(env, task_geometry_reward_config, device=device),
            "quaternion_convention": _quaternion_convention_diagnostics(env, task_geometry_reward_config),
            "force_wrench": _force_wrench_diagnostics(env),
            "frequency": _frequency_diagnostics(
                env_cfg,
                env,
                action_horizon=action_horizon,
                single_action_dim=single_action_dim,
            ),
            "act_chunk_inference": _act_chunk_inference_diagnostics(offline_cfg),
            "checkpoint_compatibility": checkpoint_compatibility,
            "act_freeze": _act_freeze_diagnostics(trainer),
            "gripper_cable_state": _gripper_cable_state_diagnostics(env),
            "randomization": _randomization_diagnostics(env_cfg),
            "lateral_reward": {
                "weight": float(task_geometry_reward_config.get("lateral_weight", 0.0)),
                "frame": "target/port frame",
                "warning": "positive lateral weight rewards lateral error" if float(task_geometry_reward_config.get("lateral_weight", 0.0)) > 0.0 else None,
            },
            "observations": _observation_diagnostics(
                obs=obs,
                policy_obs=policy_obs,
                act_obs=initial_act_obs,
                actor=trainer.actor,
                images=current_images,
                env=env,
                expert_prior=expert_prior,
            ),
            "stage_c_process": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch_device": str(device),
                "torch_cuda_current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
                "seed": int(args_cli.seed),
                "output_dir": str(run_dir),
                "checkpoint_latest_path": str(run_dir / "checkpoint_latest.pt"),
                "periodic_checkpoint_dir": str(run_dir / "checkpoints"),
                "run_name": str(args_cli.run_name),
            },
        }
        diagnostics_summary_path.write_text(
            json.dumps(_jsonable(initial_summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            "[AIC SERL][diagnostic] initial_summary " + json.dumps(_jsonable(initial_summary), sort_keys=True),
            flush=True,
        )
    updates_done = 0
    last_metrics: dict[str, float] = {}
    stop_reason = "max_steps"
    max_loop_steps = int(args_cli.debug_audit_steps) if int(args_cli.debug_audit_steps) > 0 else int(args_cli.steps)
    queued_policy_actions = torch.empty((policy_obs.shape[0], 0, single_action_dim), dtype=torch.float32, device=device)
    queued_action_components: dict[str, torch.Tensor] | None = None
    queued_action_chunk: torch.Tensor | None = None
    for step in range(1, max_loop_steps + 1):
        step_start_time = time.monotonic()
        if STOP_REQUESTED is not None:
            stop_reason = STOP_REQUESTED
            print(f"[AIC SERL] Stop requested before step {step}; saving and exiting.", flush=True)
            break
        if args_cli.ram_watchdog_min_available_gb > 0.0:
            mem_available_gb = _mem_available_gb()
            if mem_available_gb is not None and mem_available_gb < args_cli.ram_watchdog_min_available_gb:
                stop_reason = f"ram_watchdog_{mem_available_gb:.1f}gb_available"
                print(
                    f"[AIC SERL] RAM watchdog stop before step {step}: "
                    f"MemAvailable={mem_available_gb:.1f} GiB",
                    flush=True,
                )
                break
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
        constant_audit_action = getattr(args_cli, "debug_audit_constant_action", None)
        constant_audit_enabled = int(args_cli.debug_audit_steps) > 0 and constant_audit_action is not None
        axis_audit_enabled = (
            int(args_cli.debug_audit_steps) > 0
            and not constant_audit_enabled
            and float(args_cli.debug_audit_axis_magnitude) > 0.0
        )
        recompute_chunk = queued_policy_actions.shape[1] == 0
        if recompute_chunk:
            with torch.no_grad():
                if constant_audit_enabled:
                    axis_action = torch.zeros(
                        (policy_obs.shape[0], int(offline_cfg["action_dim"])),
                        dtype=torch.float32,
                        device=device,
                    )
                    constant = torch.tensor(
                        constant_audit_action,
                        dtype=torch.float32,
                        device=device,
                    ).view(1, single_action_dim)
                    for action_idx in range(n_action_steps):
                        start = action_idx * single_action_dim
                        axis_action[:, start : start + single_action_dim] = constant
                    action_components = {
                        "base_action": axis_action,
                        "raw_delta_action": torch.zeros_like(axis_action),
                        "delta_action": torch.zeros_like(axis_action),
                        "final_action": axis_action,
                    }
                    action_chunk = axis_action
                elif axis_audit_enabled:
                    axis_action = _debug_axis_action(
                        step=step,
                        batch_size=policy_obs.shape[0],
                        action_dim=int(offline_cfg["action_dim"]),
                        single_action_dim=single_action_dim,
                        magnitude=float(args_cli.debug_audit_axis_magnitude),
                        device=device,
                    )
                    action_components = {
                        "base_action": axis_action,
                        "raw_delta_action": torch.zeros_like(axis_action),
                        "delta_action": torch.zeros_like(axis_action),
                        "final_action": axis_action,
                    }
                    action_chunk = axis_action
                else:
                    action_components = trainer.actor.action_components(act_obs)
                    if args_cli.audit_act_only or args_cli.audit_zero_adapter:
                        action_chunk = action_components["base_action"]
                    else:
                        action_chunk = action_components["final_action"]
            queued_action_components = action_components
            queued_action_chunk = action_chunk
            queued_policy_actions = action_chunk.reshape(policy_obs.shape[0], action_horizon, single_action_dim)[
                :, :n_action_steps
            ].clone()
        else:
            action_components = queued_action_components
            action_chunk = queued_action_chunk
        _timing_log(args_cli.debug_timing and step == 1, "actor_forward", t0)
        if action_components is None or action_chunk is None:
            raise RuntimeError("Action chunk queue was empty without available action components")
        policy_tcp_action = queued_policy_actions[:, 0, :]
        actor_policy_tcp_action = policy_tcp_action.clone()
        queued_policy_actions = queued_policy_actions[:, 1:, :]
        guide_action_for_transition = None
        effective_guide_collect_blend = 0.0
        guide_needed = (
            float(args_cli.target_action_guide_weight) > 0.0
            or float(args_cli.target_action_guide_collect_blend) > 0.0
        )
        if guide_needed:
            guide_action_for_transition = _target_guided_policy_action(
                env,
                task_geometry_reward_config,
                step_size=float(args_cli.target_action_guide_step_size),
                axial_step_size=float(args_cli.target_action_guide_axial_step_size),
                lateral_switch_m=float(args_cli.target_action_guide_lateral_switch_m),
                axial_blend_lateral_m=float(args_cli.target_action_guide_axial_blend_lateral_m),
                action_frame=str(args_cli.tcp_action_frame),
                device=device,
            )
        collect_steps = int(args_cli.target_action_guide_collect_steps)
        collect_blend = float(args_cli.target_action_guide_collect_blend)
        if (
            guide_action_for_transition is not None
            and collect_blend > 0.0
            and (collect_steps <= 0 or step <= collect_steps)
        ):
            blend = min(max(collect_blend, 0.0), 1.0)
            if bool(args_cli.target_action_guide_collect_decay) and collect_steps > 0:
                blend *= max(0.0, 1.0 - (float(step) - 1.0) / max(float(collect_steps), 1.0))
            effective_guide_collect_blend = blend
            policy_tcp_action = (1.0 - blend) * policy_tcp_action + blend * guide_action_for_transition
        t0 = time.monotonic()
        desired_root_action = _tcp_delta_action_to_isaac_base_action(
            env,
            policy_tcp_action,
            action_frame=str(args_cli.tcp_action_frame),
            apply_ik_sign_fix=False,
        )
        env_action = _tcp_delta_action_to_isaac_base_action(
            env,
            policy_tcp_action,
            action_frame=str(args_cli.tcp_action_frame),
            apply_ik_sign_fix=True,
        )
        _timing_log(args_cli.debug_timing and step == 1, "tcp_to_isaac_action", t0)
        t0 = time.monotonic()
        before_positions = _selected_body_positions(env) if diagnostics_enabled and (step == 1 or step % diagnostics_every == 0) else {}
        before_orientations = _selected_body_orientations(env) if before_positions else {}
        before_target = _target_position_from_reward_config(env, task_geometry_reward_config) if before_positions else None
        episode_metadata_before = [_episode_metadata(env, idx) for idx in range(policy_obs.shape[0])]
        episode_length_before = getattr(env.unwrapped, "episode_length_buf", None)
        episode_length_before_cpu = None if episode_length_before is None else episode_length_before.detach().cpu().clone()
        previous_images_for_diag = current_images
        next_obs, reward, terminated, truncated, _ = env.step(env_action)
        after_positions = _selected_body_positions(env) if before_positions else {}
        after_orientations = _selected_body_orientations(env) if before_positions else {}
        after_target = _target_position_from_reward_config(env, task_geometry_reward_config) if before_positions else None
        episode_length_after = getattr(env.unwrapped, "episode_length_buf", None)
        episode_length_after_cpu = None if episode_length_after is None else episode_length_after.detach().cpu().clone()
        force_metrics = _force_delta_metrics(env, device=device)
        reward_term_metrics = _reward_term_metrics(env, device=device)
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
        done_for_bootstrap_bool = (
            torch.logical_or(terminated, truncated)
            if args_cli.treat_time_limit_truncation_as_terminal
            else terminated
        )
        done = done_for_bootstrap_bool.float().reshape(-1, 1).to(device)
        reward = reward.reshape(-1, 1).to(device)
        action_for_critic = policy_tcp_action
        guide_action = guide_action_for_transition
        for env_index in range(policy_obs.shape[0]):
            metadata = _episode_metadata(env, env_index)
            metadata["inserted_env_step"] = int(step)
            metadata["episode_before_step"] = episode_metadata_before[env_index]
            metadata["terminated"] = bool(terminated[env_index].detach().cpu())
            metadata["truncated"] = bool(truncated[env_index].detach().cpu())
            metadata["done_for_bootstrap"] = bool(done_for_bootstrap_bool[env_index].detach().cpu())
            if after_target is not None and after_positions.get(CONTROLLED_TCP_BODY) is not None:
                metadata["distance_to_target_after"] = float(
                    torch.norm(after_positions[CONTROLLED_TCP_BODY][env_index] - after_target[env_index]).detach().cpu()
                )
            transition = _transition_payload(
                act_obs=act_obs,
                next_act_obs=next_act_obs,
                action_for_critic=action_for_critic,
                guide_action=guide_action,
                reward=reward,
                done=done,
                env_index=env_index,
                metadata=metadata,
            )
            replay.append(transition)
        policy_obs = next_policy_obs
        current_images = next_images

        if (
            int(args_cli.debug_audit_steps) <= 0
            and step % max(1, int(args_cli.update_every_steps)) == 0
            and len(replay) >= args_cli.batch_size
            and len(replay) >= args_cli.warmup_steps
            and updates_done < args_cli.updates
        ):
            t0 = time.monotonic()
            batch = replay.sample(args_cli.batch_size, device)
            _timing_log(args_cli.debug_timing and step == 1, "sample_replay", t0)
            t0 = time.monotonic()
            actor_update_end_steps = int(args_cli.actor_update_end_steps)
            actor_update_enabled = step >= int(args_cli.actor_update_start_steps) and (
                actor_update_end_steps <= 0 or step <= actor_update_end_steps
            )
            last_metrics = trainer.train_step(batch, update_actor=actor_update_enabled)
            _timing_log(args_cli.debug_timing and step == 1, "train_step", t0)
            updates_done += 1
            if diagnostics_enabled and (
                float(last_metrics.get("adapter_clipped_fraction", 0.0)) > 0.5
                or (
                    updates_done < 500
                    and float(last_metrics.get("final_minus_act_norm", 0.0)) > max(float(args_cli.adapter_delta_clip), 1.0e-6) * 3.0
                )
                or abs(float(last_metrics.get("q_mean", 0.0))) > 50.0
            ):
                print(
                    "[AIC SERL][diagnostic][warning] "
                    + json.dumps(
                        {
                            "step": step,
                            "updates_done": updates_done,
                            "adapter_clipped_fraction": last_metrics.get("adapter_clipped_fraction"),
                            "final_minus_act_norm": last_metrics.get("final_minus_act_norm"),
                            "q_mean": last_metrics.get("q_mean"),
                            "reward_mean": float(reward.mean().detach().cpu()),
                            "replay_size": len(replay),
                        },
                        sort_keys=True,
                        default=str,
                    ),
                    flush=True,
                )

        diagnostic_row: dict[str, Any] = {}
        if diagnostics_enabled and (step == 1 or step % diagnostics_every == 0):
            distance_before_after: dict[str, Any] = {}
            if before_target is not None and after_target is not None:
                for body_name, before_pos in before_positions.items():
                    after_pos = after_positions.get(body_name)
                    if before_pos is None or after_pos is None:
                        continue
                    distance_before_after[body_name] = {
                        "before_distance_env0": float(torch.norm(before_pos[0] - before_target[0]).detach().cpu()),
                        "after_distance_env0": float(torch.norm(after_pos[0] - after_target[0]).detach().cpu()),
                        "delta_distance_env0": float(
                            (torch.norm(after_pos[0] - after_target[0]) - torch.norm(before_pos[0] - before_target[0]))
                            .detach()
                            .cpu()
                        ),
                    }
            multi_env_rows: list[dict[str, Any]] = []
            for env_index in range(min(int(policy_obs.shape[0]), 8)):
                multi_env_rows.append(
                    {
                        "env_id": env_index,
                        "episode_before_step": episode_metadata_before[env_index],
                        "episode": _episode_metadata(env, env_index),
                        "task_vector_tail10": _sample_vector(next_act_obs["state"][env_index : env_index + 1, -10:], limit=10),
                        "reward_target_pose": (
                            None
                            if after_target is None
                            else _sample_vector(after_target[env_index : env_index + 1], limit=3)
                        ),
                        "action_first6": _sample_vector(policy_tcp_action[env_index : env_index + 1], limit=6),
                        "env_action_first6": _sample_vector(env_action[env_index : env_index + 1], limit=6),
                        "reward": float(reward[env_index].detach().cpu()),
                        "terminated": bool(terminated[env_index].detach().cpu()),
                        "truncated": bool(truncated[env_index].detach().cpu()),
                        "done_for_bootstrap": bool(done_for_bootstrap_bool[env_index].detach().cpu()),
                        "episode_changed_after_step": episode_metadata_before[env_index].get("episode_id")
                        != _episode_metadata(env, env_index).get("episode_id"),
                    }
                )
            diagnostic_row = {
                "diagnostics": {
                    "action_components": _action_components_diagnostics(
                        action_components,
                        single_action_dim=single_action_dim,
                    ),
                    "action_scale_realization": _realized_delta_diagnostics(
                        before=before_positions,
                        after=after_positions,
                        requested_tcp_action=policy_tcp_action,
                        desired_root_action=desired_root_action,
                        env_action=env_action,
                    )
                    if before_positions
                    else {},
                    "rotation_realization": _realized_rotation_diagnostics(
                        before=before_orientations,
                        after=after_orientations,
                    )
                    if before_orientations
                    else {},
                    "reward_pose": _pose_reward_diagnostics(env, task_geometry_reward_config),
                    "insertion_geometry": _insertion_geometry_diagnostics(env, task_geometry_reward_config),
                    "reward_resolution": {
                        "resolved_terms": _resolved_reward_terms(env),
                        "all_reward_weights_after_env_creation": _reward_weight_diagnostics(env),
                        "scene_assets": _scene_asset_pose_diagnostics(env, task_geometry_reward_config),
                        "target_source": _target_source_diagnostics(env, task_geometry_reward_config),
                    },
                    "reset_control_body": _reset_control_body_diagnostics(env, task_geometry_reward_config),
                    "task_vector_scene_match": _episode_scene_diagnostics(env, task_geometry_reward_config, device=device),
                    "quaternion_convention": _quaternion_convention_diagnostics(env, task_geometry_reward_config),
                    "force_wrench": _force_wrench_diagnostics(env),
                    "lateral_reward": {
                        "weight": float(task_geometry_reward_config.get("lateral_weight", 0.0)),
                        "frame": "target/port frame",
                        "warning": "positive lateral weight rewards lateral error" if float(task_geometry_reward_config.get("lateral_weight", 0.0)) > 0.0 else None,
                    },
                    "distance_before_after": distance_before_after,
                    "progress_off_by_one": {
                        "episode_length_before_env0": None
                        if episode_length_before_cpu is None
                        else int(episode_length_before_cpu[0].item()),
                        "episode_length_after_env0": None
                        if episode_length_after_cpu is None
                        else int(episode_length_after_cpu[0].item()),
                        "reset_flag_after_step_env0": None
                        if episode_length_after_cpu is None
                        else bool(int(episode_length_after_cpu[0].item()) <= 1),
                        "body_distance_delta_to_target": distance_before_after,
                        "target_distance_progress_term": reward_term_metrics.get("reward_terms_mean", {}).get(
                            "target_distance_progress"
                        ),
                    },
                    "previous_action": _previous_action_diagnostics(
                        state=next_act_obs["state"],
                        policy_tcp_action=policy_tcp_action,
                        env_action=env_action,
                        env=env,
                    ),
                    "camera_freshness": _camera_freshness_diagnostics(previous_images_for_diag, next_images),
                    "action_representations": {
                        "full_actor_action_shape": list(action_chunk.shape),
                        "env_action_shape": list(env_action.shape),
                        "replay_action_shape": list(action_for_critic.shape),
                        "critic_action_representation": "first_executed_6d",
                        "chunk_size": action_horizon,
                        "n_action_steps": n_action_steps,
                        "chunk_recomputed_this_step": recompute_chunk,
                        "queued_actions_remaining_after_step": int(queued_policy_actions.shape[1]),
                        "tcp_action_frame": args_cli.tcp_action_frame,
                        "debug_audit_axis_magnitude": float(args_cli.debug_audit_axis_magnitude),
                        "debug_audit_constant_action": constant_audit_action,
                        "full_actor_action_env0_first12": _sample_vector(action_chunk, limit=12),
                        "actor_policy_tcp_action_env0_first6": _sample_vector(actor_policy_tcp_action, limit=6),
                        "guide_action_env0_first6": None
                        if guide_action_for_transition is None
                        else _sample_vector(guide_action_for_transition, limit=6),
                        "target_action_guide_collect_blend_effective": float(effective_guide_collect_blend),
                        "env_action_env0_first6": _sample_vector(env_action, limit=6),
                        "replay_action_env0_first12": _sample_vector(action_for_critic, limit=12),
                    },
                    "replay_storage": replay.diagnostic_snapshot(),
                    "replay_age_distribution": replay.age_diagnostics(current_step=step),
                    "multi_env_episode_metadata": multi_env_rows,
                    "gripper_cable_state": _gripper_cable_state_diagnostics(env),
                    "observations": _observation_diagnostics(
                        obs=next_obs,
                        policy_obs=next_policy_obs,
                        act_obs=next_act_obs,
                        actor=trainer.actor,
                        images=next_images,
                        env=env,
                        expert_prior=expert_prior,
                    ),
                    "done_mean": float(done.mean().detach().cpu()),
                    "done_for_bootstrap_mean": float(done.mean().detach().cpu()),
                    "terminated_mean": float(terminated.float().mean().detach().cpu()),
                    "truncated_mean": float(truncated.float().mean().detach().cpu()),
                    "treat_time_limit_truncation_as_terminal": bool(args_cli.treat_time_limit_truncation_as_terminal),
                }
            }
            if args_cli.debug_audit_steps > 0:
                with audit_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(_jsonable({"step": step, **diagnostic_row}), sort_keys=True) + "\n")

        row = {
            "step": step,
            "updates_done": updates_done,
            "replay_size": len(replay),
            "reward_mean": float(reward.mean().detach().cpu()),
            "force_norm_mean": float(force_metrics["force_norm"].mean().detach().cpu()),
            "force_delta_norm_mean": float(force_metrics["force_delta_norm"].mean().detach().cpu()),
            "force_delta_penalty_mean": float(force_metrics["force_delta_penalty"].mean().detach().cpu()),
            "force_source": str(force_metrics.get("force_source", "unknown")),
            "force_body": str(force_metrics.get("force_body", FORCE_WRENCH_BODY)),
            "target_action_guide_collect_blend_effective": float(effective_guide_collect_blend),
            "actor_policy_tcp_action_norm_mean": float(torch.norm(actor_policy_tcp_action, dim=1).mean().detach().cpu()),
            "executed_policy_tcp_action_norm_mean": float(torch.norm(policy_tcp_action, dim=1).mean().detach().cpu()),
            "guide_action_norm_mean": (
                None
                if guide_action_for_transition is None
                else float(torch.norm(guide_action_for_transition, dim=1).mean().detach().cpu())
            ),
            "actor_policy_tcp_action_translation_norm_mean": float(
                torch.norm(actor_policy_tcp_action[:, :3], dim=1).mean().detach().cpu()
            ),
            "actor_policy_tcp_action_rotation_norm_mean": float(
                torch.norm(actor_policy_tcp_action[:, 3:], dim=1).mean().detach().cpu()
            ),
            "executed_policy_tcp_action_translation_norm_mean": float(
                torch.norm(policy_tcp_action[:, :3], dim=1).mean().detach().cpu()
            ),
            "executed_policy_tcp_action_rotation_norm_mean": float(
                torch.norm(policy_tcp_action[:, 3:], dim=1).mean().detach().cpu()
            ),
            "guide_action_translation_norm_mean": (
                None
                if guide_action_for_transition is None
                else float(torch.norm(guide_action_for_transition[:, :3], dim=1).mean().detach().cpu())
            ),
            "guide_action_rotation_norm_mean": (
                None
                if guide_action_for_transition is None
                else float(torch.norm(guide_action_for_transition[:, 3:], dim=1).mean().detach().cpu())
            ),
            "actor_to_guide_l1_mean": (
                None
                if guide_action_for_transition is None
                else float(torch.mean(torch.abs(actor_policy_tcp_action - guide_action_for_transition)).detach().cpu())
            ),
            "actor_to_guide_translation_l1_mean": (
                None
                if guide_action_for_transition is None
                else float(
                    torch.mean(torch.abs(actor_policy_tcp_action[:, :3] - guide_action_for_transition[:, :3]))
                    .detach()
                    .cpu()
                )
            ),
            "actor_to_guide_rotation_l1_mean": (
                None
                if guide_action_for_transition is None
                else float(
                    torch.mean(torch.abs(actor_policy_tcp_action[:, 3:] - guide_action_for_transition[:, 3:]))
                    .detach()
                    .cpu()
                )
            ),
            "actor_to_guide_dot_mean": (
                None
                if guide_action_for_transition is None
                else float((actor_policy_tcp_action * guide_action_for_transition).sum(dim=1).mean().detach().cpu())
            ),
            "actor_to_guide_cosine_mean": (
                None
                if guide_action_for_transition is None
                else float(
                    F.cosine_similarity(
                        actor_policy_tcp_action,
                        guide_action_for_transition,
                        dim=1,
                        eps=1.0e-8,
                    )
                    .mean()
                    .detach()
                    .cpu()
                )
            ),
            "actor_to_guide_translation_cosine_mean": (
                None
                if guide_action_for_transition is None
                else float(
                    F.cosine_similarity(
                        actor_policy_tcp_action[:, :3],
                        guide_action_for_transition[:, :3],
                        dim=1,
                        eps=1.0e-8,
                    )
                    .mean()
                    .detach()
                    .cpu()
                )
            ),
            "actor_to_guide_rotation_cosine_mean": (
                None
                if guide_action_for_transition is None
                else float(
                    F.cosine_similarity(
                        actor_policy_tcp_action[:, 3:],
                        guide_action_for_transition[:, 3:],
                        dim=1,
                        eps=1.0e-8,
                    )
                    .mean()
                    .detach()
                    .cpu()
                )
            ),
            "executed_to_guide_dot_mean": (
                None
                if guide_action_for_transition is None
                else float((policy_tcp_action * guide_action_for_transition).sum(dim=1).mean().detach().cpu())
            ),
            "executed_to_guide_cosine_mean": (
                None
                if guide_action_for_transition is None
                else float(
                    F.cosine_similarity(policy_tcp_action, guide_action_for_transition, dim=1, eps=1.0e-8)
                    .mean()
                    .detach()
                    .cpu()
                )
            ),
            "executed_to_guide_translation_cosine_mean": (
                None
                if guide_action_for_transition is None
                else float(
                    F.cosine_similarity(
                        policy_tcp_action[:, :3],
                        guide_action_for_transition[:, :3],
                        dim=1,
                        eps=1.0e-8,
                    )
                    .mean()
                    .detach()
                    .cpu()
                )
            ),
            "executed_to_guide_rotation_cosine_mean": (
                None
                if guide_action_for_transition is None
                else float(
                    F.cosine_similarity(policy_tcp_action[:, 3:], guide_action_for_transition[:, 3:], dim=1, eps=1.0e-8)
                    .mean()
                    .detach()
                    .cpu()
                )
            ),
            "executed_minus_actor_l1_mean": float(
                torch.mean(torch.abs(policy_tcp_action - actor_policy_tcp_action)).detach().cpu()
            ),
            "executed_minus_actor_translation_l1_mean": float(
                torch.mean(torch.abs(policy_tcp_action[:, :3] - actor_policy_tcp_action[:, :3])).detach().cpu()
            ),
            "executed_minus_actor_rotation_l1_mean": float(
                torch.mean(torch.abs(policy_tcp_action[:, 3:] - actor_policy_tcp_action[:, 3:])).detach().cpu()
            ),
            "episodes": [_episode_metadata(env, idx) for idx in range(policy_obs.shape[0])],
            "terminated_mean": float(terminated.float().mean().detach().cpu()),
            "truncated_mean": float(truncated.float().mean().detach().cpu()),
            "done_for_bootstrap_mean": float(done.mean().detach().cpu()),
            "treat_time_limit_truncation_as_terminal": bool(args_cli.treat_time_limit_truncation_as_terminal),
            "step_wall_s": time.monotonic() - step_start_time,
            "env_steps_per_s": float(policy_obs.shape[0]) / max(time.monotonic() - step_start_time, 1.0e-9),
            **reward_term_metrics,
            **last_metrics,
            **diagnostic_row,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
        if args_cli.log_every > 0 and (step == 1 or step % args_cli.log_every == 0):
            print(
                f"[AIC SERL] step={step} updates={updates_done} replay={len(replay)} "
                f"reward={row['reward_mean']:.6f} step_wall={time.monotonic() - step_start_time:.3f}s",
                flush=True,
            )
        if args_cli.save_every_steps > 0 and step % args_cli.save_every_steps == 0:
            periodic_config = {
                **train_config,
                "result": _progress_result("periodic_checkpoint", step, updates_done),
            }
            periodic_path = run_dir / "checkpoints" / f"checkpoint_{step:06d}.pt"
            _save_checkpoint(periodic_path, trainer, periodic_config, step)
            print(f"Wrote periodic online SERL checkpoint: {periodic_path}", flush=True)
        if args_cli.save_latest_every_steps > 0 and step % args_cli.save_latest_every_steps == 0:
            latest_config = {
                **train_config,
                "result": _progress_result("latest_checkpoint", step, updates_done),
            }
            _save_checkpoint(run_dir / "checkpoint_latest.pt", trainer, latest_config, step)
            print(f"Wrote latest online SERL checkpoint: {run_dir / 'checkpoint_latest.pt'}", flush=True)
        if int(args_cli.debug_audit_steps) <= 0 and updates_done >= args_cli.updates:
            stop_reason = "target_updates"
            break

    if int(args_cli.debug_audit_steps) > 0 and stop_reason == "max_steps":
        stop_reason = "debug_audit_complete"
    final_step = step if stop_reason != "max_wall_time" else max(step - 1, 0)
    train_config["result"] = _progress_result(stop_reason, final_step, updates_done)
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
