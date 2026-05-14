#!/usr/bin/env python3
"""Probe AIC Isaac target-geometry rewards with a privileged oracle controller."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="AIC-Task-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=80)
parser.add_argument("--task_family", choices=["sfp_to_nic", "sc_to_sc"], default="sfp_to_nic")
parser.add_argument("--target_port_index", type=int, default=0)
parser.add_argument("--target_body", default="sfp_tip_link")
parser.add_argument("--max_delta", type=float, default=0.02)
parser.add_argument("--max_rotation", type=float, default=0.08)
parser.add_argument("--descent_start_step", type=int, default=90)
parser.add_argument("--approach_offset", type=float, nargs=3, default=None)
parser.add_argument("--insert_offset", type=float, nargs=3, default=None)
parser.add_argument("--record_cameras", action="store_true")
parser.add_argument("--video_dir", type=Path)
parser.add_argument("--video_fps", type=float, default=20.0)
parser.add_argument("--distance_std", type=float, default=0.02)
parser.add_argument("--close_sigma", type=float, default=0.006)
parser.add_argument("--reaching_threshold", type=float, default=0.01)
parser.add_argument("--progress_scale", type=float, default=0.003)
parser.add_argument("--progress_weight", type=float, default=0.25)
parser.add_argument("--distance_weight", type=float, default=0.25)
parser.add_argument("--close_weight", type=float, default=0.35)
parser.add_argument("--orientation_weight", type=float, default=0.10)
parser.add_argument("--orientation_std", type=float, default=0.03)
parser.add_argument("--orientation_gate_sigma", type=float, default=0.012)
parser.add_argument("--terminal_weight", type=float, default=1.0)
parser.add_argument("--force_delta_penalty_weight", type=float, default=0.2)
parser.add_argument("--force_delta_threshold", type=float, default=3.0)
parser.add_argument("--force_delta_reference", type=float, default=20.0)
parser.add_argument("--expert_dataset_root", type=Path)
parser.add_argument("--expert_bc_weight", type=float, default=0.0)
parser.add_argument("--expert_bc_action_horizon", type=int, default=4)
parser.add_argument("--expert_bc_max_samples", type=int, default=8192)
parser.add_argument("--expert_bc_neighbor_chunk", type=int, default=8192)
parser.add_argument(
    "--disable_semantic_orientation_offsets",
    action="store_true",
    help="Use root/body quaternions directly while still using semantic position offsets.",
)
parser.add_argument(
    "--use_semantic_orientation_offsets",
    action="store_true",
    help="Apply the Gazebo SDF port/tip orientation frames in addition to semantic position offsets.",
)
parser.add_argument(
    "--target_position_offset",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help=(
        "Target-local XYZ offset from target rigid-object root to reward/controller target. "
        "Defaults to (0,0,0) for sfp_to_nic/nic_card and calibrated SC-port offset for sc_to_sc."
    ),
)
parser.add_argument(
    "--body_position_offset",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help=(
        "Body-local XYZ offset from target_body origin to the point measured against the target. "
        "Defaults to the SFP module tip for sfp_to_nic and body origin for sc_to_sc."
    ),
)
parser.add_argument(
    "--controller",
    choices=("target_body_world", "cheatcode_tcp"),
    default="cheatcode_tcp",
    help=(
        "target_body_world is the old direct world-frame body delta probe. "
        "cheatcode_tcp computes a wrist target that aligns the target body to the port, "
        "then uses the same TCP-to-Isaac action conversion as online SERL."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils import math as math_utils

import aic_task.tasks  # noqa: F401

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

from contact_recovery_features import CONTACT_RECOVERY_FEATURE_NAMES, ContactRecoveryFeatureComputer
from task_encoding import encode_task_vector

CAMERA_NAMES = ("center_camera", "left_camera", "right_camera")
ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
CONTROLLED_TCP_BODY = "gripper_tcp"
FORCE_WRENCH_BODY = "wrist_3_link"
EXPERT_BC_STATE_INDICES = (0, 1, 2, 13, 14, 15, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81)
SFP_PORT_LOCAL = {
    0: (0.01295, -0.031572, 0.00501),
    1: (-0.01025, -0.031572, 0.00501),
}
SFP_PORT_RPY = (4.69895, 0.0, 0.0)
SFP_PORT_ENTRANCE_LOCAL = (0.0, 0.0, -0.0458)
SFP_TIP_LOCAL = (0.0, -0.02365, 0.0)
SFP_TIP_RPY = (1.5708, 0.0, 0.0)


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
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
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
                if src < full_count and episode[src] == episode[int(row_idx)]:
                    last_valid = np.asarray(action_series.iloc[src], dtype=np.float32).reshape(-1)
                values.append(last_valid)
            action[out_idx] = np.concatenate(values, axis=0)
        single_action_dim = int(first_action.shape[0])
    else:
        obs = _stack_vector_column(df["observation.state"], "observation.state")
        single_action = _stack_vector_column(df["action"], "action")
        action = _action_chunks(single_action, episode, int(action_horizon))
        single_action_dim = int(single_action.shape[1])
    schema = {
        "dataset_root": str(dataset_root),
        "num_frames": full_count,
        "sampled_frames": int(obs.shape[0]),
        "num_episodes": int(df["episode_index"].nunique()),
        "obs_dim": int(obs.shape[1]),
        "single_action_dim": single_action_dim,
        "action_horizon": int(action_horizon),
        "action_dim": int(action.shape[1]),
    }
    return obs, action, schema


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


def _offset_from_port_frame(offset: tuple[float, float, float]) -> tuple[float, float, float]:
    port = SFP_PORT_LOCAL[int(args_cli.target_port_index)]
    rotated = _matvec(_rpy_matrix(*SFP_PORT_RPY), offset)
    return tuple(port[i] + rotated[i] for i in range(3))


def _target_scene_name(index: int) -> str:
    if args_cli.task_family == "sfp_to_nic":
        return "nic_card"
    if args_cli.task_family != "sc_to_sc":
        raise ValueError(f"Unsupported task_family: {args_cli.task_family}")
    if index == 0:
        return "sc_port"
    if index == 1:
        return "sc_port_2"
    raise ValueError("target_port_index must be 0 or 1")


class ExpertActionPrior:
    def __init__(
        self,
        *,
        dataset_root: Path,
        action_horizon: int,
        state_dim: int,
        action_dim: int,
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
            raise ValueError(f"Expert state dim {obs_all.shape[1]} != expected {state_dim}")
        if action_all.shape[1] != action_dim:
            raise ValueError(f"Expert action dim {action_all.shape[1]} != expected {action_dim}")
        count = int(schema.get("num_frames", obs_all.shape[0]))
        sample_indices = slice(None)
        obs = obs_all[sample_indices][:, EXPERT_BC_STATE_INDICES]
        mean = obs.mean(axis=0, keepdims=True)
        std = obs.std(axis=0, keepdims=True) + 1.0e-6
        self.dataset_root = Path(dataset_root)
        self.schema = schema
        self.count = count
        self.sampled_count = int(obs.shape[0])
        self.neighbor_chunk = max(1, int(neighbor_chunk))
        self.obs = torch.as_tensor((obs - mean) / std, dtype=torch.float32, device=device)
        self.action = torch.as_tensor(action_all[sample_indices], dtype=torch.float32, device=device)
        self.mean = torch.as_tensor(mean.reshape(-1), dtype=torch.float32, device=device)
        self.std = torch.as_tensor(std.reshape(-1), dtype=torch.float32, device=device)

    def nearest_action(self, state82: torch.Tensor) -> torch.Tensor:
        query = (state82.reshape(1, -1)[:, EXPERT_BC_STATE_INDICES] - self.mean) / self.std
        best_dist = None
        best_index = None
        for start in range(0, self.obs.shape[0], self.neighbor_chunk):
            dist = torch.cdist(query, self.obs[start : start + self.neighbor_chunk])
            chunk_dist, chunk_index = dist.min(dim=1)
            if best_dist is None or bool((chunk_dist < best_dist)[0].detach().cpu()):
                best_dist = chunk_dist
                best_index = chunk_index + start
        if best_index is None:
            raise RuntimeError("Expert BC prior has no samples")
        return self.action[best_index]


def _target_position_offset() -> tuple[float, float, float]:
    if args_cli.target_position_offset is not None:
        return tuple(float(v) for v in args_cli.target_position_offset)
    if args_cli.task_family == "sfp_to_nic":
        return _offset_from_port_frame(SFP_PORT_ENTRANCE_LOCAL)
    return (0.093, 0.140, 0.020)


def _controller_approach_offset() -> tuple[float, float, float]:
    if args_cli.approach_offset is not None:
        return tuple(float(v) for v in args_cli.approach_offset)
    if args_cli.task_family == "sfp_to_nic":
        return _offset_from_port_frame(SFP_PORT_ENTRANCE_LOCAL)
    return _target_position_offset()


def _controller_insert_offset() -> tuple[float, float, float]:
    if args_cli.insert_offset is not None:
        return tuple(float(v) for v in args_cli.insert_offset)
    return _target_position_offset()


def _body_position_offset() -> tuple[float, float, float]:
    if args_cli.body_position_offset is not None:
        return tuple(float(v) for v in args_cli.body_position_offset)
    if args_cli.task_family == "sfp_to_nic":
        if args_cli.target_body == "sfp_tip_link":
            return (0.0, 0.0, 0.0)
        return SFP_TIP_LOCAL
    return (0.0, 0.0, 0.0)


def _target_orientation_offset() -> tuple[float, float, float, float] | None:
    if args_cli.disable_semantic_orientation_offsets:
        return None
    if args_cli.task_family == "sfp_to_nic":
        return _quat_from_rpy(*SFP_PORT_RPY)
    return None


def _body_orientation_offset() -> tuple[float, float, float, float] | None:
    if args_cli.disable_semantic_orientation_offsets:
        return None
    if args_cli.task_family == "sfp_to_nic":
        if args_cli.target_body == "sfp_tip_link":
            return _quat_from_rpy(0.0, math.pi, 0.0)
        return _quat_from_rpy(*SFP_TIP_RPY)
    return None


def _configure_rewards(env_cfg) -> dict[str, object]:
    rewards = env_cfg.rewards
    for name in (
        "end_effector_position_tracking",
        "end_effector_position_tracking_fine_grained",
        "end_effector_position_tracking_exp",
        "end_effector_orientation_tracking",
        "end_effector_orientation_tracking_fine_grained",
        "reaching_bonus",
    ):
        getattr(rewards, name).weight = 0.0

    target_scene = _target_scene_name(args_cli.target_port_index)
    target_offset = _target_position_offset()
    body_offset = _body_position_offset()
    target_orientation_offset = _target_orientation_offset()
    body_orientation_offset = _body_orientation_offset()
    for name in (
        "target_distance_tanh",
        "target_distance_exp",
        "target_distance_progress",
        "target_orientation_tanh",
        "target_orientation_gated_exp",
        "target_reaching_bonus",
        "target_success_once_bonus",
        "target_lateral_error",
    ):
        term = getattr(rewards, name)
        term.params["body_cfg"].body_names = [args_cli.target_body]
        term.params["target_cfg"].name = target_scene
        if "target_position_offset" in term.params:
            term.params["target_position_offset"] = target_offset
        if "body_position_offset" in term.params:
            term.params["body_position_offset"] = body_offset
        if "target_orientation_offset" in term.params:
            term.params["target_orientation_offset"] = target_orientation_offset
        if "body_orientation_offset" in term.params:
            term.params["body_orientation_offset"] = body_orientation_offset

    env_step_dt = float(env_cfg.sim.dt) * float(env_cfg.decimation)
    reward_weight_multiplier = 1.0 / max(env_step_dt, 1.0e-9)
    rewards.target_distance_tanh.weight = float(args_cli.distance_weight) * reward_weight_multiplier
    rewards.target_distance_exp.weight = float(args_cli.close_weight) * reward_weight_multiplier
    rewards.target_distance_progress.weight = float(args_cli.progress_weight) * reward_weight_multiplier
    rewards.target_orientation_tanh.weight = 0.0
    rewards.target_orientation_gated_exp.weight = float(args_cli.orientation_weight) * reward_weight_multiplier
    rewards.target_reaching_bonus.weight = 0.0
    rewards.target_success_once_bonus.weight = float(args_cli.terminal_weight) * reward_weight_multiplier
    rewards.target_lateral_error.weight = 0.0
    rewards.target_distance_tanh.params["std"] = float(args_cli.distance_std)
    rewards.target_distance_exp.params["sigma"] = float(args_cli.close_sigma)
    rewards.target_distance_progress.params["scale"] = float(args_cli.progress_scale)
    rewards.target_orientation_gated_exp.params["std"] = float(args_cli.orientation_std)
    rewards.target_orientation_gated_exp.params["gate_sigma"] = float(args_cli.orientation_gate_sigma)
    rewards.target_reaching_bonus.params["threshold"] = float(args_cli.reaching_threshold)
    rewards.target_success_once_bonus.params["threshold"] = float(args_cli.reaching_threshold)
    rewards.force_delta_penalty.weight = float(args_cli.force_delta_penalty_weight) * reward_weight_multiplier
    rewards.force_delta_penalty.params["threshold"] = float(args_cli.force_delta_threshold)
    rewards.force_delta_penalty.params["reference"] = float(args_cli.force_delta_reference)
    if "asset_cfg" in rewards.force_delta_penalty.params:
        rewards.force_delta_penalty.params["asset_cfg"].body_names = [FORCE_WRENCH_BODY]
    return {
        "target_scene": target_scene,
        "target_body": args_cli.target_body,
        "target_position_offset": [float(v) for v in target_offset],
        "body_position_offset": [float(v) for v in body_offset],
        "target_orientation_offset": None
        if target_orientation_offset is None
        else [float(v) for v in target_orientation_offset],
        "body_orientation_offset": None
        if body_orientation_offset is None
        else [float(v) for v in body_orientation_offset],
        "distance_std": float(args_cli.distance_std),
        "close_sigma": float(args_cli.close_sigma),
        "reaching_threshold": float(args_cli.reaching_threshold),
        "isaac_env_step_dt": env_step_dt,
        "isaac_reward_weight_multiplier": reward_weight_multiplier,
        "progress_weight": float(args_cli.progress_weight),
        "progress_scale": float(args_cli.progress_scale),
        "distance_weight": float(args_cli.distance_weight),
        "close_weight": float(args_cli.close_weight),
        "orientation_weight": float(args_cli.orientation_weight),
        "orientation_std": float(args_cli.orientation_std),
        "orientation_gate_sigma": float(args_cli.orientation_gate_sigma),
        "reaching_weight": 0.0,
        "terminal_weight": float(args_cli.terminal_weight),
        "lateral_weight": 0.0,
        "force_delta_penalty_weight": float(args_cli.force_delta_penalty_weight),
        "force_delta_threshold": float(args_cli.force_delta_threshold),
        "force_delta_reference": float(args_cli.force_delta_reference),
        "command_pose_rewards_disabled": True,
    }


def _named_index(names: list[str], name: str) -> int:
    try:
        return names.index(name)
    except ValueError as exc:
        raise ValueError(f"{name!r} not found in available names: {names}") from exc


def _tcp_delta_action_to_isaac_base_action(env, tcp_action: torch.Tensor) -> torch.Tensor:
    """Convert Gazebo/LeRobot gripper-tcp delta actions to Isaac IK root-frame deltas."""
    robot = env.unwrapped.scene["robot"]
    body_names = list(getattr(robot, "body_names", []))
    wrist_index = _named_index(body_names, "wrist_3_link")
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


def _clip_by_norm(vec: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0.0:
        return torch.zeros_like(vec)
    norm = torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(1.0e-9)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return vec * scale


def _offset_quat_tensor(
    offset: tuple[float, float, float, float] | None,
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    if offset is None:
        out = torch.zeros_like(like)
        out[:, 0] = 1.0
        return out
    return torch.tensor(offset, dtype=like.dtype, device=like.device).reshape(1, 4).expand_as(like)


def _smoothstep(fraction: float) -> float:
    value = max(0.0, min(1.0, float(fraction)))
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def _tensor_xyz(tensor: torch.Tensor) -> list[float]:
    return [float(v) for v in tensor.reshape(-1)[:3].detach().cpu()]


def _tensor_quat(tensor: torch.Tensor) -> list[float]:
    return [float(v) for v in tensor.reshape(-1)[:4].detach().cpu()]


def _lerp_offset(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    fraction: float,
) -> tuple[float, float, float]:
    return tuple(float(a + (b - a) * fraction) for a, b in zip(start, end))


class VideoRecorder:
    def __init__(self, output_dir: Path, fps: float):
        self.output_dir = output_dir
        self.fps = float(fps)
        self._writers: dict[str, object] = {}
        self._paths: dict[str, Path] = {}

    def add_scene(self, env) -> None:
        sensors = env.unwrapped.scene.sensors
        for camera_name in CAMERA_NAMES:
            if camera_name not in sensors:
                continue
            output = sensors[camera_name].data.output
            if "rgb" not in output:
                continue
            self.add_frame(camera_name, output["rgb"])

    def add_frame(self, camera_name: str, image: torch.Tensor) -> None:
        import cv2

        frame = image.detach().cpu()
        if frame.ndim == 4:
            frame = frame[0]
        array = frame.numpy()
        if array.ndim == 3 and array.shape[0] in (3, 4):
            array = array[:3].transpose(1, 2, 0)
        elif array.ndim == 3 and array.shape[-1] in (3, 4):
            array = array[..., :3]
        else:
            raise ValueError(f"Expected RGB camera frame for {camera_name}, got {tuple(frame.shape)}")
        if array.dtype != "uint8":
            if array.max() <= 2.0:
                array = array * 255.0
            array = array.clip(0, 255).astype("uint8")
        height, width = array.shape[:2]
        writer = self._writers.get(camera_name)
        if writer is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"isaac_{camera_name}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {path}")
            self._writers[camera_name] = writer
            self._paths[camera_name] = path
        writer.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))

    def close(self) -> dict[str, str]:
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()
        return {name: str(path) for name, path in sorted(self._paths.items())}


def _cheatcode_tcp_action(
    *,
    robot,
    target,
    wrist_index: int,
    body_index: int,
    controller_target_offset: tuple[float, float, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return a TCP-frame delta that moves the wrist to align body with target.

    This mirrors the ROS CheatCode geometry: compute the rigid transform that
    would align the plug/body frame to the target-port frame, apply that same
    transform to the controlled wrist/TCP frame, then command the relative TCP
    delta. The returned action still needs Isaac's TCP-to-base conversion.
    """
    body_pos_w = robot.data.body_pos_w[:, body_index]
    body_quat_w = robot.data.body_quat_w[:, body_index]
    wrist_pos_w = robot.data.body_pos_w[:, wrist_index]
    wrist_quat_w = robot.data.body_quat_w[:, wrist_index]
    target_pos_w = target.data.root_pos_w
    target_quat_w = target.data.root_quat_w
    body_frame_quat_w = math_utils.quat_mul(
        body_quat_w,
        _offset_quat_tensor(_body_orientation_offset(), like=body_quat_w),
    )
    target_frame_quat_w = math_utils.quat_mul(
        target_quat_w,
        _offset_quat_tensor(_target_orientation_offset(), like=target_quat_w),
    )
    target_offset = torch.tensor(
        controller_target_offset,
        dtype=target_pos_w.dtype,
        device=target_pos_w.device,
    ).reshape(1, 3)
    body_offset = torch.tensor(
        _body_position_offset(),
        dtype=body_pos_w.dtype,
        device=body_pos_w.device,
    ).reshape(1, 3)
    body_point_w = body_pos_w + math_utils.quat_apply(body_quat_w, body_offset.expand_as(body_pos_w))
    target_pos_w = target_pos_w + math_utils.quat_apply(target_quat_w, target_offset.expand_as(target_pos_w))

    if args_cli.max_rotation <= 0.0:
        q_diff = torch.zeros_like(body_frame_quat_w)
        q_diff[:, 0] = 1.0
    else:
        q_body_inv = math_utils.quat_inv(body_frame_quat_w)
        q_diff = math_utils.quat_mul(target_frame_quat_w, q_body_inv)
    target_wrist_quat_w = math_utils.quat_mul(q_diff, wrist_quat_w)
    target_wrist_pos_w = target_pos_w + (wrist_pos_w - body_point_w)

    delta_pos_tcp = math_utils.quat_apply_inverse(wrist_quat_w, target_wrist_pos_w - wrist_pos_w)
    delta_quat_tcp = math_utils.quat_mul(math_utils.quat_inv(wrist_quat_w), target_wrist_quat_w)
    delta_rot_tcp = math_utils.axis_angle_from_quat(delta_quat_tcp)

    clipped_pos_tcp = _clip_by_norm(delta_pos_tcp, args_cli.max_delta)
    clipped_rot_tcp = _clip_by_norm(delta_rot_tcp, args_cli.max_rotation)
    tcp_action = torch.cat([clipped_pos_tcp, clipped_rot_tcp], dim=-1)
    diagnostics = {
        "tcp_delta_pos_norm_m": float(torch.linalg.norm(delta_pos_tcp, dim=-1)[0].detach().cpu()),
        "tcp_delta_rot_norm_rad": float(torch.linalg.norm(delta_rot_tcp, dim=-1)[0].detach().cpu()),
        "cmd_tcp_pos_norm_m": float(torch.linalg.norm(clipped_pos_tcp, dim=-1)[0].detach().cpu()),
        "cmd_tcp_rot_norm_rad": float(torch.linalg.norm(clipped_rot_tcp, dim=-1)[0].detach().cpu()),
        "controller_target_offset_x": float(controller_target_offset[0]),
        "controller_target_offset_y": float(controller_target_offset[1]),
        "controller_target_offset_z": float(controller_target_offset[2]),
    }
    return tcp_action, diagnostics


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.observations.policy.center_rgb = None
    env_cfg.observations.policy.left_rgb = None
    env_cfg.observations.policy.right_rgb = None
    env_cfg.actions.arm_action.scale = 1.0
    reward_config = _configure_rewards(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    del obs
    video_recorder = (
        VideoRecorder(args_cli.video_dir or args_cli.output.parent / "isaac_camera_videos", args_cli.video_fps)
        if args_cli.record_cameras
        else None
    )
    video_paths: dict[str, str] = {}
    if video_recorder is not None:
        video_recorder.add_scene(env)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    target = unwrapped.scene[reward_config["target_scene"]]
    body_names = list(robot.body_names)
    body_index = _named_index(body_names, args_cli.target_body)
    force_body_index = _named_index(body_names, FORCE_WRENCH_BODY)
    controlled_tcp_index = _named_index(body_names, CONTROLLED_TCP_BODY)
    wrist_index = _named_index(body_names, "wrist_3_link")
    joint_names = list(getattr(robot, "joint_names", []))
    arm_joint_indices = [_named_index(joint_names, name) for name in ARM_JOINT_NAMES]
    feature_computer = ContactRecoveryFeatureComputer()
    task_vector_np = encode_task_vector(
        task_family=args_cli.task_family,
        target_port_index=int(args_cli.target_port_index),
        target_card_index=0 if args_cli.task_family == "sfp_to_nic" else -1,
    )
    latest_norm_index = CONTACT_RECOVERY_FEATURE_NAMES.index("force_thresh_1.latest_delta_norm")
    latest_time_index = CONTACT_RECOVERY_FEATURE_NAMES.index("force_thresh_1.time_since_latest_sec")
    expert_prior = None
    if args_cli.expert_dataset_root is not None and args_cli.expert_bc_weight > 0.0:
        expert_prior = ExpertActionPrior(
            dataset_root=args_cli.expert_dataset_root,
            action_horizon=args_cli.expert_bc_action_horizon,
            state_dim=82,
            action_dim=6 * args_cli.expert_bc_action_horizon,
            max_samples=args_cli.expert_bc_max_samples,
            neighbor_chunk=args_cli.expert_bc_neighbor_chunk,
            device=torch.device(unwrapped.device),
        )
    previous_distance: torch.Tensor | None = None
    previous_force: torch.Tensor | None = None
    success_emitted = False
    pending_bc_action_chunk: torch.Tensor | None = None

    def selected_force() -> torch.Tensor:
        incoming_wrench = getattr(robot.data, "body_incoming_wrench_w", None)
        if incoming_wrench is None:
            incoming_wrench = getattr(robot.data, "body_incoming_wrench_b", None)
        if incoming_wrench is None:
            incoming_wrench = getattr(robot.data, "body_incoming_joint_wrench_b", None)
        if incoming_wrench is None:
            return torch.zeros((args_cli.num_envs, 3), dtype=robot.data.root_pos_w.dtype, device=unwrapped.device)
        return incoming_wrench[:, force_body_index, :3]

    def lerobot_compatible_feature_probe() -> dict[str, object]:
        data = robot.data
        root_pos_w = data.root_pos_w
        root_quat_w = data.root_quat_w
        tcp_pos_w = data.body_pos_w[:, controlled_tcp_index]
        tcp_quat_w = data.body_quat_w[:, controlled_tcp_index]
        tcp_pos, tcp_quat = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, tcp_pos_w, tcp_quat_w)
        tcp_quat_xyzw = torch.cat([tcp_quat[:, 1:4], tcp_quat[:, 0:1]], dim=-1)
        tcp_lin_vel = getattr(data, "body_lin_vel_w", torch.zeros(args_cli.num_envs, len(body_names), 3, device=unwrapped.device))[
            :, controlled_tcp_index
        ]
        tcp_ang_vel = getattr(data, "body_ang_vel_w", torch.zeros(args_cli.num_envs, len(body_names), 3, device=unwrapped.device))[
            :, controlled_tcp_index
        ]
        tcp_lin_vel = math_utils.quat_apply_inverse(root_quat_w, tcp_lin_vel)
        tcp_ang_vel = math_utils.quat_apply_inverse(root_quat_w, tcp_ang_vel)
        wrench = torch.cat([selected_force(), torch.zeros(args_cli.num_envs, 3, device=unwrapped.device)], dim=-1)
        incoming_wrench = getattr(data, "body_incoming_wrench_w", None)
        if incoming_wrench is None:
            incoming_wrench = getattr(data, "body_incoming_wrench_b", None)
        if incoming_wrench is None:
            incoming_wrench = getattr(data, "body_incoming_joint_wrench_b", None)
        if incoming_wrench is not None:
            wrench = incoming_wrench[:, force_body_index, :6]
        base_state = torch.cat(
            [
                tcp_pos,
                tcp_quat_xyzw,
                tcp_lin_vel,
                tcp_ang_vel,
                torch.zeros(args_cli.num_envs, 6, device=unwrapped.device),
                data.joint_pos[:, arm_joint_indices],
                torch.full((args_cli.num_envs, 1), 0.0035405, device=unwrapped.device),
                wrench,
            ],
            dim=-1,
        )
        step_dt = float(getattr(unwrapped, "step_dt", 1.0 / 30.0))
        time_sec = float(getattr(unwrapped, "common_step_counter", 0)) * step_dt
        features = feature_computer.update(
            time_sec=time_sec,
            tcp_position_base=base_state[0, 0:3].detach().cpu().numpy(),
            tcp_orientation_xyzw=base_state[0, 3:7].detach().cpu().numpy(),
            force=base_state[0, 26:29].detach().cpu().numpy(),
            torque=base_state[0, 29:32].detach().cpu().numpy(),
        )
        state82 = torch.cat(
            [
                base_state[0],
                torch.as_tensor(features, dtype=base_state.dtype, device=base_state.device),
                torch.as_tensor(task_vector_np, dtype=base_state.dtype, device=base_state.device),
            ],
            dim=0,
        )
        bc_report: dict[str, object] = {
            "bc_loss": 0.0,
            "bc_loss_weighted": 0.0,
            "bc_weight": float(args_cli.expert_bc_weight),
            "bc_enabled": expert_prior is not None,
        }
        if expert_prior is not None and pending_bc_action_chunk is not None:
            expert_action = expert_prior.nearest_action(state82)
            bc_loss = F.smooth_l1_loss(pending_bc_action_chunk.reshape(1, -1), expert_action)
            bc_report = {
                "bc_loss": float(bc_loss.detach().cpu()),
                "bc_loss_weighted": float((float(args_cli.expert_bc_weight) * bc_loss).detach().cpu()),
                "bc_weight": float(args_cli.expert_bc_weight),
                "bc_enabled": True,
                "bc_expert_action_norm": float(expert_action.norm(dim=-1).mean().detach().cpu()),
                "bc_query_action_norm": float(pending_bc_action_chunk.reshape(1, -1).norm(dim=-1).mean().detach().cpu()),
                "bc_action_l2": float((pending_bc_action_chunk.reshape(1, -1) - expert_action).norm(dim=-1).mean().detach().cpu()),
            }
        return {
            "feature_time_sec": time_sec,
            "feature_state_dim": int(state82.numel()),
            "feature_base_dim": int(base_state.shape[-1]),
            "feature_contact_dim": int(len(features)),
            "feature_task_dim": int(len(task_vector_np)),
            "feature_tcp_pos_base": [float(v) for v in state82[0:3].detach().cpu()],
            "feature_tcp_quat_xyzw": [float(v) for v in state82[3:7].detach().cpu()],
            "feature_force_xyz": [float(v) for v in state82[26:29].detach().cpu()],
            "feature_force_thresh_1_latest_time_sec": float(features[latest_time_index]),
            "feature_force_thresh_1_latest_delta_norm": float(features[latest_norm_index]),
            "feature_task_vector": [float(v) for v in task_vector_np],
            **bc_report,
        }

    def metrics(step: int, reward_value: torch.Tensor | None = None) -> dict[str, float | int | None]:
        nonlocal previous_distance, previous_force, success_emitted
        body_pos = robot.data.body_pos_w[:, body_index]
        body_quat = robot.data.body_quat_w[:, body_index]
        target_pos = target.data.root_pos_w
        target_quat = target.data.root_quat_w
        body_frame_quat = math_utils.quat_mul(
            body_quat,
            _offset_quat_tensor(_body_orientation_offset(), like=body_quat),
        )
        target_frame_quat = math_utils.quat_mul(
            target_quat,
            _offset_quat_tensor(_target_orientation_offset(), like=target_quat),
        )
        target_offset = torch.tensor(
            _target_position_offset(),
            dtype=target_pos.dtype,
            device=target_pos.device,
        ).reshape(1, 3)
        body_offset = torch.tensor(
            _body_position_offset(),
            dtype=body_pos.dtype,
            device=body_pos.device,
        ).reshape(1, 3)
        body_point = body_pos + math_utils.quat_apply(body_quat, body_offset.expand_as(body_pos))
        target_pos = target_pos + math_utils.quat_apply(target_quat, target_offset.expand_as(target_pos))
        distance = torch.linalg.norm(body_point - target_pos, dim=1)
        body_minus_target_local = math_utils.quat_apply_inverse(target_quat, body_point - target_pos)
        dot = torch.abs(torch.sum(body_frame_quat * target_frame_quat, dim=1)).clamp(0.0, 1.0)
        orientation = 2.0 * torch.acos(dot)
        distance_reward = 1.0 - torch.tanh(distance / float(args_cli.distance_std))
        close_reward = torch.exp(-(distance * distance) / (float(args_cli.close_sigma) ** 2))
        if previous_distance is None:
            progress_reward = torch.zeros_like(distance)
        else:
            progress_reward = ((previous_distance.to(distance.device) - distance) / float(args_cli.progress_scale)).clamp(-1.0, 1.0)
        previous_distance = distance.detach().clone()
        orientation_alignment = torch.exp(-torch.square(orientation / float(args_cli.orientation_std)))
        orientation_gate = torch.exp(-torch.square(distance / float(args_cli.orientation_gate_sigma)))
        orientation_reward = orientation_alignment * orientation_gate
        force = selected_force()
        if previous_force is None:
            force_delta = torch.zeros_like(force)
        else:
            force_delta = force - previous_force.to(force.device)
        previous_force = force.detach().clone()
        force_delta_norm = torch.linalg.norm(force_delta, dim=1)
        force_denominator = max(float(args_cli.force_delta_reference) - float(args_cli.force_delta_threshold), 1.0e-6)
        force_penalty_unit = -((force_delta_norm - float(args_cli.force_delta_threshold)) / force_denominator).clamp(0.0, 1.0)
        terminal_unit_value = 0.0
        if bool((distance <= float(args_cli.reaching_threshold))[0].detach().cpu()) and not success_emitted:
            terminal_unit_value = 1.0
            success_emitted = True
        expected = (
            float(args_cli.progress_weight) * progress_reward
            + float(args_cli.distance_weight) * distance_reward
            + float(args_cli.close_weight) * close_reward
            + float(args_cli.orientation_weight) * orientation_reward
            + float(args_cli.force_delta_penalty_weight) * force_penalty_unit
            + float(args_cli.terminal_weight) * torch.full_like(distance, terminal_unit_value)
        )
        return {
            "step": step,
            "reward": None if reward_value is None else float(reward_value.reshape(-1)[0].detach().cpu()),
            "distance_m": float(distance[0].detach().cpu()),
            "orientation_error_rad": float(orientation[0].detach().cpu()),
            "progress_unit": float(progress_reward[0].detach().cpu()),
            "progress_weighted": float((float(args_cli.progress_weight) * progress_reward)[0].detach().cpu()),
            "distance_unit": float(distance_reward[0].detach().cpu()),
            "distance_weighted": float((float(args_cli.distance_weight) * distance_reward)[0].detach().cpu()),
            "close_unit": float(close_reward[0].detach().cpu()),
            "close_weighted": float((float(args_cli.close_weight) * close_reward)[0].detach().cpu()),
            "orientation_unit": float(orientation_reward[0].detach().cpu()),
            "orientation_weighted": float((float(args_cli.orientation_weight) * orientation_reward)[0].detach().cpu()),
            "force_norm": float(torch.linalg.norm(force, dim=1)[0].detach().cpu()),
            "force_delta_norm": float(force_delta_norm[0].detach().cpu()),
            "force_delta_penalty_unit": float(force_penalty_unit[0].detach().cpu()),
            "force_delta_penalty_weighted": float((float(args_cli.force_delta_penalty_weight) * force_penalty_unit)[0].detach().cpu()),
            "terminal_unit": terminal_unit_value,
            "terminal_weighted": float(args_cli.terminal_weight) * terminal_unit_value,
            "expected_dense_reward": float(expected[0].detach().cpu()),
            "body_minus_target_local_x": float(body_minus_target_local[0, 0].detach().cpu()),
            "body_minus_target_local_y": float(body_minus_target_local[0, 1].detach().cpu()),
            "body_minus_target_local_z": float(body_minus_target_local[0, 2].detach().cpu()),
            "target_root_pos_w": _tensor_xyz(target.data.root_pos_w[0]),
            "target_root_quat_w": _tensor_quat(target.data.root_quat_w[0]),
            "reward_target_pos_w": _tensor_xyz(target_pos[0]),
            "body_root_pos_w": _tensor_xyz(body_pos[0]),
            "body_root_quat_w": _tensor_quat(body_quat[0]),
            "reward_body_pos_w": _tensor_xyz(body_point[0]),
            **lerobot_compatible_feature_probe(),
        }

    rows = [metrics(0)]
    rows[-1]["controller"] = args_cli.controller
    print(json.dumps(rows[-1]), flush=True)
    for step in range(1, args_cli.steps + 1):
        diagnostics: dict[str, float] = {}
        if args_cli.controller == "target_body_world":
            target_pos = target.data.root_pos_w[:, :3]
            target_offset = torch.tensor(
                _target_position_offset(),
                dtype=target_pos.dtype,
                device=target_pos.device,
            ).reshape(1, 3)
            target_pos = target_pos + math_utils.quat_apply(
                target.data.root_quat_w, target_offset.expand_as(target_pos)
            )
            body_pos = robot.data.body_pos_w[:, body_index, :3]
            body_quat = robot.data.body_quat_w[:, body_index]
            body_offset = torch.tensor(
                _body_position_offset(),
                dtype=body_pos.dtype,
                device=body_pos.device,
            ).reshape(1, 3)
            body_point = body_pos + math_utils.quat_apply(body_quat, body_offset.expand_as(body_pos))
            delta = target_pos - body_point
            action = torch.zeros((args_cli.num_envs, 6), device=unwrapped.device)
            action[:, :3] = _clip_by_norm(delta, args_cli.max_delta)
            pending_bc_action_chunk = action.reshape(-1).repeat(args_cli.expert_bc_action_horizon).detach()
            diagnostics = {
                "world_delta_norm_m": float(torch.linalg.norm(delta, dim=-1)[0].detach().cpu()),
                "cmd_world_pos_norm_m": float(torch.linalg.norm(action[:, :3], dim=-1)[0].detach().cpu()),
            }
        else:
            if step < args_cli.descent_start_step:
                controller_target_offset = _controller_approach_offset()
            else:
                denom = max(1, args_cli.steps - args_cli.descent_start_step)
                controller_target_offset = _lerp_offset(
                    _controller_approach_offset(),
                    _controller_insert_offset(),
                    _smoothstep((step - args_cli.descent_start_step) / denom),
                )
            tcp_action, diagnostics = _cheatcode_tcp_action(
                robot=robot,
                target=target,
                wrist_index=wrist_index,
                body_index=body_index,
                controller_target_offset=controller_target_offset,
            )
            pending_bc_action_chunk = tcp_action.reshape(-1).repeat(args_cli.expert_bc_action_horizon).detach()
            action = _tcp_delta_action_to_isaac_base_action(env, tcp_action)
        _, reward, terminated, truncated, _ = env.step(action)
        if video_recorder is not None:
            video_recorder.add_scene(env)
        row = metrics(step, reward)
        row["controller"] = args_cli.controller
        row.update(diagnostics)
        rows.append(row)
        print(json.dumps(row), flush=True)
        if bool(torch.logical_or(terminated, truncated).any()):
            break

    exact = {
        "distance_m": 0.0,
        "orientation_error_rad": 0.0,
        "distance_unit": 1.0,
        "close_unit": 1.0,
        "orientation_unit": 1.0,
        "expected_dense_reward_without_progress_or_force": (
            float(args_cli.distance_weight)
            + float(args_cli.close_weight)
            + float(args_cli.orientation_weight)
            + float(args_cli.terminal_weight)
        ),
    }
    if video_recorder is not None:
        video_paths = video_recorder.close()
    summary = {
        "reward_config": reward_config,
        "expert_bc": None
        if expert_prior is None
        else {
            "dataset_root": str(expert_prior.dataset_root),
            "weight": float(args_cli.expert_bc_weight),
            "num_transitions": expert_prior.count,
            "sampled_transitions": expert_prior.sampled_count,
            "state_indices": list(EXPERT_BC_STATE_INDICES),
            "action_horizon": int(args_cli.expert_bc_action_horizon),
            "note": "Diagnostic only for cheatcode probe; cheatcode is not used as training data.",
        },
        "rows": rows,
        "best": max(rows, key=lambda row: float(row["expected_dense_reward"])),
        "exact_alignment_expected": exact,
        "video_paths": video_paths,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    env.close()
    print("SUMMARY", json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    status = 0
    try:
        main()
    except BaseException:
        status = 1
        raise
    finally:
        simulation_app.close()
    raise SystemExit(status)
