from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TaskGeometryRewardConfig:
    distance_std_m: float = 0.05
    close_sigma_m: float = 0.01
    orientation_std_rad: float = 0.25
    progress_weight: float = 1.0
    distance_weight: float = 0.5
    close_weight: float = 0.3
    orientation_weight: float = 0.05
    terminal_score_weight: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _position(pose: dict[str, Any] | None) -> np.ndarray | None:
    if not isinstance(pose, dict) or pose.get("position") is None:
        return None
    values = np.asarray(pose["position"], dtype=np.float32).reshape(-1)
    return values[:3] if values.shape[0] >= 3 else None


def _quat_xyzw(pose: dict[str, Any] | None) -> np.ndarray | None:
    if not isinstance(pose, dict) or pose.get("orientation_xyzw") is None:
        return None
    values = np.asarray(pose["orientation_xyzw"], dtype=np.float64).reshape(-1)
    if values.shape[0] < 4:
        return None
    norm = np.linalg.norm(values[:4])
    if norm <= 1.0e-12:
        return None
    return values[:4] / norm


def _oracle_poses(obs: dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    oracle = (obs or {}).get("oracle") or {}
    return oracle.get("plug_pose_base_link"), oracle.get("target_port_pose_base_link")


def plug_to_port_distance(obs: dict[str, Any] | None) -> float | None:
    plug_pose, port_pose = _oracle_poses(obs)
    plug = _position(plug_pose)
    port = _position(port_pose)
    if plug is None or port is None:
        return None
    return float(np.linalg.norm(port - plug))


def plug_to_port_orientation_error(obs: dict[str, Any] | None) -> float | None:
    plug_pose, port_pose = _oracle_poses(obs)
    plug = _quat_xyzw(plug_pose)
    port = _quat_xyzw(port_pose)
    if plug is None or port is None:
        return None
    # For unit quaternions, the shortest angular distance is 2 acos(|dot|).
    dot = float(np.clip(abs(np.dot(plug, port)), 0.0, 1.0))
    return float(2.0 * math.acos(dot))


def dense_task_geometry_reward(
    *,
    prev_obs: dict[str, Any] | None,
    obs: dict[str, Any] | None,
    terminal_score: float = 0.0,
    config: TaskGeometryRewardConfig | None = None,
) -> tuple[float, dict[str, float]]:
    """Dense online reward from ground-truth plug and target-port poses."""
    cfg = config or TaskGeometryRewardConfig()
    distance = plug_to_port_distance(obs)
    if distance is None:
        return 0.0, {"task_geometry_reward_available": 0.0}

    prev_distance = plug_to_port_distance(prev_obs)
    progress = 0.0 if prev_distance is None else float(np.clip(prev_distance - distance, -0.05, 0.05))
    distance_reward = 1.0 - math.tanh(distance / cfg.distance_std_m)
    close_reward = math.exp(-((distance * distance) / (cfg.close_sigma_m * cfg.close_sigma_m)))
    orientation_error = plug_to_port_orientation_error(obs)
    orientation_reward = (
        0.0
        if orientation_error is None
        else 1.0 - math.tanh(float(orientation_error) / cfg.orientation_std_rad)
    )
    reward = (
        cfg.progress_weight * progress
        + cfg.distance_weight * distance_reward
        + cfg.close_weight * close_reward
        + cfg.orientation_weight * orientation_reward
        + cfg.terminal_score_weight * float(terminal_score)
    )
    return float(reward), {
        "task_geometry_reward_available": 1.0,
        "plug_to_port_distance_m": float(distance),
        "plug_to_port_progress_m": float(progress),
        "plug_to_port_distance_reward": float(distance_reward),
        "plug_to_port_close_reward": float(close_reward),
        "plug_to_port_orientation_error_rad": -1.0 if orientation_error is None else float(orientation_error),
        "plug_to_port_orientation_reward": float(orientation_reward),
        "terminal_score_reward": float(terminal_score),
    }
