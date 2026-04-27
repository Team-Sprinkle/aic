from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

DEFAULT_MAX_TRANSLATION_M = 0.003
DEFAULT_MAX_ROTATION_RAD = 0.03


@dataclass(frozen=True)
class DeltaTcpAction:
    delta_position_xyz: np.ndarray
    delta_rotation_xyz: np.ndarray
    delta_quaternion_xyzw: np.ndarray
    clipped_action: np.ndarray


def _as_action_array(action: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(action), dtype=np.float64)
    if arr.shape != (6,):
        raise ValueError(f"Expected action with shape (6,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Action contains non-finite values: {arr}")
    return arr


def rotation_vector_to_quaternion_xyzw(rotvec_xyz: Iterable[float]) -> np.ndarray:
    rotvec = np.asarray(list(rotvec_xyz), dtype=np.float64)
    if rotvec.shape != (3,):
        raise ValueError(f"Expected rotation vector with shape (3,), got {rotvec.shape}")
    angle = float(np.linalg.norm(rotvec))
    if angle <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = rotvec / angle
    half_angle = 0.5 * angle
    quat = np.array(
        [
            axis[0] * math.sin(half_angle),
            axis[1] * math.sin(half_angle),
            axis[2] * math.sin(half_angle),
            math.cos(half_angle),
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if quat[3] < 0.0:
        quat = -quat
    return quat / norm


def clip_delta_tcp_action(
    action: Iterable[float],
    *,
    max_translation_m: float = DEFAULT_MAX_TRANSLATION_M,
    max_rotation_rad: float = DEFAULT_MAX_ROTATION_RAD,
) -> np.ndarray:
    arr = _as_action_array(action)
    clipped = arr.copy()
    clipped[:3] = np.clip(clipped[:3], -max_translation_m, max_translation_m)
    clipped[3:] = np.clip(clipped[3:], -max_rotation_rad, max_rotation_rad)
    return clipped


def delta_tcp_action_from_array(
    action: Iterable[float],
    *,
    max_translation_m: float = DEFAULT_MAX_TRANSLATION_M,
    max_rotation_rad: float = DEFAULT_MAX_ROTATION_RAD,
) -> DeltaTcpAction:
    clipped = clip_delta_tcp_action(
        action,
        max_translation_m=max_translation_m,
        max_rotation_rad=max_rotation_rad,
    )
    return DeltaTcpAction(
        delta_position_xyz=clipped[:3].copy(),
        delta_rotation_xyz=clipped[3:].copy(),
        delta_quaternion_xyzw=rotation_vector_to_quaternion_xyzw(clipped[3:]),
        clipped_action=clipped,
    )
