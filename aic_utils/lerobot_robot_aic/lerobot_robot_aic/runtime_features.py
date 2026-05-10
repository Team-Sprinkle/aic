"""Runtime/state feature assembly shared by dataset tooling and policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .contact_recovery_features import (
    CONTACT_RECOVERY_FEATURE_DIM,
    ContactRecoveryFeatureComputer,
)
from .task_encoding import TASK_VECTOR_DIM, validate_task_vector


BASE_OBSERVATION_STATE_DIM = 32
AIC_STATE_DIMS = {
    BASE_OBSERVATION_STATE_DIM,
    BASE_OBSERVATION_STATE_DIM + TASK_VECTOR_DIM,
    BASE_OBSERVATION_STATE_DIM + CONTACT_RECOVERY_FEATURE_DIM,
    BASE_OBSERVATION_STATE_DIM + CONTACT_RECOVERY_FEATURE_DIM + TASK_VECTOR_DIM,
}


def _fixed_len(values: Iterable[float], length: int) -> list[float]:
    out = [float(v) for v in list(values)[:length]]
    if len(out) < length:
        out.extend([0.0] * (length - len(out)))
    return out


def _stamp_to_sec(stamp: Any) -> float | None:
    if stamp is None:
        return None
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    if sec == 0 and nanosec == 0:
        return None
    return float(sec) + float(nanosec) * 1e-9


def base_state_from_gazebo_observation(obs: dict[str, Any]) -> np.ndarray:
    """Return the canonical 32D LeRobot low-dimensional state from Gazebo IPC."""
    controller = obs.get("controller") or {}
    tcp_pose = controller.get("current_tcp_pose") or {}
    tcp_velocity = controller.get("tcp_velocity") or {}
    joints = obs.get("joints") or {}
    wrench = obs.get("wrist_wrench") or {}

    values: list[float] = []
    values.extend(_fixed_len(tcp_pose.get("position") or [], 3))
    values.extend(_fixed_len(tcp_pose.get("orientation_xyzw") or [0.0, 0.0, 0.0, 1.0], 4))
    values.extend(_fixed_len((tcp_velocity or {}).get("linear") or [], 3))
    values.extend(_fixed_len((tcp_velocity or {}).get("angular") or [], 3))
    values.extend(_fixed_len(controller.get("tcp_error") or [], 6))
    values.extend(_fixed_len(joints.get("position") or [], 7))
    values.extend(_fixed_len((wrench or {}).get("force") or [], 3))
    values.extend(_fixed_len((wrench or {}).get("torque") or [], 3))
    return np.asarray(values[:BASE_OBSERVATION_STATE_DIM], dtype=np.float32)


def base_state_from_ros_observation(obs_msg: Any) -> np.ndarray:
    """Return the canonical 32D LeRobot low-dimensional state from an AIC Observation."""
    tcp_pose = obs_msg.controller_state.tcp_pose
    tcp_vel = obs_msg.controller_state.tcp_velocity
    values = [
        tcp_pose.position.x,
        tcp_pose.position.y,
        tcp_pose.position.z,
        tcp_pose.orientation.x,
        tcp_pose.orientation.y,
        tcp_pose.orientation.z,
        tcp_pose.orientation.w,
        tcp_vel.linear.x,
        tcp_vel.linear.y,
        tcp_vel.linear.z,
        tcp_vel.angular.x,
        tcp_vel.angular.y,
        tcp_vel.angular.z,
        *_fixed_len(obs_msg.controller_state.tcp_error, 6),
        *_fixed_len(obs_msg.joint_states.position, 7),
        obs_msg.wrist_wrench.wrench.force.x,
        obs_msg.wrist_wrench.wrench.force.y,
        obs_msg.wrist_wrench.wrench.force.z,
        obs_msg.wrist_wrench.wrench.torque.x,
        obs_msg.wrist_wrench.wrench.torque.y,
        obs_msg.wrist_wrench.wrench.torque.z,
    ]
    return np.asarray(values[:BASE_OBSERVATION_STATE_DIM], dtype=np.float32)


def ros_observation_time_sec(obs_msg: Any, *, fallback_index: int = 0, fallback_fps: float = 20.0) -> float:
    header = getattr(obs_msg, "header", None)
    value = _stamp_to_sec(getattr(header, "stamp", None))
    if value is not None:
        return value
    return float(fallback_index) / float(fallback_fps)


@dataclass
class AICRuntimeFeatureAssembler:
    """Assemble 32D/42D/72D/82D states with causal contact features."""

    expected_dim: int
    task_vector: np.ndarray | None = None
    fps: float = 20.0

    def __post_init__(self) -> None:
        self.expected_dim = int(self.expected_dim)
        if self.expected_dim not in AIC_STATE_DIMS:
            raise ValueError(f"unsupported AIC runtime state dim {self.expected_dim}; expected one of {sorted(AIC_STATE_DIMS)}")
        self._contact = ContactRecoveryFeatureComputer()
        self._step_index = 0
        if self.task_vector is not None:
            self.task_vector = validate_task_vector(self.task_vector).astype(np.float32)

    @property
    def uses_contact_features(self) -> bool:
        return self.expected_dim in {
            BASE_OBSERVATION_STATE_DIM + CONTACT_RECOVERY_FEATURE_DIM,
            BASE_OBSERVATION_STATE_DIM + CONTACT_RECOVERY_FEATURE_DIM + TASK_VECTOR_DIM,
        }

    @property
    def uses_task_vector(self) -> bool:
        return self.expected_dim in {
            BASE_OBSERVATION_STATE_DIM + TASK_VECTOR_DIM,
            BASE_OBSERVATION_STATE_DIM + CONTACT_RECOVERY_FEATURE_DIM + TASK_VECTOR_DIM,
        }

    def reset(self, task_vector: Iterable[float] | None = None) -> None:
        self._contact.reset()
        self._step_index = 0
        if task_vector is not None:
            self.task_vector = validate_task_vector(task_vector).astype(np.float32)

    def assemble(self, base_state: np.ndarray, *, time_sec: float | None = None) -> np.ndarray:
        state = np.asarray(base_state, dtype=np.float32).reshape(-1)
        if state.shape[0] != BASE_OBSERVATION_STATE_DIM:
            raise ValueError(f"base observation.state must be 32D, got {state.shape[0]}")
        pieces = [state]
        if time_sec is None:
            time_sec = float(self._step_index) / float(self.fps)
        if self.uses_contact_features:
            pieces.append(
                self._contact.update(
                    time_sec=float(time_sec),
                    tcp_position_base=state[0:3],
                    tcp_orientation_xyzw=state[3:7],
                    force=state[26:29],
                    torque=state[29:32],
                )
            )
        if self.uses_task_vector:
            if self.task_vector is None:
                raise ValueError(f"checkpoint expects {self.expected_dim}D task-conditioned state, but no task vector is set")
            pieces.append(self.task_vector.astype(np.float32, copy=False))
        self._step_index += 1
        out = np.concatenate(pieces).astype(np.float32)
        if out.shape[0] != self.expected_dim:
            raise ValueError(f"assembled state dim {out.shape[0]} does not match expected {self.expected_dim}")
        return out

    def assemble_gazebo(self, obs: dict[str, Any]) -> np.ndarray:
        time_sec = obs.get("timestamp", obs.get("time_sec"))
        return self.assemble(
            base_state_from_gazebo_observation(obs),
            time_sec=None if time_sec is None else float(time_sec),
        )

    def assemble_ros(self, obs_msg: Any) -> np.ndarray:
        return self.assemble(
            base_state_from_ros_observation(obs_msg),
            time_sec=ros_observation_time_sec(obs_msg, fallback_index=self._step_index, fallback_fps=self.fps),
        )
