"""Observation extraction and command application interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from .runtime import RuntimeState


class AicGazeboIO(ABC):
    """Gazebo-native observation / actuation interface."""

    @abstractmethod
    def observation_from_state(
        self,
        state: RuntimeState,
        *,
        include_images: bool,
        step_count: int,
    ) -> dict[str, Any]:
        """Convert runtime state to the public env observation."""

    @abstractmethod
    def sanitize_action(self, action: np.ndarray) -> np.ndarray:
        """Clamp and normalize an env action before it reaches the backend."""


@dataclass(frozen=True)
class GazeboNativeIOPlaceholder(AicGazeboIO):
    """Phase-2 Gazebo-native IO placeholder.

    The intended live implementation order is:
    1. direct Gazebo Transport image subscriptions
    2. Gazebo system plugin extraction
    3. isolated ROS sidecar fallback
    """

    def observation_from_state(
        self,
        state: RuntimeState,
        *,
        include_images: bool,
        step_count: int,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "Gazebo-native IO is not wired in this shell. Use MockGazeboIO for state-only "
            "tests, or inject a transport-backed implementation."
        )

    def sanitize_action(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class MockGazeboIO(AicGazeboIO):
    """State-only IO used for deterministic testing."""

    def observation_from_state(
        self,
        state: RuntimeState,
        *,
        include_images: bool,
        step_count: int,
    ) -> dict[str, Any]:
        relative = state.target_port_pose[:3] - state.plug_pose[:3]
        observation = {
            "step_count": step_count,
            "sim_tick": int(state.sim_tick),
            "sim_time": float(state.sim_time),
            "joint_positions": state.joint_positions.astype(np.float32).copy(),
            "joint_velocities": state.joint_velocities.astype(np.float32).copy(),
            "gripper_state": np.array([state.gripper_position], dtype=np.float32),
            "tcp_pose": state.tcp_pose.astype(np.float32).copy(),
            "tcp_velocity": state.tcp_velocity.astype(np.float32).copy(),
            "plug_pose": state.plug_pose.astype(np.float32).copy(),
            "target_port_pose": state.target_port_pose.astype(np.float32).copy(),
            "plug_to_port_relative": np.concatenate(
                [relative, np.array([np.linalg.norm(relative)], dtype=np.float64)]
            ).astype(np.float32),
            "wrench": state.wrench.astype(np.float32).copy(),
            "off_limit_contact": np.array([float(state.off_limit_contact)], dtype=np.float32),
        }
        if include_images:
            blank = np.zeros((3, 64, 64, 3), dtype=np.uint8)
            observation["images"] = {
                "left": blank[0],
                "center": blank[1],
                "right": blank[2],
            }
        return observation

    def sanitize_action(self, action: np.ndarray) -> np.ndarray:
        array = np.asarray(action, dtype=np.float64)
        if array.shape != (6,):
            raise ValueError(f"Expected action with shape (6,), got {array.shape}.")
        array[:3] = np.clip(array[:3], -0.25, 0.25)
        array[3:] = np.clip(array[3:], -2.0, 2.0)
        return array
