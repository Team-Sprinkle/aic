"""Minimal training task definition for the Gazebo environment."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
import random


@dataclass(frozen=True)
class PoseRandomizationConfig:
    """Axis-aligned bounded randomization around a base pose."""

    x_range: tuple[float, float] = (0.0, 0.0)
    y_range: tuple[float, float] = (0.0, 0.0)
    z_range: tuple[float, float] = (0.0, 0.0)

    def sample(
        self,
        *,
        rng: random.Random,
        base_position: list[float],
    ) -> list[float]:
        """Sample a randomized position around a base pose."""
        return [
            round(base_position[0] + rng.uniform(*self.x_range), 6),
            round(base_position[1] + rng.uniform(*self.y_range), 6),
            round(base_position[2] + rng.uniform(*self.z_range), 6),
        ]


@dataclass(frozen=True)
class MinimalTaskSamplerConfig:
    """Config-driven task sampler for board and object pose variability."""

    board_pose: PoseRandomizationConfig = PoseRandomizationConfig()
    object_pose: PoseRandomizationConfig = PoseRandomizationConfig(
        x_range=(-0.05, 0.05),
        y_range=(-0.05, 0.05),
        z_range=(-0.02, 0.02),
    )


@dataclass
class MinimalTaskState:
    """Mutable state for the minimal single-object task."""

    board_pose: dict[str, list[float]]
    target_pose: dict[str, list[float]]
    object_pose: dict[str, list[float]]
    grasped: bool = False


@dataclass(frozen=True)
class MinimalTask:
    """A fixed-board task with one target and one movable object."""

    grasp_distance_threshold: float = 0.06
    success_distance_threshold: float = 0.05
    sampler_config: MinimalTaskSamplerConfig = MinimalTaskSamplerConfig()

    def reset(self, *, seed: int | None = None) -> MinimalTaskState:
        """Create a seeded initial task state."""
        rng = random.Random(seed if seed is not None else 0)
        base_board_position = [0.0, 0.0, 0.0]
        board_position = self.sampler_config.board_pose.sample(
            rng=rng,
            base_position=base_board_position,
        )
        board_pose = {
            "position": board_position,
            "orientation": [0.0, 0.0, 0.0, 1.0],
        }
        target_pose = {
            "position": [
                round(board_position[0] + 0.3, 6),
                round(board_position[1] + 0.0, 6),
                round(board_position[2] + 0.55, 6),
            ],
            "orientation": [0.0, 0.0, 0.0, 1.0],
        }
        base_object_position = [
            round(board_position[0] + 0.0, 6),
            round(board_position[1] + 0.0, 6),
            round(board_position[2] + 0.5, 6),
        ]
        object_pose = {
            "position": self.sampler_config.object_pose.sample(
                rng=rng,
                base_position=base_object_position,
            ),
            "orientation": [0.0, 0.0, 0.0, 1.0],
        }
        return MinimalTaskState(
            board_pose=board_pose,
            target_pose=target_pose,
            object_pose=object_pose,
            grasped=False,
        )

    def advance(
        self,
        *,
        state: MinimalTaskState,
        ee_position: list[float],
    ) -> MinimalTaskState:
        """Advance task state using the latest end-effector position."""
        if (not state.grasped) and (
            self.distance(ee_position, state.object_pose["position"])
            <= self.grasp_distance_threshold
        ):
            state.grasped = True
        if state.grasped:
            state.object_pose["position"] = [float(value) for value in ee_position]
        return state

    def is_success(self, state: MinimalTaskState) -> bool:
        """Return whether the task is solved."""
        return (
            self.distance(
                state.object_pose["position"],
                state.target_pose["position"],
            )
            <= self.success_distance_threshold
        )

    def privileged_observation(self, state: MinimalTaskState) -> dict[str, object]:
        """Return privileged task state for training-time consumers."""
        return {
            "board_pose": {
                "position": list(state.board_pose["position"]),
                "orientation": list(state.board_pose["orientation"]),
            },
            "target_pose": {
                "position": list(state.target_pose["position"]),
                "orientation": list(state.target_pose["orientation"]),
            },
            "object_pose": {
                "position": list(state.object_pose["position"]),
                "orientation": list(state.object_pose["orientation"]),
            },
            "grasped": state.grasped,
        }

    def distance(self, lhs: list[float], rhs: list[float]) -> float:
        """Compute Euclidean distance between two 3D points."""
        return sqrt(sum((left - right) ** 2 for left, right in zip(lhs, rhs, strict=True)))
