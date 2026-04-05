"""Simple, stable reward computation for the training-only environment."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class Reward:
    """Reward configuration and computation helpers."""

    success_bonus: float = 10.0
    action_penalty_weight: float = 0.05

    def compute(
        self,
        *,
        ee_position: list[float],
        target_position: list[float],
        action_delta: list[float],
        success: bool,
    ) -> tuple[float, float]:
        """Compute reward and return `(reward, distance_to_target)`."""
        distance = _euclidean_distance(ee_position, target_position)
        action_penalty = self.action_penalty_weight * sum(abs(value) for value in action_delta)
        reward = -distance - action_penalty
        if success:
            reward = self.success_bonus - action_penalty
        return reward, distance


def _euclidean_distance(lhs: list[float], rhs: list[float]) -> float:
    """Compute Euclidean distance between two 3D vectors."""
    return sqrt(sum((left - right) ** 2 for left, right in zip(lhs, rhs, strict=True)))
