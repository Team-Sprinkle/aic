"""Basic termination checks for the training-only environment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Termination:
    """Termination configuration and evaluation helpers."""

    success_distance_threshold: float = 0.08
    failure_distance_threshold: float = 2.5
    max_episode_steps: int = 25

    def evaluate(
        self,
        *,
        step_count: int,
        distance_to_target: float,
        ee_position: list[float],
    ) -> tuple[bool, bool, str | None]:
        """Return `(terminated, truncated, reason)` for the current state."""
        if distance_to_target <= self.success_distance_threshold:
            return True, False, "success"
        if ee_position[2] < 0.0 or distance_to_target >= self.failure_distance_threshold:
            return True, False, "failure"
        if step_count >= self.max_episode_steps:
            return False, True, "timeout"
        return False, False, None
