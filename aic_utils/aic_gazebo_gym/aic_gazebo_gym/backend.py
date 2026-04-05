"""Backend abstraction for ROS-free Gazebo control."""

from __future__ import annotations

from typing import Protocol

from .types import Action, ResetResult, StepResult


class GazeboBackend(Protocol):
    """Defines the control surface implemented by a Gazebo backend."""

    def reset(self) -> ResetResult:
        """Reset the simulator state and return the initial observation."""

    def step(self, action: Action) -> StepResult:
        """Apply one action and return the environment transition."""
