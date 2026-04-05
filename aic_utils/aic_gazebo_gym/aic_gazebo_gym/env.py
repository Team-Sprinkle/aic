"""Gym-like facade that composes a runtime and backend."""

from __future__ import annotations

from dataclasses import dataclass

from .backend import GazeboBackend
from .runtime import GazeboRuntime
from .types import Action, ResetResult, StepResult


@dataclass
class GazeboEnv:
    """Minimal env facade for the future Gazebo training integration."""

    backend: GazeboBackend
    runtime: GazeboRuntime | None = None

    def start(self) -> None:
        """Start the configured runtime if one is provided."""
        if self.runtime is not None:
            self.runtime.start()

    def close(self) -> None:
        """Stop the configured runtime if one is provided."""
        if self.runtime is not None:
            self.runtime.stop()

    def reset(self) -> ResetResult:
        """Delegate reset to the backend."""
        return self.backend.reset()

    def step(self, action: Action) -> StepResult:
        """Delegate stepping to the backend."""
        return self.backend.step(action)
