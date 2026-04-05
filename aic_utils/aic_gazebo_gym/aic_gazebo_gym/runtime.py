"""Runtime abstraction for Gazebo process management."""

from __future__ import annotations

from typing import Protocol


class GazeboRuntime(Protocol):
    """Owns simulator process lifecycle outside the env API."""

    def start(self) -> None:
        """Start the simulator runtime."""

    def stop(self) -> None:
        """Stop the simulator runtime."""
