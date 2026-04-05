"""Shared types for the Gazebo gym facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Observation = dict[str, Any]
Action = dict[str, Any]


@dataclass(frozen=True)
class ResetResult:
    """Result returned by environment reset."""

    observation: Observation
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    """Result returned by a single environment step."""

    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)
