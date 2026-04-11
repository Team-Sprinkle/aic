"""Reusable scripted policies for parity and training smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ScriptedCartesianVelocityAction:
    linear_xyz: tuple[float, float, float]
    angular_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    frame_id: str = "base_link"
    sim_steps: int = 25

    def as_env_action(self) -> np.ndarray:
        return np.asarray(
            list(self.linear_xyz) + list(self.angular_xyz),
            dtype=np.float32,
        )


DEFAULT_DETERMINISTIC_POLICY: tuple[ScriptedCartesianVelocityAction, ...] = (
    ScriptedCartesianVelocityAction((0.0, 0.0, 0.0), sim_steps=10),
    ScriptedCartesianVelocityAction((0.01, 0.0, 0.0), sim_steps=25),
    ScriptedCartesianVelocityAction((0.0, -0.01, 0.0), sim_steps=25),
    ScriptedCartesianVelocityAction((0.0, 0.0, -0.01), sim_steps=25),
)


def deterministic_policy_actions() -> tuple[ScriptedCartesianVelocityAction, ...]:
    return DEFAULT_DETERMINISTIC_POLICY


def rollout_actions_for_env(
    actions: Iterable[ScriptedCartesianVelocityAction] = DEFAULT_DETERMINISTIC_POLICY,
) -> list[np.ndarray]:
    return [action.as_env_action() for action in actions]
