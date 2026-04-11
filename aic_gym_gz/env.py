"""Gymnasium environment entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from .io import AicGazeboIO, MockGazeboIO
from .randomizer import AicEnvRandomizer
from .runtime import AicGazeboRuntime, MockStepperBackend
from .task import AicInsertionTask


@dataclass
class AicInsertionEnv(gym.Env[dict[str, Any], np.ndarray]):
    metadata = {"render_modes": []}

    runtime: AicGazeboRuntime
    task: AicInsertionTask
    io: AicGazeboIO
    randomizer: AicEnvRandomizer
    trial_id: str | None = None

    def __post_init__(self) -> None:
        self.action_space = self.task.action_space
        self.observation_space = self.task.observation_space
        self._scenario = None
        self._state = None
        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        scenario = self.randomizer.sample(
            seed=seed,
            trial_id=str(options.get("trial_id", self.trial_id)) if options.get("trial_id", self.trial_id) else None,
        )
        state = self.runtime.reset(seed=seed, scenario=scenario)
        self.task.reset(scenario=scenario, initial_state=state)
        self._scenario = scenario
        self._state = state
        self._step_count = 0
        observation = self.io.observation_from_state(
            state,
            include_images=self.task.include_images,
            step_count=self._step_count,
        )
        info = {
            "trial_id": scenario.trial_id,
            "scenario_metadata": scenario.metadata,
            "task_definition": next(iter(scenario.tasks.values())).__dict__,
        }
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Environment must be reset before step().")
        sanitized = self.io.sanitize_action(action)
        previous_state = self._state
        current_state = self.runtime.step(sanitized, ticks=self.task.hold_action_ticks)
        self._step_count += 1
        observation = self.io.observation_from_state(
            current_state,
            include_images=self.task.include_images,
            step_count=self._step_count,
        )
        reward, terminated, truncated, info = self.task.evaluate_step(
            previous_state=previous_state,
            current_state=current_state,
            action=sanitized,
            step_count=self._step_count,
        )
        if terminated or truncated:
            info["evaluation"] = self.task.final_evaluation()
        self._state = current_state
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.runtime.close()


def make_default_env(
    *,
    include_images: bool = False,
    enable_randomization: bool = True,
    ticks_per_step: int = 8,
) -> AicInsertionEnv:
    return AicInsertionEnv(
        runtime=AicGazeboRuntime(
            backend=MockStepperBackend(),
            ticks_per_step=ticks_per_step,
        ),
        task=AicInsertionTask(
            hold_action_ticks=ticks_per_step,
            include_images=include_images,
        ),
        io=MockGazeboIO(),
        randomizer=AicEnvRandomizer(enable_randomization=enable_randomization),
    )
