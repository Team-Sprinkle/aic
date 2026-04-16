"""Gymnasium environment entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from .io import AicGazeboIO, MockGazeboIO, RosCameraSidecarIO
from .randomizer import AicEnvRandomizer
from .runtime import AicGazeboRuntime, MockStepperBackend, ScenarioGymGzBackend
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
        if self.task.include_images:
            state = self._warm_images_on_reset(state)
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
            "reward_label": "rl_step_reward",
            "score_label": "gym_final_score",
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
            final_evaluation = self.task.final_evaluation()
            info["final_evaluation"] = final_evaluation
            info["evaluation"] = final_evaluation
        self._state = current_state
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.io.close()
        self.runtime.close()

    def _warm_images_on_reset(self, state):
        try:
            self.io.observation_from_state(
                state,
                include_images=True,
                step_count=0,
            )
            return state
        except TimeoutError:
            zero_action = np.zeros(6, dtype=np.float32)
            warmed_state = state
            for _ in range(5):
                warmed_state = self.runtime.step(zero_action, ticks=2)
                try:
                    self.io.observation_from_state(
                        warmed_state,
                        include_images=True,
                        step_count=0,
                    )
                    return warmed_state
                except TimeoutError:
                    continue
            return warmed_state


def live_env_health_check(
    *,
    include_images: bool = False,
    enable_randomization: bool = False,
    ticks_per_step: int = 8,
    world_path: str | None = None,
    attach_to_existing: bool = False,
    transport_backend: str = "transport",
    seed: int = 123,
) -> dict[str, Any]:
    env = make_live_env(
        include_images=include_images,
        enable_randomization=enable_randomization,
        ticks_per_step=ticks_per_step,
        world_path=world_path,
        attach_to_existing=attach_to_existing,
        transport_backend=transport_backend,
    )
    try:
        observation, info = env.reset(seed=seed)
        summary = {
            "trial_id": info["trial_id"],
            "sim_tick": int(observation["sim_tick"]),
            "sim_time": float(observation["sim_time"]),
            "joint_count": int(observation["joint_positions"].shape[0]),
            "images_ready": False,
            "observation_keys": sorted(observation.keys()),
        }
        if include_images:
            summary["images_ready"] = all(
                observation["images"][name].shape == (64, 64, 3)
                and observation["images"][name].dtype == np.uint8
                and int(observation["images"][name].sum()) > 0
                for name in ("left", "center", "right")
            )
        return summary
    finally:
        env.close()


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


def make_live_env(
    *,
    include_images: bool = False,
    enable_randomization: bool = True,
    ticks_per_step: int = 8,
    world_path: str | None = None,
    attach_to_existing: bool = False,
    transport_backend: str = "transport",
) -> AicInsertionEnv:
    return AicInsertionEnv(
        runtime=AicGazeboRuntime(
            backend=ScenarioGymGzBackend(
                world_path=world_path,
                attach_to_existing=attach_to_existing,
                transport_backend=transport_backend,
            ),
            ticks_per_step=ticks_per_step,
        ),
        task=AicInsertionTask(
            hold_action_ticks=ticks_per_step,
            include_images=include_images,
        ),
        io=RosCameraSidecarIO() if include_images else MockGazeboIO(),
        randomizer=AicEnvRandomizer(enable_randomization=enable_randomization),
    )
