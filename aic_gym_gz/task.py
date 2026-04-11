"""Task logic for the standalone AIC insertion env."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from .reward import AicRewardBreakdown, AicScoreCalculator
from .runtime import RuntimeState
from .scenario import AicScenario, TaskDefinition


@dataclass
class EpisodeTrace:
    initial_distance: float
    sim_time: list[float] = field(default_factory=list)
    tcp_positions: list[np.ndarray] = field(default_factory=list)
    tcp_linear_velocity: list[np.ndarray] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
    force_magnitudes: list[float] = field(default_factory=list)
    off_limit_contacts: list[bool] = field(default_factory=list)
    success: bool = False
    wrong_port: bool = False


@dataclass
class AicInsertionTask:
    """Simulator-agnostic task logic and reward evaluation."""

    hold_action_ticks: int = 8
    frame: str = "base_link"
    max_episode_steps: int = 512
    include_images: bool = False
    score_calculator: AicScoreCalculator = field(default_factory=AicScoreCalculator)

    def __post_init__(self) -> None:
        self.action_space = gym.spaces.Box(
            low=np.array([-0.25, -0.25, -0.25, -2.0, -2.0, -2.0], dtype=np.float32),
            high=np.array([0.25, 0.25, 0.25, 2.0, 2.0, 2.0], dtype=np.float32),
            shape=(6,),
            dtype=np.float32,
        )
        base_spaces: dict[str, gym.Space[Any]] = {
            "joint_positions": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "joint_velocities": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "gripper_state": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            "tcp_velocity": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "plug_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            "target_port_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            "plug_to_port_relative": gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            "wrench": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "off_limit_contact": gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        }
        if self.include_images:
            image_space = gym.spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8)
            base_spaces["images"] = gym.spaces.Dict(
                {"left": image_space, "center": image_space, "right": image_space}
            )
            base_spaces["image_timestamps"] = gym.spaces.Box(
                0.0,
                np.inf,
                shape=(3,),
                dtype=np.float32,
            )
        self.observation_space = gym.spaces.Dict(base_spaces)
        self._trace: EpisodeTrace | None = None
        self._task: TaskDefinition | None = None

    def reset(self, *, scenario: AicScenario, initial_state: RuntimeState) -> None:
        self._task = next(iter(scenario.tasks.values()))
        initial_distance = float(
            np.linalg.norm(initial_state.plug_pose[:3] - initial_state.target_port_pose[:3])
        )
        self._trace = EpisodeTrace(initial_distance=initial_distance)
        self._record(initial_state)

    def evaluate_step(
        self,
        *,
        previous_state: RuntimeState,
        current_state: RuntimeState,
        action: np.ndarray,
        step_count: int,
    ) -> tuple[float, bool, bool, dict[str, Any]]:
        if self._trace is None:
            raise RuntimeError("Task must be reset before evaluate_step().")
        self._record(current_state)
        prev_dist = float(np.linalg.norm(previous_state.plug_pose[:3] - previous_state.target_port_pose[:3]))
        current_dist = self._trace.distances[-1]
        success = bool(current_state.insertion_event)
        wrong_port = bool(current_state.insertion_event) and current_state.insertion_event != (
            f"{self._task.target_module_name}/{self._task.port_name}" if self._task else ""
        )
        self._trace.success = success and not wrong_port
        self._trace.wrong_port = wrong_port
        breakdown = self.score_calculator.step_breakdown(
            previous_distance=prev_dist,
            current_distance=current_dist,
            action=action,
            force_magnitude=self._trace.force_magnitudes[-1],
            off_limit_contact=bool(current_state.off_limit_contact),
            success=self._trace.success,
            wrong_port=wrong_port,
        )
        terminated = self._trace.success or wrong_port
        truncated = step_count >= self.max_episode_steps
        if current_state.off_limit_contact:
            terminated = True
        info = {
            "reward_terms": breakdown.to_dict(),
            "distance_to_target": current_dist,
            "success": self._trace.success,
            "wrong_port": wrong_port,
        }
        return breakdown.total, terminated, truncated, info

    def final_evaluation(self) -> dict[str, Any]:
        if self._trace is None:
            raise RuntimeError("Task must be reset before final evaluation.")
        summary = self.score_calculator.evaluate(
            {
                "initial_distance": self._trace.initial_distance,
                "sim_time": self._trace.sim_time,
                "tcp_positions": self._trace.tcp_positions,
                "tcp_linear_velocity": self._trace.tcp_linear_velocity,
                "distances": self._trace.distances,
                "force_magnitudes": self._trace.force_magnitudes,
                "off_limit_contacts": self._trace.off_limit_contacts,
                "success": self._trace.success,
                "wrong_port": self._trace.wrong_port,
            }
        )
        return {
            "tier2": summary.tier2,
            "tier3": summary.tier3,
            "total_score": summary.total_score,
            "message": summary.message,
        }

    def _record(self, state: RuntimeState) -> None:
        assert self._trace is not None
        self._trace.sim_time.append(float(state.sim_time))
        self._trace.tcp_positions.append(state.tcp_pose[:3].copy())
        self._trace.tcp_linear_velocity.append(state.tcp_velocity[:3].copy())
        self._trace.distances.append(
            float(np.linalg.norm(state.plug_pose[:3] - state.target_port_pose[:3]))
        )
        self._trace.force_magnitudes.append(float(np.linalg.norm(state.wrench[:3])))
        self._trace.off_limit_contacts.append(bool(state.off_limit_contact))
