"""Task logic for the standalone AIC insertion env."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from .reward import AicRewardMetrics, AicRlRewardCalculator, AicScoreCalculator
from .runtime import RuntimeState
from .scenario import AicScenario, TaskDefinition


@dataclass
class EpisodeTrace:
    initial_distance: float
    initial_plug_pose: np.ndarray
    target_port_pose: np.ndarray
    target_port_entrance_pose: np.ndarray | None
    sim_time: list[float] = field(default_factory=list)
    tcp_positions: list[np.ndarray] = field(default_factory=list)
    tcp_linear_velocity: list[np.ndarray] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
    plug_positions: list[np.ndarray] = field(default_factory=list)
    force_magnitudes: list[float] = field(default_factory=list)
    wrench_samples: list[np.ndarray] = field(default_factory=list)
    wrench_time: list[float] = field(default_factory=list)
    off_limit_contacts: list[bool] = field(default_factory=list)
    rl_step_rewards: list[float] = field(default_factory=list)
    last_reward_metrics: AicRewardMetrics | None = None
    last_action: np.ndarray | None = None
    success: bool = False
    wrong_port: bool = False


@dataclass
class AicInsertionTask:
    """Simulator-agnostic task logic and reward evaluation."""

    hold_action_ticks: int = 8
    frame: str = "base_link"
    max_episode_steps: int = 512
    include_images: bool = False
    rl_reward_calculator: AicRlRewardCalculator = field(default_factory=AicRlRewardCalculator)
    score_calculator: AicScoreCalculator = field(default_factory=AicScoreCalculator)

    def __post_init__(self) -> None:
        self.action_space = gym.spaces.Box(
            low=np.array([-0.25, -0.25, -0.25, -2.0, -2.0, -2.0], dtype=np.float32),
            high=np.array([0.25, 0.25, 0.25, 2.0, 2.0, 2.0], dtype=np.float32),
            shape=(6,),
            dtype=np.float32,
        )
        base_spaces: dict[str, gym.Space[Any]] = {
            "step_count": gym.spaces.Box(0.0, np.inf, shape=(), dtype=np.float32),
            "sim_tick": gym.spaces.Box(0.0, np.inf, shape=(), dtype=np.float32),
            "sim_time": gym.spaces.Box(0.0, np.inf, shape=(), dtype=np.float32),
            "joint_positions": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "joint_velocities": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "gripper_state": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            "tcp_velocity": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "plug_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            "target_port_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            "target_port_entrance_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            "plug_to_port_relative": gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            "wrench": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "wrench_timestamp": gym.spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
            "off_limit_contact": gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "controller_tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            "controller_reference_tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            "controller_tcp_velocity": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "controller_tcp_error": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "controller_reference_joint_state": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "controller_target_mode": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "fts_tare_wrench": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "score_geometry": gym.spaces.Dict(
                {
                    "distance_to_target": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                    "distance_threshold": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                    "plug_to_port_depth": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                    "port_to_entrance_depth": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                    "distance_to_entrance": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                    "lateral_misalignment": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                    "orientation_error": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                    "insertion_progress": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                    "partial_insertion": gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                }
            ),
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
            base_spaces["camera_info"] = gym.spaces.Dict(
                {
                    name: gym.spaces.Dict(
                        {
                            "size": gym.spaces.Box(0.0, np.inf, shape=(2,), dtype=np.float32),
                            "k": gym.spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32),
                            "p": gym.spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32),
                        }
                    )
                    for name in ("left", "center", "right")
                }
            )
        self.observation_space = gym.spaces.Dict(base_spaces)
        self._trace: EpisodeTrace | None = None
        self._task: TaskDefinition | None = None

    def reset(self, *, scenario: AicScenario, initial_state: RuntimeState) -> None:
        self._task = next(iter(scenario.tasks.values()))
        initial_distance = float(
            np.linalg.norm(initial_state.plug_pose[:3] - initial_state.target_port_pose[:3])
        )
        self._trace = EpisodeTrace(
            initial_distance=initial_distance,
            initial_plug_pose=initial_state.plug_pose.copy(),
            target_port_pose=initial_state.target_port_pose.copy(),
            target_port_entrance_pose=(
                None
                if initial_state.target_port_entrance_pose is None
                else initial_state.target_port_entrance_pose.copy()
            ),
        )
        self._record(initial_state)
        self._trace.last_reward_metrics = self.rl_reward_calculator.metrics_from_state(initial_state)
        self._trace.last_action = np.zeros(self.action_space.shape, dtype=np.float64)

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
        previous_metrics = (
            self._trace.last_reward_metrics
            if self._trace.last_reward_metrics is not None
            else self.rl_reward_calculator.metrics_from_state(previous_state)
        )
        current_metrics = self.rl_reward_calculator.metrics_from_state(current_state)
        current_dist = current_metrics.target_distance
        success = bool(current_state.insertion_event)
        wrong_port = bool(current_state.insertion_event) and current_state.insertion_event != (
            f"{self._task.target_module_name}/{self._task.port_name}" if self._task else ""
        )
        self._trace.success = success and not wrong_port
        self._trace.wrong_port = wrong_port
        breakdown = self.rl_reward_calculator.evaluate_step(
            previous_state=previous_state,
            current_state=current_state,
            action=action,
            previous_action=self._trace.last_action,
            previous_metrics=previous_metrics,
            current_metrics=current_metrics,
            success=self._trace.success,
            wrong_port=wrong_port,
            distance_history=self._trace.distances,
        )
        rl_step_reward = breakdown.total
        self._trace.rl_step_rewards.append(rl_step_reward)
        self._trace.last_reward_metrics = current_metrics
        self._trace.last_action = action.astype(np.float64, copy=True)
        terminated = self._trace.success or wrong_port
        truncated = step_count >= self.max_episode_steps
        if current_state.off_limit_contact:
            terminated = True
        info = {
            "reward_label": "rl_step_reward",
            "rl_step_reward": rl_step_reward,
            "reward_terms": breakdown.to_dict(),
            "reward_metrics": current_metrics.to_dict(),
            "distance_to_target": current_dist,
            "distance_to_entrance": current_metrics.entrance_distance,
            "training_reward_total": float(sum(self._trace.rl_step_rewards)),
            "success": self._trace.success,
            "wrong_port": wrong_port,
        }
        return rl_step_reward, terminated, truncated, info

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
                "plug_positions": self._trace.plug_positions,
                "target_port_pose": self._trace.target_port_pose,
                "target_port_entrance_pose": self._trace.target_port_entrance_pose,
                "force_magnitudes": self._trace.force_magnitudes,
                "wrench_time": self._trace.wrench_time,
                "wrench_samples": self._trace.wrench_samples,
                "off_limit_contacts": self._trace.off_limit_contacts,
                "success": self._trace.success,
                "wrong_port": self._trace.wrong_port,
            }
        )
        return {
            "score_label": "gym_final_score",
            "gym_reward": summary.total_score,
            "gym_final_score": summary.total_score,
            "official_eval_score": None,
            "tier2": summary.tier2,
            "tier3": summary.tier3,
            "total_score": summary.total_score,
            "training_reward_label": "rl_step_reward",
            "training_reward_total": float(sum(self._trace.rl_step_rewards)),
            "message": summary.message,
            "parity_notes": summary.parity_notes,
            "approximation_notes": summary.parity_notes,
        }

    def _record(self, state: RuntimeState) -> None:
        assert self._trace is not None
        self._trace.sim_time.append(float(state.sim_time))
        self._trace.tcp_positions.append(state.tcp_pose[:3].copy())
        self._trace.tcp_linear_velocity.append(state.tcp_velocity[:3].copy())
        self._trace.plug_positions.append(state.plug_pose[:3].copy())
        self._trace.distances.append(
            float(np.linalg.norm(state.plug_pose[:3] - state.target_port_pose[:3]))
        )
        self._trace.force_magnitudes.append(float(np.linalg.norm(state.wrench[:3])))
        self._trace.wrench_samples.append(state.wrench.copy())
        self._trace.wrench_time.append(float(state.wrench_timestamp or state.sim_time))
        self._trace.off_limit_contacts.append(bool(state.off_limit_contact))
