"""Reward shaping and official-like score decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


def _inverse_score(
    *,
    max_score: float,
    min_score: float,
    max_range: float,
    min_range: float,
    measurement: float,
) -> float:
    if measurement >= max_range:
        return min_score
    if measurement <= min_range:
        return max_score
    return min_score + ((max_range - measurement) / (max_range - min_range)) * (
        max_score - min_score
    )


@dataclass(frozen=True)
class AicRewardBreakdown:
    success_reward: float = 0.0
    wrong_port_penalty: float = 0.0
    partial_insertion_reward: float = 0.0
    proximity_reward: float = 0.0
    progress_reward: float = 0.0
    duration_penalty: float = 0.0
    path_efficiency_term: float = 0.0
    smoothness_term: float = 0.0
    excessive_force_penalty: float = 0.0
    off_limit_contact_penalty: float = 0.0

    @property
    def total(self) -> float:
        return float(sum(self.__dict__.values()))

    def to_dict(self) -> dict[str, float]:
        return {**self.__dict__, "total": self.total}


@dataclass(frozen=True)
class AicEvaluationSummary:
    tier2: dict[str, float]
    tier3: dict[str, float]
    total_score: float
    message: str


@dataclass
class AicScoreCalculator:
    """Implements official-like final scoring using stable env history."""

    def evaluate(self, episode: dict[str, Any]) -> AicEvaluationSummary:
        initial_distance = float(episode["initial_distance"])
        final_distance = float(episode["distances"][-1])
        duration = float(episode["sim_time"][-1] - episode["sim_time"][0])
        jerk = float(_average_linear_jerk(episode["tcp_linear_velocity"], episode["sim_time"]))
        path_length = float(_path_length(episode["tcp_positions"]))
        excessive_force_penalty = -12.0 if _time_above_force(episode["force_magnitudes"], episode["sim_time"]) > 1.0 else 0.0
        contacts_penalty = -24.0 if any(episode["off_limit_contacts"]) else 0.0

        partial = final_distance < 0.015
        success = bool(episode["success"])
        wrong_port = bool(episode["wrong_port"])
        if success:
            tier3_score = 75.0
            tier3_message = "Cable insertion successful."
        elif wrong_port:
            tier3_score = -12.0
            tier3_message = "Cable insertion failed. Incorrect port."
        elif partial:
            tier3_score = _inverse_score(
                max_score=50.0,
                min_score=38.0,
                max_range=0.015,
                min_range=0.0,
                measurement=final_distance,
            )
            tier3_message = "Partial insertion detected."
        else:
            radius = initial_distance * 0.5
            tier3_score = _inverse_score(
                max_score=25.0,
                min_score=0.0,
                max_range=radius + 0.015,
                min_range=0.015,
                measurement=final_distance,
            )
            tier3_message = "No insertion detected."

        tier2 = {
            "duration": _inverse_score(
                max_score=12.0,
                min_score=0.0,
                max_range=60.0,
                min_range=5.0,
                measurement=duration,
            )
            if tier3_score > 0
            else 0.0,
            "trajectory_smoothness": _inverse_score(
                max_score=6.0,
                min_score=0.0,
                max_range=50.0,
                min_range=0.0,
                measurement=jerk,
            )
            if tier3_score > 0
            else 0.0,
            "trajectory_efficiency": _inverse_score(
                max_score=6.0,
                min_score=0.0,
                max_range=1.0 + initial_distance,
                min_range=initial_distance,
                measurement=path_length,
            )
            if tier3_score > 0
            else 0.0,
            "insertion_force": excessive_force_penalty,
            "contacts": contacts_penalty,
        }
        total = 1.0 + tier3_score + float(sum(tier2.values()))
        return AicEvaluationSummary(
            tier2=tier2,
            tier3={"score": tier3_score},
            total_score=total,
            message=tier3_message,
        )

    def step_breakdown(
        self,
        *,
        previous_distance: float,
        current_distance: float,
        action: np.ndarray,
        force_magnitude: float,
        off_limit_contact: bool,
        success: bool,
        wrong_port: bool,
    ) -> AicRewardBreakdown:
        progress = previous_distance - current_distance
        proximity = max(0.0, 0.05 - current_distance) * 10.0
        smoothness = -0.01 * float(np.linalg.norm(action))
        path_efficiency = -0.005 * float(np.linalg.norm(action[:3]))
        return AicRewardBreakdown(
            success_reward=75.0 if success else 0.0,
            wrong_port_penalty=-12.0 if wrong_port else 0.0,
            partial_insertion_reward=25.0 if current_distance < 0.015 and not success else 0.0,
            proximity_reward=proximity if not success else 0.0,
            progress_reward=25.0 * progress,
            duration_penalty=-0.01,
            path_efficiency_term=path_efficiency,
            smoothness_term=smoothness,
            excessive_force_penalty=-12.0 if force_magnitude > 20.0 else 0.0,
            off_limit_contact_penalty=-24.0 if off_limit_contact else 0.0,
        )


def _path_length(positions: list[np.ndarray]) -> float:
    return float(
        sum(np.linalg.norm(np.asarray(b) - np.asarray(a)) for a, b in zip(positions, positions[1:]))
    )


def _time_above_force(force_magnitudes: list[float], times: list[float]) -> float:
    total = 0.0
    for i in range(1, len(force_magnitudes)):
        if force_magnitudes[i] > 20.0:
            total += times[i] - times[i - 1]
    return total


def _average_linear_jerk(velocities: list[np.ndarray], times: list[float]) -> float:
    if len(velocities) < 3:
        return 0.0
    samples: list[float] = []
    for i in range(2, len(velocities)):
        dt0 = times[i - 1] - times[i - 2]
        dt1 = times[i] - times[i - 1]
        if dt0 <= 0.0 or dt1 <= 0.0:
            continue
        accel0 = (np.asarray(velocities[i - 1]) - np.asarray(velocities[i - 2])) / dt0
        accel1 = (np.asarray(velocities[i]) - np.asarray(velocities[i - 1])) / dt1
        jerk = (accel1 - accel0) / max(dt1, 1e-9)
        speed = np.linalg.norm(velocities[i])
        if speed > 0.01:
            samples.append(float(np.linalg.norm(jerk)))
    return float(sum(samples) / len(samples)) if samples else 0.0
