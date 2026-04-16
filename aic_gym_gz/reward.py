"""Reward shaping and gym-side official-score approximation."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    parity_notes: list[str] = field(default_factory=list)


@dataclass
class AicScoreCalculator:
    """Implements a gym-side score path aligned to `aic_scoring` where possible."""

    def evaluate(self, episode: dict[str, Any]) -> AicEvaluationSummary:
        initial_distance = float(episode["initial_distance"])
        target_port_pose = np.asarray(
            episode.get("target_port_pose", np.zeros(7, dtype=np.float64)),
            dtype=np.float64,
        )
        entrance_pose_raw = episode.get("target_port_entrance_pose")
        target_port_entrance_pose = (
            None
            if entrance_pose_raw is None
            else np.asarray(entrance_pose_raw, dtype=np.float64)
        )
        plug_positions = episode.get("plug_positions")
        if plug_positions:
            final_plug_position = np.asarray(plug_positions[-1], dtype=np.float64)
        else:
            final_distance = float(episode["distances"][-1])
            final_plug_position = target_port_pose[:3] + np.array(
                [final_distance, 0.0, 0.0],
                dtype=np.float64,
            )
        final_distance = float(np.linalg.norm(final_plug_position - target_port_pose[:3]))
        duration = float(episode["sim_time"][-1] - episode["sim_time"][0])
        jerk = float(_official_average_linear_jerk(episode["tcp_linear_velocity"], episode["sim_time"]))
        path_length = float(_path_length(episode["tcp_positions"]))
        excessive_force_penalty = (
            -12.0
            if _time_above_force(episode.get("wrench_samples", []), episode.get("wrench_time", [])) > 1.0
            else 0.0
        )
        contacts_penalty = -24.0 if any(episode["off_limit_contacts"]) else 0.0

        tier3_score, tier3_message, tier3_notes = _tier3_score(
            success=bool(episode["success"]),
            wrong_port=bool(episode["wrong_port"]),
            initial_distance=initial_distance,
            final_plug_position=final_plug_position,
            target_port_pose=target_port_pose,
            target_port_entrance_pose=target_port_entrance_pose,
        )

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
        return AicEvaluationSummary(
            tier2=tier2,
            tier3={"score": tier3_score},
            total_score=1.0 + tier3_score + float(sum(tier2.values())),
            message=tier3_message,
            parity_notes=[
                "This score is the local gazebo-gym score path (`gym_reward`), not the official toolkit evaluation.",
                "Tier-2 jerk uses the same central-window style averaging approach as `aic_scoring`.",
                "Insertion force and off-limit contact terms are exact only to the extent that live wrench/contact topics are available.",
                *tier3_notes,
            ],
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
        partial_insertion: bool = False,
    ) -> AicRewardBreakdown:
        progress = previous_distance - current_distance
        proximity = max(0.0, 0.05 - current_distance) * 10.0
        smoothness = -0.01 * float(np.linalg.norm(action))
        path_efficiency = -0.005 * float(np.linalg.norm(action[:3]))
        return AicRewardBreakdown(
            success_reward=75.0 if success else 0.0,
            wrong_port_penalty=-12.0 if wrong_port else 0.0,
            partial_insertion_reward=38.0 if partial_insertion and not success else 0.0,
            proximity_reward=proximity if not success else 0.0,
            progress_reward=25.0 * progress,
            duration_penalty=-0.01,
            path_efficiency_term=path_efficiency,
            smoothness_term=smoothness,
            excessive_force_penalty=-12.0 if force_magnitude > 20.0 else 0.0,
            off_limit_contact_penalty=-24.0 if off_limit_contact else 0.0,
        )


def _tier3_score(
    *,
    success: bool,
    wrong_port: bool,
    initial_distance: float,
    final_plug_position: np.ndarray,
    target_port_pose: np.ndarray,
    target_port_entrance_pose: np.ndarray | None,
) -> tuple[float, str, list[str]]:
    if success:
        return 75.0, "Cable insertion successful.", []
    if wrong_port:
        return -12.0, "Cable insertion failed. Incorrect port.", []

    final_distance = float(np.linalg.norm(final_plug_position - target_port_pose[:3]))
    radius = initial_distance * 0.5
    notes: list[str] = []
    if target_port_entrance_pose is None:
        notes.append(
            "Tier-3 partial insertion remains approximate because the gym path did not expose a port-entrance transform."
        )
        return (
            _inverse_score(
                max_score=25.0,
                min_score=0.0,
                max_range=0.015 + radius,
                min_range=0.015,
                measurement=final_distance,
            ),
            "No insertion detected.",
            notes,
        )

    distance_threshold = float(abs(target_port_entrance_pose[2] - target_port_pose[2]))
    in_partial_insertion = bool(
        abs(final_plug_position[0] - target_port_pose[0]) < 0.005
        and abs(final_plug_position[1] - target_port_pose[1]) < 0.005
        and final_plug_position[2] < target_port_entrance_pose[2]
        and final_plug_position[2] - target_port_pose[2] > -0.01
    )
    if in_partial_insertion:
        plug_to_port_dist = float(final_plug_position[2] - target_port_pose[2])
        port_to_entrance_dist = float(target_port_entrance_pose[2] - target_port_pose[2])
        return (
            _inverse_score(
                max_score=50.0,
                min_score=38.0,
                max_range=port_to_entrance_dist,
                min_range=0.0,
                measurement=plug_to_port_dist,
            ),
            f"Partial insertion detected with distance of {plug_to_port_dist:.4f} m.",
            [],
        )
    return (
        _inverse_score(
            max_score=25.0,
            min_score=0.0,
            max_range=distance_threshold + radius,
            min_range=distance_threshold,
            measurement=final_distance,
        ),
        f"No insertion detected. Final plug port distance: {final_distance:.4f} m.",
        [],
    )


def _path_length(positions: list[np.ndarray]) -> float:
    return float(
        sum(np.linalg.norm(np.asarray(b) - np.asarray(a)) for a, b in zip(positions, positions[1:]))
    )


def _time_above_force(wrench_samples: list[np.ndarray], times: list[float]) -> float:
    total = 0.0
    for i in range(1, min(len(wrench_samples), len(times))):
        if np.linalg.norm(np.asarray(wrench_samples[i], dtype=np.float64)[:3]) > 20.0:
            total += float(times[i] - times[i - 1])
    return total


def _official_average_linear_jerk(velocities: list[np.ndarray], times: list[float]) -> float:
    if len(velocities) < 5:
        return 0.0
    k = 2
    total_jerk_time = 0.0
    accum_linear_jerk = 0.0
    vectors = [np.asarray(sample, dtype=np.float64) for sample in velocities]
    for i in range(k, len(vectors) - k):
        speed = float(np.linalg.norm(vectors[i]))
        if speed <= 0.01:
            continue
        t0 = float(times[i - k])
        t1 = float(times[i + k])
        if t1 <= t0:
            continue
        dt = (t1 - t0) / (2 * k)
        accel_prev = (vectors[i] - vectors[i - 1]) / max(float(times[i] - times[i - 1]), 1e-9)
        accel_next = (vectors[i + 1] - vectors[i]) / max(float(times[i + 1] - times[i]), 1e-9)
        jerk = (accel_next - accel_prev) / max(dt, 1e-9)
        jerk_mag = float(np.linalg.norm(jerk))
        total_jerk_time += dt
        accum_linear_jerk += jerk_mag * dt
    return accum_linear_jerk / total_jerk_time if total_jerk_time > 1e-9 else 0.0
