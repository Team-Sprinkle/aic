"""Dense RL reward shaping and separate local final-score evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .runtime import RuntimeState


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
class AicRlRewardWeights:
    target_progress: float = 24.0
    entrance_progress: float = 18.0
    corridor_progress: float = 30.0
    orientation_progress: float = 6.0
    corridor_alignment: float = 1.5
    proximity: float = 1.25
    action_l2_penalty: float = -0.02
    action_delta_penalty: float = -0.04
    tcp_velocity_delta_penalty: float = -0.01
    force_penalty: float = -0.0015
    off_limit_contact_penalty: float = -20.0
    oscillation_penalty: float = -0.5
    time_penalty: float = -0.01
    partial_insertion_bonus: float = 1.5
    success_bonus: float = 75.0
    wrong_port_penalty: float = -12.0
    invalid_outcome_penalty: float = -8.0


@dataclass(frozen=True)
class AicRlRewardBreakdown:
    target_progress_reward: float = 0.0
    entrance_progress_reward: float = 0.0
    corridor_progress_reward: float = 0.0
    orientation_progress_reward: float = 0.0
    corridor_alignment_reward: float = 0.0
    proximity_reward: float = 0.0
    partial_insertion_bonus: float = 0.0
    action_l2_penalty: float = 0.0
    action_delta_penalty: float = 0.0
    tcp_velocity_delta_penalty: float = 0.0
    force_penalty: float = 0.0
    off_limit_contact_penalty: float = 0.0
    oscillation_penalty: float = 0.0
    time_penalty: float = 0.0
    success_bonus: float = 0.0
    wrong_port_penalty: float = 0.0
    invalid_outcome_penalty: float = 0.0

    @property
    def total(self) -> float:
        return float(sum(self.__dict__.values()))

    def to_dict(self) -> dict[str, float]:
        return {**self.__dict__, "total": self.total}


@dataclass(frozen=True)
class AicRewardMetrics:
    target_distance: float
    entrance_distance: float
    orientation_error: float
    insertion_progress: float
    lateral_misalignment: float
    force_magnitude: float
    partial_insertion: bool
    off_limit_contact: bool

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "target_distance": self.target_distance,
            "entrance_distance": self.entrance_distance,
            "orientation_error": self.orientation_error,
            "insertion_progress": self.insertion_progress,
            "lateral_misalignment": self.lateral_misalignment,
            "force_magnitude": self.force_magnitude,
            "partial_insertion": self.partial_insertion,
            "off_limit_contact": self.off_limit_contact,
        }


@dataclass(frozen=True)
class AicEvaluationSummary:
    tier2: dict[str, float]
    tier3: dict[str, float]
    total_score: float
    message: str
    parity_notes: list[str] = field(default_factory=list)


@dataclass
class AicRlRewardCalculator:
    """Isaac-Lab-style dense local reward for policy optimization."""

    weights: AicRlRewardWeights = field(default_factory=AicRlRewardWeights)
    proximity_length_scale: float = 0.05
    force_penalty_threshold: float = 10.0

    def metrics_from_state(self, state: RuntimeState) -> AicRewardMetrics:
        target_distance = float(
            state.score_geometry.get(
                "distance_to_target",
                np.linalg.norm(state.target_port_pose[:3] - state.plug_pose[:3]),
            )
        )
        entrance_pose = (
            state.target_port_entrance_pose
            if state.target_port_entrance_pose is not None
            else state.target_port_pose
        )
        entrance_distance = float(
            state.score_geometry.get(
                "distance_to_entrance",
                np.linalg.norm(entrance_pose[:3] - state.plug_pose[:3]),
            )
        )
        tracked_orientation_error = state.score_geometry.get("orientation_error")
        orientation_error = float(
            tracked_orientation_error
            if tracked_orientation_error is not None
            else abs(_wrap_to_pi(float(state.plug_pose[5] - state.target_port_pose[5])))
        )
        insertion_progress = float(np.clip(state.score_geometry.get("insertion_progress", 0.0), 0.0, 1.0))
        lateral_misalignment = float(max(state.score_geometry.get("lateral_misalignment", 0.0), 0.0))
        force_magnitude = float(np.linalg.norm(state.wrench[:3]))
        partial_insertion = bool(state.score_geometry.get("partial_insertion", False))
        off_limit_contact = bool(state.off_limit_contact)
        return AicRewardMetrics(
            target_distance=target_distance,
            entrance_distance=entrance_distance,
            orientation_error=orientation_error,
            insertion_progress=insertion_progress,
            lateral_misalignment=lateral_misalignment,
            force_magnitude=force_magnitude,
            partial_insertion=partial_insertion,
            off_limit_contact=off_limit_contact,
        )

    def evaluate_step(
        self,
        *,
        previous_state: RuntimeState,
        current_state: RuntimeState,
        action: np.ndarray,
        previous_action: np.ndarray | None,
        previous_metrics: AicRewardMetrics,
        current_metrics: AicRewardMetrics,
        success: bool,
        wrong_port: bool,
        invalid_outcome: bool = False,
        distance_history: list[float] | None = None,
    ) -> AicRlRewardBreakdown:
        weights = self.weights
        action_delta = (
            np.zeros_like(action, dtype=np.float64)
            if previous_action is None
            else action.astype(np.float64) - previous_action.astype(np.float64)
        )
        tcp_velocity_delta = current_state.tcp_velocity.astype(np.float64) - previous_state.tcp_velocity.astype(
            np.float64
        )

        target_progress_reward = weights.target_progress * (
            previous_metrics.target_distance - current_metrics.target_distance
        )
        entrance_progress_reward = weights.entrance_progress * (
            previous_metrics.entrance_distance - current_metrics.entrance_distance
        )
        corridor_progress_reward = weights.corridor_progress * (
            current_metrics.insertion_progress - previous_metrics.insertion_progress
        )
        orientation_progress_reward = weights.orientation_progress * (
            previous_metrics.orientation_error - current_metrics.orientation_error
        )
        corridor_alignment_reward = weights.corridor_alignment * (
            previous_metrics.lateral_misalignment - current_metrics.lateral_misalignment
        )
        proximity_reward = weights.proximity * (
            np.exp(-current_metrics.target_distance / self.proximity_length_scale)
            - np.exp(-previous_metrics.target_distance / self.proximity_length_scale)
        )
        partial_insertion_bonus = (
            weights.partial_insertion_bonus
            if current_metrics.partial_insertion and not previous_metrics.partial_insertion and not success
            else 0.0
        )
        action_l2_penalty = weights.action_l2_penalty * float(np.dot(action, action))
        action_delta_penalty = weights.action_delta_penalty * float(np.dot(action_delta, action_delta))
        tcp_velocity_delta_penalty = weights.tcp_velocity_delta_penalty * float(
            np.dot(tcp_velocity_delta, tcp_velocity_delta)
        )
        force_excess = max(0.0, current_metrics.force_magnitude - self.force_penalty_threshold)
        force_penalty = weights.force_penalty * float(force_excess * force_excess)
        off_limit_contact_penalty = (
            weights.off_limit_contact_penalty if current_metrics.off_limit_contact else 0.0
        )
        oscillation_penalty = 0.0
        if distance_history is not None and len(distance_history) >= 3:
            recent = distance_history[-3:]
            first_delta = recent[1] - recent[0]
            second_delta = recent[2] - recent[1]
            if first_delta * second_delta < 0.0 and abs(second_delta - first_delta) > 1e-4:
                oscillation_penalty = weights.oscillation_penalty * float(abs(second_delta - first_delta))
        return AicRlRewardBreakdown(
            target_progress_reward=target_progress_reward,
            entrance_progress_reward=entrance_progress_reward,
            corridor_progress_reward=corridor_progress_reward,
            orientation_progress_reward=orientation_progress_reward,
            corridor_alignment_reward=corridor_alignment_reward,
            proximity_reward=proximity_reward,
            partial_insertion_bonus=partial_insertion_bonus,
            action_l2_penalty=action_l2_penalty,
            action_delta_penalty=action_delta_penalty,
            tcp_velocity_delta_penalty=tcp_velocity_delta_penalty,
            force_penalty=force_penalty,
            off_limit_contact_penalty=off_limit_contact_penalty,
            oscillation_penalty=oscillation_penalty,
            time_penalty=weights.time_penalty,
            success_bonus=weights.success_bonus if success else 0.0,
            wrong_port_penalty=weights.wrong_port_penalty if wrong_port else 0.0,
            invalid_outcome_penalty=weights.invalid_outcome_penalty if invalid_outcome else 0.0,
        )


@dataclass
class AicScoreCalculator:
    """Local gazebo-gym final episode score approximation."""

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
            total_score=tier3_score + float(sum(tier2.values())),
            message=tier3_message,
            parity_notes=[
                "This report is the local gazebo-gym final score path (`gym_final_score` / `gym_reward`), not the official toolkit evaluation.",
                "Tier-2 jerk uses the same central-window style averaging approach as `aic_scoring`.",
                "Insertion force and off-limit contact terms are exact only to the extent that live wrench/contact topics are available.",
                *tier3_notes,
            ],
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

    insertion_axis = target_port_pose[:3] - target_port_entrance_pose[:3]
    insertion_axis_norm = float(np.linalg.norm(insertion_axis))
    if insertion_axis_norm <= 1e-8:
        notes.append("Target-port entrance geometry is degenerate; using target distance only.")
        return (
            _inverse_score(
                max_score=25.0,
                min_score=0.0,
                max_range=0.015 + radius,
                min_range=0.015,
                measurement=final_distance,
            ),
            f"No insertion detected. Final plug port distance: {final_distance:.4f} m.",
            notes,
        )
    axis_unit = insertion_axis / insertion_axis_norm
    plug_offset = final_plug_position - target_port_entrance_pose[:3]
    axial_depth = float(np.dot(plug_offset, axis_unit))
    lateral_offset = plug_offset - (axial_depth * axis_unit)
    lateral_distance = float(np.linalg.norm(lateral_offset))
    clipped_axial_depth = float(np.clip(axial_depth, 0.0, insertion_axis_norm))

    in_partial_insertion = bool(lateral_distance < 0.005 and clipped_axial_depth > 0.0)
    if in_partial_insertion:
        return (
            _inverse_score(
                max_score=50.0,
                min_score=38.0,
                max_range=insertion_axis_norm,
                min_range=0.0,
                measurement=clipped_axial_depth,
            ),
            f"Partial insertion detected with insertion depth of {clipped_axial_depth:.4f} m.",
            [],
        )
    return (
        _inverse_score(
            max_score=25.0,
            min_score=0.0,
            max_range=insertion_axis_norm + radius,
            min_range=insertion_axis_norm,
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
    linear_velocities = np.asarray([np.asarray(v, dtype=np.float64)[:3] for v in velocities], dtype=np.float64)
    timestamps = np.asarray(times, dtype=np.float64)
    accelerations: list[np.ndarray] = []
    acceleration_times: list[float] = []
    for i in range(1, len(linear_velocities)):
        dt = timestamps[i] - timestamps[i - 1]
        if dt <= 0.0:
            continue
        accelerations.append((linear_velocities[i] - linear_velocities[i - 1]) / dt)
        acceleration_times.append((timestamps[i] + timestamps[i - 1]) * 0.5)
    if len(accelerations) < 3:
        return 0.0
    jerk_norms: list[float] = []
    for i in range(1, len(accelerations) - 1):
        dt = acceleration_times[i + 1] - acceleration_times[i - 1]
        if dt <= 0.0:
            continue
        jerk = (accelerations[i + 1] - accelerations[i - 1]) / dt
        jerk_norms.append(float(np.linalg.norm(jerk)))
    if not jerk_norms:
        return 0.0
    return float(np.mean(jerk_norms))


def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
