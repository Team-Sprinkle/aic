"""Official-style scoring utilities for teacher candidate selection.

This mirrors the public scoring docs and the `aic_scoring` implementation where
possible from the data available in `aic_gym_gz` artifacts. Partial insertion
depth remains approximate because the gym path does not currently expose the
official port-entrance TF needed by the scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
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
class OfficialStyleScore:
    total_score: float
    tier1: float
    tier2: dict[str, Any]
    tier3: dict[str, Any]
    selection: dict[str, Any]
    parity_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_score": self.total_score,
            "tier1": self.tier1,
            "tier2": self.tier2,
            "tier3": self.tier3,
            "selection": self.selection,
            "parity_notes": list(self.parity_notes),
        }


class OfficialStyleScoreEvaluator:
    """Most official-faithful scoring available from teacher rollout artifacts."""

    def __init__(
        self,
        *,
        near_perfect_threshold: float = 90.0,
        cable_settling_threshold: float = 0.75,
        partial_insertion_xy_tol_m: float = 0.005,
        approximate_port_depth_m: float = 0.015,
    ) -> None:
        self.near_perfect_threshold = near_perfect_threshold
        self.cable_settling_threshold = cable_settling_threshold
        self.partial_insertion_xy_tol_m = partial_insertion_xy_tol_m
        self.approximate_port_depth_m = approximate_port_depth_m

    def evaluate_rollout(self, rollout: dict[str, Any]) -> OfficialStyleScore:
        step_logs = rollout.get("step_logs", [])
        if not step_logs:
            return OfficialStyleScore(
                total_score=0.0,
                tier1=1.0,
                tier2={},
                tier3={"score": 0.0, "message": "No step logs available."},
                selection={"near_perfect": False, "cable_ready": False},
                parity_notes=["No rollout data was available for official-style scoring."],
            )
        observations = [entry["observation_summary"] for entry in step_logs]
        sim_times = [float(entry["sim_time"]) for entry in step_logs]
        tcp_positions = [np.asarray(obs["tcp_pose"][:3], dtype=np.float64) for obs in observations]
        tcp_linear_velocities = [np.asarray(obs["tcp_velocity"][:3], dtype=np.float64) for obs in observations]
        plug_positions = [np.asarray(obs["plug_pose"][:3], dtype=np.float64) for obs in observations]
        target_positions = [np.asarray(obs["target_port_pose"][:3], dtype=np.float64) for obs in observations]
        force_magnitudes = [float(np.linalg.norm(np.asarray(obs["wrench"][:3], dtype=np.float64))) for obs in observations]
        distances = [float(np.linalg.norm(plug - target)) for plug, target in zip(plug_positions, target_positions)]
        initial_distance = distances[0]
        final_distance = distances[-1]
        final_info = rollout.get("final_info", {})
        success = bool(final_info.get("success", False))
        wrong_port = bool(final_info.get("wrong_port", False))
        off_limit_contact = any(bool(obs["off_limit_contact"]) for obs in observations)
        duration = max(0.0, sim_times[-1] - sim_times[0])
        jerk = self._official_like_average_jerk(tcp_linear_velocities, sim_times)
        path_length = float(sum(np.linalg.norm(b - a) for a, b in zip(tcp_positions, tcp_positions[1:])))
        time_above_force = self._time_above_threshold(force_magnitudes, sim_times, threshold=20.0)
        tier3 = self._tier3_score(
            success=success,
            wrong_port=wrong_port,
            final_distance=final_distance,
            initial_distance=initial_distance,
            final_plug=plug_positions[-1],
            final_target=target_positions[-1],
        )
        tier3_score = float(tier3["score"])
        tier2 = {
            "duration": _inverse_score(
                max_score=12.0,
                min_score=0.0,
                max_range=60.0,
                min_range=5.0,
                measurement=duration,
            )
            if tier3_score > 0.0
            else 0.0,
            "trajectory_smoothness": _inverse_score(
                max_score=6.0,
                min_score=0.0,
                max_range=50.0,
                min_range=0.0,
                measurement=jerk,
            )
            if tier3_score > 0.0
            else 0.0,
            "trajectory_efficiency": _inverse_score(
                max_score=6.0,
                min_score=0.0,
                max_range=1.0 + initial_distance,
                min_range=initial_distance,
                measurement=path_length,
            )
            if tier3_score > 0.0
            else 0.0,
            "insertion_force": -12.0 if time_above_force > 1.0 else 0.0,
            "contacts": -24.0 if off_limit_contact else 0.0,
            "duration_s": duration,
            "jerk_mps3": jerk,
            "path_length_m": path_length,
            "time_above_20n_s": time_above_force,
        }
        latest_dynamics = step_logs[-1]["dynamics_summary"]
        cable_ready = bool(
            latest_dynamics["cable_settling_score"] >= self.cable_settling_threshold
            and latest_dynamics["plug_oscillation_magnitude"] <= 0.01
        )
        cable_settling_term = 1.5 if cable_ready else -2.0
        selection = {
            "near_perfect_threshold": self.near_perfect_threshold,
            "near_perfect": False,
            "cable_ready": cable_ready,
            "cable_settling_term": cable_settling_term,
        }
        total = 1.0 + tier3_score + float(tier2["duration"]) + float(tier2["trajectory_smoothness"]) + float(tier2["trajectory_efficiency"]) + float(tier2["insertion_force"]) + float(tier2["contacts"]) + cable_settling_term
        selection["near_perfect"] = bool(total >= self.near_perfect_threshold)
        parity_notes = [
            "Tier-2 jerk uses the official quadratic-window/Savitzky-Golay-style approximation over linear velocity samples.",
            "Tier-2 duration, efficiency, insertion-force, and off-limit-contact thresholds match docs and `aic_scoring`.",
            "Tier-3 partial insertion remains approximate because the gym path does not expose the official port-entrance TF.",
            "Tier-1 is assumed valid for generated teacher trajectories; official submission validity still depends on running through `aic_model`.",
        ]
        return OfficialStyleScore(
            total_score=total,
            tier1=1.0,
            tier2=tier2,
            tier3=tier3,
            selection=selection,
            parity_notes=parity_notes,
        )

    def _tier3_score(
        self,
        *,
        success: bool,
        wrong_port: bool,
        final_distance: float,
        initial_distance: float,
        final_plug: np.ndarray,
        final_target: np.ndarray,
    ) -> dict[str, Any]:
        if success:
            return {"score": 75.0, "message": "Cable insertion successful.", "status": "success"}
        if wrong_port:
            return {"score": -12.0, "message": "Cable insertion failed. Incorrect Port.", "status": "wrong_port"}
        dx = abs(float(final_plug[0] - final_target[0]))
        dy = abs(float(final_plug[1] - final_target[1]))
        dz = max(0.0, abs(float(final_plug[2] - final_target[2])))
        if dx < self.partial_insertion_xy_tol_m and dy < self.partial_insertion_xy_tol_m and dz < self.approximate_port_depth_m:
            score = _inverse_score(
                max_score=50.0,
                min_score=38.0,
                max_range=self.approximate_port_depth_m,
                min_range=0.0,
                measurement=dz,
            )
            return {
                "score": score,
                "message": f"Approximate partial insertion detected with depth residual {dz:.4f}m.",
                "status": "partial_insertion_approximate",
            }
        radius = initial_distance * 0.5
        score = _inverse_score(
            max_score=25.0,
            min_score=0.0,
            max_range=radius + self.approximate_port_depth_m,
            min_range=self.approximate_port_depth_m,
            measurement=final_distance,
        )
        return {
            "score": score,
            "message": f"No insertion detected. Final plug port distance: {final_distance:.4f}m.",
            "status": "proximity",
        }

    def _official_like_average_jerk(
        self,
        velocities: list[np.ndarray],
        times: list[float],
    ) -> float:
        window_size = 15
        half = window_size // 2
        if len(velocities) < window_size:
            return 0.0
        total_jerk_time = 0.0
        accum_jerk = 0.0
        for index in range(half, len(velocities) - half):
            speed = float(np.linalg.norm(velocities[index]))
            if speed <= 0.01:
                continue
            center_t = times[index]
            a = np.zeros((window_size, 3), dtype=np.float64)
            y = np.zeros((window_size, 3), dtype=np.float64)
            for j in range(window_size):
                data_idx = index - half + j
                dt = float(times[data_idx] - center_t)
                a[j] = [1.0, dt, dt * dt]
                y[j] = velocities[data_idx]
            coeffs, *_ = np.linalg.lstsq(a, y, rcond=None)
            jerk_vec = 2.0 * coeffs[2]
            t0 = times[index - half]
            t1 = times[index + half]
            dt = max((t1 - t0) / window_size, 1e-9)
            total_jerk_time += dt
            accum_jerk += float(np.linalg.norm(jerk_vec)) * dt
        if total_jerk_time <= 1e-9:
            return 0.0
        return accum_jerk / total_jerk_time

    def _time_above_threshold(
        self,
        force_magnitudes: list[float],
        times: list[float],
        *,
        threshold: float,
    ) -> float:
        total = 0.0
        for i in range(1, len(force_magnitudes)):
            if force_magnitudes[i] > threshold:
                total += max(0.0, times[i] - times[i - 1])
        return total
