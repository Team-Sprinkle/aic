"""Postprocess piecewise teacher plans into smooth replay trajectories."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from aic_teacher_official.trajectory import (
    PhaseLabel,
    PiecewiseTrajectory,
    SmoothTrajectory,
    SourceLabel,
    TCPPose,
    TrajectoryWaypoint,
    assert_monotonic_timestamps,
)


def minimum_jerk_fraction(progress: float) -> float:
    """Return the classic zero-velocity endpoint minimum-jerk blend."""
    s = float(np.clip(progress, 0.0, 1.0))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def minimum_jerk_fraction_derivative(progress: float) -> float:
    s = float(np.clip(progress, 0.0, 1.0))
    return 30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4


def normalize_quaternion_xyzw(quat: Iterable[float]) -> np.ndarray:
    q = np.asarray(list(quat), dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    q = q / norm
    if q[3] < 0.0:
        q = -q
    return q


def slerp_xyzw(q0: Iterable[float], q1: Iterable[float], fraction: float) -> list[float]:
    qa = normalize_quaternion_xyzw(q0)
    qb = normalize_quaternion_xyzw(q1)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion_xyzw(qa + fraction * (qb - qa)).tolist()
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * fraction
    scale_a = math.sin(theta_0 - theta) / sin_theta_0
    scale_b = math.sin(theta) / sin_theta_0
    return normalize_quaternion_xyzw(scale_a * qa + scale_b * qb).tolist()


def _compute_c1_waypoint_velocities(
    waypoints: list[TrajectoryWaypoint],
) -> list[np.ndarray]:
    """Estimate waypoint velocities for one global C1 Hermite trajectory."""
    positions = [
        np.asarray(waypoint.tcp_pose.position, dtype=np.float64)
        for waypoint in waypoints
    ]
    velocities: list[np.ndarray] = []
    for index, waypoint in enumerate(waypoints):
        if waypoint.tcp_velocity is not None:
            velocities.append(np.asarray(waypoint.tcp_velocity, dtype=np.float64))
            continue
        if index == 0 or index == len(waypoints) - 1:
            velocities.append(np.zeros(3, dtype=np.float64))
            continue
        prev_waypoint = waypoints[index - 1]
        next_waypoint = waypoints[index + 1]
        dt = next_waypoint.timestamp - prev_waypoint.timestamp
        if dt <= 0.0:
            raise ValueError("Piecewise timestamps must be strictly increasing")
        velocities.append((positions[index + 1] - positions[index - 1]) / dt)
    return velocities


def _hermite_position_velocity(
    p0: np.ndarray,
    p1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    duration: float,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cubic Hermite interpolation with continuous waypoint velocities."""
    s = float(np.clip(fraction, 0.0, 1.0))
    h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
    h10 = s**3 - 2.0 * s**2 + s
    h01 = -2.0 * s**3 + 3.0 * s**2
    h11 = s**3 - s**2
    position = h00 * p0 + h10 * duration * v0 + h01 * p1 + h11 * duration * v1

    dh00 = 6.0 * s**2 - 6.0 * s
    dh10 = 3.0 * s**2 - 4.0 * s + 1.0
    dh01 = -6.0 * s**2 + 6.0 * s
    dh11 = 3.0 * s**2 - 2.0 * s
    velocity = (dh00 * p0 + dh10 * duration * v0 + dh01 * p1 + dh11 * duration * v1) / duration
    return position, velocity


def postprocess_piecewise_trajectory(
    piecewise: PiecewiseTrajectory,
    *,
    sample_dt: float = 0.05,
) -> SmoothTrajectory:
    """Convert a piecewise plan to a smooth sampled trajectory.

    This version computes one global C1 cubic Hermite curve over all piecewise
    waypoints. It keeps the simple minimum-jerk helper available for tests and
    future time scaling, but avoids per-segment stop-and-go behavior by sharing
    velocity estimates across boundaries. Future work should replace this with
    a constrained spline/optimizer that preserves clearance and contact limits.
    """
    if sample_dt <= 0.0:
        raise ValueError("sample_dt must be positive")
    assert_monotonic_timestamps(piecewise.waypoints)

    waypoint_velocities = _compute_c1_waypoint_velocities(piecewise.waypoints)
    smooth: list[TrajectoryWaypoint] = []
    for segment_index, (start, end) in enumerate(
        zip(piecewise.waypoints, piecewise.waypoints[1:])
    ):
        duration = end.timestamp - start.timestamp
        if duration <= 0.0:
            raise ValueError("Piecewise segment duration must be positive")

        steps = max(1, int(math.ceil(duration / sample_dt)))
        start_pos = np.asarray(start.tcp_pose.position, dtype=np.float64)
        end_pos = np.asarray(end.tcp_pose.position, dtype=np.float64)
        start_velocity = waypoint_velocities[segment_index]
        end_velocity = waypoint_velocities[segment_index + 1]
        phase = end.phase
        segment_source = SourceLabel.CHEATCODE if phase == PhaseLabel.FINAL_INSERTION else end.source

        first_sample = 0 if segment_index == 0 else 1
        for i in range(first_sample, steps + 1):
            raw = i / steps
            timestamp = start.timestamp + duration * raw
            position, velocity = _hermite_position_velocity(
                start_pos,
                end_pos,
                start_velocity,
                end_velocity,
                duration,
                raw,
            )
            sample_phase = start.phase if segment_index == 0 and i == 0 else phase
            sample_source = start.source if segment_index == 0 and i == 0 else segment_source
            waypoint = TrajectoryWaypoint(
                timestamp=float(timestamp),
                tcp_pose=TCPPose(
                    position=position.tolist(),
                    orientation_xyzw=slerp_xyzw(
                        start.tcp_pose.orientation_xyzw,
                        end.tcp_pose.orientation_xyzw,
                        raw,
                    ),
                ),
                tcp_velocity=velocity.tolist(),
                gripper_state=end.gripper_state,
                cable_state=end.cable_state,
                port_state=end.port_state,
                phase=sample_phase,
                source=sample_source,
                diagnostics={
                    "postprocessor": "minimum_jerk_v0",
                    "global_smoother": "c1_cubic_hermite_v0",
                    "segment_index": segment_index,
                    "raw_fraction": raw,
                    "input_start_source": start.source.value,
                    "input_end_source": end.source.value,
                    "cheatcode_derived": sample_phase == PhaseLabel.FINAL_INSERTION,
                    "todo": "Replace independent segment interpolation with constrained continuous optimizer.",
                },
            )
            smooth.append(waypoint)

    assert_monotonic_timestamps(smooth)
    metadata = replace(
        piecewise.metadata,
        postprocessing={
            **piecewise.metadata.postprocessing,
            "method": "global_c1_cubic_hermite_v0",
            "sample_dt": sample_dt,
            "guarantees": [
                "strictly_monotonic_timestamps",
                "continuous_position",
                "continuous_velocity_at_piece_boundaries",
                "explicit_phase_labels",
                "source_labels_preserved_except_postprocessor_diagnostics",
                "final_insertion_marked_cheatcode_derived",
            ],
        },
    )
    return SmoothTrajectory(waypoints=smooth, metadata=metadata)


def postprocess_file(input_path: str | Path, output_path: str | Path, sample_dt: float) -> SmoothTrajectory:
    piecewise = PiecewiseTrajectory.load_json(input_path)
    smooth = postprocess_piecewise_trajectory(piecewise, sample_dt=sample_dt)
    smooth.save_json(output_path)
    return smooth
