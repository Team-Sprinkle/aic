"""Postprocess piecewise teacher plans into smooth replay trajectories."""

from __future__ import annotations

from dataclasses import replace
import math
import os
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


def insertion_start_waypoint_index(waypoints: list[TrajectoryWaypoint]) -> int | None:
    for index, waypoint in enumerate(waypoints):
        if waypoint.phase == PhaseLabel.FINAL_INSERTION:
            return max(0, index - 1)
    return None


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


def _linear_position_velocity(
    p0: np.ndarray,
    p1: np.ndarray,
    duration: float,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    s = float(np.clip(fraction, 0.0, 1.0))
    velocity = (p1 - p0) / duration
    return p0 + s * (p1 - p0), velocity


def _quat_angle_rad(q0: Iterable[float], q1: Iterable[float]) -> float:
    qa = normalize_quaternion_xyzw(q0)
    qb = normalize_quaternion_xyzw(q1)
    dot = abs(float(np.dot(qa, qb)))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def compact_stall_intervals(
    trajectory: SmoothTrajectory,
    *,
    speedup: float = 2.0,
    min_segment_sec: float = 0.05,
    stall_translation_m: float = 0.00025,
    stall_rotation_rad: float = 0.002,
) -> SmoothTrajectory:
    """Drop repeated low-motion samples and retime the remaining waypoints.

    This is intentionally geometric: it compacts intervals where adjacent TCP
    poses barely move, then recomputes timestamps and TCP velocities. It does
    not change waypoint poses or phases.
    """
    if speedup <= 0.0:
        raise ValueError("speedup must be positive")
    if min_segment_sec <= 0.0:
        raise ValueError("min_segment_sec must be positive")
    if len(trajectory.waypoints) < 2:
        return trajectory

    compacted = [trajectory.waypoints[0]]
    removed = 0
    for waypoint in trajectory.waypoints[1:-1]:
        prev = compacted[-1]
        distance = float(
            np.linalg.norm(
                np.asarray(waypoint.tcp_pose.position, dtype=np.float64)
                - np.asarray(prev.tcp_pose.position, dtype=np.float64)
            )
        )
        angle = _quat_angle_rad(prev.tcp_pose.orientation_xyzw, waypoint.tcp_pose.orientation_xyzw)
        same_phase = waypoint.phase == prev.phase
        if same_phase and distance < stall_translation_m and angle < stall_rotation_rad:
            removed += 1
            continue
        compacted.append(waypoint)
    compacted.append(trajectory.waypoints[-1])

    retimed: list[TrajectoryWaypoint] = []
    current_time = 0.0
    for index, waypoint in enumerate(compacted):
        if index == 0:
            retimed.append(replace(waypoint, timestamp=0.0))
            continue
        prev_original = compacted[index - 1]
        original_dt = max(min_segment_sec, waypoint.timestamp - prev_original.timestamp)
        new_dt = max(min_segment_sec, original_dt / speedup)
        current_time += new_dt
        position_delta = (
            np.asarray(waypoint.tcp_pose.position, dtype=np.float64)
            - np.asarray(prev_original.tcp_pose.position, dtype=np.float64)
        )
        retimed.append(
            replace(
                waypoint,
                timestamp=float(current_time),
                tcp_velocity=(position_delta / new_dt).tolist(),
                diagnostics={
                    **waypoint.diagnostics,
                    "stall_compaction": "retimed_after_low_motion_removal_v1",
                },
            )
        )
    assert_monotonic_timestamps(retimed)
    metadata = replace(
        trajectory.metadata,
        postprocessing={
            **trajectory.metadata.postprocessing,
            "stall_compaction": {
                "method": "drop_low_motion_samples_and_retime_v1",
                "speedup": speedup,
                "min_segment_sec": min_segment_sec,
                "stall_translation_m": stall_translation_m,
                "stall_rotation_rad": stall_rotation_rad,
                "input_waypoints": len(trajectory.waypoints),
                "output_waypoints": len(retimed),
                "removed_waypoints": removed,
                "input_duration_sec": trajectory.waypoints[-1].timestamp - trajectory.waypoints[0].timestamp,
                "output_duration_sec": retimed[-1].timestamp - retimed[0].timestamp,
            },
        },
    )
    return SmoothTrajectory(waypoints=retimed, metadata=metadata)


def retime_tcp_speed_profile(
    trajectory: SmoothTrajectory,
    *,
    max_tcp_speed_mps: float = 0.025,
    max_tcp_accel_mps2: float = 0.08,
    approach_max_tcp_speed_mps: float | None = 0.14,
    approach_max_tcp_accel_mps2: float | None = 0.45,
    min_segment_sec: float = 0.05,
    approach_stall_translation_m: float = 0.00025,
) -> SmoothTrajectory:
    """Retimestamp waypoints to bound replay TCP speed and speed changes.

    MoveIt can return geometrically valid plans whose segment timing changes
    sharply between replans. This pass keeps the waypoint poses and joint
    targets unchanged, but stretches timestamps so replay commands do not jump
    between very different TCP speeds.
    """
    if max_tcp_speed_mps <= 0.0:
        raise ValueError("max_tcp_speed_mps must be positive")
    if max_tcp_accel_mps2 <= 0.0:
        raise ValueError("max_tcp_accel_mps2 must be positive")
    if approach_max_tcp_speed_mps is not None and approach_max_tcp_speed_mps <= 0.0:
        raise ValueError("approach_max_tcp_speed_mps must be positive")
    if approach_max_tcp_accel_mps2 is not None and approach_max_tcp_accel_mps2 <= 0.0:
        raise ValueError("approach_max_tcp_accel_mps2 must be positive")
    if min_segment_sec <= 0.0:
        raise ValueError("min_segment_sec must be positive")
    if approach_stall_translation_m < 0.0:
        raise ValueError("approach_stall_translation_m must be non-negative")
    if len(trajectory.waypoints) < 2:
        return trajectory

    filtered_waypoints: list[TrajectoryWaypoint] = [trajectory.waypoints[0]]
    skipped_approach_stalls = 0
    for waypoint in trajectory.waypoints[1:-1]:
        prev = filtered_waypoints[-1]
        same_phase = waypoint.phase == prev.phase
        distance = float(
            np.linalg.norm(
                np.asarray(waypoint.tcp_pose.position, dtype=np.float64)
                - np.asarray(prev.tcp_pose.position, dtype=np.float64)
            )
        )
        if (
            same_phase
            and _is_approach_phase(waypoint.phase)
            and distance <= approach_stall_translation_m
        ):
            skipped_approach_stalls += 1
            continue
        filtered_waypoints.append(waypoint)
    filtered_waypoints.append(trajectory.waypoints[-1])

    retimed: list[TrajectoryWaypoint] = [
        replace(
            filtered_waypoints[0],
            timestamp=0.0,
            tcp_velocity=[0.0, 0.0, 0.0],
            diagnostics={
                **filtered_waypoints[0].diagnostics,
                "tcp_speed_retiming": "bounded_tcp_speed_profile_v1",
            },
        )
    ]
    current_time = 0.0
    previous_speed = 0.0
    max_observed_speed = 0.0
    max_observed_speed_delta = 0.0
    for waypoint in filtered_waypoints[1:]:
        prev = retimed[-1]
        segment_speed_limit = _phase_speed_limit(
            waypoint.phase,
            default=max_tcp_speed_mps,
            approach=approach_max_tcp_speed_mps,
        )
        segment_accel_limit = _phase_speed_limit(
            waypoint.phase,
            default=max_tcp_accel_mps2,
            approach=approach_max_tcp_accel_mps2,
        )
        preserve_original_dt = not _is_approach_phase(waypoint.phase)
        if preserve_original_dt:
            original_prev = filtered_waypoints[len(retimed) - 1]
            dt = max(min_segment_sec, waypoint.timestamp - original_prev.timestamp)
        else:
            dt = min_segment_sec
        delta = np.asarray(waypoint.tcp_pose.position, dtype=np.float64) - np.asarray(
            prev.tcp_pose.position,
            dtype=np.float64,
        )
        distance = float(np.linalg.norm(delta))
        if distance > 1e-12:
            dt = max(dt, distance / segment_speed_limit)
            for _ in range(6):
                speed = distance / dt
                dt = max(dt, abs(speed - previous_speed) / segment_accel_limit)
        current_time += dt
        velocity = (delta / dt).tolist() if dt > 0.0 else [0.0, 0.0, 0.0]
        speed = float(np.linalg.norm(velocity))
        max_observed_speed = max(max_observed_speed, speed)
        max_observed_speed_delta = max(max_observed_speed_delta, abs(speed - previous_speed))
        previous_speed = speed
        retimed.append(
            replace(
                waypoint,
                timestamp=float(current_time),
                tcp_velocity=velocity,
                diagnostics={
                    **waypoint.diagnostics,
                    "tcp_speed_retiming": "bounded_tcp_speed_profile_v1",
                },
            )
        )

    assert_monotonic_timestamps(retimed)
    metadata = replace(
        trajectory.metadata,
        postprocessing={
            **trajectory.metadata.postprocessing,
            "tcp_speed_retiming": {
                "method": "bounded_tcp_speed_profile_v1",
                "max_tcp_speed_mps": max_tcp_speed_mps,
                "max_tcp_accel_mps2": max_tcp_accel_mps2,
                "approach_max_tcp_speed_mps": approach_max_tcp_speed_mps,
                "approach_max_tcp_accel_mps2": approach_max_tcp_accel_mps2,
                "min_segment_sec": min_segment_sec,
                "approach_stall_translation_m": approach_stall_translation_m,
                "skipped_approach_stall_waypoints": skipped_approach_stalls,
                "input_duration_sec": trajectory.waypoints[-1].timestamp - trajectory.waypoints[0].timestamp,
                "output_duration_sec": retimed[-1].timestamp - retimed[0].timestamp,
                "max_observed_speed_mps": max_observed_speed,
                "max_observed_speed_delta_mps": max_observed_speed_delta,
            },
        },
    )
    return SmoothTrajectory(waypoints=retimed, metadata=metadata)


def resample_transport_minimum_jerk(
    trajectory: SmoothTrajectory,
    *,
    sample_dt: float = 0.05,
    min_segment_distance_m: float = 1e-6,
) -> SmoothTrajectory:
    """Resample the initial transport path with one minimum-jerk arc-length profile."""
    if sample_dt <= 0.0:
        raise ValueError("sample_dt must be positive")
    if min_segment_distance_m < 0.0:
        raise ValueError("min_segment_distance_m must be non-negative")
    if len(trajectory.waypoints) < 3 or not _is_approach_phase(trajectory.waypoints[0].phase):
        return trajectory

    transport_end = 0
    while (
        transport_end + 1 < len(trajectory.waypoints)
        and _is_approach_phase(trajectory.waypoints[transport_end + 1].phase)
    ):
        transport_end += 1
    if transport_end < 2:
        return trajectory

    transport = trajectory.waypoints[: transport_end + 1]
    duration = transport[-1].timestamp - transport[0].timestamp
    if duration <= sample_dt:
        return trajectory

    positions = [np.asarray(waypoint.tcp_pose.position, dtype=np.float64) for waypoint in transport]
    cumulative = [0.0]
    for prev, curr in zip(positions, positions[1:]):
        cumulative.append(cumulative[-1] + float(np.linalg.norm(curr - prev)))
    total_distance = cumulative[-1]
    if total_distance <= min_segment_distance_m:
        return trajectory

    steps = max(1, int(math.ceil(duration / sample_dt)))
    resampled: list[TrajectoryWaypoint] = []
    for sample_index in range(steps + 1):
        time_fraction = sample_index / steps
        timestamp = transport[0].timestamp + duration * time_fraction
        arc = total_distance * minimum_jerk_fraction(time_fraction)
        segment_index = _arc_segment_index(cumulative, arc)
        left = transport[segment_index]
        right = transport[min(segment_index + 1, len(transport) - 1)]
        left_arc = cumulative[segment_index]
        right_arc = cumulative[min(segment_index + 1, len(cumulative) - 1)]
        segment_distance = max(min_segment_distance_m, right_arc - left_arc)
        raw = float(np.clip((arc - left_arc) / segment_distance, 0.0, 1.0))
        left_pos = positions[segment_index]
        right_pos = positions[min(segment_index + 1, len(positions) - 1)]
        position = left_pos + raw * (right_pos - left_pos)
        direction = right_pos - left_pos
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > 1e-12:
            direction = direction / direction_norm
        speed = total_distance * minimum_jerk_fraction_derivative(time_fraction) / duration
        joint_names = right.joint_names or left.joint_names
        joint_positions = _interpolate_optional_vector(left.joint_positions, right.joint_positions, raw)
        waypoint = replace(
            right,
            timestamp=float(timestamp),
            tcp_pose=TCPPose(
                position=position.tolist(),
                orientation_xyzw=slerp_xyzw(
                    left.tcp_pose.orientation_xyzw,
                    right.tcp_pose.orientation_xyzw,
                    raw,
                ),
            ),
            tcp_velocity=(speed * direction).tolist(),
            joint_names=joint_names,
            joint_positions=joint_positions,
            joint_velocities=None,
            diagnostics={
                **right.diagnostics,
                "minimum_jerk_transport": "arc_length_resample_v1",
                "minimum_jerk_time_fraction": time_fraction,
                "minimum_jerk_arc_fraction": arc / total_distance,
            },
        )
        if sample_index == 0:
            waypoint = replace(
                waypoint,
                phase=transport[0].phase,
                source=transport[0].source,
                diagnostics={
                    **waypoint.diagnostics,
                    "input_start_source": transport[0].source.value,
                },
            )
        resampled.append(waypoint)

    assert_monotonic_timestamps(resampled + trajectory.waypoints[transport_end + 1 :])
    metadata = replace(
        trajectory.metadata,
        postprocessing={
            **trajectory.metadata.postprocessing,
            "minimum_jerk_transport": {
                "method": "arc_length_resample_v1",
                "sample_dt": sample_dt,
                "transport_input_waypoints": len(transport),
                "transport_output_waypoints": len(resampled),
                "transport_duration_sec": duration,
                "transport_arc_length_m": total_distance,
            },
        },
    )
    return SmoothTrajectory(
        waypoints=resampled + trajectory.waypoints[transport_end + 1 :],
        metadata=metadata,
    )


def resample_precontact_joint_minimum_jerk(
    trajectory: SmoothTrajectory,
    *,
    sample_dt: float = 0.05,
    max_joint_speed_rad_s: float | None = None,
    approach_speedup: float = 1.0,
    preinsertion_speedup: float = 1.0,
    min_joint_distance_rad: float = 1e-6,
) -> SmoothTrajectory:
    """Rewrite pre-contact joint targets with one minimum-jerk joint-space profile."""
    if sample_dt <= 0.0:
        raise ValueError("sample_dt must be positive")
    if max_joint_speed_rad_s is not None and max_joint_speed_rad_s <= 0.0:
        raise ValueError("max_joint_speed_rad_s must be positive")
    if approach_speedup <= 0.0:
        raise ValueError("approach_speedup must be positive")
    if preinsertion_speedup <= 0.0:
        raise ValueError("preinsertion_speedup must be positive")
    if min_joint_distance_rad < 0.0:
        raise ValueError("min_joint_distance_rad must be non-negative")
    if len(trajectory.waypoints) < 3:
        return trajectory

    final_index = next(
        (
            index
            for index, waypoint in enumerate(trajectory.waypoints)
            if waypoint.phase == PhaseLabel.FINAL_INSERTION
        ),
        None,
    )
    precontact_end = len(trajectory.waypoints) - 1 if final_index is None else final_index - 1
    if precontact_end < 2:
        return trajectory

    precontact = trajectory.waypoints[: precontact_end + 1]
    reference_names = precontact[0].joint_names
    reference_dim = len(precontact[0].joint_positions or [])
    if reference_names is None or reference_dim == 0:
        return trajectory
    if any(
        waypoint.joint_positions is None
        or waypoint.joint_names != reference_names
        or len(waypoint.joint_positions) != reference_dim
        for waypoint in precontact
    ):
        return trajectory

    joint_positions = [
        np.asarray(waypoint.joint_positions, dtype=np.float64)
        for waypoint in precontact
    ]
    cumulative = [0.0]
    for prev, curr in zip(joint_positions, joint_positions[1:]):
        cumulative.append(cumulative[-1] + float(np.linalg.norm(curr - prev)))
    total_joint_distance = cumulative[-1]
    if total_joint_distance <= min_joint_distance_rad:
        return trajectory

    original_duration = precontact[-1].timestamp - precontact[0].timestamp
    if original_duration <= sample_dt:
        return trajectory
    phase_scaled_times = _phase_scaled_precontact_times(
        precontact,
        approach_speedup=approach_speedup,
        preinsertion_speedup=preinsertion_speedup,
    )
    duration = phase_scaled_times[-1]
    if max_joint_speed_rad_s is not None:
        min_duration = total_joint_distance * 1.875 / max_joint_speed_rad_s
        if min_duration > duration:
            scale = min_duration / duration
            phase_scaled_times = [value * scale for value in phase_scaled_times]
            duration = phase_scaled_times[-1]

    steps = max(2, int(math.ceil(duration / sample_dt)))
    resampled: list[TrajectoryWaypoint] = []
    max_observed_joint_speed = 0.0
    use_phase_scaled_timing = (
        approach_speedup > 1.0
        or preinsertion_speedup > 1.0
    )
    for sample_index in range(steps + 1):
        time_fraction = sample_index / steps
        timestamp = precontact[0].timestamp + duration * time_fraction
        if use_phase_scaled_timing:
            segment_index = _time_segment_index(phase_scaled_times, duration * time_fraction)
            left_time = phase_scaled_times[segment_index]
            right_time = phase_scaled_times[min(segment_index + 1, len(phase_scaled_times) - 1)]
            segment_time = max(sample_dt, right_time - left_time)
            raw = float(np.clip((duration * time_fraction - left_time) / segment_time, 0.0, 1.0))
            arc = cumulative[segment_index] + raw * (
                cumulative[min(segment_index + 1, len(cumulative) - 1)] - cumulative[segment_index]
            )
        else:
            arc = total_joint_distance * minimum_jerk_fraction(time_fraction)
            segment_index = _arc_segment_index(cumulative, arc)
        left = precontact[segment_index]
        right = precontact[min(segment_index + 1, len(precontact) - 1)]
        left_arc = cumulative[segment_index]
        right_arc = cumulative[min(segment_index + 1, len(cumulative) - 1)]
        segment_distance = max(min_joint_distance_rad, right_arc - left_arc)
        if not use_phase_scaled_timing:
            raw = float(np.clip((arc - left_arc) / segment_distance, 0.0, 1.0))

        left_joints = joint_positions[segment_index]
        right_joints = joint_positions[min(segment_index + 1, len(joint_positions) - 1)]
        joint_delta = right_joints - left_joints
        joint_direction = np.zeros(reference_dim, dtype=np.float64)
        joint_delta_norm = float(np.linalg.norm(joint_delta))
        if joint_delta_norm > 1e-12:
            joint_direction = joint_delta / joint_delta_norm
        if use_phase_scaled_timing:
            left_time = phase_scaled_times[segment_index]
            right_time = phase_scaled_times[min(segment_index + 1, len(phase_scaled_times) - 1)]
            segment_time = max(sample_dt, right_time - left_time)
            joint_speed = joint_delta_norm / segment_time
        else:
            joint_speed = total_joint_distance * minimum_jerk_fraction_derivative(time_fraction) / duration
        max_observed_joint_speed = max(max_observed_joint_speed, abs(joint_speed))

        left_pos = np.asarray(left.tcp_pose.position, dtype=np.float64)
        right_pos = np.asarray(right.tcp_pose.position, dtype=np.float64)
        tcp_delta = right_pos - left_pos
        tcp_direction = np.zeros(3, dtype=np.float64)
        tcp_delta_norm = float(np.linalg.norm(tcp_delta))
        if tcp_delta_norm > 1e-12:
            tcp_direction = tcp_delta / tcp_delta_norm
        tcp_speed = tcp_delta_norm * joint_speed / segment_distance

        waypoint = replace(
            right,
            timestamp=float(timestamp),
            tcp_pose=TCPPose(
                position=(left_pos + raw * tcp_delta).tolist(),
                orientation_xyzw=slerp_xyzw(
                    left.tcp_pose.orientation_xyzw,
                    right.tcp_pose.orientation_xyzw,
                    raw,
                ),
            ),
            tcp_velocity=(tcp_speed * tcp_direction).tolist(),
            joint_names=reference_names,
            joint_positions=(left_joints + raw * joint_delta).tolist(),
            joint_velocities=(joint_speed * joint_direction).tolist(),
            diagnostics={
                **right.diagnostics,
                "joint_minimum_jerk_retime": "precontact_joint_arc_length_v1",
                "joint_minimum_jerk_time_fraction": time_fraction,
                "joint_minimum_jerk_arc_fraction": arc / total_joint_distance,
                "joint_retime_approach_speedup": approach_speedup,
                "joint_retime_preinsertion_speedup": preinsertion_speedup,
            },
        )
        if sample_index == 0:
            waypoint = replace(
                waypoint,
                phase=precontact[0].phase,
                source=precontact[0].source,
                diagnostics={
                    **waypoint.diagnostics,
                    "input_start_source": precontact[0].source.value,
                },
            )
        resampled.append(waypoint)

    timestamp_shift = resampled[-1].timestamp - precontact[-1].timestamp
    suffix = [
        replace(waypoint, timestamp=float(waypoint.timestamp + timestamp_shift))
        for waypoint in trajectory.waypoints[precontact_end + 1 :]
    ]
    output = resampled + suffix
    assert_monotonic_timestamps(output)
    metadata = replace(
        trajectory.metadata,
        postprocessing={
            **trajectory.metadata.postprocessing,
            "joint_minimum_jerk_retime": {
                "method": "precontact_joint_arc_length_v1",
                "sample_dt": sample_dt,
                "max_joint_speed_rad_s": max_joint_speed_rad_s,
                "approach_speedup": approach_speedup,
                "preinsertion_speedup": preinsertion_speedup,
                "precontact_input_waypoints": len(precontact),
                "precontact_output_waypoints": len(resampled),
                "precontact_original_duration_sec": original_duration,
                "precontact_output_duration_sec": duration,
                "precontact_joint_arc_length_rad": total_joint_distance,
                "max_observed_joint_speed_rad_s": max_observed_joint_speed,
            },
        },
    )
    return SmoothTrajectory(waypoints=output, metadata=metadata)


def _phase_scaled_precontact_times(
    waypoints: list[TrajectoryWaypoint],
    *,
    approach_speedup: float,
    preinsertion_speedup: float,
) -> list[float]:
    times = [0.0]
    for prev, curr in zip(waypoints, waypoints[1:]):
        dt = curr.timestamp - prev.timestamp
        if dt <= 0.0:
            raise ValueError("Pre-contact timestamps must be strictly increasing")
        if curr.phase == PhaseLabel.APPROACH and prev.phase == PhaseLabel.APPROACH:
            dt /= approach_speedup
        elif curr.phase == PhaseLabel.PRE_INSERTION:
            dt /= preinsertion_speedup
        times.append(times[-1] + dt)
    return times


def _time_segment_index(times: list[float], value: float) -> int:
    if value <= times[0]:
        return 0
    for index in range(len(times) - 1):
        if times[index] <= value <= times[index + 1]:
            return index
    return max(0, len(times) - 2)


def _arc_segment_index(cumulative: list[float], arc: float) -> int:
    if arc <= cumulative[0]:
        return 0
    for index in range(len(cumulative) - 1):
        if cumulative[index] <= arc <= cumulative[index + 1]:
            return index
    return max(0, len(cumulative) - 2)


def _is_approach_phase(phase: PhaseLabel) -> bool:
    return phase in {
        PhaseLabel.APPROACH,
        PhaseLabel.ALIGNMENT,
        PhaseLabel.OBSTACLE_AVOIDANCE,
        PhaseLabel.PRE_INSERTION,
    }


def _phase_speed_limit(
    phase: PhaseLabel,
    *,
    default: float,
    approach: float | None,
) -> float:
    if approach is None:
        return default
    if _is_approach_phase(phase):
        return approach
    return default


def postprocess_piecewise_trajectory(
    piecewise: PiecewiseTrajectory,
    *,
    sample_dt: float = 0.05,
) -> SmoothTrajectory:
    """Convert a piecewise plan to a smooth sampled trajectory.

    This version computes one C1 cubic Hermite curve over approach/alignment
    waypoints only. Final insertion is kept as deterministic CheatCode geometry
    and is sampled linearly from the pre-insertion pose to the inserted pose, so
    a global curve cannot bend the insertion path away from the port axis.
    """
    if sample_dt <= 0.0:
        raise ValueError("sample_dt must be positive")
    assert_monotonic_timestamps(piecewise.waypoints)

    insertion_start_index = insertion_start_waypoint_index(piecewise.waypoints)
    if insertion_start_index is None:
        waypoint_velocities = _compute_c1_waypoint_velocities(piecewise.waypoints)
    else:
        smoothing_waypoints = piecewise.waypoints[: insertion_start_index + 1]
        waypoint_velocities = _compute_c1_waypoint_velocities(smoothing_waypoints)
        waypoint_velocities.extend(
            np.zeros(3, dtype=np.float64)
            for _ in piecewise.waypoints[insertion_start_index + 1 :]
        )
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
        insertion_segment = (
            insertion_start_index is not None and segment_index >= insertion_start_index
        )
        segment_source = SourceLabel.CHEATCODE if insertion_segment else end.source

        first_sample = 0 if segment_index == 0 else 1
        for i in range(first_sample, steps + 1):
            raw = i / steps
            timestamp = start.timestamp + duration * raw
            if insertion_segment:
                position, velocity = _linear_position_velocity(start_pos, end_pos, duration, raw)
            else:
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
                joint_names=end.joint_names,
                joint_positions=_interpolate_optional_vector(
                    start.joint_positions,
                    end.joint_positions,
                    raw,
                ),
                joint_velocities=_interpolate_optional_vector(
                    start.joint_velocities,
                    end.joint_velocities,
                    raw,
                ),
                gripper_state=end.gripper_state,
                cable_state=end.cable_state,
                port_state=end.port_state,
                phase=sample_phase,
                source=sample_source,
                diagnostics={
                    **end.diagnostics,
                    "postprocessor": "phase_aware_c1_cubic_hermite_v1",
                    "global_smoother": (
                        "disabled_for_cheatcode_insertion"
                        if insertion_segment
                        else "c1_cubic_hermite_v0"
                    ),
                    "segment_index": segment_index,
                    "raw_fraction": raw,
                    "input_start_source": start.source.value,
                    "input_end_source": end.source.value,
                    "cheatcode_derived": insertion_segment,
                    "insertion_smoothing_protected": insertion_segment,
                    "todo": "Replace non-insertion interpolation with constrained continuous optimizer.",
                },
            )
            smooth.append(waypoint)

    assert_monotonic_timestamps(smooth)
    metadata = replace(
        piecewise.metadata,
        postprocessing={
            **piecewise.metadata.postprocessing,
            "method": "phase_aware_c1_cubic_hermite_v1",
            "sample_dt": sample_dt,
            "insertion_start_waypoint_index": insertion_start_index,
            "insertion_start_timestamp": (
                None
                if insertion_start_index is None
                else piecewise.waypoints[insertion_start_index].timestamp
            ),
            "guarantees": [
                "strictly_monotonic_timestamps",
                "continuous_position",
                "continuous_velocity_at_non_insertion_piece_boundaries",
                "explicit_phase_labels",
                "source_labels_preserved_except_postprocessor_diagnostics",
                "final_insertion_marked_cheatcode_derived",
                "global_smoothing_excludes_final_insertion",
                "final_insertion_linear_between_cheatcode_geometry_waypoints",
            ],
        },
    )
    return SmoothTrajectory(waypoints=smooth, metadata=metadata)


def _interpolate_optional_vector(
    start: list[float] | None,
    end: list[float] | None,
    fraction: float,
) -> list[float] | None:
    if start is None or end is None or len(start) != len(end):
        return end
    left = np.asarray(start, dtype=np.float64)
    right = np.asarray(end, dtype=np.float64)
    return (left + float(np.clip(fraction, 0.0, 1.0)) * (right - left)).tolist()


def postprocess_file(
    input_path: str | Path,
    output_path: str | Path,
    sample_dt: float,
    *,
    compact_stalls: bool = False,
    speedup: float = 2.0,
) -> SmoothTrajectory:
    piecewise = PiecewiseTrajectory.load_json(input_path)
    smooth = postprocess_piecewise_trajectory(piecewise, sample_dt=sample_dt)
    retime_speed_profile = _env_bool("AIC_EXPERT_POSTPROCESS_RETIME_TCP_SPEED", True)
    if retime_speed_profile:
        smooth = retime_tcp_speed_profile(
            smooth,
            max_tcp_speed_mps=_env_float("AIC_EXPERT_POSTPROCESS_MAX_TCP_SPEED_MPS", 0.025),
            max_tcp_accel_mps2=_env_float("AIC_EXPERT_POSTPROCESS_MAX_TCP_ACCEL_MPS2", 0.08),
            approach_max_tcp_speed_mps=_env_optional_float(
                "AIC_EXPERT_POSTPROCESS_APPROACH_MAX_TCP_SPEED_MPS",
                0.14,
            ),
            approach_max_tcp_accel_mps2=_env_optional_float(
                "AIC_EXPERT_POSTPROCESS_APPROACH_MAX_TCP_ACCEL_MPS2",
                0.45,
            ),
            approach_stall_translation_m=_env_float(
                "AIC_EXPERT_POSTPROCESS_APPROACH_STALL_TRANSLATION_M",
                0.00025,
            ),
            min_segment_sec=sample_dt,
        )
    if _env_bool("AIC_EXPERT_POSTPROCESS_MIN_JERK_TRANSPORT", True):
        smooth = resample_transport_minimum_jerk(
            smooth,
            sample_dt=sample_dt,
            min_segment_distance_m=_env_float(
                "AIC_EXPERT_POSTPROCESS_MIN_JERK_MIN_SEGMENT_M",
                1e-6,
            ),
        )
    if _env_bool("AIC_EXPERT_POSTPROCESS_RETIME_JOINTS", False):
        smooth = resample_precontact_joint_minimum_jerk(
            smooth,
            sample_dt=sample_dt,
            max_joint_speed_rad_s=_env_optional_float(
                "AIC_EXPERT_POSTPROCESS_MAX_JOINT_SPEED_RAD_S",
                None,
            ),
            approach_speedup=_env_float(
                "AIC_EXPERT_POSTPROCESS_JOINT_APPROACH_SPEEDUP",
                1.0,
            ),
            preinsertion_speedup=_env_float(
                "AIC_EXPERT_POSTPROCESS_JOINT_PREINSERTION_SPEEDUP",
                1.0,
            ),
            min_joint_distance_rad=_env_float(
                "AIC_EXPERT_POSTPROCESS_MIN_JOINT_SEGMENT_RAD",
                1e-6,
            ),
        )
    if compact_stalls:
        smooth = compact_stall_intervals(smooth, speedup=speedup, min_segment_sec=sample_dt)
    smooth.save_json(output_path)
    return smooth


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_optional_float(name: str, default: float | None) -> float | None:
    value = os.environ.get(name)
    if value is None:
        return default
    if value.strip().lower() in {"", "none", "off", "disabled"}:
        return None
    try:
        return float(value)
    except ValueError:
        return default
