"""Pre-contact nominal trajectory repair utilities."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

import numpy as np

from aic_teacher_official.postprocess import minimum_jerk_fraction
from aic_teacher_official.replay import SmoothTrajectoryReplayPolicy
from aic_teacher_official.trajectory import PhaseLabel, SmoothTrajectory, TrajectoryWaypoint


def repair_precontact_approach(
    trajectory: SmoothTrajectory,
    *,
    sample_dt: float,
    method: str = "minimum_jerk_precontact_retime_v1",
) -> tuple[SmoothTrajectory, dict[str, Any]]:
    """Retiming repair through the pinned pre-insertion hold point.

    The final insertion geometry is copied unchanged except for a timestamp
    shift. All pre-contact executable targets are resampled from the original
    replay trajectory with a minimum-jerk time law, so joint-position replay
    targets are actually rewritten rather than only relabeled.
    """
    if sample_dt <= 0.0:
        raise ValueError("sample_dt must be positive")
    waypoints = trajectory.waypoints
    final_index = next(
        (index for index, waypoint in enumerate(waypoints) if waypoint.phase == PhaseLabel.FINAL_INSERTION),
        None,
    )
    if final_index is None or final_index < 2:
        return trajectory, {
            "action_recomputed": False,
            "repair_method": method,
            "repair_success": False,
            "repair_reason": "no_precontact_region",
        }
    pre_end_index = final_index - 1
    original_pre_start = waypoints[0].timestamp
    original_pre_end = waypoints[pre_end_index].timestamp
    original_duration = max(sample_dt, original_pre_end - original_pre_start)
    replay = SmoothTrajectoryReplayPolicy(trajectory)
    steps = max(2, int(math.ceil(original_duration / sample_dt)))
    repaired: list[TrajectoryWaypoint] = []
    for step in range(steps + 1):
        progress = step / steps
        warped_elapsed = minimum_jerk_fraction(progress) * original_duration
        target = replay.sample(warped_elapsed)
        source_wp = target.waypoint
        if target.timestamp > original_pre_end:
            target = replay.sample(original_duration)
            source_wp = target.waypoint
        timestamp = original_pre_start + progress * original_duration
        repaired.append(
            replace(
                source_wp,
                timestamp=float(timestamp),
                tcp_pose=target.tcp_pose,
                tcp_velocity=target.tcp_velocity,
                joint_names=target.joint_names,
                joint_positions=target.joint_positions,
                joint_velocities=target.joint_velocities,
                diagnostics={
                    **source_wp.diagnostics,
                    "action_recomputed": True,
                    "repair_method": method,
                    "minimum_jerk_retime": True,
                },
            )
        )
    pinned = waypoints[pre_end_index]
    repaired[-1] = replace(
        repaired[-1],
        tcp_pose=pinned.tcp_pose,
        phase=pinned.phase,
        source=pinned.source,
        joint_names=pinned.joint_names,
        joint_positions=pinned.joint_positions,
        joint_velocities=[0.0] * len(pinned.joint_positions) if pinned.joint_positions else pinned.joint_velocities,
        diagnostics={
            **pinned.diagnostics,
            "action_recomputed": True,
            "repair_method": method,
            "minimum_jerk_retime": True,
            "preinsert_pose_pinned": True,
        },
    )
    timestamp_shift = repaired[-1].timestamp - pinned.timestamp
    repaired.extend(
        replace(waypoint, timestamp=float(waypoint.timestamp + timestamp_shift))
        for waypoint in waypoints[final_index:]
    )
    metrics = {
        "action_recomputed": True,
        "repair_method": method,
        "repair_success": True,
        "repaired_duration_sec": repaired[steps].timestamp - repaired[0].timestamp,
        "action_mode": "joint_position_then_cheatcode",
        "action_frame": "base_link_until_guarded_insert_then_gripper/tcp",
        "original_action_count": pre_end_index + 1,
        "repaired_action_count": len(repaired[: steps + 1]),
        "preinsert_pose_pinned": True,
        "max_action_delta_before": _max_joint_delta(waypoints[: pre_end_index + 1]),
        "max_action_delta_after": _max_joint_delta(repaired[: steps + 1]),
        "max_joint_jerk_before": _max_joint_jerk(waypoints[: pre_end_index + 1]),
        "max_joint_jerk_after": _max_joint_jerk(repaired[: steps + 1]),
        "max_tcp_speed_before": _max_tcp_speed(waypoints[: pre_end_index + 1]),
        "max_tcp_speed_after": _max_tcp_speed(repaired[: steps + 1]),
    }
    metadata = replace(
        trajectory.metadata,
        postprocessing={
            **trajectory.metadata.postprocessing,
            "precontact_repair": metrics,
        },
    )
    return SmoothTrajectory(waypoints=repaired, metadata=metadata), metrics


def _max_joint_delta(waypoints: list[TrajectoryWaypoint]) -> float | None:
    values = []
    for before, after in zip(waypoints, waypoints[1:]):
        if before.joint_positions is None or after.joint_positions is None:
            continue
        if len(before.joint_positions) != len(after.joint_positions):
            continue
        values.append(float(np.linalg.norm(np.asarray(after.joint_positions) - np.asarray(before.joint_positions))))
    return max(values) if values else None


def _max_joint_jerk(waypoints: list[TrajectoryWaypoint]) -> float | None:
    velocities = []
    times = []
    for before, after in zip(waypoints, waypoints[1:]):
        if before.joint_positions is None or after.joint_positions is None:
            continue
        dt = after.timestamp - before.timestamp
        if dt <= 0.0 or len(before.joint_positions) != len(after.joint_positions):
            continue
        velocities.append((np.asarray(after.joint_positions) - np.asarray(before.joint_positions)) / dt)
        times.append(after.timestamp)
    if len(velocities) < 3:
        return None
    accelerations = []
    for index in range(1, len(velocities)):
        dt = max(1e-9, times[index] - times[index - 1])
        accelerations.append((velocities[index] - velocities[index - 1]) / dt)
    jerks = []
    for index in range(1, len(accelerations)):
        dt = max(1e-9, times[index + 1] - times[index])
        jerks.append(float(np.linalg.norm((accelerations[index] - accelerations[index - 1]) / dt)))
    return max(jerks) if jerks else None


def _max_tcp_speed(waypoints: list[TrajectoryWaypoint]) -> float | None:
    speeds = []
    for before, after in zip(waypoints, waypoints[1:]):
        dt = after.timestamp - before.timestamp
        if dt <= 0.0:
            continue
        speeds.append(
            float(
                np.linalg.norm(
                    np.asarray(after.tcp_pose.position, dtype=np.float64)
                    - np.asarray(before.tcp_pose.position, dtype=np.float64)
                )
                / dt
            )
        )
    return max(speeds) if speeds else None
