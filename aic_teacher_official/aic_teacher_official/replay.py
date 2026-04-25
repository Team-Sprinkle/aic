"""Replay-time sampling for smooth official teacher trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aic_teacher_official.postprocess import slerp_xyzw
from aic_teacher_official.trajectory import SmoothTrajectory, TCPPose, TrajectoryWaypoint


@dataclass(frozen=True)
class ReplayTarget:
    timestamp: float
    tcp_pose: TCPPose
    tcp_velocity: list[float] | None
    waypoint: TrajectoryWaypoint


class SmoothTrajectoryReplayPolicy:
    """Time-indexed replay sampler.

    This class is deliberately VLM-free: it only reads a smooth trajectory JSON
    and interpolates between adjacent samples. The ROS wrapper is responsible
    for converting `ReplayTarget` into the official action message.
    """

    def __init__(self, trajectory: SmoothTrajectory):
        self.trajectory = trajectory
        self._waypoints = trajectory.waypoints
        self.start_time = self._waypoints[0].timestamp
        self.end_time = self._waypoints[-1].timestamp

    @classmethod
    def from_json(cls, path: str | Path) -> "SmoothTrajectoryReplayPolicy":
        return cls(SmoothTrajectory.load_json(path))

    def sample(self, elapsed_sec: float) -> ReplayTarget:
        query = self.start_time + max(0.0, float(elapsed_sec))
        if query <= self._waypoints[0].timestamp:
            return self._target_from_waypoint(self._waypoints[0])
        if query >= self._waypoints[-1].timestamp:
            return self._target_from_waypoint(self._waypoints[-1])

        for left, right in zip(self._waypoints, self._waypoints[1:]):
            if left.timestamp <= query <= right.timestamp:
                duration = right.timestamp - left.timestamp
                fraction = 0.0 if duration <= 0.0 else (query - left.timestamp) / duration
                left_pos = np.asarray(left.tcp_pose.position, dtype=np.float64)
                right_pos = np.asarray(right.tcp_pose.position, dtype=np.float64)
                position = left_pos + fraction * (right_pos - left_pos)
                velocity = (
                    (right_pos - left_pos).tolist()
                    if duration <= 0.0
                    else ((right_pos - left_pos) / duration).tolist()
                )
                return ReplayTarget(
                    timestamp=query,
                    tcp_pose=TCPPose(
                        position=position.tolist(),
                        orientation_xyzw=slerp_xyzw(
                            left.tcp_pose.orientation_xyzw,
                            right.tcp_pose.orientation_xyzw,
                            fraction,
                        ),
                    ),
                    tcp_velocity=velocity,
                    waypoint=right,
                )
        return self._target_from_waypoint(self._waypoints[-1])

    def is_finished(self, elapsed_sec: float) -> bool:
        return self.start_time + max(0.0, float(elapsed_sec)) >= self.end_time

    @staticmethod
    def _target_from_waypoint(waypoint: TrajectoryWaypoint) -> ReplayTarget:
        return ReplayTarget(
            timestamp=waypoint.timestamp,
            tcp_pose=waypoint.tcp_pose,
            tcp_velocity=waypoint.tcp_velocity,
            waypoint=waypoint,
        )
