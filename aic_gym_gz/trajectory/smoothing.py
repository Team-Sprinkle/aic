"""Sparse-waypoint to dense low-jerk segment conversion."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..runtime import RuntimeState
from ..teacher.types import DenseTrajectoryPoint, TeacherPlan, TrajectorySegment


@dataclass(frozen=True)
class MinimumJerkSmoother:
    base_dt: float = 0.02
    max_linear_speed: float = 0.25
    max_angular_speed: float = 2.0
    max_linear_accel: float = 0.25
    max_linear_jerk: float = 2.0

    def smooth(self, *, state: RuntimeState, plan: TeacherPlan) -> TrajectorySegment:
        current_pose = np.asarray(state.tcp_pose, dtype=np.float64).copy()
        plug_to_tcp_offset = np.asarray(state.tcp_pose[:3] - state.plug_pose[:3], dtype=np.float64)
        points: list[DenseTrajectoryPoint] = []
        for waypoint in plan.waypoints:
            target_pose = current_pose.copy()
            # Teacher waypoints are specified in plug space. Convert them into
            # TCP targets by preserving the current plug-to-TCP offset.
            target_pose[:3] = np.asarray(waypoint.position_xyz, dtype=np.float64) + plug_to_tcp_offset
            target_pose[5] = float(waypoint.yaw)
            distance = float(np.linalg.norm(target_pose[:3] - current_pose[:3]))
            steps = max(
                2,
                int(
                    math.ceil(
                        distance
                        / max(
                            self.base_dt
                            * self.max_linear_speed
                            * max(waypoint.speed_scale, 0.1),
                            1e-4,
                        )
                    )
                ),
            )
            if plan.segment_granularity != "coarse":
                steps = max(steps, 4)
            for step in range(1, steps + 1):
                tau = step / steps
                blend = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
                pose = current_pose + (target_pose - current_pose) * blend
                prev_pose = current_pose if step == 1 else np.asarray(points[-1].target_tcp_pose, dtype=np.float64)
                delta = pose - prev_pose
                linear_velocity = np.clip(delta[:3] / self.base_dt, -self.max_linear_speed, self.max_linear_speed)
                angular_velocity = np.clip(delta[3:6] / self.base_dt, -self.max_angular_speed, self.max_angular_speed)
                action = np.concatenate([linear_velocity, angular_velocity])
                points.append(
                    DenseTrajectoryPoint(
                        dt=self.base_dt,
                        action=tuple(float(value) for value in action),
                        target_tcp_pose=tuple(float(value) for value in pose),
                    )
                )
            current_pose = target_pose
        duration_s = sum(point.dt for point in points)
        return TrajectorySegment(
            phase=plan.next_phase,
            motion_mode=plan.motion_mode,
            rationale_summary=plan.rationale_summary,
            points=tuple(points),
            expected_duration_s=duration_s,
        )
