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
    planner_output_mode: str = "absolute_cartesian_waypoint"

    def smooth(self, *, state: RuntimeState, plan: TeacherPlan) -> TrajectorySegment:
        current_pose = np.asarray(state.tcp_pose, dtype=np.float64).copy()
        plug_to_tcp_offset = np.asarray(state.tcp_pose[:3] - state.plug_pose[:3], dtype=np.float64)
        targets: list[np.ndarray] = []
        ideal_steps: list[int] = []
        conversion_steps: list[dict[str, object]] = []
        native_action_override: np.ndarray | None = None
        for waypoint in plan.waypoints:
            target_pose = current_pose.copy()
            target_pose, native_action_override, conversion_note = self._convert_waypoint(
                state=state,
                current_pose=current_pose,
                plug_to_tcp_offset=plug_to_tcp_offset,
                waypoint=waypoint,
            )
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
            targets.append(target_pose)
            ideal_steps.append(steps)
            conversion_steps.append(conversion_note)
            current_pose = target_pose
        allocated_steps = self._allocate_steps(
            ideal_steps=ideal_steps,
            horizon_steps=max(int(plan.segment_horizon_steps), 1),
        )
        current_pose = np.asarray(state.tcp_pose, dtype=np.float64).copy()
        points: list[DenseTrajectoryPoint] = []
        for target_pose, steps in zip(targets, allocated_steps):
            if steps <= 0:
                current_pose = target_pose
                continue
            for step in range(1, steps + 1):
                tau = step / steps
                blend = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
                pose = current_pose + (target_pose - current_pose) * blend
                prev_pose = current_pose if step == 1 else np.asarray(points[-1].target_tcp_pose, dtype=np.float64)
                delta = pose - prev_pose
                if native_action_override is not None:
                    action = native_action_override.copy()
                else:
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
            conversion_metadata={
                "planner_output_mode": self.planner_output_mode,
                "dense_point_dt_s": float(self.base_dt),
                "conversion_steps": conversion_steps,
                "native_action_override_used": bool(native_action_override is not None),
            },
        )

    def _convert_waypoint(
        self,
        *,
        state: RuntimeState,
        current_pose: np.ndarray,
        plug_to_tcp_offset: np.ndarray,
        waypoint,
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, object]]:
        target_pose = current_pose.copy()
        if self.planner_output_mode == "absolute_cartesian_waypoint":
            target_pose[:3] = np.asarray(waypoint.position_xyz, dtype=np.float64) + plug_to_tcp_offset
            target_pose[5] = float(waypoint.yaw)
            return target_pose, None, {
                "mode": self.planner_output_mode,
                "input_position_xyz": list(waypoint.position_xyz),
                "target_space": "absolute_plug_waypoint_to_absolute_tcp_pose",
            }
        if self.planner_output_mode == "delta_cartesian_waypoint":
            current_plug = np.asarray(state.plug_pose[:3], dtype=np.float64)
            delta_xyz = np.asarray(waypoint.position_xyz, dtype=np.float64)
            target_pose[:3] = current_plug + delta_xyz + plug_to_tcp_offset
            target_pose[5] = float(current_pose[5] + waypoint.yaw)
            return target_pose, None, {
                "mode": self.planner_output_mode,
                "input_position_xyz": list(waypoint.position_xyz),
                "target_space": "delta_plug_waypoint_to_absolute_tcp_pose",
            }
        if self.planner_output_mode == "native_6d_action":
            native_action = np.array(
                [
                    float(waypoint.position_xyz[0]),
                    float(waypoint.position_xyz[1]),
                    float(waypoint.position_xyz[2]),
                    0.0,
                    0.0,
                    float(waypoint.yaw),
                ],
                dtype=np.float64,
            )
            target_pose[:3] = current_pose[:3] + native_action[:3] * self.base_dt
            target_pose[5] = float(current_pose[5] + native_action[5] * self.base_dt)
            return target_pose, native_action, {
                "mode": self.planner_output_mode,
                "input_position_xyz": list(waypoint.position_xyz),
                "target_space": "native_6d_action_passthrough",
            }
        raise ValueError(f"Unsupported planner_output_mode={self.planner_output_mode!r}")

    def _allocate_steps(self, *, ideal_steps: list[int], horizon_steps: int) -> list[int]:
        if not ideal_steps:
            return []
        total_ideal_steps = sum(ideal_steps)
        if total_ideal_steps <= horizon_steps:
            return ideal_steps
        scale = horizon_steps / max(total_ideal_steps, 1)
        allocated = [max(1, int(math.floor(step_count * scale))) for step_count in ideal_steps]
        allocated_total = sum(allocated)
        if allocated_total > horizon_steps:
            reduction_candidates = sorted(
                range(len(allocated)),
                key=lambda index: (allocated[index], ideal_steps[index]),
            )
            for index in reduction_candidates:
                if allocated_total <= horizon_steps:
                    break
                reducible = min(allocated[index] - 1, allocated_total - horizon_steps)
                if reducible <= 0:
                    continue
                allocated[index] -= reducible
                allocated_total -= reducible
        elif allocated_total < horizon_steps:
            remainders = [
                (ideal_steps[index] * scale - allocated[index], index)
                for index in range(len(allocated))
            ]
            for _, index in sorted(remainders, reverse=True):
                if allocated_total >= horizon_steps:
                    break
                allocated[index] += 1
                allocated_total += 1
        return allocated
