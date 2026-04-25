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
    max_linear_speed: float = 0.12
    max_angular_speed: float = 0.5
    max_linear_accel: float = 0.25
    max_linear_jerk: float = 2.0
    max_segment_steps: int = 512
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
        optimizer_horizon_steps = min(
            max(sum(ideal_steps), 1),
            max(int(self.max_segment_steps), 1),
        )
        horizon_steps = max(
            int(plan.segment_horizon_steps),
            self._minimum_horizon_steps(plan.segment_granularity),
            optimizer_horizon_steps,
            1,
        )
        horizon_steps = min(horizon_steps, max(int(self.max_segment_steps), 1))
        allocated_steps = self._allocate_steps(
            ideal_steps=ideal_steps,
            horizon_steps=horizon_steps,
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
                    max_linear_speed = self._linear_speed_limit_for_pose(
                        state=state,
                        tcp_pose=pose,
                        plug_to_tcp_offset=plug_to_tcp_offset,
                        nominal_limit=self.max_linear_speed,
                    )
                    linear_velocity = self._clip_vector_norm(
                        delta[:3] / self.base_dt,
                        max_norm=max_linear_speed,
                    )
                    angular_velocity = self._clip_vector_norm(
                        delta[3:6] / self.base_dt,
                        max_norm=self.max_angular_speed,
                    )
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
        optimizer_metrics = self._optimizer_metrics(
            start_pose=np.asarray(state.tcp_pose, dtype=np.float64),
            sparse_targets=targets,
            points=points,
        )
        return TrajectorySegment(
            phase=plan.next_phase,
            motion_mode=plan.motion_mode,
            rationale_summary=plan.rationale_summary,
            points=tuple(points),
            expected_duration_s=duration_s,
            conversion_metadata={
                "planner_output_mode": self.planner_output_mode,
                "dense_point_dt_s": float(self.base_dt),
                "requested_horizon_steps": int(plan.segment_horizon_steps),
                "effective_horizon_steps": int(horizon_steps),
                "optimizer_horizon_steps": int(optimizer_horizon_steps),
                "max_segment_steps": int(self.max_segment_steps),
                "conversion_steps": conversion_steps,
                "native_action_override_used": bool(native_action_override is not None),
                "optimizer_metrics": optimizer_metrics,
            },
        )

    def _minimum_horizon_steps(self, segment_granularity: str) -> int:
        del segment_granularity
        return 4

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
            target_pose[5] = self._resolve_absolute_yaw(state=state, waypoint_yaw=float(waypoint.yaw))
            return target_pose, None, {
                "mode": self.planner_output_mode,
                "input_position_xyz": list(waypoint.position_xyz),
                "input_yaw": float(waypoint.yaw),
                "resolved_target_yaw": float(target_pose[5]),
                "target_space": "absolute_plug_waypoint_to_absolute_tcp_pose",
            }
        if self.planner_output_mode == "delta_cartesian_waypoint":
            current_plug = np.asarray(state.plug_pose[:3], dtype=np.float64)
            delta_xyz = np.asarray(waypoint.position_xyz, dtype=np.float64)
            target_pose[:3] = current_plug + delta_xyz + plug_to_tcp_offset
            target_pose[5] = self._resolve_delta_yaw(
                state=state,
                current_yaw=float(current_pose[5]),
                waypoint_yaw=float(waypoint.yaw),
            )
            return target_pose, None, {
                "mode": self.planner_output_mode,
                "input_position_xyz": list(waypoint.position_xyz),
                "input_yaw": float(waypoint.yaw),
                "resolved_target_yaw": float(target_pose[5]),
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

    def _resolve_absolute_yaw(self, *, state: RuntimeState, waypoint_yaw: float) -> float:
        target_yaw = float(state.target_port_pose[5])
        if abs(waypoint_yaw) <= 1e-9 and abs(target_yaw) > 1e-6:
            return target_yaw
        return waypoint_yaw

    def _resolve_delta_yaw(self, *, state: RuntimeState, current_yaw: float, waypoint_yaw: float) -> float:
        target_yaw = float(state.target_port_pose[5])
        if abs(waypoint_yaw) <= 1e-9 and abs(target_yaw - current_yaw) > 1e-6:
            return target_yaw
        return float(current_yaw + waypoint_yaw)

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

    def _clip_vector_norm(self, vector: np.ndarray, *, max_norm: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= max_norm or norm <= 1e-12:
            return np.asarray(vector, dtype=np.float64)
        return np.asarray(vector, dtype=np.float64) * (float(max_norm) / norm)

    def _linear_speed_limit_for_pose(
        self,
        *,
        state: RuntimeState,
        tcp_pose: np.ndarray,
        plug_to_tcp_offset: np.ndarray,
        nominal_limit: float,
    ) -> float:
        target = np.asarray(state.target_port_pose[:3], dtype=np.float64)
        entrance = (
            np.asarray(state.target_port_entrance_pose[:3], dtype=np.float64)
            if state.target_port_entrance_pose is not None
            else target
        )
        plug_position = np.asarray(tcp_pose[:3], dtype=np.float64) - plug_to_tcp_offset
        distance_to_target = float(np.linalg.norm(target - plug_position))
        axis = target - entrance
        axis_norm = float(np.linalg.norm(axis))
        insertion_progress = 0.0
        if axis_norm > 1e-8:
            axial_depth = float(np.dot(plug_position - entrance, axis / axis_norm))
            insertion_progress = float(np.clip(axial_depth / axis_norm, 0.0, 1.0))
        if distance_to_target <= 0.015 or insertion_progress >= 0.75:
            return min(float(nominal_limit), 0.035)
        return float(nominal_limit)

    def _optimizer_metrics(
        self,
        *,
        start_pose: np.ndarray,
        sparse_targets: list[np.ndarray],
        points: list[DenseTrajectoryPoint],
    ) -> dict[str, object]:
        sparse_poses = [start_pose, *sparse_targets]
        sparse_path_length_m = self._pose_path_length([pose[:3] for pose in sparse_poses])
        dense_poses = [start_pose, *[np.asarray(point.target_tcp_pose, dtype=np.float64) for point in points]]
        dense_path_length_m = self._pose_path_length([pose[:3] for pose in dense_poses])
        actions = [np.asarray(point.action, dtype=np.float64) for point in points]
        linear_speeds = [float(np.linalg.norm(action[:3])) for action in actions]
        angular_speeds = [float(np.linalg.norm(action[3:6])) for action in actions]
        linear_accels = [
            float(np.linalg.norm((actions[index][:3] - actions[index - 1][:3]) / self.base_dt))
            for index in range(1, len(actions))
        ]
        linear_jerks = [
            float(
                np.linalg.norm(
                    (
                        actions[index][:3]
                        - 2.0 * actions[index - 1][:3]
                        + actions[index - 2][:3]
                    )
                    / (self.base_dt * self.base_dt)
                )
            )
            for index in range(2, len(actions))
        ]
        return {
            "sparse_waypoint_count": int(len(sparse_targets)),
            "dense_point_count": int(len(points)),
            "sparse_path_length_m": float(sparse_path_length_m),
            "dense_path_length_m": float(dense_path_length_m),
            "path_length_ratio_dense_to_sparse": float(
                dense_path_length_m / sparse_path_length_m
                if sparse_path_length_m > 1e-9
                else 0.0
            ),
            "max_command_linear_speed_mps": float(max(linear_speeds, default=0.0)),
            "final_approach_speed_limit_mps": 0.035,
            "max_command_angular_speed_radps": float(max(angular_speeds, default=0.0)),
            "max_command_linear_accel_mps2": float(max(linear_accels, default=0.0)),
            "mean_command_linear_jerk_mps3": float(np.mean(linear_jerks) if linear_jerks else 0.0),
        }

    def _pose_path_length(self, positions: list[np.ndarray]) -> float:
        if len(positions) < 2:
            return 0.0
        return float(
            sum(
                np.linalg.norm(np.asarray(positions[index], dtype=np.float64) - np.asarray(positions[index - 1], dtype=np.float64))
                for index in range(1, len(positions))
            )
        )
