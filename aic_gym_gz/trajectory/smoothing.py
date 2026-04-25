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
    port_alignment_lateral_threshold_m: float = 0.008
    port_alignment_yaw_threshold_rad: float = 0.10
    pre_insert_standoff_m: float = 0.025
    guarded_entry_standoff_m: float = 0.005
    guarded_insert_speed_limit: float = 0.010
    port_alignment_unaligned_speed_limit: float = 0.030

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
        targets, conversion_steps, port_gate_metadata = self._apply_port_frame_alignment_gates(
            state=state,
            plan=plan,
            targets=targets,
            conversion_steps=conversion_steps,
            plug_to_tcp_offset=plug_to_tcp_offset,
        )
        self._normalize_target_yaw_path(
            start_yaw=float(np.asarray(state.tcp_pose, dtype=np.float64)[5]),
            targets=targets,
        )
        segment_linear_speed_limit = (
            float(self.port_alignment_unaligned_speed_limit)
            if port_gate_metadata.get("active")
            and port_gate_metadata.get("gate_action") == "align_lateral_and_yaw_at_pre_insert_standoff"
            else float(self.max_linear_speed)
        )
        ideal_steps = self._ideal_step_counts(
            start_pose=np.asarray(state.tcp_pose, dtype=np.float64),
            targets=targets,
            plan=plan,
            linear_speed_limit=segment_linear_speed_limit,
            min_speed_scale=(
                0.8
                if port_gate_metadata.get("active")
                and port_gate_metadata.get("gate_action") == "align_lateral_and_yaw_at_pre_insert_standoff"
                else 0.1
            ),
        )
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
                        nominal_limit=segment_linear_speed_limit,
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
                "port_frame_alignment_gate": port_gate_metadata,
                "native_action_override_used": bool(native_action_override is not None),
                "optimizer_metrics": optimizer_metrics,
                "segment_linear_speed_limit_mps": float(segment_linear_speed_limit),
                "gated_alignment_min_speed_scale": (
                    0.8
                    if port_gate_metadata.get("active")
                    and port_gate_metadata.get("gate_action") == "align_lateral_and_yaw_at_pre_insert_standoff"
                    else None
                ),
            },
        )

    def _minimum_horizon_steps(self, segment_granularity: str) -> int:
        del segment_granularity
        return 4

    def _ideal_step_counts(
        self,
        *,
        start_pose: np.ndarray,
        targets: list[np.ndarray],
        plan: TeacherPlan,
        linear_speed_limit: float | None = None,
        min_speed_scale: float = 0.1,
    ) -> list[int]:
        if not targets:
            return []
        current_pose = np.asarray(start_pose, dtype=np.float64)
        counts: list[int] = []
        waypoint_speed_scales = [
            max(float(waypoint.speed_scale), float(min_speed_scale))
            for waypoint in plan.waypoints
        ]
        if not waypoint_speed_scales:
            waypoint_speed_scales = [max(float(min_speed_scale), 1.0)]
        for index, target_pose in enumerate(targets):
            speed_scale = waypoint_speed_scales[min(index, len(waypoint_speed_scales) - 1)]
            distance = float(np.linalg.norm(target_pose[:3] - current_pose[:3]))
            speed_limit = float(linear_speed_limit) if linear_speed_limit is not None else float(self.max_linear_speed)
            steps = max(
                2,
                int(math.ceil(distance / max(self.base_dt * speed_limit * speed_scale, 1e-4))),
            )
            if plan.segment_granularity != "coarse":
                steps = max(steps, 4)
            counts.append(steps)
            current_pose = np.asarray(target_pose, dtype=np.float64)
        return counts

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

    def _apply_port_frame_alignment_gates(
        self,
        *,
        state: RuntimeState,
        plan: TeacherPlan,
        targets: list[np.ndarray],
        conversion_steps: list[dict[str, object]],
        plug_to_tcp_offset: np.ndarray,
    ) -> tuple[list[np.ndarray], list[dict[str, object]], dict[str, object]]:
        """Enforce the port-frame sequence recommended by failure analysis.

        The VLM still owns sparse global planning, but near the port the dense
        optimizer must not smooth through an invalid shortcut. This gate rewrites
        near-port sparse targets into the sequence: align yaw/lateral at the
        entrance standoff, move to guarded entry, then advance along the
        insertion axis only when the plug is already aligned.
        """
        if self.planner_output_mode == "native_6d_action" or state.target_port_entrance_pose is None:
            return targets, conversion_steps, {"active": False, "reason": "unsupported_mode_or_missing_entrance"}
        if plan.next_phase not in {"pre_insert_align", "guarded_insert"} and plan.motion_mode != "guarded_insert":
            return targets, conversion_steps, {"active": False, "reason": "phase_not_near_port"}
        target = np.asarray(state.target_port_pose[:3], dtype=np.float64)
        entrance = np.asarray(state.target_port_entrance_pose[:3], dtype=np.float64)
        axis = target - entrance
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-8:
            return targets, conversion_steps, {"active": False, "reason": "degenerate_insertion_axis"}
        axis_unit = axis / axis_norm
        plug = np.asarray(state.plug_pose[:3], dtype=np.float64)
        plug_offset = plug - entrance
        axial_depth = float(np.dot(plug_offset, axis_unit))
        lateral_vector = plug_offset - axial_depth * axis_unit
        lateral_norm = float(np.linalg.norm(lateral_vector))
        yaw_error = self._wrap_to_pi(float(state.target_port_pose[5] - state.plug_pose[5]))
        aligned = (
            lateral_norm <= float(self.port_alignment_lateral_threshold_m)
            and abs(yaw_error) <= float(self.port_alignment_yaw_threshold_rad)
        )

        def pose_for_plug(plug_xyz: np.ndarray) -> np.ndarray:
            pose = np.asarray(state.tcp_pose, dtype=np.float64).copy()
            pose[:3] = np.asarray(plug_xyz, dtype=np.float64) + plug_to_tcp_offset
            pose[5] = float(state.target_port_pose[5])
            return pose

        pre_insert_plug = entrance - float(self.pre_insert_standoff_m) * axis_unit
        guarded_entry_plug = entrance - float(self.guarded_entry_standoff_m) * axis_unit
        rewritten: list[np.ndarray]
        gate_action: str
        if not aligned:
            rewritten = [pose_for_plug(pre_insert_plug)]
            gate_action = "align_lateral_and_yaw_at_pre_insert_standoff"
        elif plan.next_phase == "guarded_insert" or plan.motion_mode == "guarded_insert":
            inserted_depth = min(axis_norm, max(0.0, axial_depth) + 0.010)
            inserted_plug = entrance + inserted_depth * axis_unit
            rewritten = [pose_for_plug(guarded_entry_plug), pose_for_plug(inserted_plug)]
            gate_action = "guarded_axis_advance_after_alignment"
        else:
            rewritten = [pose_for_plug(pre_insert_plug)]
            gate_action = "hold_pre_insert_alignment"

        rewritten_steps = list(conversion_steps)
        rewritten_steps.append(
            {
                "mode": "port_frame_alignment_gate",
                "coordinate_frame": "port_entrance_frame",
                "gate_action": gate_action,
                "axis_unit_world_xyz": axis_unit.astype(float).tolist(),
                "pre_insert_standoff_m": float(self.pre_insert_standoff_m),
                "guarded_entry_standoff_m": float(self.guarded_entry_standoff_m),
                "input_target_count": int(len(targets)),
                "rewritten_target_count": int(len(rewritten)),
            }
        )
        final_plug = np.asarray(rewritten[-1][:3], dtype=np.float64) - plug_to_tcp_offset
        final_offset = final_plug - entrance
        final_axial_depth = float(np.dot(final_offset, axis_unit))
        final_lateral_vector = final_offset - final_axial_depth * axis_unit
        final_yaw_error = self._wrap_to_pi(float(state.target_port_pose[5] - rewritten[-1][5]))
        return rewritten, rewritten_steps, {
            "active": True,
            "gate_action": gate_action,
            "coordinate_frame": "port_entrance_frame",
            "origin_world_xyz": entrance.astype(float).tolist(),
            "axis_unit_world_xyz": axis_unit.astype(float).tolist(),
            "plug_lateral_error_m": lateral_norm,
            "plug_axial_depth_m": axial_depth,
            "yaw_error_rad": yaw_error,
            "alignment_threshold_lateral_m": float(self.port_alignment_lateral_threshold_m),
            "alignment_threshold_yaw_rad": float(self.port_alignment_yaw_threshold_rad),
            "aligned_before_gate": bool(aligned),
            "final_target_lateral_error_m": float(np.linalg.norm(final_lateral_vector)),
            "final_target_axial_depth_m": final_axial_depth,
            "final_target_yaw_error_rad": final_yaw_error,
            "guarded_insert_speed_limit_mps": float(self.guarded_insert_speed_limit),
            "unaligned_port_alignment_speed_limit_mps": float(self.port_alignment_unaligned_speed_limit),
        }

    def _normalize_target_yaw_path(self, *, start_yaw: float, targets: list[np.ndarray]) -> None:
        previous_yaw = float(start_yaw)
        for target in targets:
            desired_yaw = float(target[5])
            target[5] = previous_yaw + self._wrap_to_pi(desired_yaw - previous_yaw)
            previous_yaw = float(target[5])

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
            return min(float(nominal_limit), float(self.guarded_insert_speed_limit))
        if axis_norm > 1e-8:
            lateral_vector = (plug_position - entrance) - float(np.dot(plug_position - entrance, axis / axis_norm)) * (axis / axis_norm)
            distance_to_entrance = float(np.linalg.norm(entrance - plug_position))
            if distance_to_entrance <= 0.08 and float(np.linalg.norm(lateral_vector)) <= self.port_alignment_lateral_threshold_m:
                return min(float(nominal_limit), 0.035)
        return float(nominal_limit)

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)

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
            "near_entrance_aligned_speed_limit_mps": 0.035,
            "guarded_insert_speed_limit_mps": float(self.guarded_insert_speed_limit),
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
