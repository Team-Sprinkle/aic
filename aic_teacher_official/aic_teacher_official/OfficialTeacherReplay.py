"""Official AIC policy wrapper for replaying a smooth teacher trajectory.

This module intentionally does not import any VLM/planner backends. Replay must
be deterministic and local to the official ROS/Gazebo execution.
"""

from __future__ import annotations

import os
import json
import math
from pathlib import Path

import numpy as np
from aic_control_interfaces.msg import JointMotionUpdate, TrajectoryGenerationMode
from aic_model_interfaces.msg import Observation
from aic_model.policy import (
    compute_delta_pose,
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    quaternion_xyzw_to_rotation_matrix,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform, WrenchStamped
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp
from trajectory_msgs.msg import JointTrajectoryPoint

from aic_teacher_official.replay import ReplayTarget, SmoothTrajectoryReplayPolicy

QuaternionTuple = tuple[float, float, float, float]


class OfficialTeacherReplay(Policy):
    """Replay a precomputed smooth trajectory through the official Policy API."""

    def __init__(
        self,
        parent_node,
        trajectory_path: str | None = None,
        action_mode: str | None = None,
    ):
        super().__init__(parent_node)
        self._trajectory_path = trajectory_path or os.environ.get(
            "AIC_OFFICIAL_TEACHER_TRAJECTORY", ""
        )
        if not self._trajectory_path:
            raise RuntimeError(
                "AIC_OFFICIAL_TEACHER_TRAJECTORY must point to a smooth trajectory JSON. "
                "Run scripts/official_teacher_postprocess.py first, then export "
                "AIC_OFFICIAL_TEACHER_TRAJECTORY=/absolute/path/to/smooth_trajectory.json "
                "or pass --teacher-trajectory to the official recording launcher."
            )
        self._replay = SmoothTrajectoryReplayPolicy.from_json(self._trajectory_path)
        self._action_mode = action_mode or os.environ.get(
            "AIC_OFFICIAL_TEACHER_ACTION_MODE",
            "relative_delta_gripper_tcp",
        )
        self._online_cheatcode_final_insertion = (
            os.environ.get("AIC_OFFICIAL_TEACHER_ONLINE_CHEATCODE_INSERTION", "true").lower()
            == "true"
        )
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05
        self._expert_mode = os.environ.get("AIC_EXPERT_MODE", "nominal")
        self._ft_threshold_n = float(os.environ.get("AIC_OFFICIAL_TEACHER_FT_THRESHOLD_N", "1.0"))
        self._tracking_gate_threshold_m = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_TRACKING_GATE_THRESHOLD_M", "0.02")
        )
        self._tracking_gate_timeout_sec = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_TRACKING_GATE_TIMEOUT_SEC", "2.0")
        )
        self._local_preinsert_align_done = False
        self._preinsert_settle_done = False
        self._last_tracking_gate_force_delta_n = None
        self._last_tracking_gate_error_m = None
        self._last_tracking_gate_speed_mps = None
        self._last_precontact_lateral_offset_base = np.zeros(3, dtype=np.float64)
        runtime_trace = os.environ.get("AIC_OFFICIAL_TEACHER_RUNTIME_TRACE", "")
        self._runtime_trace_path = Path(runtime_trace) if runtime_trace else None
        if self._action_mode not in {
            "relative_delta_gripper_tcp",
            "absolute_cartesian_pose_base_link",
            "joint_position_then_cheatcode",
        }:
            raise RuntimeError(
                "AIC_OFFICIAL_TEACHER_ACTION_MODE must be one of: "
                "relative_delta_gripper_tcp, absolute_cartesian_pose_base_link, "
                "joint_position_then_cheatcode"
            )
        self.get_logger().info(
            "Loaded official teacher replay trajectory: "
            f"{self._trajectory_path}; action_mode={self._action_mode}"
        )
        self._latest_insertion_event_namespace = ""
        self._insertion_event_sub = None
        self._latest_observation = None
        self._latest_wrench = None
        self._observation_sub = None
        self._wrench_sub = None
        if hasattr(self._parent_node, "create_subscription"):
            self._insertion_event_sub = self._parent_node.create_subscription(
                String,
                "/scoring/insertion_event",
                self._insertion_event_callback,
                10,
            )
            self._observation_sub = self._parent_node.create_subscription(
                Observation,
                "observations",
                self._observation_callback,
                10,
            )
            self._wrench_sub = self._parent_node.create_subscription(
                WrenchStamped,
                "/fts_broadcaster/wrench",
                self._wrench_callback,
                10,
            )

    def _trace_event(self, event: str, **payload) -> None:
        if self._runtime_trace_path is None:
            return
        try:
            self._runtime_trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self._runtime_trace_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "event": event,
                            "time_sec": self.time_now().nanoseconds * 1e-9,
                            **payload,
                        }
                    )
                    + "\n"
                )
        except Exception as ex:
            self.get_logger().warn(f"Unable to write runtime trace event {event!r}: {ex}")

    def _insertion_event_callback(self, msg: String) -> None:
        self._latest_insertion_event_namespace = msg.data.strip().strip("/")

    def _observation_callback(self, msg: Observation) -> None:
        self._latest_observation = msg

    def _wrench_callback(self, msg: WrenchStamped) -> None:
        self._latest_wrench = msg

    def _task_completed_in_simulation(self, task: Task) -> bool:
        namespace = self._latest_insertion_event_namespace
        if not namespace:
            return False
        tokens = [token for token in namespace.split("/") if token]
        if len(tokens) < 2:
            return False
        return tokens[-2] == task.target_module_name and tokens[-1] == task.port_name

    def _wait_for_tf(self, target_frame: str, source_frame: str, timeout_sec: float = 10.0) -> bool:
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(target_frame, source_frame, Time())
                return True
            except TransformException:
                self.sleep_for(0.1)
        self.get_logger().error(f"Timed out waiting for TF: {source_frame} -> {target_frame}")
        return False

    def _calc_cheatcode_gripper_pose(
        self,
        task: Task,
        port_transform: Transform,
        *,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        z_offset: float = 0.1,
        reset_xy_integrator: bool = False,
        preserve_current_z: bool = False,
        lateral_offset_base: np.ndarray | None = None,
    ) -> Pose:
        q_port = (
            port_transform.rotation.w,
            port_transform.rotation.x,
            port_transform.rotation.y,
            port_transform.rotation.z,
        )
        plug_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            f"{task.cable_name}/{task.plug_name}_link",
            Time(),
        )
        q_plug = (
            plug_tf_stamped.transform.rotation.w,
            plug_tf_stamped.transform.rotation.x,
            plug_tf_stamped.transform.rotation.y,
            plug_tf_stamped.transform.rotation.z,
        )
        q_plug_inv = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])
        gripper_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            "gripper/tcp",
            Time(),
        )
        q_gripper = (
            gripper_tf_stamped.transform.rotation.w,
            gripper_tf_stamped.transform.rotation.x,
            gripper_tf_stamped.transform.rotation.y,
            gripper_tf_stamped.transform.rotation.z,
        )
        q_gripper_target = quaternion_multiply(quaternion_multiply(q_port, q_plug_inv), q_gripper)
        q_gripper_slerp = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)

        gripper_xyz = np.asarray(
            [
                gripper_tf_stamped.transform.translation.x,
                gripper_tf_stamped.transform.translation.y,
                gripper_tf_stamped.transform.translation.z,
            ],
            dtype=np.float64,
        )
        plug_xyz = np.asarray(
            [
                plug_tf_stamped.transform.translation.x,
                plug_tf_stamped.transform.translation.y,
                plug_tf_stamped.transform.translation.z,
            ],
            dtype=np.float64,
        )
        plug_tip_gripper_offset = gripper_xyz - plug_xyz
        tip_x_error = port_transform.translation.x - plug_tf_stamped.transform.translation.x
        tip_y_error = port_transform.translation.y - plug_tf_stamped.transform.translation.y
        if reset_xy_integrator:
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0
        else:
            self._tip_x_error_integrator = float(
                np.clip(
                    self._tip_x_error_integrator + tip_x_error,
                    -self._max_integrator_windup,
                    self._max_integrator_windup,
                )
            )
            self._tip_y_error_integrator = float(
                np.clip(
                    self._tip_y_error_integrator + tip_y_error,
                    -self._max_integrator_windup,
                    self._max_integrator_windup,
                )
            )

        i_gain = 0.15
        target_xyz = np.array(
            [
                port_transform.translation.x + i_gain * self._tip_x_error_integrator,
                port_transform.translation.y + i_gain * self._tip_y_error_integrator,
                port_transform.translation.z + z_offset + plug_tip_gripper_offset[2],
            ],
            dtype=np.float64,
        )
        if lateral_offset_base is not None:
            lateral_offset = np.asarray(lateral_offset_base, dtype=np.float64)
            target_xyz[:2] += lateral_offset[:2]
        if preserve_current_z:
            target_xyz[2] = gripper_xyz[2]
        blend_xyz = position_fraction * target_xyz + (1.0 - position_fraction) * gripper_xyz
        return Pose(
            position=Point(x=float(blend_xyz[0]), y=float(blend_xyz[1]), z=float(blend_xyz[2])),
            orientation=Quaternion(
                w=float(q_gripper_slerp[0]),
                x=float(q_gripper_slerp[1]),
                y=float(q_gripper_slerp[2]),
                z=float(q_gripper_slerp[3]),
            ),
        )

    def _send_relative_target(
        self,
        move_robot: MoveRobotCallback,
        target_pose: Pose,
        *,
        max_translation_step_m: float | None = None,
    ) -> None:
        delta_pose = compute_delta_pose(self._current_tcp_pose(), target_pose)
        if max_translation_step_m is None:
            max_translation_step_m = float(
                os.environ.get("AIC_OFFICIAL_TEACHER_MAX_RELATIVE_TRANSLATION_STEP_M", "0.0")
            )
        if max_translation_step_m > 0.0:
            delta = np.asarray(
                [
                    delta_pose.position.x,
                    delta_pose.position.y,
                    delta_pose.position.z,
                ],
                dtype=np.float64,
            )
            norm = float(np.linalg.norm(delta))
            if norm > max_translation_step_m:
                delta *= max_translation_step_m / norm
                delta_pose.position.x = float(delta[0])
                delta_pose.position.y = float(delta[1])
                delta_pose.position.z = float(delta[2])
        self.set_delta_pose_target(
            move_robot=move_robot,
            delta_pose=delta_pose,
            frame_id="gripper/tcp",
        )

    def _send_absolute_target(self, move_robot: MoveRobotCallback, target_pose: Pose) -> None:
        self.set_pose_target(move_robot=move_robot, pose=target_pose, frame_id="base_link")

    @staticmethod
    def _minimum_jerk_fraction(progress: float) -> float:
        s = float(np.clip(progress, 0.0, 1.0))
        return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5

    @staticmethod
    def _pose_position_array(pose: Pose) -> np.ndarray:
        return np.asarray([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)

    @staticmethod
    def _pose_to_trace_dict(pose: Pose | None) -> dict | None:
        if pose is None:
            return None
        return {
            "position": [float(pose.position.x), float(pose.position.y), float(pose.position.z)],
            "orientation_xyzw": [
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ],
        }

    @staticmethod
    def _transform_to_trace_dict(transform: Transform | None) -> dict | None:
        if transform is None:
            return None
        return {
            "translation": [
                float(transform.translation.x),
                float(transform.translation.y),
                float(transform.translation.z),
            ],
            "rotation_xyzw": [
                float(transform.rotation.x),
                float(transform.rotation.y),
                float(transform.rotation.z),
                float(transform.rotation.w),
            ],
        }

    @staticmethod
    def _pose_quat_wxyz(pose: Pose) -> QuaternionTuple:
        return (pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z)

    @staticmethod
    def _make_pose(position: np.ndarray, quat_wxyz: QuaternionTuple) -> Pose:
        return Pose(
            position=Point(x=float(position[0]), y=float(position[1]), z=float(position[2])),
            orientation=Quaternion(
                w=float(quat_wxyz[0]),
                x=float(quat_wxyz[1]),
                y=float(quat_wxyz[2]),
                z=float(quat_wxyz[3]),
            ),
        )

    @staticmethod
    def _observation_dict(observation) -> dict:
        if isinstance(observation, dict):
            return observation
        controller = getattr(observation, "controller_state", None)
        wrench = getattr(getattr(observation, "wrist_wrench", None), "wrench", None)
        values = {}
        tcp_error_raw = getattr(controller, "tcp_error", None) if controller is not None else None
        tcp_error = list(tcp_error_raw) if tcp_error_raw is not None else []
        if len(tcp_error) >= 6:
            values.update(
                {
                    "tcp_error.x": float(tcp_error[0]),
                    "tcp_error.y": float(tcp_error[1]),
                    "tcp_error.z": float(tcp_error[2]),
                    "tcp_error.rx": float(tcp_error[3]),
                    "tcp_error.ry": float(tcp_error[4]),
                    "tcp_error.rz": float(tcp_error[5]),
                }
            )
        tcp_velocity = getattr(controller, "tcp_velocity", None) if controller is not None else None
        if tcp_velocity is not None:
            try:
                velocity_values = list(tcp_velocity)
            except TypeError:
                velocity_values = []
            if len(velocity_values) >= 6:
                values.update(
                    {
                        "tcp_velocity.linear.x": float(velocity_values[0]),
                        "tcp_velocity.linear.y": float(velocity_values[1]),
                        "tcp_velocity.linear.z": float(velocity_values[2]),
                        "tcp_velocity.angular.x": float(velocity_values[3]),
                        "tcp_velocity.angular.y": float(velocity_values[4]),
                        "tcp_velocity.angular.z": float(velocity_values[5]),
                    }
                )
            elif hasattr(tcp_velocity, "linear"):
                values.update(
                    {
                        "tcp_velocity.linear.x": float(tcp_velocity.linear.x),
                        "tcp_velocity.linear.y": float(tcp_velocity.linear.y),
                        "tcp_velocity.linear.z": float(tcp_velocity.linear.z),
                        "tcp_velocity.angular.x": float(tcp_velocity.angular.x),
                        "tcp_velocity.angular.y": float(tcp_velocity.angular.y),
                        "tcp_velocity.angular.z": float(tcp_velocity.angular.z),
                    }
                )
        if wrench is not None:
            values.update(
                {
                    "wrist_wrench.force.x": float(wrench.force.x),
                    "wrist_wrench.force.y": float(wrench.force.y),
                    "wrist_wrench.force.z": float(wrench.force.z),
                    "wrist_wrench.torque.x": float(wrench.torque.x),
                    "wrist_wrench.torque.y": float(wrench.torque.y),
                    "wrist_wrench.torque.z": float(wrench.torque.z),
                }
            )
        if values:
            return values
        return {}

    def _tracking_error_norm(self, get_observation: GetObservationCallback) -> float | None:
        try:
            obs = self._observation_dict(get_observation())
        except Exception as ex:
            self.get_logger().warn(f"Tracking gate observation failed: {ex}")
            return None
        if not obs and self._latest_observation is not None:
            obs = self._observation_dict(self._latest_observation)
        values = [obs.get("tcp_error.x"), obs.get("tcp_error.y"), obs.get("tcp_error.z")]
        if any(value is None for value in values):
            return None
        return float(np.linalg.norm(np.asarray(values, dtype=np.float64)))

    def _tcp_speed_norm(self, get_observation: GetObservationCallback) -> float | None:
        try:
            obs = self._observation_dict(get_observation())
        except Exception as ex:
            self.get_logger().warn(f"TCP speed observation failed: {ex}")
            return None
        if not obs and self._latest_observation is not None:
            obs = self._observation_dict(self._latest_observation)
        values = [
            obs.get("tcp_velocity.linear.x"),
            obs.get("tcp_velocity.linear.y"),
            obs.get("tcp_velocity.linear.z"),
        ]
        if any(value is None for value in values):
            return None
        return float(np.linalg.norm(np.asarray(values, dtype=np.float64)))

    def _pose_position_error_norm(self, target_pose: Pose) -> float | None:
        try:
            current = self._current_tcp_pose()
        except TransformException as ex:
            self.get_logger().warn(f"Tracking gate TF fallback failed: {ex}")
            return None
        return float(np.linalg.norm(self._pose_position_array(current) - self._pose_position_array(target_pose)))

    def _force_vector(self, get_observation: GetObservationCallback) -> np.ndarray:
        if self._latest_wrench is not None:
            wrench = self._latest_wrench.wrench
            return np.asarray(
                [
                    float(wrench.force.x),
                    float(wrench.force.y),
                    float(wrench.force.z),
                ],
                dtype=np.float64,
            )
        try:
            obs = self._observation_dict(get_observation())
        except Exception as ex:
            self.get_logger().warn(f"F/T observation failed: {ex}")
            return np.zeros(3, dtype=np.float64)
        if not obs and self._latest_observation is not None:
            obs = self._observation_dict(self._latest_observation)
        return np.asarray(
            [
                float(obs.get("wrist_wrench.force.x", 0.0)),
                float(obs.get("wrist_wrench.force.y", 0.0)),
                float(obs.get("wrist_wrench.force.z", 0.0)),
            ],
            dtype=np.float64,
        )

    def _force_delta_norm(
        self,
        get_observation: GetObservationCallback,
        baseline_force: np.ndarray,
    ) -> float:
        return float(np.linalg.norm(self._force_vector(get_observation) - baseline_force))

    def _force_trigger_confirmed(
        self,
        get_observation: GetObservationCallback,
        baseline_force: np.ndarray,
        *,
        dt: float,
    ) -> tuple[bool, float]:
        confirm_sec = float(os.environ.get("AIC_OFFICIAL_TEACHER_FORCE_CONFIRM_SEC", str(dt)))
        if confirm_sec > 0.0:
            self.sleep_for(confirm_sec)
        confirmed_delta = self._force_delta_norm(get_observation, baseline_force)
        return confirmed_delta >= self._ft_threshold_n, confirmed_delta

    def _post_insert_force_check(
        self,
        get_observation: GetObservationCallback,
        baseline_force: np.ndarray,
        *,
        dt: float,
        retry_count: int,
    ) -> tuple[bool, float]:
        check_sec = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_POST_INSERT_FORCE_CHECK_SEC", "0.35")
        )
        if check_sec <= 0.0:
            return True, 0.0
        started = self.time_now()
        max_force_delta = 0.0
        while (self.time_now() - started) < Duration(seconds=check_sec):
            force_delta = self._force_delta_norm(get_observation, baseline_force)
            max_force_delta = max(max_force_delta, force_delta)
            if force_delta >= self._ft_threshold_n:
                self._trace_event(
                    "post_insert_force_check_failed",
                    retry_count=retry_count,
                    force_delta_n=force_delta,
                    max_force_delta_n=max_force_delta,
                    threshold_n=self._ft_threshold_n,
                    check_sec=check_sec,
                )
                return False, max_force_delta
            self.sleep_for(dt)
        self._trace_event(
            "post_insert_force_check_passed",
            retry_count=retry_count,
            max_force_delta_n=max_force_delta,
            threshold_n=self._ft_threshold_n,
            check_sec=check_sec,
        )
        return True, max_force_delta

    def _hold_guarded_insert_exact_target(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        *,
        target_pose: Pose,
        baseline_force: np.ndarray,
        retry_count: int,
        z_offset: float,
        dt: float,
        min_settle_sec: float,
    ) -> tuple[bool, float, float, str]:
        max_settle_sec = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_INSERTION_EXACT_POSITION_MAX_SETTLE_SEC", "1.20")
        )
        speed_threshold = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_INSERTION_EXACT_POSITION_SPEED_GATE_MPS", "0.004")
        )
        started = self.time_now()
        max_force_delta = 0.0
        final_speed = None
        speed_gate_passed = False
        held_count = 0
        while (self.time_now() - started) < Duration(seconds=max_settle_sec):
            self._send_absolute_target(move_robot, target_pose)
            force_delta = self._force_delta_norm(get_observation, baseline_force)
            max_force_delta = max(max_force_delta, force_delta)
            if force_delta >= self._ft_threshold_n:
                confirmed, confirmed_delta = self._force_trigger_confirmed(
                    get_observation,
                    baseline_force,
                    dt=dt,
                )
                return confirmed, force_delta, confirmed_delta, "during_exact_position_settle"
            final_speed = self._tcp_speed_norm(get_observation)
            speed_gate_passed = final_speed is None or final_speed <= speed_threshold
            if (self.time_now() - started) >= Duration(seconds=min_settle_sec) and speed_gate_passed:
                break
            held_count += 1
            self.sleep_for(dt)
        waited = (self.time_now() - started).nanoseconds * 1e-9
        self._trace_event(
            "guarded_insert_exact_position_settle_checked",
            retry_count=retry_count,
            guarded_insert_speed_gate_checked=True,
            guarded_insert_speed_gate_held=held_count > 0,
            held_depth_step_count=held_count,
            z_offset=z_offset,
            min_settle_sec=min_settle_sec,
            max_settle_sec=max_settle_sec,
            time_waited_sec=waited,
            speed_threshold_mps=speed_threshold,
            final_tcp_speed_mps=final_speed,
            speed_gate_passed=speed_gate_passed,
            max_force_delta_n=max_force_delta,
            target_tcp_pose=self._pose_to_trace_dict(target_pose),
        )
        return False, max_force_delta, max_force_delta, "after_exact_position_settle"

    def _run_local_preinsert_align(
        self,
        task: Task,
        port_transform: Transform,
        move_robot: MoveRobotCallback,
        *,
        duration_sec: float,
        dt: float,
        z_offset: float,
        lateral_offset_base: np.ndarray | None = None,
        preserve_current_z: bool = True,
        max_speed_mps: float | None = None,
    ) -> None:
        start_pose = self._current_tcp_pose()
        target_pose = self._calc_cheatcode_gripper_pose(
            task,
            port_transform,
            slerp_fraction=1.0,
            position_fraction=1.0,
            z_offset=z_offset,
            reset_xy_integrator=True,
            preserve_current_z=preserve_current_z,
            lateral_offset_base=lateral_offset_base,
        )
        start_xyz = self._pose_position_array(start_pose)
        target_xyz = self._pose_position_array(target_pose)
        start_quat = self._pose_quat_wxyz(start_pose)
        target_quat = self._pose_quat_wxyz(target_pose)
        distance_m = float(np.linalg.norm(target_xyz - start_xyz))
        speed_limit_mps = (
            float(max_speed_mps)
            if max_speed_mps is not None
            else float(os.environ.get("AIC_OFFICIAL_TEACHER_LOCAL_PREINSERT_ALIGN_SPEED_MPS", "0.08"))
        )
        effective_duration_sec = max(float(duration_sec), distance_m / max(speed_limit_mps, 1e-6))
        steps = max(1, int(effective_duration_sec / dt))
        self._trace_event(
            "local_preinsert_align_started",
            requested_duration_sec=duration_sec,
            duration_sec=effective_duration_sec,
            max_speed_mps=speed_limit_mps,
            z_offset=z_offset,
            preserve_current_z=preserve_current_z,
            lateral_offset_base=(
                np.asarray(lateral_offset_base, dtype=np.float64).tolist()
                if lateral_offset_base is not None
                else None
            ),
            distance_m=distance_m,
        )
        for step in range(steps + 1):
            fraction = self._minimum_jerk_fraction(step / steps)
            pose = self._make_pose(
                start_xyz + fraction * (target_xyz - start_xyz),
                quaternion_slerp(start_quat, target_quat, fraction),
            )
            self._send_absolute_target(move_robot, pose)
            self.sleep_for(dt)
        self._trace_event("local_preinsert_align_completed")

    def _run_preinsert_micro_align(
        self,
        task: Task,
        port_transform: Transform,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        *,
        duration_sec: float,
        dt: float,
        z_offset: float,
    ) -> bool:
        if duration_sec <= 0.0:
            self._trace_event("preinsert_micro_align_skipped", reason="duration_disabled")
            return True
        steps = max(1, int(duration_sec / dt))
        max_xy_step = float(os.environ.get("AIC_OFFICIAL_TEACHER_MICRO_ALIGN_MAX_XY_STEP_M", "0.0005"))
        gain = float(os.environ.get("AIC_OFFICIAL_TEACHER_MICRO_ALIGN_GAIN", "0.45"))
        force_abort_fraction = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_MICRO_ALIGN_FORCE_ABORT_FRACTION", "0.75")
        )
        baseline_force = self._force_vector(get_observation)
        start_pose = self._current_tcp_pose()
        self._trace_event(
            "preinsert_micro_align_started",
            duration_sec=duration_sec,
            max_xy_step_m=max_xy_step,
            gain=gain,
            frame="gripper/tcp",
            representation="relative_delta",
            start_tcp_pose=self._pose_to_trace_dict(start_pose),
        )
        final_tip_error = None
        aborted = False
        for _ in range(steps):
            try:
                plug_transform = self._parent_node._tf_buffer.lookup_transform(
                    "base_link",
                    f"{task.cable_name}/{task.plug_name}_link",
                    Time(),
                ).transform
                current_pose = self._current_tcp_pose()
                force_delta = self._force_delta_norm(get_observation, baseline_force)
                if force_delta >= self._ft_threshold_n * force_abort_fraction:
                    aborted = True
                    self._trace_event(
                        "preinsert_micro_align_aborted_force",
                        force_delta_n=force_delta,
                        abort_threshold_n=self._ft_threshold_n * force_abort_fraction,
                    )
                    break
                tip_error_base = np.asarray(
                    [
                        port_transform.translation.x - plug_transform.translation.x,
                        port_transform.translation.y - plug_transform.translation.y,
                        0.0,
                    ],
                    dtype=np.float64,
                )
                current_quat_xyzw = np.asarray(
                    [
                        current_pose.orientation.x,
                        current_pose.orientation.y,
                        current_pose.orientation.z,
                        current_pose.orientation.w,
                    ],
                    dtype=np.float64,
                )
                rot_tcp_to_base = quaternion_xyzw_to_rotation_matrix(current_quat_xyzw)
                correction_tcp = gain * (rot_tcp_to_base.T @ tip_error_base)
                correction_tcp[2] = 0.0
                norm = float(np.linalg.norm(correction_tcp))
                if norm > max_xy_step:
                    correction_tcp *= max_xy_step / norm
                self.set_delta_pose_target_from_components(
                    move_robot=move_robot,
                    delta_position_xyz=correction_tcp,
                    delta_rotation_xyz=np.zeros(3, dtype=np.float64),
                    frame_id="gripper/tcp",
                    max_translation=max_xy_step,
                )
                final_tip_error = float(np.linalg.norm(tip_error_base[:2]))
            except TransformException as ex:
                self.get_logger().warn(f"Pre-insertion micro-align TF lookup failed: {ex}")
            self.sleep_for(dt)
        self._trace_event(
            "preinsert_micro_align_completed",
            final_tip_error_m=final_tip_error,
            aborted=aborted,
            final_tcp_pose=self._pose_to_trace_dict(self._current_tcp_pose()),
        )
        return not aborted

    def _run_port_frame_precontact_align(
        self,
        task: Task,
        port_transform: Transform,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        *,
        duration_sec: float,
        dt: float,
        z_offset: float,
        target_z_m: float | None,
    ) -> bool:
        if duration_sec <= 0.0:
            self._trace_event("precontact_port_align_skipped", reason="duration_disabled")
            return True
        max_offset = float(os.environ.get("AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_MAX_OFFSET_M", "0.00025"))
        residual_threshold = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_RESIDUAL_M", "0.0006")
        )
        speed_threshold = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_SPEED_MPS", "0.004")
        )
        force_abort_fraction = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_FORCE_ABORT_FRACTION", "0.60")
        )
        gain = float(os.environ.get("AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_GAIN", "0.15"))
        baseline_force = self._force_vector(get_observation)
        steps = max(1, int(duration_sec / dt))
        final_residual = None
        final_offset = np.zeros(3, dtype=np.float64)
        final_speed = None
        aborted = False
        converged = False
        self._trace_event(
            "precontact_port_align_started",
            duration_sec=duration_sec,
            max_offset_m=max_offset,
            residual_threshold_m=residual_threshold,
            speed_threshold_mps=speed_threshold,
            force_abort_threshold_n=self._ft_threshold_n * force_abort_fraction,
            gain=gain,
            frame="base_link",
            representation="absolute_cartesian_pose",
        )
        for _ in range(steps):
            try:
                plug_transform = self._parent_node._tf_buffer.lookup_transform(
                    "base_link",
                    f"{task.cable_name}/{task.plug_name}_link",
                    Time(),
                ).transform
                tip_error_base = np.asarray(
                    [
                        port_transform.translation.x - plug_transform.translation.x,
                        port_transform.translation.y - plug_transform.translation.y,
                        0.0,
                    ],
                    dtype=np.float64,
                )
                final_residual = float(np.linalg.norm(tip_error_base[:2]))
                force_delta = self._force_delta_norm(get_observation, baseline_force)
                final_speed = self._tcp_speed_norm(get_observation)
                if force_delta >= self._ft_threshold_n:
                    aborted = True
                    self._trace_event(
                        "precontact_port_align_aborted_force",
                        force_delta_n=force_delta,
                        abort_threshold_n=self._ft_threshold_n,
                        residual_m=final_residual,
                        fatal=True,
                    )
                    break
                if force_delta >= self._ft_threshold_n * force_abort_fraction:
                    self._trace_event(
                        "precontact_port_align_stopped_subthreshold_force",
                        force_delta_n=force_delta,
                        stop_threshold_n=self._ft_threshold_n * force_abort_fraction,
                        residual_m=final_residual,
                    )
                    break
                speed_ok = final_speed is None or final_speed <= speed_threshold
                if final_residual <= residual_threshold and speed_ok:
                    converged = True
                    break
                correction = gain * tip_error_base
                correction_norm = float(np.linalg.norm(correction[:2]))
                if correction_norm > max_offset > 0.0:
                    correction *= max_offset / correction_norm
                final_offset = correction
                target_pose = self._calc_cheatcode_gripper_pose(
                    task,
                    port_transform,
                    slerp_fraction=1.0,
                    position_fraction=1.0,
                    z_offset=z_offset,
                    reset_xy_integrator=False,
                    preserve_current_z=True,
                    lateral_offset_base=correction,
                )
                if target_z_m is not None:
                    target_pose.position.z = float(target_z_m)
                self._send_absolute_target(move_robot, target_pose)
            except TransformException as ex:
                self.get_logger().warn(f"Pre-contact port align TF lookup failed: {ex}")
            self.sleep_for(dt)
        self._trace_event(
            "precontact_port_align_completed",
            converged=converged,
            aborted=aborted,
            final_residual_m=final_residual,
            final_offset_base_m=final_offset.tolist(),
            final_tcp_speed_mps=final_speed,
        )
        self._last_precontact_lateral_offset_base = (
            np.zeros(3, dtype=np.float64) if aborted else final_offset.copy()
        )
        return not aborted

    def _hold_preinsert_until_tracking_gate(
        self,
        task: Task,
        port_transform: Transform,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        *,
        settle_sec: float,
        dt: float,
        z_offset: float,
        preserve_current_z: bool = True,
        target_z_m: float | None = None,
        lateral_offset_base: np.ndarray | None = None,
    ) -> bool:
        threshold = self._tracking_gate_threshold_m
        speed_threshold = float(os.environ.get("AIC_OFFICIAL_TEACHER_TRACKING_GATE_SPEED_MPS", "0.005"))
        force_fraction = float(os.environ.get("AIC_OFFICIAL_TEACHER_TRACKING_GATE_FORCE_FRACTION", "1.0"))
        force_threshold = self._ft_threshold_n * force_fraction
        timeout_sec = max(settle_sec, self._tracking_gate_timeout_sec)
        started = self.time_now()
        baseline_force = self._force_vector(get_observation)
        passed = False
        final_error = None
        final_speed = None
        final_force_delta = None
        gate_threshold = threshold
        gate_source = "unavailable"
        while (self.time_now() - started) < Duration(seconds=timeout_sec):
            try:
                target_pose = self._calc_cheatcode_gripper_pose(
                    task,
                    port_transform,
                    slerp_fraction=1.0,
                    position_fraction=1.0,
                    z_offset=z_offset,
                    reset_xy_integrator=True,
                    preserve_current_z=preserve_current_z,
                    lateral_offset_base=lateral_offset_base,
                )
                if target_z_m is not None:
                    target_pose.position.z = float(target_z_m)
                hold_command_mode = os.environ.get(
                    "AIC_OFFICIAL_TEACHER_PREINSERT_HOLD_COMMAND_MODE",
                    "absolute",
                )
                if preserve_current_z or hold_command_mode == "absolute":
                    self._send_absolute_target(move_robot, target_pose)
                else:
                    self._send_relative_target(move_robot, target_pose)
            except TransformException as ex:
                self.get_logger().warn(f"Pre-insertion tracking-gate TF lookup failed: {ex}")
                target_pose = None
            final_error = self._tracking_error_norm(get_observation)
            gate_source = "controller_tcp_error"
            if final_error is None and target_pose is not None:
                final_error = self._pose_position_error_norm(target_pose)
                gate_threshold = float(
                    os.environ.get("AIC_OFFICIAL_TEACHER_TRACKING_GATE_FALLBACK_THRESHOLD_M", "0.015")
                )
                gate_source = "tf_pose_error_fallback"
            final_speed = self._tcp_speed_norm(get_observation)
            final_force_delta = self._force_delta_norm(get_observation, baseline_force)
            speed_ok = final_speed is None or final_speed <= speed_threshold
            force_ok = final_force_delta < force_threshold
            passed = bool(final_error is not None and final_error <= gate_threshold and speed_ok and force_ok)
            if passed:
                if (self.time_now() - started) >= Duration(seconds=settle_sec):
                    break
            self.sleep_for(dt)
        waited = (self.time_now() - started).nanoseconds * 1e-9
        current_pose = None
        plug_transform = None
        try:
            current_pose = self._current_tcp_pose()
        except TransformException:
            current_pose = None
        try:
            plug_transform = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                f"{task.cable_name}/{task.plug_name}_link",
                Time(),
            ).transform
        except TransformException:
            plug_transform = None
        self._trace_event(
            "tracking_gate_checked",
            tracking_gate_passed=passed,
            threshold_m=gate_threshold if final_error is not None else threshold,
            nominal_controller_threshold_m=threshold,
            gate_source=gate_source if final_error is not None else "unavailable",
            timeout_sec=timeout_sec,
            final_tracking_error_m=final_error,
            speed_threshold_mps=speed_threshold,
            final_tcp_speed_mps=final_speed,
            force_delta_n=final_force_delta,
            ft_threshold_n=self._ft_threshold_n,
            force_gate_threshold_n=force_threshold,
            force_gate_fraction=force_fraction,
            current_tcp_pose=self._pose_to_trace_dict(current_pose),
            target_tcp_pose=self._pose_to_trace_dict(target_pose),
            pinned_target_z_m=target_z_m,
            lateral_offset_base=(
                np.asarray(lateral_offset_base, dtype=np.float64).tolist()
                if lateral_offset_base is not None
                else None
            ),
            port_transform=self._transform_to_trace_dict(port_transform),
            plug_transform=self._transform_to_trace_dict(plug_transform),
            time_waited_sec=waited,
        )
        self._last_tracking_gate_force_delta_n = final_force_delta
        self._last_tracking_gate_error_m = final_error
        self._last_tracking_gate_speed_mps = final_speed
        return passed

    def _compute_recovery_retry_lateral_offset(
        self,
        task: Task,
        port_transform: Transform,
        *,
        retry_count: int,
    ) -> np.ndarray:
        if retry_count <= 0:
            return np.zeros(3, dtype=np.float64)
        if os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_RETRY_XY_BIAS", "false").lower() != "true":
            return np.zeros(3, dtype=np.float64)
        try:
            plug_transform = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                f"{task.cable_name}/{task.plug_name}_link",
                Time(),
            ).transform
        except TransformException as ex:
            self.get_logger().warn(f"Recovery retry XY-bias TF lookup failed: {ex}")
            return np.zeros(3, dtype=np.float64)
        tip_error_base = np.asarray(
            [
                port_transform.translation.x - plug_transform.translation.x,
                port_transform.translation.y - plug_transform.translation.y,
                0.0,
            ],
            dtype=np.float64,
        )
        gain = float(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_RETRY_XY_BIAS_GAIN", "0.75"))
        step_limit = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_RETRY_XY_BIAS_STEP_LIMIT_M", "0.0015")
        )
        max_limit = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_RETRY_XY_BIAS_MAX_M", "0.003")
        )
        limit = min(max_limit, max(step_limit, step_limit * retry_count))
        offset = gain * tip_error_base
        norm = float(np.linalg.norm(offset[:2]))
        if norm > limit > 0.0:
            offset *= limit / norm
        self._trace_event(
            "recovery_retry_xy_bias_computed",
            retry_count=retry_count,
            tip_error_base_m=tip_error_base.tolist(),
            lateral_offset_base_m=offset.tolist(),
            gain=gain,
            limit_m=limit,
            port_transform=self._transform_to_trace_dict(port_transform),
            plug_transform=self._transform_to_trace_dict(plug_transform),
        )
        return offset

    def _hold_recovery_return_to_preinsert_gate(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        *,
        target_pose: Pose,
        dt: float,
        gate_context: str,
    ) -> bool:
        threshold = float(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_RETURN_Z_THRESHOLD_M", "0.004"))
        timeout_sec = float(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_RETURN_Z_TIMEOUT_SEC", "2.0"))
        speed_threshold = float(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_RETURN_SPEED_MPS", "0.010"))
        baseline_force = self._force_vector(get_observation)
        started = self.time_now()
        passed = False
        final_z_error = None
        final_speed = None
        final_force_delta = None
        while (self.time_now() - started) < Duration(seconds=timeout_sec):
            self._send_absolute_target(move_robot, target_pose)
            try:
                current_pose = self._current_tcp_pose()
                final_z_error = abs(float(current_pose.position.z) - float(target_pose.position.z))
            except TransformException:
                final_z_error = None
            final_speed = self._tcp_speed_norm(get_observation)
            final_force_delta = self._force_delta_norm(get_observation, baseline_force)
            speed_ok = final_speed is None or final_speed <= speed_threshold
            force_ok = final_force_delta < self._ft_threshold_n
            if final_z_error is not None and final_z_error <= threshold and speed_ok and force_ok:
                passed = True
                break
            self.sleep_for(dt)
        waited = (self.time_now() - started).nanoseconds * 1e-9
        current_pose = None
        try:
            current_pose = self._current_tcp_pose()
        except TransformException:
            pass
        self._trace_event(
            "recovery_return_to_preinsert_gate_checked",
            gate_context=gate_context,
            tracking_gate_passed=passed,
            target_z_m=float(target_pose.position.z),
            threshold_m=threshold,
            timeout_sec=timeout_sec,
            speed_threshold_mps=speed_threshold,
            final_z_error_m=final_z_error,
            final_tcp_speed_mps=final_speed,
            force_delta_n=final_force_delta,
            ft_threshold_n=self._ft_threshold_n,
            current_tcp_pose=self._pose_to_trace_dict(current_pose),
            target_tcp_pose=self._pose_to_trace_dict(target_pose),
            time_waited_sec=waited,
        )
        return passed

    def _compute_guarded_insertion_depth(
        self,
        task: Task,
        port_transform: Transform,
        *,
        fallback_depth_m: float,
    ) -> float:
        min_depth = float(os.environ.get("AIC_OFFICIAL_TEACHER_MIN_INSERTION_DEPTH_M", "0.020"))
        max_depth = float(os.environ.get("AIC_OFFICIAL_TEACHER_MAX_INSERTION_DEPTH_M", "0.080"))
        seat_margin = float(os.environ.get("AIC_OFFICIAL_TEACHER_INSERTION_SEAT_MARGIN_M", "0.005"))
        try:
            plug_transform = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                f"{task.cable_name}/{task.plug_name}_link",
                Time(),
            ).transform
        except TransformException as ex:
            depth = float(np.clip(fallback_depth_m, min_depth, max_depth))
            self._trace_event(
                "guarded_insert_depth_estimate_failed",
                fallback_depth_m=fallback_depth_m,
                insertion_depth_m=depth,
                error=str(ex),
            )
            return depth

        axial_gap = float(plug_transform.translation.z - port_transform.translation.z)
        required_depth = max(fallback_depth_m, axial_gap + seat_margin)
        depth = float(np.clip(required_depth, min_depth, max_depth))
        self._trace_event(
            "guarded_insert_depth_estimated",
            port_z_m=float(port_transform.translation.z),
            plug_z_m=float(plug_transform.translation.z),
            axial_gap_m=axial_gap,
            seat_margin_m=seat_margin,
            fallback_depth_m=fallback_depth_m,
            min_depth_m=min_depth,
            max_depth_m=max_depth,
            insertion_depth_m=depth,
            clipped=depth != required_depth,
        )
        return depth

    def _current_cheatcode_z_offset(self, task: Task, port_transform: Transform) -> float:
        plug_transform = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            f"{task.cable_name}/{task.plug_name}_link",
            Time(),
        ).transform
        return float(plug_transform.translation.z - port_transform.translation.z)

    def _run_cheatcode_handoff_to_insert_start(
        self,
        task: Task,
        port_transform: Transform,
        move_robot: MoveRobotCallback,
        *,
        from_z_offset: float,
        to_z_offset: float,
        dt: float,
        lateral_offset_base: np.ndarray | None = None,
    ) -> None:
        handoff_blend_sec = float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_HANDOFF_SEC", "2.0"))
        handoff_speed_mps = float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_HANDOFF_SPEED_MPS", "0.02"))
        rate_limited_sec = abs(from_z_offset - to_z_offset) / max(handoff_speed_mps, 1e-6)
        steps = max(1, int(max(handoff_blend_sec, rate_limited_sec) / dt))
        if abs(from_z_offset - to_z_offset) <= 1e-9:
            self._trace_event(
                "cheatcode_handoff_skipped",
                from_z_offset=from_z_offset,
                to_z_offset=to_z_offset,
                reason="already_at_insertion_start_offset",
            )
            return
        self._trace_event(
            "cheatcode_handoff_started",
            from_z_offset=from_z_offset,
            to_z_offset=to_z_offset,
            duration_sec=steps * dt,
            handoff_speed_mps=handoff_speed_mps,
            command_mode="absolute_cartesian_pose_base_link",
            lateral_offset_base=(
                np.asarray(lateral_offset_base, dtype=np.float64).tolist()
                if lateral_offset_base is not None
                else None
            ),
        )
        for step in range(steps):
            fraction = (step + 1) / steps
            z_offset = from_z_offset + fraction * (to_z_offset - from_z_offset)
            try:
                self._send_absolute_target(
                    move_robot,
                    self._calc_cheatcode_gripper_pose(
                        task,
                        port_transform,
                        slerp_fraction=1.0,
                        position_fraction=1.0,
                        z_offset=z_offset,
                        reset_xy_integrator=True,
                        preserve_current_z=False,
                        lateral_offset_base=lateral_offset_base,
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(f"CheatCode handoff TF lookup failed: {ex}")
            self.sleep_for(dt)
        self._trace_event("cheatcode_handoff_completed")

    def _backoff_and_realign(
        self,
        task: Task,
        port_transform: Transform,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        *,
        current_z_offset: float,
        dt: float,
        release_baseline_force: np.ndarray | None = None,
        retry_start_z_m: float | None = None,
    ) -> bool:
        backoff_increment = float(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_BACKOFF_DISTANCE_M", "0.005"))
        max_backoff_distance = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_MAX_BACKOFF_DISTANCE_M", "0.03")
        )
        min_backoff_distance = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_MIN_BACKOFF_DISTANCE_M", "0.0")
        )
        min_backoff_distance = min(max_backoff_distance, max(0.0, min_backoff_distance))
        backoff_duration = float(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_BACKOFF_SEC", "0.45"))
        release_timeout = float(os.environ.get("AIC_OFFICIAL_TEACHER_FORCE_RELEASE_TIMEOUT_SEC", "5.0"))
        release_check_sec = float(os.environ.get("AIC_OFFICIAL_TEACHER_FORCE_RELEASE_STAGE_CHECK_SEC", "0.10"))
        release_threshold = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_RELEASE_FORCE_THRESHOLD_N", str(self._ft_threshold_n))
        )
        baseline = (
            np.asarray(release_baseline_force, dtype=np.float64)
            if release_baseline_force is not None
            else self._force_vector(get_observation)
        )
        start_pose = self._current_tcp_pose()
        start_xyz = self._pose_position_array(start_pose)
        start_quat = self._pose_quat_wxyz(start_pose)
        self._trace_event(
            "recovery_backoff_started",
            backoff_increment_m=backoff_increment,
            max_backoff_distance_m=max_backoff_distance,
            min_backoff_distance_m=min_backoff_distance,
            release_timeout_sec=release_timeout,
            release_threshold_n=release_threshold,
            start_tcp_pose=self._pose_to_trace_dict(start_pose),
        )
        force_released = False
        release_started = self.time_now()
        total_backoff = 0.0
        backoff_pose = start_pose
        stage_index = 0
        while total_backoff < max_backoff_distance and (self.time_now() - release_started) < Duration(seconds=release_timeout):
            stage_index += 1
            next_backoff = min(max_backoff_distance, total_backoff + backoff_increment)
            stage_start_xyz = self._pose_position_array(self._current_tcp_pose())
            stage_target_xyz = start_xyz.copy()
            stage_target_xyz[2] += next_backoff
            backoff_pose = self._make_pose(stage_target_xyz, start_quat)
            steps = max(1, int(backoff_duration / dt))
            self._trace_event(
                "recovery_backoff_stage_started",
                stage_index=stage_index,
                target_total_backoff_m=next_backoff,
                target_tcp_pose=self._pose_to_trace_dict(backoff_pose),
            )
            for step in range(steps + 1):
                fraction = self._minimum_jerk_fraction(step / steps)
                pose = self._make_pose(
                    stage_start_xyz + fraction * (stage_target_xyz - stage_start_xyz),
                    start_quat,
                )
                self._send_absolute_target(move_robot, pose)
                self.sleep_for(dt)
            total_backoff = next_backoff
            self._trace_event(
                "recovery_backoff_stage_completed",
                stage_index=stage_index,
                backoff_distance_achieved_m=total_backoff,
            )

            if total_backoff < min_backoff_distance:
                continue

            stage_release_started = self.time_now()
            while (
                (self.time_now() - stage_release_started) < Duration(seconds=release_check_sec)
                and (self.time_now() - release_started) < Duration(seconds=release_timeout)
            ):
                force_delta = float(np.linalg.norm(self._force_vector(get_observation) - baseline))
                self._send_absolute_target(move_robot, backoff_pose)
                if force_delta < release_threshold:
                    force_released = True
                    break
                self.sleep_for(dt)
            if force_released:
                break
        self._trace_event("recovery_backoff_completed", backoff_distance_achieved_m=total_backoff)
        self._trace_event(
            "recovery_force_release_wait",
            force_released=force_released,
            threshold_n=release_threshold,
            contact_threshold_n=self._ft_threshold_n,
            backoff_distance_achieved_m=total_backoff,
            min_backoff_distance_m=min_backoff_distance,
            max_backoff_distance_m=max_backoff_distance,
        )
        if not force_released:
            return False
        if retry_start_z_m is not None:
            lift_start_pose = self._current_tcp_pose()
            lift_start_xyz = self._pose_position_array(lift_start_pose)
            lift_target_xyz = lift_start_xyz.copy()
            lift_target_xyz[2] = max(float(retry_start_z_m), float(lift_start_xyz[2]))
            lift_distance = float(abs(lift_target_xyz[2] - lift_start_xyz[2]))
            lift_duration = float(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_RETURN_TO_PREINSERT_SEC", "2.5"))
            lift_steps = max(1, int(lift_duration / dt))
            lift_target_pose = self._make_pose(lift_target_xyz, start_quat)
            self._trace_event(
                "recovery_return_to_preinsert_started",
                target_z_m=float(lift_target_xyz[2]),
                lift_distance_m=lift_distance,
                duration_sec=lift_duration,
            )
            for step in range(lift_steps + 1):
                fraction = self._minimum_jerk_fraction(step / lift_steps)
                pose = self._make_pose(
                    lift_start_xyz + fraction * (lift_target_xyz - lift_start_xyz),
                    start_quat,
                )
                self._send_absolute_target(move_robot, pose)
                self.sleep_for(dt)
            self._trace_event(
                "recovery_return_to_preinsert_completed",
                target_z_m=float(lift_target_xyz[2]),
            )
            if not self._hold_recovery_return_to_preinsert_gate(
                get_observation,
                move_robot,
                target_pose=lift_target_pose,
                dt=dt,
                gate_context="after_force_release",
            ):
                return False
        recovery_realign_preserve_current_z = os.environ.get(
            "AIC_OFFICIAL_TEACHER_RECOVERY_REALIGN_PRESERVE_CURRENT_Z",
            "true",
        ).lower() in {"1", "true", "yes", "on"}
        recovery_realign_speed_mps = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_REALIGN_SPEED_MPS", "0.02")
        )
        self._trace_event(
            "recovery_realign_started",
            preserve_current_z=recovery_realign_preserve_current_z,
            max_speed_mps=recovery_realign_speed_mps,
        )
        self._run_local_preinsert_align(
            task,
            port_transform,
            move_robot,
            duration_sec=float(os.environ.get("AIC_OFFICIAL_TEACHER_LOCAL_PREINSERT_ALIGN_SEC", "2.25")),
            dt=dt,
            z_offset=float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_START_Z_OFFSET", "0.03")),
            preserve_current_z=recovery_realign_preserve_current_z,
            max_speed_mps=recovery_realign_speed_mps,
        )
        self._trace_event("recovery_realign_completed")
        if retry_start_z_m is not None:
            post_align_pose = self._current_tcp_pose()
            post_align_xyz = self._pose_position_array(post_align_pose)
            post_align_xyz[2] = max(float(retry_start_z_m), float(post_align_xyz[2]))
            post_align_target_pose = self._make_pose(post_align_xyz, self._pose_quat_wxyz(post_align_pose))
            post_align_start_xyz = self._pose_position_array(post_align_pose)
            post_align_return_duration = float(
                os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_POST_ALIGN_RETURN_SEC", "1.25")
            )
            post_align_steps = max(1, int(post_align_return_duration / dt))
            self._trace_event(
                "recovery_post_realign_return_started",
                target_z_m=float(post_align_xyz[2]),
                lift_distance_m=float(abs(post_align_xyz[2] - post_align_start_xyz[2])),
                duration_sec=post_align_return_duration,
            )
            for step in range(post_align_steps + 1):
                fraction = self._minimum_jerk_fraction(step / post_align_steps)
                pose = self._make_pose(
                    post_align_start_xyz + fraction * (post_align_xyz - post_align_start_xyz),
                    self._pose_quat_wxyz(post_align_pose),
                )
                self._send_absolute_target(move_robot, pose)
                self.sleep_for(dt)
            self._trace_event(
                "recovery_post_realign_return_completed",
                target_z_m=float(post_align_xyz[2]),
            )
            if not self._hold_recovery_return_to_preinsert_gate(
                get_observation,
                move_robot,
                target_pose=post_align_target_pose,
                dt=dt,
                gate_context="after_realign",
            ):
                return False
        return True

    def _send_joint_target(self, move_robot: MoveRobotCallback, target: ReplayTarget) -> None:
        if target.joint_positions is None:
            self.get_logger().warn(
                "Joint replay target is missing joint_positions; falling back to Cartesian pose command."
            )
            self.set_pose_target(
                move_robot=move_robot,
                pose=self.target_to_pose(target),
                frame_id="base_link",
            )
            return
        n_joints = len(target.joint_positions)
        velocities = (
            list(target.joint_velocities)
            if target.joint_velocities is not None
            else [0.0] * n_joints
        )
        move_robot(
            joint_motion_update=JointMotionUpdate(
                target_state=JointTrajectoryPoint(
                    positions=[float(v) for v in target.joint_positions],
                    velocities=[float(v) for v in velocities],
                ),
                target_stiffness=[100.0] * n_joints,
                target_damping=[20.0] * n_joints,
                trajectory_generation_mode=TrajectoryGenerationMode(
                    mode=TrajectoryGenerationMode.MODE_POSITION,
                ),
                target_feedforward_torque=[0.0] * n_joints,
            )
        )

    def _run_online_cheatcode_insertion(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        plug_frame = f"{task.cable_name}/{task.plug_name}_link"
        for frame in (port_frame, plug_frame, "gripper/tcp"):
            if not self._wait_for_tf("base_link", frame):
                return False
        port_transform = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            port_frame,
            Time(),
        ).transform
        send_feedback("official_teacher_replay_online_cheatcode_final_insertion")
        dt = float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_DT", "0.05"))
        z_offset = float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_START_Z_OFFSET", "0.03"))
        settle_sec = float(os.environ.get("AIC_OFFICIAL_TEACHER_PRE_INSERT_SETTLE_SEC", "1.0"))
        cheatcode_z_mode = os.environ.get(
            "AIC_OFFICIAL_TEACHER_CHEATCODE_Z_MODE",
            "cheatcode_offsets",
        )
        preinsert_align_preserve_current_z = os.environ.get(
            "AIC_OFFICIAL_TEACHER_PREINSERT_ALIGN_PRESERVE_CURRENT_Z",
            "true",
        ).lower() in {"1", "true", "yes", "on"}
        preinsert_gate_preserve_current_z = os.environ.get(
            "AIC_OFFICIAL_TEACHER_PREINSERT_GATE_PRESERVE_CURRENT_Z",
            "true",
        ).lower() in {"1", "true", "yes", "on"}
        if not self._local_preinsert_align_done:
            self._run_local_preinsert_align(
                task,
                port_transform,
                move_robot,
                duration_sec=float(os.environ.get("AIC_OFFICIAL_TEACHER_LOCAL_PREINSERT_ALIGN_SEC", "2.25")),
                dt=dt,
                z_offset=z_offset,
                preserve_current_z=preinsert_align_preserve_current_z,
            )
            self._local_preinsert_align_done = True
        if not self._preinsert_settle_done:
            gate_passed = self._hold_preinsert_until_tracking_gate(
                task,
                port_transform,
                get_observation,
                move_robot,
                settle_sec=settle_sec,
                dt=dt,
                z_offset=z_offset,
                preserve_current_z=preinsert_gate_preserve_current_z,
            )
            self._preinsert_settle_done = True
            if not gate_passed and self._expert_mode == "nominal":
                send_feedback("official_teacher_replay_tracking_gate_failed")
                self._trace_event("nominal_rejected_tracking_gate")
                return False
            if not gate_passed:
                send_feedback("official_teacher_replay_tracking_gate_realign")
                self._trace_event("recovery_tracking_gate_realign_started")
                self._run_local_preinsert_align(
                    task,
                    port_transform,
                    move_robot,
                    duration_sec=float(
                        os.environ.get("AIC_OFFICIAL_TEACHER_LOCAL_PREINSERT_ALIGN_SEC", "2.25")
                    ),
                    dt=dt,
                    z_offset=z_offset,
                    preserve_current_z=preinsert_align_preserve_current_z,
                )
                gate_passed = self._hold_preinsert_until_tracking_gate(
                    task,
                    port_transform,
                    get_observation,
                    move_robot,
                    settle_sec=settle_sec,
                    dt=dt,
                    z_offset=z_offset,
                    preserve_current_z=preinsert_gate_preserve_current_z,
                )
                self._trace_event(
                    "recovery_tracking_gate_realign_completed",
                    tracking_gate_passed=gate_passed,
                )
                if not gate_passed:
                    send_feedback("official_teacher_replay_tracking_gate_failed")
                    return False

        insertion_speed_mps = float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_INSERTION_SPEED_MPS", "0.0013"))
        if "AIC_OFFICIAL_TEACHER_CHEATCODE_HANDOFF_START_Z_OFFSET" in os.environ:
            handoff_start_z_offset = float(os.environ["AIC_OFFICIAL_TEACHER_CHEATCODE_HANDOFF_START_Z_OFFSET"])
        else:
            handoff_start_z_offset = (
                self._current_cheatcode_z_offset(task, port_transform)
                if cheatcode_z_mode == "cheatcode_offsets"
                else z_offset
            )
        fixed_end_z_offset = float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_END_Z_OFFSET", "-0.015"))
        fallback_distance = max(0.0, z_offset - fixed_end_z_offset)
        if cheatcode_z_mode == "tf_depth":
            planned_insertion_distance = self._compute_guarded_insertion_depth(
                task,
                port_transform,
                fallback_depth_m=fallback_distance,
            )
        else:
            planned_insertion_distance = fallback_distance
        insertion_command_mode = os.environ.get(
            "AIC_OFFICIAL_TEACHER_INSERTION_COMMAND_MODE",
            "exact_position",
        )
        exact_position_step_m = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_INSERTION_EXACT_POSITION_STEP_M", "0.0005")
        )
        exact_position_settle_sec = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_INSERTION_EXACT_POSITION_SETTLE_SEC", "0.45")
        )
        insertion_max_step_m = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_INSERTION_MAX_TRANSLATION_STEP_M", "0.0")
        )
        min_success_z_offset = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_MIN_INSERTION_SUCCESS_Z_OFFSET", "1.0")
        )
        max_retries = int(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_MAX_RETRIES", "2"))
        retry_count = 0
        original_preinsert_z = float(self._current_tcp_pose().position.z)
        while retry_count <= max_retries:
            retry_lateral_offset = self._compute_recovery_retry_lateral_offset(
                task,
                port_transform,
                retry_count=retry_count,
            )
            if retry_count > 0 and float(np.linalg.norm(retry_lateral_offset[:2])) > 1e-6:
                send_feedback("official_teacher_replay_recovery_retry_xy_bias_align")
                self._trace_event(
                    "recovery_retry_xy_bias_align_started",
                    retry_count=retry_count,
                    lateral_offset_base_m=retry_lateral_offset.tolist(),
                )
                self._run_local_preinsert_align(
                    task,
                    port_transform,
                    move_robot,
                    duration_sec=float(
                        os.environ.get(
                            "AIC_OFFICIAL_TEACHER_RECOVERY_RETRY_XY_BIAS_ALIGN_SEC",
                            "1.0",
                        )
                    ),
                    dt=dt,
                    z_offset=z_offset,
                    lateral_offset_base=retry_lateral_offset,
                )
                self._trace_event(
                    "recovery_retry_xy_bias_align_completed",
                    retry_count=retry_count,
                    lateral_offset_base_m=retry_lateral_offset.tolist(),
                )
            micro_align_sec = float(os.environ.get("AIC_OFFICIAL_TEACHER_PREINSERT_MICRO_ALIGN_SEC", "0.0"))
            port_align_sec = float(os.environ.get("AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_SEC", "0.0"))
            micro_align_ok = self._run_preinsert_micro_align(
                task,
                port_transform,
                get_observation,
                move_robot,
                duration_sec=micro_align_sec,
                dt=dt,
                z_offset=z_offset,
            )
            if not micro_align_ok:
                send_feedback("official_teacher_replay_preinsert_micro_align_force_abort")
                return False
            port_align_ok = self._run_port_frame_precontact_align(
                task,
                port_transform,
                get_observation,
                move_robot,
                duration_sec=port_align_sec,
                dt=dt,
                z_offset=z_offset,
                target_z_m=original_preinsert_z if retry_count > 0 else None,
            )
            if not port_align_ok:
                send_feedback("official_teacher_replay_precontact_port_align_force_abort")
                return False
            active_lateral_offset = retry_lateral_offset + self._last_precontact_lateral_offset_base
            skip_retry_preinsert_gate = os.environ.get(
                "AIC_OFFICIAL_TEACHER_SKIP_PREINSERT_GATE_ON_RECOVERY_RETRY",
                "true",
            ).lower() in {"1", "true", "yes", "on"}
            should_check_preinsert_gate = (
                micro_align_sec > 0.0
                or port_align_sec > 0.0
                or (retry_count > 0 and not skip_retry_preinsert_gate)
            )
            if should_check_preinsert_gate:
                retry_gate_target_z = original_preinsert_z if retry_count > 0 and cheatcode_z_mode == "tf_depth" else None
                gate_passed = self._hold_preinsert_until_tracking_gate(
                    task,
                    port_transform,
                    get_observation,
                    move_robot,
                    settle_sec=float(os.environ.get("AIC_OFFICIAL_TEACHER_MICRO_ALIGN_SETTLE_SEC", "0.25")),
                    dt=dt,
                    z_offset=z_offset,
                    preserve_current_z=True,
                    target_z_m=retry_gate_target_z,
                    lateral_offset_base=active_lateral_offset,
                )
                self._trace_event(
                    "preinsert_micro_align_gate_checked",
                    retry_count=retry_count,
                    tracking_gate_passed=gate_passed,
                )
                if not gate_passed:
                    send_feedback("official_teacher_replay_tracking_gate_failed")
                    return False
            else:
                self._trace_event(
                    "preinsert_micro_align_gate_skipped",
                    retry_count=retry_count,
                    reason=(
                        "recovery_retry_gate_skipped"
                        if retry_count > 0 and skip_retry_preinsert_gate
                        else "no_precontact_alignment_stage"
                    ),
                )
            baseline_force = self._force_vector(get_observation)
            if cheatcode_z_mode == "cheatcode_offsets":
                live_z_offset = self._current_cheatcode_z_offset(task, port_transform)
                max_live_start_z_offset = float(
                    os.environ.get(
                        "AIC_OFFICIAL_TEACHER_MAX_LIVE_INSERTION_START_Z_OFFSET",
                        "0.08",
                    )
                )
                skip_handoff = os.environ.get(
                    "AIC_OFFICIAL_TEACHER_SKIP_CHEATCODE_HANDOFF_WHEN_LIVE_OFFSET",
                    "false",
                ).lower() in {"1", "true", "yes", "on"}
                if skip_handoff and z_offset < live_z_offset <= max_live_start_z_offset:
                    guarded_start_z_offset = live_z_offset
                    guarded_insertion_distance = max(0.0, live_z_offset - fixed_end_z_offset)
                    gate_passed = True
                    self._trace_event(
                        "cheatcode_handoff_skipped_live_z_offset",
                        retry_count=retry_count,
                        live_z_offset=live_z_offset,
                        nominal_start_z_offset=z_offset,
                        guarded_insertion_distance_m=guarded_insertion_distance,
                        max_live_start_z_offset=max_live_start_z_offset,
                    )
                else:
                    self._run_cheatcode_handoff_to_insert_start(
                        task,
                        port_transform,
                        move_robot,
                        from_z_offset=handoff_start_z_offset,
                        to_z_offset=z_offset,
                        dt=dt,
                        lateral_offset_base=active_lateral_offset,
                    )
                    gate_passed = self._hold_preinsert_until_tracking_gate(
                        task,
                        port_transform,
                        get_observation,
                        move_robot,
                        settle_sec=settle_sec,
                        dt=dt,
                        z_offset=z_offset,
                        preserve_current_z=False,
                        lateral_offset_base=active_lateral_offset,
                    )
                    guarded_start_z_offset = z_offset
                    guarded_insertion_distance = planned_insertion_distance
                    self._trace_event(
                        "cheatcode_handoff_gate_checked",
                        retry_count=retry_count,
                        tracking_gate_passed=gate_passed,
                    )
                if not gate_passed:
                    last_gate_force_delta = (
                        float(self._last_tracking_gate_force_delta_n)
                        if self._last_tracking_gate_force_delta_n is not None
                        else 0.0
                    )
                    if last_gate_force_delta >= self._ft_threshold_n:
                        self._trace_event(
                            "contact_detected",
                            retry_count=retry_count,
                            force_delta_n=last_gate_force_delta,
                            threshold_n=self._ft_threshold_n,
                            z_offset=live_z_offset,
                            sample_source="handoff_gate",
                        )
                        if self._expert_mode == "nominal":
                            send_feedback("official_teacher_replay_nominal_contact_before_insert")
                            return False
                        if not self._backoff_and_realign(
                            task,
                            port_transform,
                            get_observation,
                            move_robot,
                            current_z_offset=live_z_offset,
                            dt=dt,
                            release_baseline_force=baseline_force,
                            retry_start_z_m=original_preinsert_z,
                        ):
                            self._trace_event("recovery_force_release_failed", retry_count=retry_count)
                            return False
                        retry_count += 1
                        continue
                    if z_offset < live_z_offset <= max_live_start_z_offset:
                        guarded_start_z_offset = live_z_offset
                        guarded_insertion_distance = max(0.0, live_z_offset - fixed_end_z_offset)
                        preserve_live_lateral = os.environ.get(
                            "AIC_OFFICIAL_TEACHER_PRESERVE_LIVE_LATERAL_ON_Z_REPAIR",
                            "true",
                        ).lower() in {"1", "true", "yes", "on"}
                        if preserve_live_lateral:
                            try:
                                nominal_live_pose = self._calc_cheatcode_gripper_pose(
                                    task,
                                    port_transform,
                                    z_offset=live_z_offset,
                                    lateral_offset_base=active_lateral_offset,
                                )
                                current_xyz = self._pose_position_array(self._current_tcp_pose())
                                nominal_xyz = self._pose_position_array(nominal_live_pose)
                                live_lateral_offset = np.array(
                                    [
                                        current_xyz[0] - nominal_xyz[0],
                                        current_xyz[1] - nominal_xyz[1],
                                        0.0,
                                    ],
                                    dtype=np.float64,
                                )
                                max_live_lateral_offset = float(
                                    os.environ.get(
                                        "AIC_OFFICIAL_TEACHER_MAX_LIVE_LATERAL_REPAIR_M",
                                        "0.02",
                                    )
                                )
                                live_lateral_norm = float(np.linalg.norm(live_lateral_offset[:2]))
                                if 0.0 < live_lateral_norm <= max_live_lateral_offset:
                                    active_lateral_offset = active_lateral_offset + live_lateral_offset
                            except TransformException as ex:
                                self.get_logger().warn(f"Live lateral repair TF lookup failed: {ex}")
                        self._trace_event(
                            "cheatcode_handoff_gate_repaired_with_live_z_offset",
                            retry_count=retry_count,
                            live_z_offset=live_z_offset,
                            nominal_start_z_offset=z_offset,
                            guarded_insertion_distance_m=guarded_insertion_distance,
                            max_live_start_z_offset=max_live_start_z_offset,
                            lateral_offset_base_m=active_lateral_offset.tolist(),
                        )
                        gate_passed = True
                    else:
                        send_feedback("official_teacher_replay_tracking_gate_failed")
                        return False
            else:
                guarded_start_z_offset = z_offset
                guarded_insertion_distance = planned_insertion_distance
            insertion_duration = max(dt, guarded_insertion_distance / insertion_speed_mps)
            insertion_steps = max(1, int(math.ceil(insertion_duration / dt)))
            if insertion_command_mode == "exact_position":
                insertion_steps = max(
                    insertion_steps,
                    int(math.ceil(guarded_insertion_distance / max(exact_position_step_m, 1e-6))),
                )
                insertion_duration = insertion_steps * dt
            insertion_start_pose = self._current_tcp_pose()
            insertion_start_z = float(insertion_start_pose.position.z)
            pin_insertion_target = os.environ.get(
                "AIC_OFFICIAL_TEACHER_PIN_INSERTION_TARGET",
                "false",
            ).lower() in {"1", "true", "yes", "on"}
            try:
                insertion_reference_pose = self._calc_cheatcode_gripper_pose(
                    task,
                    port_transform,
                    z_offset=guarded_start_z_offset,
                    reset_xy_integrator=True,
                    preserve_current_z=(cheatcode_z_mode == "tf_depth"),
                    lateral_offset_base=active_lateral_offset,
                )
                if cheatcode_z_mode == "tf_depth":
                    insertion_reference_pose.position.z = insertion_start_z
            except TransformException as ex:
                self.get_logger().warn(f"Unable to pin guarded insertion start pose: {ex}")
                insertion_reference_pose = insertion_start_pose
            current_z = guarded_start_z_offset
            retry_requested = False
            self._trace_event(
                "guarded_insert_started",
                retry_count=retry_count,
                insertion_speed_mps=insertion_speed_mps,
                ft_threshold_n=self._ft_threshold_n,
                descent_profile="minimum_jerk",
                cheatcode_z_mode=cheatcode_z_mode,
                insertion_command_mode=insertion_command_mode,
                insertion_start_z_offset=guarded_start_z_offset,
                insertion_depth_m=guarded_insertion_distance,
                fallback_insertion_depth_m=fallback_distance,
                insertion_duration_sec=insertion_duration,
                insertion_steps=insertion_steps,
                exact_position_step_m=exact_position_step_m if insertion_command_mode == "exact_position" else None,
                exact_position_settle_sec=(
                    exact_position_settle_sec if insertion_command_mode == "exact_position" else None
                ),
                insertion_max_translation_step_m=insertion_max_step_m,
                min_success_z_offset=min_success_z_offset,
                pinned_insertion_reference_pose=self._pose_to_trace_dict(insertion_reference_pose),
                target_policy=(
                    "pinned_xy_orientation_depth_only"
                    if pin_insertion_target
                    else "dynamic_cheatcode_exact_position"
                ),
            )
            def recover_after_contact() -> bool:
                nonlocal retry_count, retry_requested
                if self._expert_mode == "nominal":
                    send_feedback("official_teacher_replay_nominal_contact_rejected")
                    return False
                send_feedback("official_teacher_replay_recovery_backoff")
                if not self._backoff_and_realign(
                    task,
                    port_transform,
                    get_observation,
                    move_robot,
                    current_z_offset=current_z,
                    dt=dt,
                    release_baseline_force=baseline_force,
                    retry_start_z_m=original_preinsert_z,
                ):
                    send_feedback("official_teacher_replay_recovery_force_release_failed")
                    self._trace_event("recovery_force_release_failed", retry_count=retry_count)
                    return False
                skip_retry_gate = os.environ.get(
                    "AIC_OFFICIAL_TEACHER_SKIP_RETRY_GATE_AFTER_RECOVERY",
                    "true",
                ).lower() in {"1", "true", "yes", "on"}
                if skip_retry_gate:
                    gate_passed = True
                    self._trace_event(
                        "recovery_retry_tracking_gate_skipped",
                        retry_count=retry_count,
                        reason="backoff_realign_already_gated",
                    )
                else:
                    gate_passed = self._hold_preinsert_until_tracking_gate(
                        task,
                        port_transform,
                        get_observation,
                        move_robot,
                        settle_sec=settle_sec,
                        dt=dt,
                        z_offset=z_offset,
                        target_z_m=original_preinsert_z,
                    )
                    self._trace_event(
                        "recovery_retry_tracking_gate_checked",
                        retry_count=retry_count,
                        tracking_gate_passed=gate_passed,
                    )
                if not gate_passed:
                    send_feedback("official_teacher_replay_tracking_gate_failed")
                    return False
                retry_count += 1
                if retry_count > max_retries:
                    send_feedback("official_teacher_replay_recovery_max_retries_exhausted")
                    self._trace_event("recovery_max_retries_exhausted", max_retries=max_retries)
                    return False
                send_feedback("official_teacher_replay_retry_insert")
                retry_requested = True
                return True

            def accept_insert_success() -> bool | None:
                if not self._task_completed_in_simulation(task):
                    return None
                if current_z > min_success_z_offset:
                    self._trace_event(
                        "guarded_insert_success_ignored_before_depth",
                        retry_count=retry_count,
                        z_offset=current_z,
                        min_success_z_offset=min_success_z_offset,
                    )
                    return None
                self._trace_event("guarded_insert_success_candidate", retry_count=retry_count)
                force_ok, max_force_delta = self._post_insert_force_check(
                    get_observation,
                    baseline_force,
                    dt=dt,
                    retry_count=retry_count,
                )
                if force_ok:
                    self.get_logger().info("Online CheatCode insertion observed insertion event.")
                    self._trace_event(
                        "guarded_insert_success",
                        retry_count=retry_count,
                        post_insert_max_force_delta_n=max_force_delta,
                    )
                    return True
                return recover_after_contact()

            speed_gate_mps = float(
                os.environ.get("AIC_OFFICIAL_TEACHER_GUARDED_INSERT_SPEED_GATE_MPS", "0.012")
            )
            speed_gate_max_hold_count = int(
                max(
                    1,
                    math.ceil(
                        float(
                            os.environ.get(
                                "AIC_OFFICIAL_TEACHER_GUARDED_INSERT_SPEED_GATE_MAX_HOLD_SEC",
                                "1.20",
                            )
                        )
                        / dt
                    ),
                )
            )
            speed_gate_hold_count = 0
            last_target_pose = None
            descent_step = 1
            while descent_step <= insertion_steps:
                actual_tcp_speed = (
                    self._tcp_speed_norm(get_observation)
                    if insertion_command_mode == "exact_position"
                    else None
                )
                if (
                    insertion_command_mode == "exact_position"
                    and actual_tcp_speed is not None
                    and actual_tcp_speed > speed_gate_mps
                    and speed_gate_hold_count < speed_gate_max_hold_count
                ):
                    if last_target_pose is not None:
                        self._send_absolute_target(move_robot, last_target_pose)
                    self._trace_event(
                        "guarded_insert_speed_gate_hold",
                        retry_count=retry_count,
                        guarded_insert_speed_gate_checked=True,
                        guarded_insert_speed_gate_held=True,
                        held_depth_step_count=speed_gate_hold_count + 1,
                        actual_tcp_speed_mps=actual_tcp_speed,
                        speed_threshold_mps=speed_gate_mps,
                        target_z_offset=current_z,
                    )
                    speed_gate_hold_count += 1
                    force_delta = self._force_delta_norm(get_observation, baseline_force)
                    if force_delta >= self._ft_threshold_n:
                        confirmed, confirmed_delta = self._force_trigger_confirmed(
                            get_observation,
                            baseline_force,
                            dt=dt,
                        )
                        if confirmed:
                            self._trace_event(
                                "contact_detected",
                                retry_count=retry_count,
                                force_delta_n=max(force_delta, confirmed_delta),
                                threshold_n=self._ft_threshold_n,
                                z_offset=current_z,
                                sample_source="during_speed_gate_hold",
                            )
                            if not recover_after_contact():
                                return False
                            break
                    self.sleep_for(dt)
                    continue
                speed_gate_hold_count = 0
                fraction = self._minimum_jerk_fraction(descent_step / insertion_steps)
                current_z = guarded_start_z_offset - fraction * guarded_insertion_distance
                contact_confirmed = False
                try:
                    if pin_insertion_target:
                        target_xyz = self._pose_position_array(insertion_reference_pose)
                        target_xyz[2] = float(insertion_reference_pose.position.z) - fraction * guarded_insertion_distance
                        target_pose = self._make_pose(
                            target_xyz,
                            self._pose_quat_wxyz(insertion_reference_pose),
                        )
                    else:
                        target_pose = self._calc_cheatcode_gripper_pose(
                            task,
                            port_transform,
                            z_offset=current_z,
                            preserve_current_z=(cheatcode_z_mode == "tf_depth"),
                            lateral_offset_base=active_lateral_offset,
                        )
                    if cheatcode_z_mode == "tf_depth":
                        target_pose.position.z = insertion_start_z - fraction * guarded_insertion_distance
                    if insertion_command_mode == "exact_position":
                        self._send_absolute_target(move_robot, target_pose)
                        last_target_pose = target_pose
                        self._trace_event(
                            "guarded_insert_speed_gate_advance",
                            retry_count=retry_count,
                            guarded_insert_speed_gate_checked=actual_tcp_speed is not None,
                            guarded_insert_speed_gate_held=False,
                            actual_tcp_speed_mps=actual_tcp_speed,
                            speed_threshold_mps=speed_gate_mps,
                            target_z_offset=current_z,
                        )
                    else:
                        self._send_relative_target(
                            move_robot,
                            target_pose,
                            max_translation_step_m=insertion_max_step_m,
                        )
                        contact_confirmed = False
                except TransformException as ex:
                    self.get_logger().warn(f"Online CheatCode TF lookup failed: {ex}")
                    continue

                for sample_source in ("after_command", "after_settle"):
                    if sample_source == "after_settle":
                        self.sleep_for(dt)
                    force_delta = self._force_delta_norm(get_observation, baseline_force)
                    if force_delta < self._ft_threshold_n:
                        continue
                    confirmed, confirmed_delta = self._force_trigger_confirmed(
                        get_observation,
                        baseline_force,
                        dt=dt,
                    )
                    if confirmed:
                        contact_confirmed = True
                        break
                    self._trace_event(
                        "contact_trigger_ignored_single_sample",
                        retry_count=retry_count,
                        force_delta_n=force_delta,
                        confirmed_force_delta_n=confirmed_delta,
                        threshold_n=self._ft_threshold_n,
                        z_offset=current_z,
                        sample_source=sample_source,
                    )
                if contact_confirmed:
                    self._trace_event(
                        "contact_detected",
                        retry_count=retry_count,
                        force_delta_n=max(force_delta, confirmed_delta),
                        threshold_n=self._ft_threshold_n,
                        z_offset=current_z,
                        sample_source=sample_source,
                    )
                    if not recover_after_contact():
                        return False
                    break
                success_result = accept_insert_success()
                if success_result is True:
                    return True
                if success_result is False:
                    return False
                descent_step += 1
                if retry_requested:
                    break
            if retry_requested:
                continue
            break
        wait_started = self.time_now()
        wait_timeout = Duration(
            seconds=float(os.environ.get("AIC_OFFICIAL_TEACHER_FINAL_HOLD_SEC", "5.0"))
        )
        while (self.time_now() - wait_started) < wait_timeout:
            if self._task_completed_in_simulation(task):
                self.get_logger().info("Online CheatCode insertion event observed during hold.")
                return True
            self.sleep_for(dt)
        self._trace_event("guarded_insert_failed_no_insertion_event")
        return False

    @staticmethod
    def target_to_delta_pose(target: ReplayTarget, current_tcp_pose: Pose) -> Pose:
        """Convert an absolute replay target to a gripper-frame delta pose."""
        return compute_delta_pose(current_tcp_pose, OfficialTeacherReplay.target_to_pose(target))

    def _current_tcp_pose(self) -> Pose:
        transform = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            "gripper/tcp",
            Time(),
        ).transform
        return Pose(
            position=Point(
                x=transform.translation.x,
                y=transform.translation.y,
                z=transform.translation.z,
            ),
            orientation=Quaternion(
                x=transform.rotation.x,
                y=transform.rotation.y,
                z=transform.rotation.z,
                w=transform.rotation.w,
            ),
        )

    @staticmethod
    def target_to_pose(target: ReplayTarget) -> Pose:
        """Convert replay target to the current official Cartesian action shape."""
        return Pose(
            position=Point(
                x=float(target.tcp_pose.position[0]),
                y=float(target.tcp_pose.position[1]),
                z=float(target.tcp_pose.position[2]),
            ),
            orientation=Quaternion(
                x=float(target.tcp_pose.orientation_xyzw[0]),
                y=float(target.tcp_pose.orientation_xyzw[1]),
                z=float(target.tcp_pose.orientation_xyzw[2]),
                w=float(target.tcp_pose.orientation_xyzw[3]),
            ),
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(f"OfficialTeacherReplay.insert_cable() task: {task}")
        send_feedback("official_teacher_replay_started")
        self._latest_insertion_event_namespace = ""

        start_time = self.time_now()
        command_dt_sec = float(
            self._replay.trajectory.metadata.recording.get("command_dt_sec", 0.05)
        )
        while True:
            elapsed = (self.time_now() - start_time).nanoseconds * 1e-9
            target = self._replay.sample(elapsed)
            pose = self.target_to_pose(target)
            phase = getattr(target.waypoint.phase, "value", str(target.waypoint.phase))
            command_source = str(target.waypoint.diagnostics.get("command_source", ""))
            if (
                self._online_cheatcode_final_insertion
                and phase == "local_preinsert_align"
                and not self._local_preinsert_align_done
            ):
                cheatcode_z_mode = os.environ.get(
                    "AIC_OFFICIAL_TEACHER_CHEATCODE_Z_MODE",
                    "cheatcode_offsets",
                )
                port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
                if self._wait_for_tf("base_link", port_frame) and self._wait_for_tf("base_link", "gripper/tcp"):
                    port_transform = self._parent_node._tf_buffer.lookup_transform(
                        "base_link",
                        port_frame,
                        Time(),
                    ).transform
                    self._run_local_preinsert_align(
                        task,
                        port_transform,
                        move_robot,
                        duration_sec=float(
                            target.waypoint.diagnostics.get(
                                "local_preinsert_align_sec",
                                os.environ.get("AIC_OFFICIAL_TEACHER_LOCAL_PREINSERT_ALIGN_SEC", "2.25"),
                            )
                        ),
                        dt=command_dt_sec,
                        z_offset=float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_START_Z_OFFSET", "0.03")),
                        preserve_current_z=False,
                    )
                    self._local_preinsert_align_done = True
                self.sleep_for(command_dt_sec)
                continue
            if (
                self._online_cheatcode_final_insertion
                and phase == "pre_insertion"
                and command_source == "blended_handoff"
            ):
                cheatcode_z_mode = os.environ.get(
                    "AIC_OFFICIAL_TEACHER_CHEATCODE_Z_MODE",
                    "cheatcode_offsets",
                )
                port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
                if self._wait_for_tf("base_link", port_frame):
                    port_transform = self._parent_node._tf_buffer.lookup_transform(
                        "base_link",
                        port_frame,
                        Time(),
                    ).transform
                    try:
                        self._send_absolute_target(
                            move_robot,
                            self._calc_cheatcode_gripper_pose(
                                task,
                                port_transform,
                                slerp_fraction=1.0,
                                position_fraction=1.0,
                                z_offset=float(
                                    os.environ.get(
                                        "AIC_OFFICIAL_TEACHER_CHEATCODE_START_Z_OFFSET", "0.03"
                                    )
                                ),
                                reset_xy_integrator=True,
                                preserve_current_z=False,
                            ),
                        )
                    except TransformException as ex:
                        self.get_logger().warn(f"Blended handoff TF lookup failed: {ex}")
                self.sleep_for(command_dt_sec)
                continue
            if (
                self._online_cheatcode_final_insertion
                and phase == "hold"
                and command_source == "pre_insert_settle"
                and not self._preinsert_settle_done
            ):
                cheatcode_z_mode = os.environ.get(
                    "AIC_OFFICIAL_TEACHER_CHEATCODE_Z_MODE",
                    "cheatcode_offsets",
                )
                port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
                if self._wait_for_tf("base_link", port_frame):
                    port_transform = self._parent_node._tf_buffer.lookup_transform(
                        "base_link",
                        port_frame,
                        Time(),
                    ).transform
                    preinsert_align_preserve_current_z = os.environ.get(
                        "AIC_OFFICIAL_TEACHER_PREINSERT_ALIGN_PRESERVE_CURRENT_Z",
                        "true",
                    ).lower() in {"1", "true", "yes", "on"}
                    preinsert_gate_preserve_current_z = os.environ.get(
                        "AIC_OFFICIAL_TEACHER_PREINSERT_GATE_PRESERVE_CURRENT_Z",
                        "true",
                    ).lower() in {"1", "true", "yes", "on"}
                    gate_passed = self._hold_preinsert_until_tracking_gate(
                        task,
                        port_transform,
                        get_observation,
                        move_robot,
                        settle_sec=float(
                            target.waypoint.diagnostics.get(
                                "pre_insert_settle_sec",
                                os.environ.get("AIC_OFFICIAL_TEACHER_PRE_INSERT_SETTLE_SEC", "1.0"),
                            )
                        ),
                        dt=command_dt_sec,
                        z_offset=float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_START_Z_OFFSET", "0.03")),
                        preserve_current_z=preinsert_gate_preserve_current_z,
                    )
                    self._preinsert_settle_done = True
                    if not gate_passed and self._expert_mode == "nominal":
                        send_feedback("official_teacher_replay_tracking_gate_failed")
                        self._trace_event("nominal_rejected_tracking_gate")
                        return False
                    if not gate_passed:
                        send_feedback("official_teacher_replay_tracking_gate_realign")
                        self._trace_event("recovery_tracking_gate_realign_started")
                        self._run_local_preinsert_align(
                            task,
                            port_transform,
                            move_robot,
                            duration_sec=float(
                                os.environ.get("AIC_OFFICIAL_TEACHER_LOCAL_PREINSERT_ALIGN_SEC", "2.25")
                            ),
                            dt=command_dt_sec,
                            z_offset=float(
                                os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_START_Z_OFFSET", "0.03")
                            ),
                            preserve_current_z=preinsert_align_preserve_current_z,
                        )
                        gate_passed = self._hold_preinsert_until_tracking_gate(
                            task,
                            port_transform,
                            get_observation,
                            move_robot,
                            settle_sec=float(
                                target.waypoint.diagnostics.get(
                                    "pre_insert_settle_sec",
                                    os.environ.get("AIC_OFFICIAL_TEACHER_PRE_INSERT_SETTLE_SEC", "1.0"),
                                )
                            ),
                            dt=command_dt_sec,
                            z_offset=float(
                                os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_START_Z_OFFSET", "0.03")
                            ),
                            preserve_current_z=preinsert_gate_preserve_current_z,
                        )
                        self._trace_event(
                            "recovery_tracking_gate_realign_completed",
                            tracking_gate_passed=gate_passed,
                        )
                        if not gate_passed:
                            send_feedback("official_teacher_replay_tracking_gate_failed")
                            return False
                self.sleep_for(command_dt_sec)
                continue
            if self._online_cheatcode_final_insertion and phase == "final_insertion":
                return self._run_online_cheatcode_insertion(
                    task,
                    get_observation,
                    move_robot,
                    send_feedback,
                )
            if self._action_mode == "relative_delta_gripper_tcp":
                try:
                    delta_pose = self.target_to_delta_pose(target, self._current_tcp_pose())
                    self.set_delta_pose_target(
                        move_robot=move_robot,
                        delta_pose=delta_pose,
                        frame_id="gripper/tcp",
                    )
                except TransformException as ex:
                    self.get_logger().warn(
                        "TF lookup failed for relative replay; falling back to "
                        f"absolute base_link pose for this tick: {ex}"
                    )
                    self.set_pose_target(move_robot=move_robot, pose=pose, frame_id="base_link")
            else:
                if self._action_mode == "joint_position_then_cheatcode":
                    self._send_joint_target(move_robot, target)
                else:
                    self.set_pose_target(move_robot=move_robot, pose=pose, frame_id="base_link")

            if self._task_completed_in_simulation(task):
                send_feedback("official_teacher_replay_insertion_event")
                self.get_logger().info("OfficialTeacherReplay observed insertion event.")
                return True
            if self._replay.is_finished(elapsed):
                hold_timeout = Duration(
                    seconds=float(
                        os.environ.get("AIC_OFFICIAL_TEACHER_FINAL_HOLD_SEC", "5.0")
                    )
                )
                hold_started = self.time_now()
                while (self.time_now() - hold_started) < hold_timeout:
                    if self._task_completed_in_simulation(task):
                        send_feedback("official_teacher_replay_insertion_event")
                        self.get_logger().info(
                            "OfficialTeacherReplay observed insertion event during final hold."
                        )
                        return True
                    if self._action_mode == "relative_delta_gripper_tcp":
                        try:
                            self.set_delta_pose_target(
                                move_robot=move_robot,
                                delta_pose=self.target_to_delta_pose(
                                    target,
                                    self._current_tcp_pose(),
                                ),
                                frame_id="gripper/tcp",
                            )
                        except TransformException:
                            self.set_pose_target(
                                move_robot=move_robot,
                                pose=pose,
                                frame_id="base_link",
                            )
                    else:
                        if self._action_mode == "joint_position_then_cheatcode":
                            self._send_joint_target(move_robot, target)
                        else:
                            self.set_pose_target(move_robot=move_robot, pose=pose, frame_id="base_link")
                    self.sleep_for(command_dt_sec)
                send_feedback("official_teacher_replay_finished")
                return True
            self.sleep_for(command_dt_sec)
