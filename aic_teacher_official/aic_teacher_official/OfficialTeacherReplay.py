"""Official AIC policy wrapper for replaying a smooth teacher trajectory.

This module intentionally does not import any VLM/planner backends. Replay must
be deterministic and local to the official ROS/Gazebo execution.
"""

from __future__ import annotations

import os
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
from aic_control_interfaces.msg import JointMotionUpdate, MotionUpdate, TrajectoryGenerationMode
from aic_model_interfaces.msg import Observation
from aic_model.policy import (
    DEFAULT_CARTESIAN_DAMPING,
    DEFAULT_CARTESIAN_STIFFNESS,
    compute_delta_pose,
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    quaternion_xyzw_to_rotation_matrix,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform, Wrench, WrenchStamped
from geometry_msgs.msg import Vector3
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import Header, String
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
        self._post_insert_force_threshold_n = float(
            os.environ.get(
                "AIC_OFFICIAL_TEACHER_POST_INSERT_FORCE_THRESHOLD_N",
                str(max(6.0, self._ft_threshold_n)),
            )
        )
        self._tracking_gate_threshold_m = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_TRACKING_GATE_THRESHOLD_M", "0.02")
        )
        self._tracking_gate_timeout_sec = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_TRACKING_GATE_TIMEOUT_SEC", "1.0")
        )
        self._local_preinsert_align_done = False
        self._preinsert_settle_done = False
        self._last_tracking_gate_force_delta_n = None
        self._last_tracking_gate_baseline_force = None
        self._last_tracking_gate_error_m = None
        self._last_tracking_gate_lateral_error_m = None
        self._last_tracking_gate_axial_error_m = None
        self._last_tracking_gate_speed_mps = None
        self._last_precontact_lateral_offset_base = np.zeros(3, dtype=np.float64)
        self._last_precontact_align_baseline_force = None
        self._last_precontact_align_force_delta_n = None
        self._active_task = None
        self._active_port_transform = None
        self._last_commanded_target_pose = None
        self._force_delta_history = deque(maxlen=200)
        self._recovery_context_history = deque(maxlen=240)
        self._last_recovery_context_sample_time_sec = -1e9
        self._force_trend_window_samples = int(
            os.environ.get("AIC_OFFICIAL_TEACHER_FORCE_TREND_WINDOW_SAMPLES", "5")
        )
        self._force_trend_center_gap_samples = int(
            os.environ.get("AIC_OFFICIAL_TEACHER_FORCE_TREND_CENTER_GAP_SAMPLES", "5")
        )
        self._recovery_context_sample_period_sec = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_CONTEXT_SAMPLE_PERIOD_SEC", "0.25")
        )
        self._cartesian_stiffness = self._env_vector(
            "AIC_OFFICIAL_TEACHER_CARTESIAN_STIFFNESS",
            DEFAULT_CARTESIAN_STIFFNESS,
            expected_len=6,
        )
        self._cartesian_damping = self._env_vector(
            "AIC_OFFICIAL_TEACHER_CARTESIAN_DAMPING",
            DEFAULT_CARTESIAN_DAMPING,
            expected_len=6,
        )
        self._recovery_cartesian_stiffness = self._env_vector(
            "AIC_OFFICIAL_TEACHER_RECOVERY_CARTESIAN_STIFFNESS",
            self._cartesian_stiffness,
            expected_len=6,
        )
        self._recovery_cartesian_damping = self._env_vector(
            "AIC_OFFICIAL_TEACHER_RECOVERY_CARTESIAN_DAMPING",
            self._cartesian_damping,
            expected_len=6,
        )
        self._cartesian_wrench_feedback_gains_default = self._env_vector(
            "AIC_OFFICIAL_TEACHER_CARTESIAN_WRENCH_FEEDBACK_GAINS",
            [0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            expected_len=6,
        )
        self._recovery_cartesian_wrench_feedback_gains_default = self._env_vector(
            "AIC_OFFICIAL_TEACHER_RECOVERY_CARTESIAN_WRENCH_FEEDBACK_GAINS",
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            expected_len=6,
        )
        self._joint_stiffness_default = self._env_vector(
            "AIC_OFFICIAL_TEACHER_JOINT_STIFFNESS",
            [100.0],
            expected_len=None,
        )
        self._joint_damping_default = self._env_vector(
            "AIC_OFFICIAL_TEACHER_JOINT_DAMPING",
            [20.0],
            expected_len=None,
        )
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

    @staticmethod
    def _env_vector(name: str, default: list[float], *, expected_len: int | None) -> list[float]:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return [float(v) for v in default]
        try:
            values = [float(token) for token in raw.replace(",", " ").split()]
        except ValueError as ex:
            raise RuntimeError(f"{name} must contain numeric values") from ex
        if expected_len is not None and len(values) == 1:
            values = values * expected_len
        if expected_len is not None and len(values) != expected_len:
            raise RuntimeError(f"{name} must contain either 1 or {expected_len} values")
        if not values:
            raise RuntimeError(f"{name} must not be empty")
        return values

    @staticmethod
    def _expand_gain_vector(values: list[float], length: int, name: str) -> list[float]:
        if len(values) == 1:
            return [float(values[0])] * length
        if len(values) != length:
            raise RuntimeError(f"{name} must contain either 1 or {length} values")
        return [float(v) for v in values]

    def _cartesian_gains(self, profile: str = "default") -> tuple[list[float], list[float]]:
        if profile == "recovery":
            return self._recovery_cartesian_stiffness, self._recovery_cartesian_damping
        return self._cartesian_stiffness, self._cartesian_damping

    def _cartesian_wrench_feedback_gains(self, profile: str = "default") -> list[float]:
        if profile == "recovery":
            return list(self._recovery_cartesian_wrench_feedback_gains_default)
        return list(self._cartesian_wrench_feedback_gains_default)

    def _trace_event(self, event: str, **payload) -> None:
        if self._runtime_trace_path is None:
            return
        try:
            if event in {"contact_detected", "recovery_force_release_failed", "recovery_max_retries_exhausted"}:
                payload = {
                    **payload,
                    "force_trend": self._force_trend_snapshot(),
                    "recent_recovery_context": list(self._recovery_context_history)[
                        -int(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_CONTEXT_MAX_TRACE_SAMPLES", "80")) :
                    ],
                }
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

    @staticmethod
    def _vector_to_trace_list(values: np.ndarray | list[float] | tuple[float, ...] | None) -> list[float] | None:
        if values is None:
            return None
        return [float(v) for v in np.asarray(values, dtype=np.float64).tolist()]

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
        use_plug_xy_offset_setting = os.environ.get("AIC_OFFICIAL_TEACHER_USE_PLUG_XY_OFFSET", "auto").lower()
        use_plug_xy_offset = (
            use_plug_xy_offset_setting in {"1", "true", "yes", "on"}
            or (
                use_plug_xy_offset_setting == "auto"
                and str(getattr(task, "plug_name", "")).startswith("sc")
            )
        )
        target_xy_offset = plug_tip_gripper_offset[:2] if use_plug_xy_offset else np.zeros(2, dtype=np.float64)
        target_xyz = np.array(
            [
                port_transform.translation.x + target_xy_offset[0] + i_gain * self._tip_x_error_integrator,
                port_transform.translation.y + target_xy_offset[1] + i_gain * self._tip_y_error_integrator,
                port_transform.translation.z + z_offset + plug_tip_gripper_offset[2],
            ],
            dtype=np.float64,
        )
        vertical_bias_m = float(os.environ.get("AIC_OFFICIAL_TEACHER_PREINSERT_VERTICAL_BIAS_M", "0.0"))
        if abs(vertical_bias_m) > 0.0:
            target_xyz[2] += vertical_bias_m
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

    def _calc_near_gate_gripper_pose(
        self,
        task: Task,
        *,
        axial_offset_m: float,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        lateral_offset_base: np.ndarray | None = None,
    ) -> Pose:
        entrance_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            f"task_board/{task.target_module_name}/{task.port_name}_link_entrance",
            Time(),
        )
        entrance_transform = entrance_tf_stamped.transform
        q_port = (
            entrance_transform.rotation.w,
            entrance_transform.rotation.x,
            entrance_transform.rotation.y,
            entrance_transform.rotation.z,
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
        port_rotation = quaternion_xyzw_to_rotation_matrix(
            np.asarray(
                [
                    float(entrance_transform.rotation.x),
                    float(entrance_transform.rotation.y),
                    float(entrance_transform.rotation.z),
                    float(entrance_transform.rotation.w),
                ],
                dtype=np.float64,
            )
        )
        port_axis_base = port_rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        entrance_xyz = np.asarray(
            [
                float(entrance_transform.translation.x),
                float(entrance_transform.translation.y),
                float(entrance_transform.translation.z),
            ],
            dtype=np.float64,
        )
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
        target_xyz = entrance_xyz + port_axis_base * float(axial_offset_m) + (gripper_xyz - plug_xyz)
        if lateral_offset_base is not None:
            lateral_offset = np.asarray(lateral_offset_base, dtype=np.float64)
            target_xyz[:2] += lateral_offset[:2]
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
        gain_profile: str = "default",
    ) -> None:
        self._last_commanded_target_pose = target_pose
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
        stiffness, damping = self._cartesian_gains(gain_profile)
        self.set_delta_pose_target(
            move_robot=move_robot,
            delta_pose=delta_pose,
            frame_id="gripper/tcp",
            stiffness=stiffness,
            damping=damping,
        )

    def _send_absolute_target(
        self,
        move_robot: MoveRobotCallback,
        target_pose: Pose,
        *,
        gain_profile: str = "default",
    ) -> None:
        self._last_commanded_target_pose = target_pose
        stiffness, damping = self._cartesian_gains(gain_profile)
        wrench_feedback_gains = self._cartesian_wrench_feedback_gains(gain_profile)
        motion_update = MotionUpdate(
            header=Header(frame_id="base_link", stamp=self._parent_node.get_clock().now().to_msg()),
            pose=target_pose,
            target_stiffness=np.diag(stiffness).flatten().tolist(),
            target_damping=np.diag(damping).flatten().tolist(),
            feedforward_wrench_at_tip=Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0),
                torque=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            wrench_feedback_gains_at_tip=wrench_feedback_gains,
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )
        try:
            move_robot(motion_update=motion_update)
        except Exception as ex:
            self.get_logger().info(f"move_robot exception: {ex}")

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

    def _transform_translation_array(self, transform: Transform | None) -> np.ndarray | None:
        if transform is None:
            return None
        return np.asarray(
            [
                float(transform.translation.x),
                float(transform.translation.y),
                float(transform.translation.z),
            ],
            dtype=np.float64,
        )

    def _pose_orientation_rotation_tcp_to_base(self, pose: Pose) -> np.ndarray:
        return quaternion_xyzw_to_rotation_matrix(
            np.asarray(
                [
                    float(pose.orientation.x),
                    float(pose.orientation.y),
                    float(pose.orientation.z),
                    float(pose.orientation.w),
                ],
                dtype=np.float64,
            )
        )

    def _lookup_active_plug_transform(self) -> Transform | None:
        task = self._active_task
        if task is None:
            return None
        try:
            return self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                f"{task.cable_name}/{task.plug_name}_link",
                Time(),
            ).transform
        except TransformException:
            return None

    def _append_force_delta_sample(self, force_delta_n: float) -> None:
        self._force_delta_history.append(
            {
                "time_sec": self.time_now().nanoseconds * 1e-9,
                "force_delta_n": float(force_delta_n),
            }
        )

    def _force_trend_snapshot(self) -> dict:
        window = max(1, int(self._force_trend_window_samples))
        center_gap = max(1, int(self._force_trend_center_gap_samples))
        # Two odd-sized windows whose centers are separated by center_gap samples.
        # With dt=0.05, window=5 and center_gap=5 compares medians centered 250 ms apart.
        previous_end = -center_gap
        previous_start = previous_end - window
        current_values = [float(row["force_delta_n"]) for row in list(self._force_delta_history)[-window:]]
        history = list(self._force_delta_history)
        previous_values = (
            [float(row["force_delta_n"]) for row in history[previous_start:previous_end]]
            if len(history) >= center_gap + window
            else []
        )
        current_median = float(np.median(current_values)) if len(current_values) == window else None
        previous_median = float(np.median(previous_values)) if len(previous_values) == window else None
        rising_delta = (
            float(current_median - previous_median)
            if current_median is not None and previous_median is not None
            else None
        )
        return {
            "schema_version": "aic_force_trend/v1",
            "window_samples": window,
            "center_gap_samples": center_gap,
            "center_gap_sec": center_gap * float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_DT", "0.05")),
            "current_median_force_delta_n": current_median,
            "previous_median_force_delta_n": previous_median,
            "median_delta_n": rising_delta,
            "sample_count": len(history),
        }

    def _force_trigger_decision(
        self,
        force_delta_n: float,
    ) -> tuple[bool, dict]:
        self._append_force_delta_sample(force_delta_n)
        trend = self._force_trend_snapshot()
        median_threshold = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_FORCE_MEDIAN_THRESHOLD_N", str(self._ft_threshold_n))
        )
        rise_threshold = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_FORCE_MEDIAN_RISE_THRESHOLD_N", "0.35")
        )
        sustained_threshold = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_FORCE_SUSTAINED_THRESHOLD_N", str(self._ft_threshold_n))
        )
        current_median = trend.get("current_median_force_delta_n")
        median_delta = trend.get("median_delta_n")
        median_trigger = (
            current_median is not None
            and current_median >= median_threshold
            and (median_delta is None or median_delta >= rise_threshold)
        )
        sustained_trigger = current_median is not None and current_median >= sustained_threshold
        instant_trigger_enabled = os.environ.get(
            "AIC_OFFICIAL_TEACHER_FORCE_SINGLE_SAMPLE_TRIGGER_ENABLED",
            "false",
        ).lower() in {"1", "true", "yes", "on"}
        instant_trigger = bool(
            instant_trigger_enabled
            and force_delta_n >= self._ft_threshold_n
            and current_median is None
        )
        confirmed = bool(median_trigger or sustained_trigger or instant_trigger)
        return confirmed, {
            **trend,
            "instant_force_delta_n": float(force_delta_n),
            "instant_threshold_n": self._ft_threshold_n,
            "median_threshold_n": median_threshold,
            "median_rise_threshold_n": rise_threshold,
            "sustained_threshold_n": sustained_threshold,
            "instant_trigger_enabled": instant_trigger_enabled,
            "median_trigger": bool(median_trigger),
            "sustained_trigger": bool(sustained_trigger),
            "instant_trigger": instant_trigger,
            "confirmed": confirmed,
        }

    def _record_recovery_context_sample(
        self,
        get_observation: GetObservationCallback,
        *,
        force_base_or_tcp: np.ndarray,
        baseline_force: np.ndarray | None,
        phase: str,
        target_pose: Pose | None = None,
        force_delta_n: float | None = None,
        z_offset: float | None = None,
        retry_count: int | None = None,
    ) -> None:
        now_sec = self.time_now().nanoseconds * 1e-9
        if now_sec - self._last_recovery_context_sample_time_sec < self._recovery_context_sample_period_sec - 1e-9:
            return
        self._last_recovery_context_sample_time_sec = now_sec
        tcp_pose = None
        plug_transform = None
        try:
            tcp_pose = self._current_tcp_pose()
            plug_transform = self._lookup_active_plug_transform()
        except TransformException:
            tcp_pose = None
        port_transform = self._active_port_transform
        obs = {}
        try:
            obs = self._observation_dict(get_observation())
        except Exception:
            obs = {}
        if not obs and self._latest_observation is not None:
            obs = self._observation_dict(self._latest_observation)
        force_tcp_assumed = np.asarray(force_base_or_tcp, dtype=np.float64)
        force_base_assumed = None
        vectors = {}
        if tcp_pose is not None:
            tcp_xyz = self._pose_position_array(tcp_pose)
            rot_tcp_to_base = self._pose_orientation_rotation_tcp_to_base(tcp_pose)
            force_base_assumed = rot_tcp_to_base @ force_tcp_assumed
            for name, xyz in {
                "port_minus_tcp": self._transform_translation_array(port_transform),
                "plug_minus_tcp": self._transform_translation_array(plug_transform),
            }.items():
                if xyz is None:
                    continue
                vector_base = xyz - tcp_xyz
                vectors[name] = {
                    "base_link": self._vector_to_trace_list(vector_base),
                    "gripper_tcp": self._vector_to_trace_list(rot_tcp_to_base.T @ vector_base),
                }
        sample = {
            "schema_version": "aic_recovery_context_sample/v1",
            "time_sec": now_sec,
            "phase": phase,
            "retry_count": retry_count,
            "z_offset": z_offset,
            "frames": {
                "world_frame": "base_link",
                "tcp_frame": "gripper/tcp",
                "force_frame_note": (
                    "Raw wrist wrench is treated as TCP-local for tcp values and rotated by current "
                    "TCP orientation to estimate base_link values."
                ),
                "action_frame": "base_link",
                "action_representation": "absolute_cartesian_pose",
            },
            "tcp_pose_base_link": self._pose_to_trace_dict(tcp_pose),
            "target_tcp_pose_base_link": self._pose_to_trace_dict(target_pose or self._last_commanded_target_pose),
            "port_transform_base_link": self._transform_to_trace_dict(port_transform),
            "plug_transform_base_link": self._transform_to_trace_dict(plug_transform),
            "vectors_from_tcp": vectors,
            "wrist_force_tcp_assumed_n": self._vector_to_trace_list(force_tcp_assumed),
            "wrist_force_base_link_estimated_n": self._vector_to_trace_list(force_base_assumed),
            "baseline_force_tcp_assumed_n": self._vector_to_trace_list(baseline_force),
            "force_delta_n": (
                float(force_delta_n)
                if force_delta_n is not None
                else (
                    float(np.linalg.norm(force_tcp_assumed - np.asarray(baseline_force, dtype=np.float64)))
                    if baseline_force is not None
                    else None
                )
            ),
            "tcp_speed_mps": self._tcp_speed_norm(get_observation),
            "tracking_error_m": self._tracking_error_norm(get_observation),
            "observation_tcp_velocity": {
                "linear": [
                    obs.get("tcp_velocity.linear.x"),
                    obs.get("tcp_velocity.linear.y"),
                    obs.get("tcp_velocity.linear.z"),
                ],
                "angular": [
                    obs.get("tcp_velocity.angular.x"),
                    obs.get("tcp_velocity.angular.y"),
                    obs.get("tcp_velocity.angular.z"),
                ],
            },
        }
        self._recovery_context_history.append(sample)

    def _trace_recent_recovery_context(self, event: str, **payload) -> None:
        max_samples = int(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_CONTEXT_MAX_TRACE_SAMPLES", "80"))
        samples = list(self._recovery_context_history)[-max_samples:]
        self._trace_event(
            event,
            recovery_context_sample_period_sec=self._recovery_context_sample_period_sec,
            recovery_context_samples=samples,
            force_trend=self._force_trend_snapshot(),
            **payload,
        )

    def _tcp_away_from_port_direction(self, port_transform: Transform) -> np.ndarray:
        current_pose = self._current_tcp_pose()
        tcp_xyz = self._pose_position_array(current_pose)
        port_xyz = self._transform_translation_array(port_transform)
        if port_xyz is None:
            return np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        rot_tcp_to_base = self._pose_orientation_rotation_tcp_to_base(current_pose)
        port_from_tcp_base = port_xyz - tcp_xyz
        port_from_tcp_tcp = rot_tcp_to_base.T @ port_from_tcp_base
        norm = float(np.linalg.norm(port_from_tcp_tcp))
        if norm <= 1e-9:
            return np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        direction = -port_from_tcp_tcp / norm
        # The port is usually close to the TCP +Z axis near insertion. Keep the
        # retreat dominated by axial separation but allow the small lateral
        # component needed to unload a side rub.
        lateral_norm = float(np.linalg.norm(direction[:2]))
        max_lateral = float(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_BACKOFF_MAX_LATERAL_FRACTION", "0.35"))
        if lateral_norm > max_lateral > 0.0:
            direction[:2] *= max_lateral / lateral_norm
            direction /= max(float(np.linalg.norm(direction)), 1e-9)
        return direction

    def _should_ignore_shallow_contact(
        self,
        *,
        task: Task | None = None,
        z_offset: float,
        force_delta_n: float,
        retry_count: int,
        sample_source: str,
    ) -> bool:
        is_sc_plug = bool(task is not None and str(getattr(task, "plug_name", "")).startswith("sc"))
        min_contact_z_offset = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_CONTACT_MIN_Z_OFFSET", "0.02")
        )
        early_force_threshold = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_EARLY_CONTACT_FORCE_THRESHOLD_N", "5.0")
        )
        above_contact_soft_realign_threshold = float(
            os.environ.get(
                "AIC_OFFICIAL_TEACHER_ABOVE_CONTACT_SOFT_REALIGN_FORCE_THRESHOLD_N",
                "2.0",
            )
        )
        near_insert_min_z_offset = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_LOW_FORCE_INSERTION_IGNORE_MIN_Z_OFFSET", "-0.012")
        )
        near_insert_max_z_offset = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_LOW_FORCE_INSERTION_IGNORE_MAX_Z_OFFSET", "0.0")
        )
        near_insert_force_threshold = float(
            os.environ.get(
                "AIC_OFFICIAL_TEACHER_LOW_FORCE_INSERTION_IGNORE_THRESHOLD_N",
                "6.0",
            )
        )
        if is_sc_plug:
            near_insert_force_threshold = float(
                os.environ.get(
                    "AIC_OFFICIAL_TEACHER_SC_LOW_FORCE_INSERTION_IGNORE_THRESHOLD_N",
                    str(near_insert_force_threshold),
                )
            )
        ignore_above_contact = bool(
            z_offset > min_contact_z_offset
            and force_delta_n < early_force_threshold
            and force_delta_n < above_contact_soft_realign_threshold
        )
        if ignore_above_contact:
            self._trace_event(
                "contact_trigger_ignored_above_contact_zone",
                retry_count=retry_count,
                force_delta_n=force_delta_n,
                threshold_n=self._ft_threshold_n,
                early_force_threshold_n=early_force_threshold,
                z_offset=z_offset,
                min_contact_z_offset=min_contact_z_offset,
                sample_source=sample_source,
            )
            return True
        if (
            z_offset > min_contact_z_offset
            and force_delta_n < early_force_threshold
            and force_delta_n >= above_contact_soft_realign_threshold
        ):
            self._trace_event(
                "contact_trigger_soft_realign_above_contact_zone",
                retry_count=retry_count,
                force_delta_n=force_delta_n,
                threshold_n=self._ft_threshold_n,
                early_force_threshold_n=early_force_threshold,
                soft_realign_threshold_n=above_contact_soft_realign_threshold,
                z_offset=z_offset,
                min_contact_z_offset=min_contact_z_offset,
                sample_source=sample_source,
            )
            return False
        ignore_near_insert = bool(
            z_offset > near_insert_min_z_offset
            and z_offset <= near_insert_max_z_offset
            and force_delta_n < near_insert_force_threshold
        )
        if ignore_near_insert:
            self._trace_event(
                "contact_trigger_ignored_low_force_near_insertion",
                retry_count=retry_count,
                force_delta_n=force_delta_n,
                threshold_n=self._ft_threshold_n,
                near_insert_force_threshold_n=near_insert_force_threshold,
                z_offset=z_offset,
                near_insert_min_z_offset=near_insert_min_z_offset,
                near_insert_max_z_offset=near_insert_max_z_offset,
                sample_source=sample_source,
            )
            return True
        return False

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
        force = self._force_vector(get_observation)
        force_delta = float(np.linalg.norm(force - baseline_force))
        self._record_recovery_context_sample(
            get_observation,
            force_base_or_tcp=force,
            baseline_force=baseline_force,
            phase="force_monitor",
            target_pose=self._last_commanded_target_pose,
            force_delta_n=force_delta,
        )
        return force_delta

    def _force_trigger_confirmed(
        self,
        get_observation: GetObservationCallback,
        baseline_force: np.ndarray,
        *,
        dt: float,
    ) -> tuple[bool, float]:
        confirm_sec = float(os.environ.get("AIC_OFFICIAL_TEACHER_FORCE_CONFIRM_SEC", str(dt)))
        trend_enabled = os.environ.get(
            "AIC_OFFICIAL_TEACHER_FORCE_TREND_CONFIRM_ENABLED",
            "true",
        ).lower() in {"1", "true", "yes", "on"}
        if trend_enabled:
            self._force_delta_history.clear()
            started = self.time_now()
            max_delta = 0.0
            decision = {}
            # Collect enough samples for two 5-point median windows whose centers
            # are separated by 5 controller ticks. This keeps the contact response
            # bounded while avoiding single-sample force spikes.
            required_samples = max(
                self._force_trend_window_samples + self._force_trend_center_gap_samples,
                self._force_trend_window_samples,
            )
            while (
                len(self._force_delta_history) < required_samples
                or (confirm_sec > 0.0 and (self.time_now() - started) < Duration(seconds=confirm_sec))
            ):
                delta = self._force_delta_norm(get_observation, baseline_force)
                max_delta = max(max_delta, delta)
                confirmed, decision = self._force_trigger_decision(delta)
                if confirmed:
                    break
                self.sleep_for(dt)
                if (self.time_now() - started) >= Duration(seconds=max(confirm_sec, required_samples * dt)):
                    break
            if not decision:
                delta = self._force_delta_norm(get_observation, baseline_force)
                max_delta = max(max_delta, delta)
                confirmed, decision = self._force_trigger_decision(delta)
            else:
                confirmed = bool(decision.get("confirmed"))
            self._trace_event(
                "force_trigger_confirmed_checked",
                force_trigger_confirmed=confirmed,
                confirmed_force_delta_n=max_delta,
                **decision,
            )
            return confirmed, max_delta
        if confirm_sec > 0.0:
            self.sleep_for(confirm_sec)
        confirmed_delta = self._force_delta_norm(get_observation, baseline_force)
        confirmed = confirmed_delta >= self._ft_threshold_n
        self._trace_event(
            "force_trigger_confirmed_checked",
            confirmed=confirmed,
            confirmed_force_delta_n=confirmed_delta,
            instant_threshold_n=self._ft_threshold_n,
            trend_enabled=False,
        )
        return confirmed, confirmed_delta

    def _post_insert_force_check(
        self,
        get_observation: GetObservationCallback,
        baseline_force: np.ndarray,
        *,
        dt: float,
        retry_count: int,
    ) -> tuple[bool, float]:
        check_sec = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_POST_INSERT_FORCE_CHECK_SEC", "0.05")
        )
        if check_sec <= 0.0:
            return True, 0.0
        started = self.time_now()
        max_force_delta = 0.0
        while (self.time_now() - started) < Duration(seconds=check_sec):
            force_delta = self._force_delta_norm(get_observation, baseline_force)
            max_force_delta = max(max_force_delta, force_delta)
            if force_delta >= self._post_insert_force_threshold_n:
                self._trace_event(
                    "post_insert_force_check_failed",
                    retry_count=retry_count,
                    force_delta_n=force_delta,
                    max_force_delta_n=max_force_delta,
                    threshold_n=self._post_insert_force_threshold_n,
                    approach_contact_threshold_n=self._ft_threshold_n,
                    check_sec=check_sec,
                )
                return False, max_force_delta
            self.sleep_for(dt)
        self._trace_event(
            "post_insert_force_check_passed",
            retry_count=retry_count,
            max_force_delta_n=max_force_delta,
            threshold_n=self._post_insert_force_threshold_n,
            approach_contact_threshold_n=self._ft_threshold_n,
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
        gain_profile: str = "default",
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
            vertical_bias_m=float(os.environ.get("AIC_OFFICIAL_TEACHER_PREINSERT_VERTICAL_BIAS_M", "0.0")),
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
            self._send_absolute_target(move_robot, pose, gain_profile=gain_profile)
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
        command_mode = os.environ.get("AIC_OFFICIAL_TEACHER_MICRO_ALIGN_COMMAND_MODE", "tcp_delta")
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
            command_mode=command_mode,
            frame="base_link" if command_mode == "base_absolute" else "gripper/tcp",
            representation="absolute_cartesian_pose" if command_mode == "base_absolute" else "relative_delta",
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
                if command_mode == "base_absolute":
                    correction_base = gain * tip_error_base
                    correction_base[2] = 0.0
                    norm = float(np.linalg.norm(correction_base))
                    if norm > max_xy_step:
                        correction_base *= max_xy_step / norm
                    current_xyz = self._pose_position_array(current_pose)
                    self._send_absolute_target(
                        move_robot,
                        self._make_pose(
                            current_xyz + correction_base,
                            self._pose_quat_wxyz(current_pose),
                        ),
                    )
                else:
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
        is_sc_plug = str(getattr(task, "plug_name", "")).startswith("sc")
        if is_sc_plug:
            duration_sec = max(
                duration_sec,
                float(os.environ.get("AIC_OFFICIAL_TEACHER_SC_PRECONTACT_PORT_ALIGN_SEC", "0.35")),
            )
        if duration_sec <= 0.0:
            self._trace_event("precontact_port_align_skipped", reason="duration_disabled")
            return True
        max_offset_default = "0.0030" if is_sc_plug else "0.00025"
        gain_default = "0.50" if is_sc_plug else "0.15"
        generic_max_offset = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_MAX_OFFSET_M", max_offset_default)
        )
        if is_sc_plug:
            max_offset = float(
                os.environ.get(
                    "AIC_OFFICIAL_TEACHER_SC_PRECONTACT_PORT_ALIGN_MAX_OFFSET_M",
                    str(max(generic_max_offset, float(max_offset_default))),
                )
            )
        else:
            max_offset = generic_max_offset
        residual_threshold = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_RESIDUAL_M", "0.0007")
        )
        speed_threshold = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_SPEED_MPS", "0.004")
        )
        force_abort_fraction = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_FORCE_ABORT_FRACTION", "0.60")
        )
        generic_gain = float(os.environ.get("AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_GAIN", gain_default))
        if is_sc_plug:
            gain = float(
                os.environ.get(
                    "AIC_OFFICIAL_TEACHER_SC_PRECONTACT_PORT_ALIGN_GAIN",
                    str(max(generic_gain, float(gain_default))),
                )
            )
        else:
            gain = generic_gain
        baseline_force = self._force_vector(get_observation)
        self._last_precontact_align_baseline_force = baseline_force.copy()
        self._last_precontact_align_force_delta_n = None
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
                self._last_precontact_align_force_delta_n = force_delta
                final_speed = self._tcp_speed_norm(get_observation)
                speed_ok = final_speed is None or final_speed <= speed_threshold
                if final_residual <= residual_threshold and speed_ok:
                    converged = True
                    break
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
        preserve_unconverged = os.environ.get(
            "AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_PRESERVE_UNCONVERGED_OFFSET",
            "true" if is_sc_plug else "false",
        ).lower() in {"1", "true", "yes", "on"}
        if aborted or (not converged and not preserve_unconverged):
            self._last_precontact_lateral_offset_base = np.zeros(3, dtype=np.float64)
        else:
            self._last_precontact_lateral_offset_base = final_offset.copy()
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
        max_lateral_error_m: float | None = None,
        near_gate_axial_target_m: float | None = None,
    ) -> bool:
        threshold = self._tracking_gate_threshold_m
        speed_threshold = float(os.environ.get("AIC_OFFICIAL_TEACHER_TRACKING_GATE_SPEED_MPS", "0.006"))
        if max_lateral_error_m is None:
            max_lateral_error_m = float(
                os.environ.get("AIC_OFFICIAL_TEACHER_TRACKING_GATE_MAX_LATERAL_ERROR_M", "0.0022")
            )
        is_sc_plug = str(getattr(task, "plug_name", "")).startswith("sc")
        if is_sc_plug:
            max_lateral_error_m = float(
                os.environ.get(
                    "AIC_OFFICIAL_TEACHER_SC_TRACKING_GATE_MAX_LATERAL_ERROR_M",
                    str(max_lateral_error_m),
                )
            )
        force_fraction = float(os.environ.get("AIC_OFFICIAL_TEACHER_TRACKING_GATE_FORCE_FRACTION", "1.0"))
        force_threshold = self._ft_threshold_n * force_fraction
        timeout_sec = max(settle_sec, self._tracking_gate_timeout_sec)
        if is_sc_plug:
            timeout_sec = max(
                timeout_sec,
                float(os.environ.get("AIC_OFFICIAL_TEACHER_SC_TRACKING_GATE_TIMEOUT_SEC", "4.0")),
            )
        started = self.time_now()
        baseline_force = self._force_vector(get_observation)
        self._last_tracking_gate_baseline_force = baseline_force.copy()
        passed = False
        final_error = None
        final_speed = None
        final_force_delta = None
        final_lateral_error = None
        final_axial_error = None
        final_commanded_lateral_error = None
        servo_command_bias_base = np.zeros(3, dtype=np.float64)
        gate_threshold = threshold
        gate_source = "unavailable"
        servo_compensation_enabled = os.environ.get(
            "AIC_OFFICIAL_TEACHER_PREINSERT_SERVO_COMPENSATION",
            "true",
        ).lower() in {"1", "true", "yes", "on"}
        servo_gain = float(os.environ.get("AIC_OFFICIAL_TEACHER_PREINSERT_SERVO_GAIN", "0.5"))
        servo_step_limit_m = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_PREINSERT_SERVO_STEP_LIMIT_M", "0.0005")
        )
        servo_max_bias_m = float(os.environ.get("AIC_OFFICIAL_TEACHER_PREINSERT_SERVO_MAX_BIAS_M", "0.0015"))
        if is_sc_plug:
            servo_max_bias_m = max(
                servo_max_bias_m,
                float(os.environ.get("AIC_OFFICIAL_TEACHER_SC_PREINSERT_SERVO_MAX_BIAS_M", "0.012")),
            )
        servo_deadband_m = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_PREINSERT_SERVO_DEADBAND_M", "0.00025")
        )
        servo_update_speed_mps = float(
            os.environ.get(
                "AIC_OFFICIAL_TEACHER_PREINSERT_SERVO_UPDATE_SPEED_MPS",
                str(speed_threshold),
            )
        )
        desired_lateral_offset_base = (
            np.asarray(lateral_offset_base, dtype=np.float64)
            if lateral_offset_base is not None
            else np.zeros(3, dtype=np.float64)
        )
        desired_target_pose = None
        while (self.time_now() - started) < Duration(seconds=timeout_sec):
            try:
                if near_gate_axial_target_m is None:
                    desired_target_pose = self._calc_cheatcode_gripper_pose(
                        task,
                        port_transform,
                        slerp_fraction=1.0,
                        position_fraction=1.0,
                        z_offset=z_offset,
                        reset_xy_integrator=True,
                        preserve_current_z=preserve_current_z,
                        lateral_offset_base=desired_lateral_offset_base,
                    )
                else:
                    desired_target_pose = self._calc_near_gate_gripper_pose(
                        task,
                        axial_offset_m=near_gate_axial_target_m,
                        slerp_fraction=1.0,
                        position_fraction=1.0,
                        lateral_offset_base=desired_lateral_offset_base,
                    )
                if target_z_m is not None:
                    desired_target_pose.position.z = float(target_z_m)
                target_pose = desired_target_pose
                if servo_compensation_enabled:
                    try:
                        if is_sc_plug:
                            plug_transform_for_servo = self._parent_node._tf_buffer.lookup_transform(
                                "base_link",
                                f"{task.cable_name}/{task.plug_name}_link",
                                Time(),
                            ).transform
                            lateral_error_for_servo = -np.asarray(
                                [
                                    port_transform.translation.x - plug_transform_for_servo.translation.x,
                                    port_transform.translation.y - plug_transform_for_servo.translation.y,
                                    0.0,
                                ],
                                dtype=np.float64,
                            )
                        else:
                            current_pose_for_servo = self._current_tcp_pose()
                            current_xyz_for_servo = self._pose_position_array(current_pose_for_servo)
                            desired_xyz_for_servo = self._pose_position_array(desired_target_pose)
                            lateral_error_for_servo = current_xyz_for_servo - desired_xyz_for_servo
                        speed_for_servo = self._tcp_speed_norm(get_observation)
                        update_ok = speed_for_servo is None or speed_for_servo <= servo_update_speed_mps
                        if update_ok and float(np.linalg.norm(lateral_error_for_servo[:2])) > servo_deadband_m:
                            step = -servo_gain * lateral_error_for_servo
                            step[2] = 0.0
                            step_norm = float(np.linalg.norm(step[:2]))
                            if step_norm > servo_step_limit_m > 0.0:
                                step *= servo_step_limit_m / step_norm
                            servo_command_bias_base += step
                            bias_norm = float(np.linalg.norm(servo_command_bias_base[:2]))
                            if bias_norm > servo_max_bias_m > 0.0:
                                servo_command_bias_base *= servo_max_bias_m / bias_norm
                    except TransformException:
                        pass
                    if near_gate_axial_target_m is None:
                        target_pose = self._calc_cheatcode_gripper_pose(
                            task,
                            port_transform,
                            slerp_fraction=1.0,
                            position_fraction=1.0,
                            z_offset=z_offset,
                            reset_xy_integrator=True,
                            preserve_current_z=preserve_current_z,
                            lateral_offset_base=desired_lateral_offset_base + servo_command_bias_base,
                        )
                    else:
                        target_pose = self._calc_near_gate_gripper_pose(
                            task,
                            axial_offset_m=near_gate_axial_target_m,
                            slerp_fraction=1.0,
                            position_fraction=1.0,
                            lateral_offset_base=desired_lateral_offset_base + servo_command_bias_base,
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
                desired_target_pose = None
            final_error = None
            if servo_compensation_enabled and desired_target_pose is not None:
                final_error = self._pose_position_error_norm(desired_target_pose)
                gate_source = "tf_desired_pose_error_servo"
            if final_error is None:
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
            final_lateral_error = None
            final_axial_error = None
            if target_pose is not None:
                try:
                    plug_transform_for_gate = None
                    entrance_transform_for_gate = None
                    try:
                        plug_transform_for_gate = self._parent_node._tf_buffer.lookup_transform(
                            "base_link",
                            f"{task.cable_name}/{task.plug_name}_link",
                            Time(),
                        ).transform
                    except TransformException:
                        plug_transform_for_gate = None
                    try:
                        entrance_transform_for_gate = self._parent_node._tf_buffer.lookup_transform(
                            "base_link",
                            f"task_board/{task.target_module_name}/{task.port_name}_link_entrance",
                            Time(),
                        ).transform
                    except TransformException:
                        entrance_transform_for_gate = None
                    if plug_transform_for_gate is not None and entrance_transform_for_gate is not None:
                        port_rotation = quaternion_xyzw_to_rotation_matrix(
                            np.asarray(
                                [
                                    float(entrance_transform_for_gate.rotation.x),
                                    float(entrance_transform_for_gate.rotation.y),
                                    float(entrance_transform_for_gate.rotation.z),
                                    float(entrance_transform_for_gate.rotation.w),
                                ],
                                dtype=np.float64,
                            )
                        )
                        port_axis_base = port_rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
                        plug_minus_entrance = np.asarray(
                            [
                                float(plug_transform_for_gate.translation.x - entrance_transform_for_gate.translation.x),
                                float(plug_transform_for_gate.translation.y - entrance_transform_for_gate.translation.y),
                                float(plug_transform_for_gate.translation.z - entrance_transform_for_gate.translation.z),
                            ],
                            dtype=np.float64,
                        )
                        final_axial_error = float(abs(np.dot(plug_minus_entrance, port_axis_base)))
                    if is_sc_plug:
                        if plug_transform_for_gate is not None:
                            lateral_reference_transform = entrance_transform_for_gate or port_transform
                            final_lateral_error = float(
                                np.linalg.norm(
                                    [
                                        lateral_reference_transform.translation.x - plug_transform_for_gate.translation.x,
                                        lateral_reference_transform.translation.y - plug_transform_for_gate.translation.y,
                                    ]
                                )
                            )
                            final_commanded_lateral_error = final_lateral_error
                    else:
                        current_pose_for_gate = self._current_tcp_pose()
                        current_xyz_for_gate = self._pose_position_array(current_pose_for_gate)
                        desired_pose_for_gate = desired_target_pose if desired_target_pose is not None else target_pose
                        target_xyz_for_gate = self._pose_position_array(desired_pose_for_gate)
                        final_lateral_error = float(
                            np.linalg.norm((current_xyz_for_gate - target_xyz_for_gate)[:2])
                        )
                        commanded_xyz_for_gate = self._pose_position_array(target_pose)
                        final_commanded_lateral_error = float(
                            np.linalg.norm((current_xyz_for_gate - commanded_xyz_for_gate)[:2])
                        )
                except TransformException:
                    final_lateral_error = None
                    final_commanded_lateral_error = None
            speed_ok = final_speed is None or final_speed <= speed_threshold
            force_ok = final_force_delta < force_threshold
            lateral_ok = (
                max_lateral_error_m <= 0.0
                or final_lateral_error is None
                or final_lateral_error <= max_lateral_error_m
            )
            passed = bool(
                final_error is not None
                and final_error <= gate_threshold
                and speed_ok
                and force_ok
                and lateral_ok
            )
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
        final_axial_error = None
        if current_pose is not None and target_pose is not None:
            if is_sc_plug and plug_transform is not None:
                lateral_reference_transform = port_transform
                try:
                    lateral_reference_transform = self._parent_node._tf_buffer.lookup_transform(
                        "base_link",
                        f"task_board/{task.target_module_name}/{task.port_name}_link_entrance",
                        Time(),
                    ).transform
                except TransformException:
                    lateral_reference_transform = port_transform
                final_lateral_error = float(
                    np.linalg.norm(
                        [
                            lateral_reference_transform.translation.x - plug_transform.translation.x,
                            lateral_reference_transform.translation.y - plug_transform.translation.y,
                        ]
                    )
                )
                final_commanded_lateral_error = final_lateral_error
            else:
                current_xyz = self._pose_position_array(current_pose)
                desired_pose_for_gate = desired_target_pose if desired_target_pose is not None else target_pose
                target_xyz = self._pose_position_array(desired_pose_for_gate)
                final_lateral_error = float(np.linalg.norm((current_xyz - target_xyz)[:2]))
                commanded_xyz = self._pose_position_array(target_pose)
                final_commanded_lateral_error = float(np.linalg.norm((current_xyz - commanded_xyz)[:2]))
        if plug_transform is not None:
            try:
                entrance_transform = self._parent_node._tf_buffer.lookup_transform(
                    "base_link",
                    f"task_board/{task.target_module_name}/{task.port_name}_link_entrance",
                    Time(),
                ).transform
                port_rotation = quaternion_xyzw_to_rotation_matrix(
                    np.asarray(
                        [
                            float(entrance_transform.rotation.x),
                            float(entrance_transform.rotation.y),
                            float(entrance_transform.rotation.z),
                            float(entrance_transform.rotation.w),
                        ],
                        dtype=np.float64,
                    )
                )
                port_axis_base = port_rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
                plug_minus_entrance = np.asarray(
                    [
                        float(plug_transform.translation.x - entrance_transform.translation.x),
                        float(plug_transform.translation.y - entrance_transform.translation.y),
                        float(plug_transform.translation.z - entrance_transform.translation.z),
                    ],
                    dtype=np.float64,
                )
                final_axial_error = float(abs(np.dot(plug_minus_entrance, port_axis_base)))
            except TransformException:
                final_axial_error = None
        self._trace_event(
            "tracking_gate_checked",
            tracking_gate_passed=passed,
            threshold_m=gate_threshold if final_error is not None else threshold,
            nominal_controller_threshold_m=threshold,
            gate_source=gate_source if final_error is not None else "unavailable",
            timeout_sec=timeout_sec,
            final_tracking_error_m=final_error,
            final_lateral_error_m=final_lateral_error,
            final_axial_error_m=final_axial_error,
            speed_threshold_mps=speed_threshold,
            max_lateral_error_m=max_lateral_error_m if max_lateral_error_m > 0.0 else None,
            final_tcp_speed_mps=final_speed,
            force_delta_n=final_force_delta,
            ft_threshold_n=self._ft_threshold_n,
            force_gate_threshold_n=force_threshold,
            force_gate_fraction=force_fraction,
            servo_compensation_enabled=servo_compensation_enabled,
            servo_command_bias_base_m=servo_command_bias_base.tolist()
            if servo_compensation_enabled
            else None,
            final_commanded_lateral_error_m=final_commanded_lateral_error,
            current_tcp_pose=self._pose_to_trace_dict(current_pose),
            target_tcp_pose=self._pose_to_trace_dict(target_pose),
            desired_target_tcp_pose=self._pose_to_trace_dict(desired_target_pose),
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
        self._last_tracking_gate_baseline_force = baseline_force.copy()
        self._last_tracking_gate_error_m = final_error
        self._last_tracking_gate_lateral_error_m = final_lateral_error
        self._last_tracking_gate_axial_error_m = final_axial_error
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
        pattern = os.environ.get(
            "AIC_OFFICIAL_TEACHER_RECOVERY_RETRY_PATTERN",
            "error_bias",
        ).lower()
        if pattern in {"cross", "spiral"}:
            step = float(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_RETRY_PATTERN_STEP_M", "0.003"))
            radius_index = (retry_count - 1) // 4 + 1
            direction_index = (retry_count - 1) % 4
            directions = [
                np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
                np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
                np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
                np.asarray([0.0, -1.0, 0.0], dtype=np.float64),
            ]
            if pattern == "cross":
                radius_index = 1
            offset = directions[direction_index] * step * radius_index
            self._trace_event(
                "recovery_retry_xy_pattern_computed",
                retry_count=retry_count,
                pattern=pattern,
                lateral_offset_base_m=offset.tolist(),
                step_m=step,
                radius_index=radius_index,
            )
            return offset
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
        default_threshold = "0.008" if self._expert_mode == "recovery" else "0.004"
        threshold = float(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_RETURN_Z_THRESHOLD_M", default_threshold))
        timeout_sec = float(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_RETURN_Z_TIMEOUT_SEC", "2.5"))
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
        handoff_blend_sec = float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_HANDOFF_SEC", "0.5"))
        handoff_speed_mps = float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_HANDOFF_SPEED_MPS", "0.04"))
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
        profile = os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_HANDOFF_PROFILE", "minimum_jerk").lower()
        self._trace_event(
            "cheatcode_handoff_started",
            from_z_offset=from_z_offset,
            to_z_offset=to_z_offset,
            vertical_bias_m=float(os.environ.get("AIC_OFFICIAL_TEACHER_PREINSERT_VERTICAL_BIAS_M", "0.0")),
            duration_sec=steps * dt,
            handoff_speed_mps=handoff_speed_mps,
            profile=profile,
            command_mode="absolute_cartesian_pose_base_link",
            lateral_offset_base=(
                np.asarray(lateral_offset_base, dtype=np.float64).tolist()
                if lateral_offset_base is not None
                else None
            ),
        )
        for step in range(steps):
            progress = (step + 1) / steps
            fraction = self._minimum_jerk_fraction(progress) if profile == "minimum_jerk" else progress
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

    def _nominal_realign_and_recheck_tracking_gate(
        self,
        task: Task,
        port_transform: Transform,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        *,
        settle_sec: float,
        dt: float,
        z_offset: float,
        preserve_current_z: bool,
        target_z_m: float | None = None,
        lateral_offset_base: np.ndarray | None = None,
        reason: str,
    ) -> bool:
        enabled = os.environ.get(
            "AIC_OFFICIAL_TEACHER_NOMINAL_PRECONTACT_REALIGN_ON_GATE_FAIL",
            "true",
        ).lower() in {"1", "true", "yes", "on"}
        last_force_delta = (
            float(self._last_tracking_gate_force_delta_n)
            if self._last_tracking_gate_force_delta_n is not None
            else 0.0
        )
        if not enabled or last_force_delta >= self._ft_threshold_n:
            self._trace_event(
                "nominal_precontact_realign_skipped",
                reason=reason,
                enabled=enabled,
                force_delta_n=last_force_delta,
                ft_threshold_n=self._ft_threshold_n,
            )
            return False
        realign_sec = float(
            os.environ.get(
                "AIC_OFFICIAL_TEACHER_NOMINAL_PRECONTACT_REALIGN_SEC",
                os.environ.get(
                    "AIC_OFFICIAL_TEACHER_TRACKING_GATE_REALIGN_SEC",
                    os.environ.get("AIC_OFFICIAL_TEACHER_LOCAL_PREINSERT_ALIGN_SEC", "1.2"),
                ),
            )
        )
        realign_speed_mps = float(
            os.environ.get(
                "AIC_OFFICIAL_TEACHER_NOMINAL_PRECONTACT_REALIGN_SPEED_MPS",
                os.environ.get("AIC_OFFICIAL_TEACHER_LOCAL_PREINSERT_ALIGN_SPEED_MPS", "0.08"),
            )
        )
        self._trace_event(
            "nominal_precontact_realign_started",
            reason=reason,
            duration_sec=realign_sec,
            preserve_current_z=preserve_current_z,
            target_z_m=target_z_m,
            lateral_offset_base=(
                np.asarray(lateral_offset_base, dtype=np.float64).tolist()
                if lateral_offset_base is not None
                else None
            ),
        )
        self._run_local_preinsert_align(
            task,
            port_transform,
            move_robot,
            duration_sec=realign_sec,
            dt=dt,
            z_offset=z_offset,
            preserve_current_z=preserve_current_z,
            lateral_offset_base=lateral_offset_base,
            max_speed_mps=realign_speed_mps,
        )
        gate_passed = self._hold_preinsert_until_tracking_gate(
            task,
            port_transform,
            get_observation,
            move_robot,
            settle_sec=settle_sec,
            dt=dt,
            z_offset=z_offset,
            preserve_current_z=preserve_current_z,
            target_z_m=target_z_m,
            lateral_offset_base=lateral_offset_base,
        )
        if not gate_passed:
            allowed_lateral_m = float(
                os.environ.get("AIC_OFFICIAL_TEACHER_NOMINAL_ALLOW_LOW_FORCE_GATE_MISS_M", "0.0")
            )
            last_lateral_error = (
                float(self._last_tracking_gate_lateral_error_m)
                if self._last_tracking_gate_lateral_error_m is not None
                else None
            )
            last_force_delta = (
                float(self._last_tracking_gate_force_delta_n)
                if self._last_tracking_gate_force_delta_n is not None
                else 0.0
            )
            last_speed = (
                float(self._last_tracking_gate_speed_mps)
                if self._last_tracking_gate_speed_mps is not None
                else None
            )
            speed_threshold = float(os.environ.get("AIC_OFFICIAL_TEACHER_TRACKING_GATE_SPEED_MPS", "0.006"))
            allow_small_low_force_miss = (
                allowed_lateral_m > 0.0
                and last_lateral_error is not None
                and last_lateral_error <= allowed_lateral_m
                and last_force_delta < self._ft_threshold_n
                and (last_speed is None or last_speed <= speed_threshold)
            )
            if allow_small_low_force_miss:
                self._trace_event(
                    "nominal_precontact_gate_miss_allowed",
                    reason=reason,
                    lateral_error_m=last_lateral_error,
                    allowed_lateral_m=allowed_lateral_m,
                    force_delta_n=last_force_delta,
                    ft_threshold_n=self._ft_threshold_n,
                    tcp_speed_mps=last_speed,
                    speed_threshold_mps=speed_threshold,
                )
                gate_passed = True
        self._trace_event(
            "nominal_precontact_realign_completed",
            reason=reason,
            tracking_gate_passed=gate_passed,
        )
        return gate_passed

    def _recovery_induced_lateral_offset_base(self, task: Task, port_transform: Transform) -> np.ndarray:
        enabled_default = "true" if self._expert_mode == "recovery" else "false"
        enabled = os.environ.get(
            "AIC_OFFICIAL_TEACHER_RECOVERY_INDUCE_FAILURE",
            enabled_default,
        ).lower() in {"1", "true", "yes", "on"}
        if not enabled:
            return np.zeros(3, dtype=np.float64)

        offset_m = float(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_INDUCE_LATERAL_OFFSET_M", "0.003"))
        if offset_m <= 0.0:
            return np.zeros(3, dtype=np.float64)

        port_xyz = self._transform_translation_array(port_transform)
        plug_transform = self._lookup_active_plug_transform()
        plug_xyz = self._transform_translation_array(plug_transform)
        direction = None
        direction_source = "port_x_axis"
        if port_xyz is not None and plug_xyz is not None:
            plug_error_xy = np.asarray(
                [plug_xyz[0] - port_xyz[0], plug_xyz[1] - port_xyz[1], 0.0],
                dtype=np.float64,
            )
            error_norm = float(np.linalg.norm(plug_error_xy[:2]))
            if error_norm > 1e-6:
                direction = plug_error_xy / error_norm
                direction_source = "plug_port_error"

        if direction is None:
            port_rotation = quaternion_xyzw_to_rotation_matrix(
                np.asarray(
                    [
                        float(port_transform.rotation.x),
                        float(port_transform.rotation.y),
                        float(port_transform.rotation.z),
                        float(port_transform.rotation.w),
                    ],
                    dtype=np.float64,
                )
            )
            direction = np.asarray([port_rotation[0, 0], port_rotation[1, 0], 0.0], dtype=np.float64)
            direction_norm = float(np.linalg.norm(direction[:2]))
            if direction_norm <= 1e-6:
                direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                direction = direction / direction_norm

        induced_offset = direction * offset_m
        self._trace_event(
            "recovery_induced_lateral_offset",
            lateral_offset_base_m=induced_offset.tolist(),
            requested_offset_m=offset_m,
            direction_source=direction_source,
        )
        return induced_offset

    def _plug_port_lateral_error_base(self, port_transform: Transform) -> np.ndarray | None:
        port_xyz = self._transform_translation_array(port_transform)
        plug_transform = self._lookup_active_plug_transform()
        plug_xyz = self._transform_translation_array(plug_transform)
        if port_xyz is None or plug_xyz is None:
            return None
        return np.asarray([plug_xyz[0] - port_xyz[0], plug_xyz[1] - port_xyz[1], 0.0], dtype=np.float64)

    def _run_recovery_lateral_unwedge(
        self,
        port_transform: Transform,
        move_robot: MoveRobotCallback,
        *,
        dt: float,
        stage_index: int,
        retry_count: int | None,
    ) -> bool:
        enabled = os.environ.get(
            "AIC_OFFICIAL_TEACHER_RECOVERY_ENABLE_LATERAL_UNWEDGE",
            "true" if self._expert_mode == "recovery" else "false",
        ).lower() in {"1", "true", "yes", "on"}
        if not enabled:
            return False
        lateral_error = self._plug_port_lateral_error_base(port_transform)
        if lateral_error is None:
            return False
        lateral_norm = float(np.linalg.norm(lateral_error[:2]))
        min_error = float(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_UNWEDGE_MIN_ERROR_M", "0.0007"))
        if lateral_norm < min_error:
            return False
        distance = float(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_UNWEDGE_DISTANCE_M", "0.002"))
        max_distance = float(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_UNWEDGE_MAX_DISTANCE_M", "0.003"))
        distance = min(max_distance, max(0.0, distance))
        if distance <= 0.0:
            return False
        duration = float(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_UNWEDGE_SEC", "0.35"))
        steps = max(1, int(math.ceil(duration / dt)))
        start_pose = self._current_tcp_pose()
        start_xyz = self._pose_position_array(start_pose)
        start_quat = self._pose_quat_wxyz(start_pose)
        direction = -lateral_error / lateral_norm
        target_xyz = start_xyz + direction * distance
        target_pose = self._make_pose(target_xyz, start_quat)
        self._trace_event(
            "recovery_lateral_unwedge_started",
            retry_count=retry_count,
            stage_index=stage_index,
            lateral_error_base_m=lateral_error.tolist(),
            lateral_error_norm_m=lateral_norm,
            unwedge_delta_base_m=(direction * distance).tolist(),
            target_tcp_pose=self._pose_to_trace_dict(target_pose),
        )
        for step in range(steps + 1):
            fraction = self._minimum_jerk_fraction(step / steps)
            pose = self._make_pose(start_xyz + fraction * (target_xyz - start_xyz), start_quat)
            self._send_absolute_target(move_robot, pose, gain_profile="recovery")
            self.sleep_for(dt)
        self._trace_event(
            "recovery_lateral_unwedge_completed",
            retry_count=retry_count,
            stage_index=stage_index,
            current_tcp_pose=self._pose_to_trace_dict(self._current_tcp_pose()),
        )
        return True

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
        retry_count: int | None = None,
    ) -> bool:
        is_sc_plug = str(getattr(task, "plug_name", "")).startswith("sc")
        backoff_increment = float(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_BACKOFF_DISTANCE_M", "0.005"))
        max_backoff_distance = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_MAX_BACKOFF_DISTANCE_M", "0.03")
        )
        min_backoff_distance = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_MIN_BACKOFF_DISTANCE_M", "0.002")
        )
        backoff_duration = float(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_BACKOFF_SEC", "0.45"))
        release_timeout = float(os.environ.get("AIC_OFFICIAL_TEACHER_FORCE_RELEASE_TIMEOUT_SEC", "5.0"))
        release_check_sec = float(os.environ.get("AIC_OFFICIAL_TEACHER_FORCE_RELEASE_STAGE_CHECK_SEC", "0.10"))
        release_check_sec = max(
            release_check_sec,
            float(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_RELEASE_STABLE_SEC", "0.25")),
        )
        configured_release_threshold = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_RELEASE_FORCE_THRESHOLD_N", str(self._ft_threshold_n))
        )
        if is_sc_plug:
            backoff_increment = float(
                os.environ.get("AIC_OFFICIAL_TEACHER_SC_RECOVERY_BACKOFF_DISTANCE_M", str(backoff_increment))
            )
            max_backoff_distance = float(
                os.environ.get("AIC_OFFICIAL_TEACHER_SC_RECOVERY_MAX_BACKOFF_DISTANCE_M", str(max_backoff_distance))
            )
            min_backoff_distance = float(
                os.environ.get("AIC_OFFICIAL_TEACHER_SC_RECOVERY_MIN_BACKOFF_DISTANCE_M", str(min_backoff_distance))
            )
            backoff_duration = float(
                os.environ.get("AIC_OFFICIAL_TEACHER_SC_RECOVERY_BACKOFF_SEC", str(backoff_duration))
            )
            release_timeout = float(
                os.environ.get("AIC_OFFICIAL_TEACHER_SC_FORCE_RELEASE_TIMEOUT_SEC", str(release_timeout))
            )
            release_check_sec = max(
                release_check_sec,
                float(
                    os.environ.get(
                        "AIC_OFFICIAL_TEACHER_SC_RECOVERY_RELEASE_STABLE_SEC",
                        str(release_check_sec),
                    )
                ),
            )
            configured_release_threshold = float(
                os.environ.get(
                    "AIC_OFFICIAL_TEACHER_SC_RECOVERY_RELEASE_FORCE_THRESHOLD_N",
                    str(configured_release_threshold),
                )
            )
        min_backoff_distance = min(max_backoff_distance, max(0.0, min_backoff_distance))
        strict_release_default = str(configured_release_threshold) if self._expert_mode == "recovery" else "1.0"
        strict_release_value = os.environ.get(
            "AIC_OFFICIAL_TEACHER_RECOVERY_STRICT_RELEASE_FORCE_THRESHOLD_N",
            strict_release_default,
        )
        if is_sc_plug:
            strict_release_value = os.environ.get(
                "AIC_OFFICIAL_TEACHER_SC_RECOVERY_STRICT_RELEASE_FORCE_THRESHOLD_N",
                strict_release_value,
            )
        release_threshold = min(
            configured_release_threshold,
            float(strict_release_value),
        )
        required_measured_backoff_value = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_REQUIRED_MEASURED_BACKOFF_M", "0.002")
        )
        if is_sc_plug:
            required_measured_backoff_value = float(
                os.environ.get(
                    "AIC_OFFICIAL_TEACHER_SC_RECOVERY_REQUIRED_MEASURED_BACKOFF_M",
                    str(required_measured_backoff_value),
                )
            )
        required_measured_backoff = min(
            max_backoff_distance,
            max(
                min_backoff_distance,
                required_measured_backoff_value,
            ),
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
            required_measured_backoff_m=required_measured_backoff,
            release_timeout_sec=release_timeout,
            release_threshold_n=release_threshold,
            configured_release_threshold_n=configured_release_threshold,
            release_stable_sec=release_check_sec,
            start_tcp_pose=self._pose_to_trace_dict(start_pose),
        )
        force_released = False
        release_started = self.time_now()
        total_backoff = 0.0
        max_measured_backoff = 0.0
        backoff_pose = start_pose
        stage_index = 0
        backoff_mode = os.environ.get(
            "AIC_OFFICIAL_TEACHER_RECOVERY_BACKOFF_MODE",
            "tcp_away_from_port",
        )
        if is_sc_plug:
            backoff_mode = os.environ.get("AIC_OFFICIAL_TEACHER_SC_RECOVERY_BACKOFF_MODE", backoff_mode)
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
                backoff_mode=backoff_mode,
                target_tcp_pose=self._pose_to_trace_dict(backoff_pose),
            )
            for step in range(steps + 1):
                if backoff_mode == "tcp_away_from_port":
                    direction_tcp = self._tcp_away_from_port_direction(port_transform)
                    step_distance = max(0.0, next_backoff - total_backoff) / float(steps + 1)
                    self.set_delta_pose_target_from_components(
                        move_robot=move_robot,
                        delta_position_xyz=direction_tcp * step_distance,
                        delta_rotation_xyz=np.zeros(3, dtype=np.float64),
                        frame_id="gripper/tcp",
                        stiffness=self._recovery_cartesian_stiffness,
                        damping=self._recovery_cartesian_damping,
                        max_translation=max(step_distance, 1e-6),
                    )
                    backoff_pose = self._current_tcp_pose()
                else:
                    fraction = self._minimum_jerk_fraction(step / steps)
                    pose = self._make_pose(
                        stage_start_xyz + fraction * (stage_target_xyz - stage_start_xyz),
                        start_quat,
                    )
                    self._send_absolute_target(move_robot, pose, gain_profile="recovery")
                    backoff_pose = pose
                self.sleep_for(dt)
                current_xyz = self._pose_position_array(self._current_tcp_pose())
                measured_distance = float(np.linalg.norm(current_xyz - start_xyz))
                if backoff_mode == "base_z_absolute":
                    measured_distance = current_xyz[2] - float(start_xyz[2])
                max_measured_backoff = max(max_measured_backoff, measured_distance)
            total_backoff = next_backoff
            current_pose = self._current_tcp_pose()
            current_xyz = self._pose_position_array(current_pose)
            measured_distance = float(np.linalg.norm(current_xyz - start_xyz))
            if backoff_mode == "base_z_absolute":
                measured_distance = current_xyz[2] - float(start_xyz[2])
            max_measured_backoff = max(max_measured_backoff, measured_distance)
            self._trace_event(
                "recovery_backoff_stage_completed",
                stage_index=stage_index,
                backoff_mode=backoff_mode,
                backoff_distance_achieved_m=total_backoff,
                measured_backoff_distance_m=max_measured_backoff,
                current_tcp_pose=self._pose_to_trace_dict(current_pose),
            )

            if total_backoff < min_backoff_distance:
                continue

            stage_release_started = self.time_now()
            stage_release_stable = True
            stage_release_sample_count = 0
            while (
                (self.time_now() - stage_release_started) < Duration(seconds=release_check_sec)
                and (self.time_now() - release_started) < Duration(seconds=release_timeout)
            ):
                current_force = self._force_vector(get_observation)
                force_delta = float(np.linalg.norm(current_force - baseline))
                stage_release_sample_count += 1
                self._record_recovery_context_sample(
                    get_observation,
                    force_base_or_tcp=current_force,
                    baseline_force=baseline,
                    phase="recovery_backoff_release_check",
                    target_pose=backoff_pose,
                    force_delta_n=force_delta,
                    z_offset=current_z_offset,
                    retry_count=retry_count,
                )
                self._send_absolute_target(move_robot, backoff_pose, gain_profile="recovery")
                current_xyz = self._pose_position_array(self._current_tcp_pose())
                measured_distance = float(np.linalg.norm(current_xyz - start_xyz))
                if backoff_mode == "base_z_absolute":
                    measured_distance = current_xyz[2] - float(start_xyz[2])
                max_measured_backoff = max(max_measured_backoff, measured_distance)
                if force_delta >= release_threshold:
                    stage_release_stable = False
                self.sleep_for(dt)
            force_released = stage_release_stable and stage_release_sample_count > 0
            measured_backoff_occurred = max_measured_backoff >= required_measured_backoff
            if force_released and measured_backoff_occurred:
                break
            if (
                not force_released
                and total_backoff >= min_backoff_distance
                and stage_index
                <= int(os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_MAX_UNWEDGE_STAGES", "2"))
            ):
                self._run_recovery_lateral_unwedge(
                    port_transform,
                    move_robot,
                    dt=dt,
                    stage_index=stage_index,
                    retry_count=retry_count,
                )
        current_pose = self._current_tcp_pose()
        current_xyz = self._pose_position_array(current_pose)
        measured_distance = float(np.linalg.norm(current_xyz - start_xyz))
        if backoff_mode == "base_z_absolute":
            measured_distance = current_xyz[2] - float(start_xyz[2])
        max_measured_backoff = max(max_measured_backoff, measured_distance)
        self._trace_event(
            "recovery_backoff_completed",
            backoff_mode=backoff_mode,
            backoff_distance_achieved_m=total_backoff,
            measured_backoff_distance_m=max_measured_backoff,
            measured_backoff_occurred=bool(max_measured_backoff >= required_measured_backoff),
            required_measured_backoff_m=required_measured_backoff,
            current_tcp_pose=self._pose_to_trace_dict(current_pose),
        )
        self._trace_event(
            "recovery_force_release_wait",
            force_released=force_released,
            threshold_n=release_threshold,
            contact_threshold_n=self._ft_threshold_n,
            backoff_distance_achieved_m=total_backoff,
            measured_backoff_distance_m=max_measured_backoff,
            measured_backoff_occurred=bool(max_measured_backoff >= required_measured_backoff),
            min_backoff_distance_m=min_backoff_distance,
            required_measured_backoff_m=required_measured_backoff,
            max_backoff_distance_m=max_backoff_distance,
        )
        measured_backoff_occurred = max_measured_backoff >= required_measured_backoff
        if not measured_backoff_occurred:
            fallback_enabled = os.environ.get(
                "AIC_OFFICIAL_TEACHER_RECOVERY_MEASURED_BACKOFF_FALLBACK",
                "true",
            ).lower() in {"1", "true", "yes", "on"}
            if fallback_enabled:
                fallback_distance = min(
                    max_backoff_distance,
                    max(required_measured_backoff, backoff_increment),
                )
                fallback_sec = float(
                    os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_MEASURED_BACKOFF_FALLBACK_SEC", "0.6")
                )
                fallback_start_pose = self._current_tcp_pose()
                fallback_start_xyz = self._pose_position_array(fallback_start_pose)
                fallback_target_xyz = fallback_start_xyz.copy()
                fallback_target_xyz[2] = max(
                    float(fallback_target_xyz[2]),
                    float(start_xyz[2]) + fallback_distance,
                )
                fallback_pose = self._make_pose(fallback_target_xyz, start_quat)
                fallback_steps = max(1, int(math.ceil(fallback_sec / dt)))
                self._trace_event(
                    "recovery_measured_backoff_fallback_started",
                    fallback_distance_m=fallback_distance,
                    fallback_sec=fallback_sec,
                    target_tcp_pose=self._pose_to_trace_dict(fallback_pose),
                    measured_backoff_distance_m=max_measured_backoff,
                    min_backoff_distance_m=min_backoff_distance,
                    required_measured_backoff_m=required_measured_backoff,
                )
                for step in range(fallback_steps + 1):
                    fraction = self._minimum_jerk_fraction(step / fallback_steps)
                    pose = self._make_pose(
                        fallback_start_xyz + fraction * (fallback_target_xyz - fallback_start_xyz),
                        start_quat,
                    )
                    self._send_absolute_target(move_robot, pose, gain_profile="recovery")
                    self.sleep_for(dt)
                    current_xyz = self._pose_position_array(self._current_tcp_pose())
                    measured_distance = float(np.linalg.norm(current_xyz - start_xyz))
                    if backoff_mode == "base_z_absolute":
                        measured_distance = current_xyz[2] - float(start_xyz[2])
                    max_measured_backoff = max(max_measured_backoff, measured_distance)
                fallback_release_started = self.time_now()
                fallback_release_stable = True
                fallback_release_sample_count = 0
                while (
                    (self.time_now() - fallback_release_started) < Duration(seconds=release_check_sec)
                    and (self.time_now() - release_started) < Duration(seconds=release_timeout)
                ):
                    current_force = self._force_vector(get_observation)
                    force_delta = float(np.linalg.norm(current_force - baseline))
                    fallback_release_sample_count += 1
                    self._record_recovery_context_sample(
                        get_observation,
                        force_base_or_tcp=current_force,
                        baseline_force=baseline,
                        phase="recovery_measured_backoff_fallback_release_check",
                        target_pose=fallback_pose,
                        force_delta_n=force_delta,
                        z_offset=current_z_offset,
                        retry_count=retry_count,
                    )
                    self._send_absolute_target(move_robot, fallback_pose, gain_profile="recovery")
                    if force_delta >= release_threshold:
                        fallback_release_stable = False
                    self.sleep_for(dt)
                force_released = fallback_release_stable and fallback_release_sample_count > 0
                current_pose = self._current_tcp_pose()
                current_xyz = self._pose_position_array(current_pose)
                measured_distance = float(np.linalg.norm(current_xyz - start_xyz))
                if backoff_mode == "base_z_absolute":
                    measured_distance = current_xyz[2] - float(start_xyz[2])
                max_measured_backoff = max(max_measured_backoff, measured_distance)
                measured_backoff_occurred = max_measured_backoff >= required_measured_backoff
                self._trace_event(
                    "recovery_measured_backoff_fallback_completed",
                    force_released=force_released,
                    measured_backoff_distance_m=max_measured_backoff,
                    measured_backoff_occurred=bool(measured_backoff_occurred),
                    required_measured_backoff_m=required_measured_backoff,
                    current_tcp_pose=self._pose_to_trace_dict(current_pose),
                )
        require_measured_backoff = os.environ.get(
            "AIC_OFFICIAL_TEACHER_REQUIRE_MEASURED_BACKOFF",
            "true",
        ).lower() in {"1", "true", "yes", "on"}
        if require_measured_backoff and not measured_backoff_occurred:
            self._trace_recent_recovery_context(
                "recovery_measured_backoff_failed",
                retry_count=retry_count,
                measured_backoff_distance_m=max_measured_backoff,
                min_backoff_distance_m=min_backoff_distance,
                required_measured_backoff_m=required_measured_backoff,
                force_released=force_released,
            )
            return False
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
                self._send_absolute_target(move_robot, pose, gain_profile="recovery")
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
            duration_sec=float(os.environ.get("AIC_OFFICIAL_TEACHER_LOCAL_PREINSERT_ALIGN_SEC", "0.8")),
            dt=dt,
            z_offset=float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_START_Z_OFFSET", "0.045")),
            preserve_current_z=recovery_realign_preserve_current_z,
            max_speed_mps=recovery_realign_speed_mps,
            gain_profile="recovery",
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
                self._send_absolute_target(move_robot, pose, gain_profile="recovery")
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
                stiffness=self._cartesian_stiffness,
                damping=self._cartesian_damping,
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
                target_stiffness=self._expand_gain_vector(
                    self._joint_stiffness_default,
                    n_joints,
                    "AIC_OFFICIAL_TEACHER_JOINT_STIFFNESS",
                ),
                target_damping=self._expand_gain_vector(
                    self._joint_damping_default,
                    n_joints,
                    "AIC_OFFICIAL_TEACHER_JOINT_DAMPING",
                ),
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
        self._active_task = task
        self._active_port_transform = port_transform
        send_feedback("official_teacher_replay_online_cheatcode_final_insertion")
        dt = float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_DT", "0.05"))
        z_offset = float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_START_Z_OFFSET", "0.045"))
        settle_sec = float(os.environ.get("AIC_OFFICIAL_TEACHER_PRE_INSERT_SETTLE_SEC", "0.35"))
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
        initial_recovery_gate_max_lateral_m = None
        if self._expert_mode == "recovery":
            initial_recovery_gate_max_lateral_m = float(
                os.environ.get(
                    "AIC_OFFICIAL_TEACHER_RECOVERY_INITIAL_TRACKING_GATE_MAX_LATERAL_ERROR_M",
                    os.environ.get("AIC_OFFICIAL_TEACHER_TRACKING_GATE_MAX_LATERAL_ERROR_M", "0.0022"),
                )
            )
        if not self._local_preinsert_align_done:
            self._run_local_preinsert_align(
                task,
                port_transform,
                move_robot,
                duration_sec=float(os.environ.get("AIC_OFFICIAL_TEACHER_LOCAL_PREINSERT_ALIGN_SEC", "0.8")),
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
                max_lateral_error_m=initial_recovery_gate_max_lateral_m,
            )
            self._preinsert_settle_done = True
            if not gate_passed and self._expert_mode == "nominal":
                gate_passed = self._nominal_realign_and_recheck_tracking_gate(
                    task,
                    port_transform,
                    get_observation,
                    move_robot,
                    settle_sec=settle_sec,
                    dt=dt,
                    z_offset=z_offset,
                    preserve_current_z=preinsert_align_preserve_current_z,
                    reason="initial_preinsert_tracking_gate",
                )
                if not gate_passed:
                    send_feedback("official_teacher_replay_tracking_gate_failed")
                    self._trace_event("nominal_rejected_tracking_gate")
                    return False
            if not gate_passed:
                last_gate_force_delta = (
                    float(self._last_tracking_gate_force_delta_n)
                    if self._last_tracking_gate_force_delta_n is not None
                    else 0.0
                )
                if last_gate_force_delta >= self._ft_threshold_n:
                    self._trace_event(
                        "contact_detected",
                        retry_count=0,
                        force_delta_n=last_gate_force_delta,
                        threshold_n=self._ft_threshold_n,
                        z_offset=z_offset,
                        sample_source="initial_preinsert_tracking_gate",
                    )
                    send_feedback("official_teacher_replay_recovery_backoff")
                    if not self._backoff_and_realign(
                        task,
                        port_transform,
                        get_observation,
                        move_robot,
                        current_z_offset=z_offset,
                        dt=dt,
                        release_baseline_force=self._last_tracking_gate_baseline_force,
                        retry_start_z_m=float(self._current_tcp_pose().position.z),
                        retry_count=0,
                    ):
                        send_feedback("official_teacher_replay_recovery_force_release_failed")
                        self._trace_event("recovery_force_release_failed", retry_count=0)
                        return False
                    self._trace_event("recovery_initial_tracking_gate_backoff_completed")
                    gate_passed = True
            if not gate_passed:
                send_feedback("official_teacher_replay_tracking_gate_realign")
                self._trace_event("recovery_tracking_gate_realign_started")
                self._run_local_preinsert_align(
                    task,
                    port_transform,
                    move_robot,
                    duration_sec=float(
                        os.environ.get(
                            "AIC_OFFICIAL_TEACHER_TRACKING_GATE_REALIGN_SEC",
                            os.environ.get("AIC_OFFICIAL_TEACHER_LOCAL_PREINSERT_ALIGN_SEC", "1.2"),
                        )
                    ),
                    dt=dt,
                    z_offset=z_offset,
                    preserve_current_z=preinsert_align_preserve_current_z,
                    gain_profile="recovery",
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
                    max_lateral_error_m=initial_recovery_gate_max_lateral_m,
                )
                self._trace_event(
                    "recovery_tracking_gate_realign_completed",
                    tracking_gate_passed=gate_passed,
                )
                if not gate_passed:
                    allowed_lateral_m = float(
                        os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_ALLOW_INITIAL_LOW_FORCE_GATE_MISS_M", "0.003")
                    )
                    last_lateral_error = (
                        float(self._last_tracking_gate_lateral_error_m)
                        if self._last_tracking_gate_lateral_error_m is not None
                        else None
                    )
                    last_gate_force_delta = (
                        float(self._last_tracking_gate_force_delta_n)
                        if self._last_tracking_gate_force_delta_n is not None
                        else 0.0
                    )
                    last_speed = (
                        float(self._last_tracking_gate_speed_mps)
                        if self._last_tracking_gate_speed_mps is not None
                        else None
                    )
                    speed_threshold = float(
                        os.environ.get("AIC_OFFICIAL_TEACHER_TRACKING_GATE_SPEED_MPS", "0.006")
                    )
                    allow_small_low_force_miss = (
                        self._expert_mode == "recovery"
                        and allowed_lateral_m > 0.0
                        and last_lateral_error is not None
                        and last_lateral_error <= allowed_lateral_m
                        and last_gate_force_delta < self._ft_threshold_n
                        and (last_speed is None or last_speed <= speed_threshold)
                    )
                    if allow_small_low_force_miss:
                        self._trace_event(
                            "recovery_initial_low_force_gate_miss_allowed",
                            lateral_error_m=last_lateral_error,
                            allowed_lateral_m=allowed_lateral_m,
                            force_delta_n=last_gate_force_delta,
                            ft_threshold_n=self._ft_threshold_n,
                            tcp_speed_mps=last_speed,
                            speed_threshold_mps=speed_threshold,
                        )
                        gate_passed = True
                if not gate_passed:
                    send_feedback("official_teacher_replay_tracking_gate_failed")
                    return False

        insertion_speed_mps = float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_INSERTION_SPEED_MPS", "0.012"))
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
            os.environ.get("AIC_OFFICIAL_TEACHER_INSERTION_EXACT_POSITION_STEP_M", "0.0012")
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
            port_align_sec = float(os.environ.get("AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_SEC", "0.4"))
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
                if self._expert_mode == "nominal":
                    self._trace_event(
                        "nominal_precontact_port_align_force_abort",
                        retry_count=retry_count,
                        force_delta_n=(
                            float(self._last_precontact_align_force_delta_n)
                            if self._last_precontact_align_force_delta_n is not None
                            else None
                        ),
                    )
                    return False
                self._trace_event(
                    "recovery_precontact_port_align_backoff_started",
                    retry_count=retry_count,
                    force_delta_n=(
                        float(self._last_precontact_align_force_delta_n)
                        if self._last_precontact_align_force_delta_n is not None
                        else None
                    ),
                )
                if not self._backoff_and_realign(
                    task,
                    port_transform,
                    get_observation,
                    move_robot,
                    current_z_offset=z_offset,
                    dt=dt,
                    release_baseline_force=self._last_precontact_align_baseline_force,
                    retry_start_z_m=original_preinsert_z,
                    retry_count=retry_count,
                ):
                    return False
                retry_count += 1
                continue
            active_lateral_offset = retry_lateral_offset + self._last_precontact_lateral_offset_base
            recovery_induced_offset = np.zeros(3, dtype=np.float64)
            if self._expert_mode == "recovery" and retry_count == 0:
                recovery_induced_offset = self._recovery_induced_lateral_offset_base(task, port_transform)
                active_lateral_offset = active_lateral_offset + recovery_induced_offset
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
                if (
                    not gate_passed
                    and self._expert_mode == "recovery"
                    and retry_count == 0
                    and float(np.linalg.norm(recovery_induced_offset[:2])) > 1e-6
                    and os.environ.get(
                        "AIC_OFFICIAL_TEACHER_RECOVERY_ALLOW_INDUCED_GATE_MISS",
                        "true",
                    ).lower() in {"1", "true", "yes", "on"}
                ):
                    self._trace_event(
                        "recovery_induced_preinsert_gate_miss_allowed",
                        retry_count=retry_count,
                        lateral_offset_base_m=active_lateral_offset.tolist(),
                        force_delta_n=(
                            float(self._last_tracking_gate_force_delta_n)
                            if self._last_tracking_gate_force_delta_n is not None
                            else None
                        ),
                        lateral_error_m=(
                            float(self._last_tracking_gate_lateral_error_m)
                            if self._last_tracking_gate_lateral_error_m is not None
                            else None
                        ),
                    )
                    gate_passed = True
                if self._expert_mode == "recovery" and retry_count > 0 and not gate_passed:
                    allowed_lateral_m = float(
                        os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_ALLOW_LOW_FORCE_GATE_MISS_M", "0.003")
                    )
                    last_lateral_error = (
                        float(self._last_tracking_gate_lateral_error_m)
                        if self._last_tracking_gate_lateral_error_m is not None
                        else None
                    )
                    last_gate_force_delta = (
                        float(self._last_tracking_gate_force_delta_n)
                        if self._last_tracking_gate_force_delta_n is not None
                        else 0.0
                    )
                    last_speed = (
                        float(self._last_tracking_gate_speed_mps)
                        if self._last_tracking_gate_speed_mps is not None
                        else None
                    )
                    speed_threshold = float(
                        os.environ.get("AIC_OFFICIAL_TEACHER_TRACKING_GATE_SPEED_MPS", "0.006")
                    )
                    allow_low_force_retry_miss = (
                        allowed_lateral_m > 0.0
                        and last_lateral_error is not None
                        and last_lateral_error <= allowed_lateral_m
                        and last_gate_force_delta < self._ft_threshold_n
                        and (last_speed is None or last_speed <= speed_threshold)
                    )
                    if allow_low_force_retry_miss:
                        self._trace_event(
                            "recovery_retry_preinsert_gate_miss_allowed",
                            retry_count=retry_count,
                            lateral_error_m=last_lateral_error,
                            allowed_lateral_m=allowed_lateral_m,
                            force_delta_n=last_gate_force_delta,
                            ft_threshold_n=self._ft_threshold_n,
                            tcp_speed_mps=last_speed,
                            speed_threshold_mps=speed_threshold,
                        )
                        gate_passed = True
                if not gate_passed:
                    send_feedback("official_teacher_replay_tracking_gate_realign")
                    self._trace_event(
                        "recovery_preinsert_micro_align_gate_realign_started",
                        retry_count=retry_count,
                    )
                    self._run_local_preinsert_align(
                        task,
                        port_transform,
                        move_robot,
                        duration_sec=float(
                            os.environ.get(
                                "AIC_OFFICIAL_TEACHER_TRACKING_GATE_REALIGN_SEC",
                                os.environ.get("AIC_OFFICIAL_TEACHER_LOCAL_PREINSERT_ALIGN_SEC", "1.2"),
                            )
                        ),
                        dt=dt,
                        z_offset=z_offset,
                        preserve_current_z=True,
                        lateral_offset_base=active_lateral_offset,
                        gain_profile="recovery",
                    )
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
                        "recovery_preinsert_micro_align_gate_realign_completed",
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
            stop_at_near_gate = (
                os.environ.get("AIC_OFFICIAL_TEACHER_STOP_AT_NEAR_GATE", "false").lower()
                in {"1", "true", "yes", "on"}
            )
            stop_near_gate_axial_target_m = (
                float(os.environ.get("AIC_OFFICIAL_TEACHER_STOP_NEAR_GATE_TARGET_AXIAL_OFFSET_M", "0.0"))
                if stop_at_near_gate
                else None
            )
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
                        near_gate_axial_target_m=stop_near_gate_axial_target_m,
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
                    if (
                        self._expert_mode == "recovery"
                        and retry_count == 0
                        and float(np.linalg.norm(recovery_induced_offset[:2])) > 1e-6
                        and last_gate_force_delta < self._ft_threshold_n
                        and os.environ.get(
                            "AIC_OFFICIAL_TEACHER_RECOVERY_ALLOW_INDUCED_GATE_MISS",
                            "true",
                        ).lower() in {"1", "true", "yes", "on"}
                    ):
                        self._trace_event(
                            "recovery_induced_handoff_gate_miss_allowed",
                            retry_count=retry_count,
                            lateral_offset_base_m=active_lateral_offset.tolist(),
                            force_delta_n=last_gate_force_delta,
                            ft_threshold_n=self._ft_threshold_n,
                            lateral_error_m=(
                                float(self._last_tracking_gate_lateral_error_m)
                                if self._last_tracking_gate_lateral_error_m is not None
                                else None
                            ),
                        )
                        gate_passed = True
                    if self._expert_mode in {"nominal", "nominalrecovery"}:
                        allowed_lateral_m = float(
                            os.environ.get("AIC_OFFICIAL_TEACHER_NOMINAL_ALLOW_LOW_FORCE_GATE_MISS_M", "0.0")
                        )
                        last_lateral_error = (
                            float(self._last_tracking_gate_lateral_error_m)
                            if self._last_tracking_gate_lateral_error_m is not None
                            else None
                        )
                        last_speed = (
                            float(self._last_tracking_gate_speed_mps)
                            if self._last_tracking_gate_speed_mps is not None
                            else None
                        )
                        speed_threshold = float(
                            os.environ.get("AIC_OFFICIAL_TEACHER_TRACKING_GATE_SPEED_MPS", "0.006")
                        )
                        allow_small_low_force_miss = (
                            allowed_lateral_m > 0.0
                            and last_lateral_error is not None
                            and last_lateral_error <= allowed_lateral_m
                            and last_gate_force_delta < self._ft_threshold_n
                            and (last_speed is None or last_speed <= speed_threshold)
                        )
                        if allow_small_low_force_miss:
                            self._trace_event(
                                "nominal_handoff_gate_miss_allowed",
                                expert_mode=self._expert_mode,
                                retry_count=retry_count,
                                lateral_error_m=last_lateral_error,
                                allowed_lateral_m=allowed_lateral_m,
                                force_delta_n=last_gate_force_delta,
                                ft_threshold_n=self._ft_threshold_n,
                                tcp_speed_mps=last_speed,
                                speed_threshold_mps=speed_threshold,
                            )
                            gate_passed = True
                    if self._expert_mode == "recovery" and retry_count > 0 and not gate_passed:
                        allowed_lateral_m = float(
                            os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_ALLOW_LOW_FORCE_GATE_MISS_M", "0.003")
                        )
                        last_lateral_error = (
                            float(self._last_tracking_gate_lateral_error_m)
                            if self._last_tracking_gate_lateral_error_m is not None
                            else None
                        )
                        last_speed = (
                            float(self._last_tracking_gate_speed_mps)
                            if self._last_tracking_gate_speed_mps is not None
                            else None
                        )
                        speed_threshold = float(
                            os.environ.get("AIC_OFFICIAL_TEACHER_TRACKING_GATE_SPEED_MPS", "0.006")
                        )
                        allow_low_force_retry_miss = (
                            allowed_lateral_m > 0.0
                            and last_lateral_error is not None
                            and last_lateral_error <= allowed_lateral_m
                            and last_gate_force_delta < self._ft_threshold_n
                            and (last_speed is None or last_speed <= speed_threshold)
                        )
                        if allow_low_force_retry_miss:
                            self._trace_event(
                                "recovery_retry_handoff_gate_miss_allowed",
                                retry_count=retry_count,
                                lateral_error_m=last_lateral_error,
                                allowed_lateral_m=allowed_lateral_m,
                                force_delta_n=last_gate_force_delta,
                                ft_threshold_n=self._ft_threshold_n,
                                tcp_speed_mps=last_speed,
                                speed_threshold_mps=speed_threshold,
                            )
                            gate_passed = True
                    if not gate_passed and last_gate_force_delta >= self._ft_threshold_n:
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
                            retry_count=retry_count,
                        ):
                            self._trace_event("recovery_force_release_failed", retry_count=retry_count)
                            return False
                        retry_count += 1
                        continue
                    if not gate_passed:
                        live_z_repair_default = "false" if self._expert_mode == "nominal" else "true"
                        live_z_repair_enabled = os.environ.get(
                            "AIC_OFFICIAL_TEACHER_ENABLE_LIVE_Z_REPAIR",
                            live_z_repair_default,
                        ).lower() in {"1", "true", "yes", "on"}
                        live_z_repair_max_start = float(
                            os.environ.get(
                                "AIC_OFFICIAL_TEACHER_LIVE_Z_REPAIR_MAX_START_Z_OFFSET_M",
                                "0.035",
                            )
                        )
                        live_z_repair_lateral_threshold = float(
                            os.environ.get("AIC_OFFICIAL_TEACHER_LIVE_Z_REPAIR_MAX_LATERAL_ERROR_M", "0.003")
                        )
                        live_z_repair_force_threshold = float(
                            os.environ.get("AIC_OFFICIAL_TEACHER_LIVE_Z_REPAIR_MAX_FORCE_DELTA_N", "0.7")
                        )
                        if str(getattr(task, "plug_name", "")).startswith("sc"):
                            live_z_repair_enabled = os.environ.get(
                                "AIC_OFFICIAL_TEACHER_SC_ENABLE_LIVE_Z_REPAIR",
                                "true" if live_z_repair_enabled else "false",
                            ).lower() in {"1", "true", "yes", "on"}
                            live_z_repair_max_start = float(
                                os.environ.get(
                                    "AIC_OFFICIAL_TEACHER_SC_LIVE_Z_REPAIR_MAX_START_Z_OFFSET_M",
                                    str(live_z_repair_max_start),
                                )
                            )
                            live_z_repair_lateral_threshold = float(
                                os.environ.get(
                                    "AIC_OFFICIAL_TEACHER_SC_LIVE_Z_REPAIR_MAX_LATERAL_ERROR_M",
                                    str(live_z_repair_lateral_threshold),
                                )
                            )
                            live_z_repair_force_threshold = float(
                                os.environ.get(
                                    "AIC_OFFICIAL_TEACHER_SC_LIVE_Z_REPAIR_MAX_FORCE_DELTA_N",
                                    str(live_z_repair_force_threshold),
                                )
                            )
                        last_lateral_error = (
                            float(self._last_tracking_gate_lateral_error_m)
                            if self._last_tracking_gate_lateral_error_m is not None
                            else None
                        )
                        live_z_repair_allowed = (
                            live_z_repair_enabled
                            and z_offset < live_z_offset <= min(max_live_start_z_offset, live_z_repair_max_start)
                            and (last_lateral_error is None or last_lateral_error <= live_z_repair_lateral_threshold)
                            and last_gate_force_delta <= live_z_repair_force_threshold
                        )
                        if z_offset < live_z_offset <= max_live_start_z_offset and not live_z_repair_allowed:
                            self._trace_event(
                                "cheatcode_handoff_live_z_repair_rejected",
                                retry_count=retry_count,
                                live_z_offset=live_z_offset,
                                nominal_start_z_offset=z_offset,
                                live_z_repair_enabled=live_z_repair_enabled,
                                max_live_start_z_offset=max_live_start_z_offset,
                                live_z_repair_max_start_z_offset=live_z_repair_max_start,
                                lateral_error_m=last_lateral_error,
                                lateral_threshold_m=live_z_repair_lateral_threshold,
                                force_delta_n=last_gate_force_delta,
                                force_threshold_n=live_z_repair_force_threshold,
                            )
                        if live_z_repair_allowed:
                            guarded_start_z_offset = live_z_offset
                            guarded_insertion_distance = max(0.0, live_z_offset - fixed_end_z_offset)
                            preserve_live_lateral = os.environ.get(
                                "AIC_OFFICIAL_TEACHER_PRESERVE_LIVE_LATERAL_ON_Z_REPAIR",
                                "true",
                            ).lower() in {"1", "true", "yes", "on"}
                            if str(getattr(task, "plug_name", "")).startswith("sc"):
                                preserve_live_lateral = os.environ.get(
                                    "AIC_OFFICIAL_TEACHER_SC_PRESERVE_LIVE_LATERAL_ON_Z_REPAIR",
                                    "false",
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
                is_sc_plug = str(getattr(task, "plug_name", "")).startswith("sc")
                if gate_passed and stop_at_near_gate:
                    near_gate_max_lateral_m = float(
                        os.environ.get("AIC_OFFICIAL_TEACHER_NEAR_GATE_MAX_LATERAL_ERROR_M", "0.003")
                    )
                    near_gate_max_axial_text = os.environ.get("AIC_OFFICIAL_TEACHER_NEAR_GATE_MAX_AXIAL_ERROR_M")
                    near_gate_max_axial_m = (
                        float(near_gate_max_axial_text) if near_gate_max_axial_text not in (None, "") else None
                    )
                    near_gate_lateral_error = (
                        float(self._last_tracking_gate_lateral_error_m)
                        if self._last_tracking_gate_lateral_error_m is not None
                        else None
                    )
                    near_gate_axial_error = (
                        float(self._last_tracking_gate_axial_error_m)
                        if self._last_tracking_gate_axial_error_m is not None
                        else None
                    )
                    axial_ok = (
                        near_gate_max_axial_m is None
                        or (near_gate_axial_error is not None and near_gate_axial_error <= near_gate_max_axial_m)
                    )
                    if (
                        near_gate_lateral_error is not None
                        and near_gate_lateral_error <= near_gate_max_lateral_m
                        and axial_ok
                    ):
                        self._trace_event(
                            "near_gate_stop_satisfied",
                            retry_count=retry_count,
                            lateral_error_m=near_gate_lateral_error,
                            max_lateral_error_m=near_gate_max_lateral_m,
                            axial_error_m=near_gate_axial_error,
                            max_axial_error_m=near_gate_max_axial_m,
                            tcp_speed_mps=(
                                float(self._last_tracking_gate_speed_mps)
                                if self._last_tracking_gate_speed_mps is not None
                                else None
                            ),
                            force_delta_n=(
                                float(self._last_tracking_gate_force_delta_n)
                                if self._last_tracking_gate_force_delta_n is not None
                                else None
                            ),
                            z_offset=guarded_start_z_offset,
                        )
                        send_feedback("official_teacher_replay_near_gate_stop")
                        return True
                final_align_sec = (
                    float(os.environ.get("AIC_OFFICIAL_TEACHER_NOMINAL_FINAL_PORT_ALIGN_SEC", "0.0"))
                    if self._expert_mode == "nominal"
                    else 0.0
                )
                if is_sc_plug and self._expert_mode != "nominal":
                    final_align_sec = max(
                        final_align_sec,
                        float(os.environ.get("AIC_OFFICIAL_TEACHER_SC_FINAL_PORT_ALIGN_SEC", "0.0")),
                    )
                if final_align_sec > 0.0:
                    final_align_ok = self._run_port_frame_precontact_align(
                        task,
                        port_transform,
                        get_observation,
                        move_robot,
                        duration_sec=final_align_sec,
                        dt=dt,
                        z_offset=guarded_start_z_offset,
                        target_z_m=None,
                    )
                    if not final_align_ok:
                        send_feedback("official_teacher_replay_final_align_force_abort")
                        self._trace_event(
                            "final_port_align_force_abort",
                            retry_count=retry_count,
                            force_delta_n=(
                                float(self._last_precontact_align_force_delta_n)
                                if self._last_precontact_align_force_delta_n is not None
                                else None
                            ),
                        )
                        return False
                    if is_sc_plug and self._expert_mode != "nominal":
                        active_lateral_offset = self._last_precontact_lateral_offset_base.copy()
                    else:
                        active_lateral_offset = active_lateral_offset + self._last_precontact_lateral_offset_base
                    final_gate_passed = self._hold_preinsert_until_tracking_gate(
                        task,
                        port_transform,
                        get_observation,
                        move_robot,
                        settle_sec=float(
                            os.environ.get(
                                "AIC_OFFICIAL_TEACHER_SC_FINAL_ALIGN_SETTLE_SEC"
                                if is_sc_plug and self._expert_mode != "nominal"
                                else "AIC_OFFICIAL_TEACHER_NOMINAL_FINAL_ALIGN_SETTLE_SEC",
                                "0.2",
                            )
                        ),
                        dt=dt,
                        z_offset=guarded_start_z_offset,
                        preserve_current_z=False,
                        lateral_offset_base=active_lateral_offset,
                    )
                    self._trace_event(
                        "final_port_align_gate_checked",
                        retry_count=retry_count,
                        tracking_gate_passed=final_gate_passed,
                        lateral_offset_base_m=active_lateral_offset.tolist(),
                    )
                    if not final_gate_passed:
                        final_gate_passed = self._nominal_realign_and_recheck_tracking_gate(
                            task,
                            port_transform,
                            get_observation,
                            move_robot,
                            settle_sec=float(
                                os.environ.get(
                                    "AIC_OFFICIAL_TEACHER_SC_FINAL_ALIGN_SETTLE_SEC"
                                    if is_sc_plug and self._expert_mode != "nominal"
                                    else "AIC_OFFICIAL_TEACHER_NOMINAL_FINAL_ALIGN_SETTLE_SEC",
                                    "0.2",
                                )
                            ),
                            dt=dt,
                            z_offset=guarded_start_z_offset,
                            preserve_current_z=False,
                            lateral_offset_base=active_lateral_offset,
                            reason="final_port_align_gate",
                        )
                    if not final_gate_passed:
                        send_feedback("official_teacher_replay_tracking_gate_failed")
                        self._trace_event("rejected_final_port_align_gate")
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
            planned_recovery_intervention_used = False
            self._trace_event(
                "guarded_insert_started",
                retry_count=retry_count,
                insertion_speed_mps=insertion_speed_mps,
                ft_threshold_n=self._ft_threshold_n,
                descent_profile="minimum_jerk",
                cheatcode_z_mode=cheatcode_z_mode,
                insertion_command_mode=insertion_command_mode,
                insertion_start_z_offset=guarded_start_z_offset,
                vertical_bias_m=float(os.environ.get("AIC_OFFICIAL_TEACHER_PREINSERT_VERTICAL_BIAS_M", "0.0")),
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
                nominal_sc_recovery_enabled = (
                    str(getattr(task, "plug_name", "")).startswith("sc")
                    and os.environ.get(
                        "AIC_OFFICIAL_TEACHER_SC_NO_EVENT_RECOVERY",
                        "false",
                    ).lower()
                    in {"1", "true", "yes", "on"}
                )
                if self._expert_mode == "nominal" and not nominal_sc_recovery_enabled:
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
                    retry_count=retry_count,
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
                os.environ.get("AIC_OFFICIAL_TEACHER_GUARDED_INSERT_SPEED_GATE_MPS", "0.018")
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
            is_sc_plug = str(getattr(task, "plug_name", "")).startswith("sc")
            sc_guarded_servo_enabled = (
                is_sc_plug
                and os.environ.get(
                    "AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO",
                    "false",
                ).lower()
                in {"1", "true", "yes", "on"}
                and insertion_command_mode == "exact_position"
            )
            sc_guarded_servo_gain = float(
                os.environ.get("AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO_GAIN", "0.35")
            )
            sc_guarded_servo_step_limit_m = float(
                os.environ.get("AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO_STEP_LIMIT_M", "0.0004")
            )
            sc_guarded_servo_max_bias_m = float(
                os.environ.get("AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO_MAX_BIAS_M", "0.004")
            )
            sc_guarded_servo_deadband_m = float(
                os.environ.get("AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO_DEADBAND_M", "0.0004")
            )
            sc_guarded_servo_force_limit_n = float(
                os.environ.get("AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO_FORCE_LIMIT_N", "5.0")
            )
            sc_guarded_servo_allow_force_relief = (
                os.environ.get(
                    "AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO_ALLOW_FORCE_RELIEF",
                    "false",
                ).lower()
                in {"1", "true", "yes", "on"}
            )
            sc_guarded_servo_force_relief_gain = float(
                os.environ.get("AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO_FORCE_RELIEF_GAIN", "0.5")
            )
            sc_guarded_servo_force_relief_hold_z = (
                os.environ.get(
                    "AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO_FORCE_RELIEF_HOLD_Z",
                    "true",
                ).lower()
                in {"1", "true", "yes", "on"}
            )
            sc_guarded_servo_bias = np.zeros(3, dtype=np.float64)

            def sc_final_seating_probe() -> bool | None:
                if (
                    not is_sc_plug
                    or insertion_command_mode != "exact_position"
                    or os.environ.get(
                        "AIC_OFFICIAL_TEACHER_SC_FINAL_SEAT_ENABLED",
                        "false",
                    ).lower()
                    not in {"1", "true", "yes", "on"}
                ):
                    return None
                duration_sec = float(os.environ.get("AIC_OFFICIAL_TEACHER_SC_FINAL_SEAT_SEC", "2.5"))
                if duration_sec <= 0.0:
                    return None
                radius_m = max(0.0, float(os.environ.get("AIC_OFFICIAL_TEACHER_SC_FINAL_SEAT_RADIUS_M", "0.0008")))
                extra_depth_m = max(
                    0.0,
                    float(os.environ.get("AIC_OFFICIAL_TEACHER_SC_FINAL_SEAT_EXTRA_DEPTH_M", "0.003")),
                )
                cycles = float(os.environ.get("AIC_OFFICIAL_TEACHER_SC_FINAL_SEAT_CYCLES", "2.0"))
                force_limit_n = float(
                    os.environ.get("AIC_OFFICIAL_TEACHER_SC_FINAL_SEAT_FORCE_LIMIT_N", "8.0")
                )
                steps = max(1, int(math.ceil(duration_sec / dt)))
                self._trace_event(
                    "sc_final_seating_probe_started",
                    retry_count=retry_count,
                    duration_sec=duration_sec,
                    steps=steps,
                    radius_m=radius_m,
                    extra_depth_m=extra_depth_m,
                    cycles=cycles,
                    force_limit_n=force_limit_n,
                    start_z_offset=current_z,
                    servo_bias_base_m=sc_guarded_servo_bias.tolist(),
                )
                for step in range(1, steps + 1):
                    if self._task_completed_in_simulation(task):
                        self._trace_event("sc_final_seating_probe_success", retry_count=retry_count)
                        return True
                    force_delta = self._force_delta_norm(get_observation, baseline_force)
                    if force_delta >= force_limit_n:
                        self._trace_event(
                            "sc_final_seating_probe_stopped_force",
                            retry_count=retry_count,
                            force_delta_n=force_delta,
                            force_limit_n=force_limit_n,
                        )
                        return None
                    fraction = self._minimum_jerk_fraction(step / steps)
                    theta = 2.0 * math.pi * cycles * step / steps
                    dither_bias = np.asarray(
                        [radius_m * math.cos(theta), radius_m * math.sin(theta), 0.0],
                        dtype=np.float64,
                    )
                    probe_z = current_z - fraction * extra_depth_m
                    try:
                        if pin_insertion_target:
                            target_xyz = self._pose_position_array(insertion_reference_pose)
                            target_xyz[:2] = target_xyz[:2] + sc_guarded_servo_bias[:2] + dither_bias[:2]
                            target_xyz[2] = float(insertion_reference_pose.position.z) - (
                                guarded_start_z_offset - probe_z
                            )
                            target_pose = self._make_pose(
                                target_xyz,
                                self._pose_quat_wxyz(insertion_reference_pose),
                            )
                        else:
                            target_pose = self._calc_cheatcode_gripper_pose(
                                task,
                                port_transform,
                                z_offset=probe_z,
                                preserve_current_z=(cheatcode_z_mode == "tf_depth"),
                                lateral_offset_base=active_lateral_offset + sc_guarded_servo_bias + dither_bias,
                            )
                        if cheatcode_z_mode == "tf_depth":
                            target_pose.position.z = insertion_start_z - (guarded_start_z_offset - probe_z)
                        self._send_absolute_target(move_robot, target_pose)
                        self._trace_event(
                            "sc_final_seating_probe_step",
                            retry_count=retry_count,
                            step=step,
                            z_offset=probe_z,
                            dither_bias_base_m=dither_bias.tolist(),
                            force_delta_n=force_delta,
                        )
                    except TransformException as ex:
                        self.get_logger().warn(f"SC final seating TF lookup failed: {ex}")
                    self.sleep_for(dt)
                if self._task_completed_in_simulation(task):
                    self._trace_event("sc_final_seating_probe_success", retry_count=retry_count)
                    return True
                self._trace_event("sc_final_seating_probe_completed_no_event", retry_count=retry_count)
                return None

            def hold_low_force_near_insert(
                *,
                force_delta_n: float,
                z_offset: float,
                sample_source: str,
            ) -> None:
                hold_threshold_n = float(
                    os.environ.get(
                        "AIC_OFFICIAL_TEACHER_LOW_FORCE_INSERTION_HOLD_THRESHOLD_N",
                        "5.0",
                    )
                )
                hold_sec = float(
                    os.environ.get(
                        "AIC_OFFICIAL_TEACHER_LOW_FORCE_INSERTION_HOLD_SEC",
                        "0.15",
                    )
                )
                if hold_sec <= 0.0 or force_delta_n < hold_threshold_n:
                    return
                hold_steps = max(1, int(math.ceil(hold_sec / dt)))
                self._trace_event(
                    "low_force_near_insertion_hold_started",
                    retry_count=retry_count,
                    force_delta_n=force_delta_n,
                    hold_threshold_n=hold_threshold_n,
                    hold_sec=hold_sec,
                    hold_steps=hold_steps,
                    z_offset=z_offset,
                    sample_source=sample_source,
                )
                for _ in range(hold_steps):
                    if insertion_command_mode == "exact_position" and last_target_pose is not None:
                        self._send_absolute_target(move_robot, last_target_pose)
                    self.sleep_for(dt)
                release_delta = self._force_delta_norm(get_observation, baseline_force)
                self._trace_event(
                    "low_force_near_insertion_hold_completed",
                    retry_count=retry_count,
                    release_force_delta_n=release_delta,
                    z_offset=z_offset,
                    sample_source=sample_source,
                )

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
                            confirmed_force = max(force_delta, confirmed_delta)
                            if self._should_ignore_shallow_contact(
                                z_offset=current_z,
                                force_delta_n=confirmed_force,
                                retry_count=retry_count,
                                sample_source="during_speed_gate_hold",
                            ):
                                hold_low_force_near_insert(
                                    force_delta_n=confirmed_force,
                                    z_offset=current_z,
                                    sample_source="during_speed_gate_hold",
                                )
                                self.sleep_for(dt)
                                continue
                            self._trace_event(
                                "contact_detected",
                                retry_count=retry_count,
                                force_delta_n=confirmed_force,
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
                commanded_z = current_z
                contact_confirmed = False
                try:
                    if sc_guarded_servo_enabled:
                        plug_transform = self._lookup_active_plug_transform()
                        if plug_transform is not None:
                            force_delta_for_servo = self._force_delta_norm(get_observation, baseline_force)
                            plug_error_base = np.asarray(
                                [
                                    float(port_transform.translation.x - plug_transform.translation.x),
                                    float(port_transform.translation.y - plug_transform.translation.y),
                                    0.0,
                                ],
                                dtype=np.float64,
                            )
                            plug_error_norm = float(np.linalg.norm(plug_error_base[:2]))
                            over_servo_force_limit = force_delta_for_servo > sc_guarded_servo_force_limit_n
                            can_force_relieve = (
                                sc_guarded_servo_allow_force_relief
                                and over_servo_force_limit
                                and plug_error_norm > sc_guarded_servo_deadband_m
                            )
                            if plug_error_norm > sc_guarded_servo_deadband_m and (
                                not over_servo_force_limit or can_force_relieve
                            ):
                                correction_gain = (
                                    sc_guarded_servo_gain * sc_guarded_servo_force_relief_gain
                                    if can_force_relieve
                                    else sc_guarded_servo_gain
                                )
                                correction = correction_gain * plug_error_base
                                correction_norm = float(np.linalg.norm(correction[:2]))
                                if correction_norm > sc_guarded_servo_step_limit_m:
                                    correction *= sc_guarded_servo_step_limit_m / correction_norm
                                proposed_bias = sc_guarded_servo_bias + correction
                                proposed_norm = float(np.linalg.norm(proposed_bias[:2]))
                                if proposed_norm > sc_guarded_servo_max_bias_m:
                                    proposed_bias *= sc_guarded_servo_max_bias_m / proposed_norm
                                sc_guarded_servo_bias = proposed_bias
                                self._trace_event(
                                    (
                                        "sc_guarded_insert_lateral_servo_force_relief_updated"
                                        if can_force_relieve
                                        else "sc_guarded_insert_lateral_servo_updated"
                                    ),
                                    retry_count=retry_count,
                                    z_offset=commanded_z,
                                    plug_tip_lateral_error_m=plug_error_norm,
                                    correction_base_m=correction.tolist(),
                                    servo_bias_base_m=sc_guarded_servo_bias.tolist(),
                                    force_delta_n=force_delta_for_servo,
                                    force_limit_n=sc_guarded_servo_force_limit_n,
                                )
                                if can_force_relieve and sc_guarded_servo_force_relief_hold_z:
                                    commanded_z = guarded_start_z_offset - self._minimum_jerk_fraction(
                                        max(0, descent_step - 1) / insertion_steps
                                    ) * guarded_insertion_distance
                                    self._trace_event(
                                        "sc_guarded_insert_force_relief_z_hold",
                                        retry_count=retry_count,
                                        requested_z_offset=current_z,
                                        held_z_offset=commanded_z,
                                        force_delta_n=force_delta_for_servo,
                                        force_limit_n=sc_guarded_servo_force_limit_n,
                                    )
                            elif plug_error_norm > sc_guarded_servo_deadband_m:
                                self._trace_event(
                                    "sc_guarded_insert_lateral_servo_skipped_force",
                                    retry_count=retry_count,
                                    z_offset=commanded_z,
                                    plug_tip_lateral_error_m=plug_error_norm,
                                    force_delta_n=force_delta_for_servo,
                                    force_limit_n=sc_guarded_servo_force_limit_n,
                                )
                    if pin_insertion_target:
                        target_xyz = self._pose_position_array(insertion_reference_pose)
                        if sc_guarded_servo_enabled:
                            target_xyz[:2] = target_xyz[:2] + sc_guarded_servo_bias[:2]
                        target_xyz[2] = float(insertion_reference_pose.position.z) - (
                            guarded_start_z_offset - commanded_z
                        )
                        target_pose = self._make_pose(
                            target_xyz,
                            self._pose_quat_wxyz(insertion_reference_pose),
                        )
                    else:
                        target_pose = self._calc_cheatcode_gripper_pose(
                            task,
                            port_transform,
                            z_offset=commanded_z,
                            preserve_current_z=(cheatcode_z_mode == "tf_depth"),
                            lateral_offset_base=active_lateral_offset + sc_guarded_servo_bias,
                        )
                    if cheatcode_z_mode == "tf_depth":
                        target_pose.position.z = insertion_start_z - (guarded_start_z_offset - commanded_z)
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
                            target_z_offset=commanded_z,
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
                        confirmed_force = max(force_delta, confirmed_delta)
                        if self._should_ignore_shallow_contact(
                            task=task,
                            z_offset=current_z,
                            force_delta_n=confirmed_force,
                            retry_count=retry_count,
                            sample_source=sample_source,
                        ):
                            hold_low_force_near_insert(
                                force_delta_n=confirmed_force,
                                z_offset=current_z,
                                sample_source=sample_source,
                            )
                            continue
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
                planned_intervention_z_offset = float(
                    os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_PLANNED_INTERVENTION_Z_OFFSET", "0.043")
                )
                should_force_recovery_intervention = (
                    self._expert_mode == "recovery"
                    and retry_count == 0
                    and not planned_recovery_intervention_used
                    and float(np.linalg.norm(recovery_induced_offset[:2])) > 1e-6
                    and current_z <= planned_intervention_z_offset
                    and os.environ.get(
                        "AIC_OFFICIAL_TEACHER_RECOVERY_ENABLE_PLANNED_INTERVENTION",
                        "true",
                    ).lower() in {"1", "true", "yes", "on"}
                )
                if should_force_recovery_intervention:
                    planned_recovery_intervention_used = True
                    force_delta = self._force_delta_norm(get_observation, baseline_force)
                    self._trace_event(
                        "recovery_planned_intervention_started",
                        retry_count=retry_count,
                        force_delta_n=force_delta,
                        threshold_n=self._ft_threshold_n,
                        z_offset=current_z,
                        planned_intervention_z_offset=planned_intervention_z_offset,
                        lateral_offset_base_m=active_lateral_offset.tolist(),
                    )
                    self._trace_event(
                        "contact_detected",
                        retry_count=retry_count,
                        force_delta_n=force_delta,
                        threshold_n=self._ft_threshold_n,
                        z_offset=current_z,
                        sample_source="planned_recovery_intervention",
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
            sc_no_event_recovery_enabled = (
                is_sc_plug
                and os.environ.get(
                    "AIC_OFFICIAL_TEACHER_SC_NO_EVENT_RECOVERY",
                    "false",
                ).lower()
                in {"1", "true", "yes", "on"}
            )
            if sc_no_event_recovery_enabled:
                no_event_force_delta = self._force_delta_norm(get_observation, baseline_force)
                no_event_force_threshold = float(
                    os.environ.get("AIC_OFFICIAL_TEACHER_SC_NO_EVENT_RECOVERY_FORCE_THRESHOLD_N", "5.0")
                )
                no_event_z_threshold = float(
                    os.environ.get("AIC_OFFICIAL_TEACHER_SC_NO_EVENT_RECOVERY_Z_OFFSET_M", "-0.010")
                )
                if no_event_force_delta >= no_event_force_threshold and current_z <= no_event_z_threshold:
                    self._trace_event(
                        "sc_no_event_recovery_triggered",
                        retry_count=retry_count,
                        force_delta_n=no_event_force_delta,
                        force_threshold_n=no_event_force_threshold,
                        z_offset=current_z,
                        z_threshold_m=no_event_z_threshold,
                    )
                    if not recover_after_contact():
                        return False
                    if retry_requested:
                        continue
            break
        final_seating_result = sc_final_seating_probe()
        if final_seating_result is True:
            return True
        if final_seating_result is False:
            return False
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
        self._active_task = task
        self._active_port_transform = None
        self._recovery_context_history.clear()
        self._force_delta_history.clear()

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
                    self._active_task = task
                    self._active_port_transform = port_transform
                    self._run_local_preinsert_align(
                        task,
                        port_transform,
                        move_robot,
                        duration_sec=float(
                            target.waypoint.diagnostics.get(
                                "local_preinsert_align_sec",
                                os.environ.get("AIC_OFFICIAL_TEACHER_LOCAL_PREINSERT_ALIGN_SEC", "0.8"),
                            )
                        ),
                        dt=command_dt_sec,
                        z_offset=float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_START_Z_OFFSET", "0.045")),
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
                                        "AIC_OFFICIAL_TEACHER_CHEATCODE_START_Z_OFFSET", "0.045"
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
                                os.environ.get("AIC_OFFICIAL_TEACHER_PRE_INSERT_SETTLE_SEC", "0.35"),
                            )
                        ),
                        dt=command_dt_sec,
                        z_offset=float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_START_Z_OFFSET", "0.045")),
                        preserve_current_z=preinsert_gate_preserve_current_z,
                    )
                    self._preinsert_settle_done = True
                    if not gate_passed and self._expert_mode == "nominal":
                        gate_passed = self._nominal_realign_and_recheck_tracking_gate(
                            task,
                            port_transform,
                            get_observation,
                            move_robot,
                            settle_sec=float(
                                target.waypoint.diagnostics.get(
                                    "pre_insert_settle_sec",
                                    os.environ.get("AIC_OFFICIAL_TEACHER_PRE_INSERT_SETTLE_SEC", "0.35"),
                                )
                            ),
                            dt=command_dt_sec,
                            z_offset=float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_START_Z_OFFSET", "0.045")),
                            preserve_current_z=preinsert_align_preserve_current_z,
                            reason="trajectory_preinsert_tracking_gate",
                        )
                        if not gate_passed:
                            send_feedback("official_teacher_replay_tracking_gate_failed")
                            self._trace_event("nominal_rejected_tracking_gate")
                            return False
                    if not gate_passed:
                        last_gate_force_delta = (
                            float(self._last_tracking_gate_force_delta_n)
                            if self._last_tracking_gate_force_delta_n is not None
                            else 0.0
                        )
                        if last_gate_force_delta >= self._ft_threshold_n:
                            start_z_offset = float(
                                os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_START_Z_OFFSET", "0.045")
                            )
                            self._trace_event(
                                "contact_detected",
                                retry_count=0,
                                force_delta_n=last_gate_force_delta,
                                threshold_n=self._ft_threshold_n,
                                z_offset=start_z_offset,
                                sample_source="trajectory_preinsert_tracking_gate",
                            )
                            send_feedback("official_teacher_replay_recovery_backoff")
                            if not self._backoff_and_realign(
                                task,
                                port_transform,
                                get_observation,
                                move_robot,
                                current_z_offset=start_z_offset,
                                dt=command_dt_sec,
                                release_baseline_force=self._last_tracking_gate_baseline_force,
                                retry_start_z_m=float(self._current_tcp_pose().position.z),
                                retry_count=0,
                            ):
                                send_feedback("official_teacher_replay_recovery_force_release_failed")
                                self._trace_event("recovery_force_release_failed", retry_count=0)
                                return False
                            self._trace_event("recovery_initial_tracking_gate_backoff_completed")
                            gate_passed = True
                    if not gate_passed:
                        send_feedback("official_teacher_replay_tracking_gate_realign")
                        self._trace_event("recovery_tracking_gate_realign_started")
                        self._run_local_preinsert_align(
                            task,
                            port_transform,
                            move_robot,
                            duration_sec=float(
                                os.environ.get(
                                    "AIC_OFFICIAL_TEACHER_TRACKING_GATE_REALIGN_SEC",
                                    os.environ.get("AIC_OFFICIAL_TEACHER_LOCAL_PREINSERT_ALIGN_SEC", "1.2"),
                                )
                            ),
                            dt=command_dt_sec,
                            z_offset=float(
                                os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_START_Z_OFFSET", "0.045")
                            ),
                            preserve_current_z=preinsert_align_preserve_current_z,
                            gain_profile="recovery",
                        )
                        gate_passed = self._hold_preinsert_until_tracking_gate(
                            task,
                            port_transform,
                            get_observation,
                            move_robot,
                            settle_sec=float(
                                target.waypoint.diagnostics.get(
                                    "pre_insert_settle_sec",
                                    os.environ.get("AIC_OFFICIAL_TEACHER_PRE_INSERT_SETTLE_SEC", "0.35"),
                                )
                            ),
                            dt=command_dt_sec,
                            z_offset=float(
                                os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_START_Z_OFFSET", "0.045")
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
                        stiffness=self._cartesian_stiffness,
                        damping=self._cartesian_damping,
                    )
                except TransformException as ex:
                    self.get_logger().warn(
                        "TF lookup failed for relative replay; falling back to "
                        f"absolute base_link pose for this tick: {ex}"
                    )
                    self._send_absolute_target(move_robot, pose)
            else:
                if self._action_mode == "joint_position_then_cheatcode":
                    self._send_joint_target(move_robot, target)
                else:
                    self._send_absolute_target(move_robot, pose)

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
                                stiffness=self._cartesian_stiffness,
                                damping=self._cartesian_damping,
                            )
                        except TransformException:
                            self._send_absolute_target(move_robot, pose)
                    else:
                        if self._action_mode == "joint_position_then_cheatcode":
                            self._send_joint_target(move_robot, target)
                        else:
                            self._send_absolute_target(move_robot, pose)
                    self.sleep_for(command_dt_sec)
                send_feedback("official_teacher_replay_finished")
                return True
            self.sleep_for(command_dt_sec)
