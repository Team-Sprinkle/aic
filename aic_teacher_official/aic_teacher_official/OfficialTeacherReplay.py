"""Official AIC policy wrapper for replaying a smooth teacher trajectory.

This module intentionally does not import any VLM/planner backends. Replay must
be deterministic and local to the official ROS/Gazebo execution.
"""

from __future__ import annotations

import os

import numpy as np
from aic_model.policy import (
    compute_delta_pose,
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

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
            os.environ.get("AIC_OFFICIAL_TEACHER_ONLINE_CHEATCODE_INSERTION", "false").lower()
            == "true"
        )
        if self._action_mode not in {
            "relative_delta_gripper_tcp",
            "absolute_cartesian_pose_base_link",
        }:
            raise RuntimeError(
                "AIC_OFFICIAL_TEACHER_ACTION_MODE must be one of: "
                "relative_delta_gripper_tcp, absolute_cartesian_pose_base_link"
            )
        self.get_logger().info(
            "Loaded official teacher replay trajectory: "
            f"{self._trajectory_path}; action_mode={self._action_mode}"
        )
        self._latest_insertion_event_namespace = ""
        self._insertion_event_sub = None
        if hasattr(self._parent_node, "create_subscription"):
            self._insertion_event_sub = self._parent_node.create_subscription(
                String,
                "/scoring/insertion_event",
                self._insertion_event_callback,
                10,
            )

    def _insertion_event_callback(self, msg: String) -> None:
        self._latest_insertion_event_namespace = msg.data.strip().strip("/")

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
        target_xyz = np.asarray(
            [
                port_transform.translation.x,
                port_transform.translation.y,
                port_transform.translation.z + z_offset - plug_tip_gripper_offset[2],
            ],
            dtype=np.float64,
        )
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

    def _send_relative_target(self, move_robot: MoveRobotCallback, target_pose: Pose) -> None:
        self.set_delta_pose_target(
            move_robot=move_robot,
            delta_pose=compute_delta_pose(self._current_tcp_pose(), target_pose),
            frame_id="gripper/tcp",
        )

    def _run_online_cheatcode_insertion(
        self,
        task: Task,
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
        interpolation_duration_sec = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_INTERPOLATION_SEC", "5.5")
        )
        dt = float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_DT", "0.05"))
        steps = max(1, int(interpolation_duration_sec / dt))
        for step in range(steps + 1):
            if self._task_completed_in_simulation(task):
                self.get_logger().info(
                    "Online CheatCode insertion observed insertion event during alignment."
                )
                return True
            fraction = step / steps
            try:
                target_pose = self._calc_cheatcode_gripper_pose(
                    task,
                    port_transform,
                    slerp_fraction=fraction,
                    position_fraction=fraction,
                    z_offset=0.2,
                )
                self._send_relative_target(move_robot, target_pose)
            except TransformException as ex:
                self.get_logger().warn(f"Online CheatCode alignment TF lookup failed: {ex}")
            self.sleep_for(dt)

        z_offset = float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_START_Z_OFFSET", "0.03"))
        end_z_offset = float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_END_Z_OFFSET", "-0.015"))
        dz = float(os.environ.get("AIC_OFFICIAL_TEACHER_CHEATCODE_DZ", "0.0005"))
        while z_offset >= end_z_offset:
            if self._task_completed_in_simulation(task):
                self.get_logger().info("Online CheatCode insertion observed insertion event.")
                return True
            z_offset -= dz
            try:
                target_pose = self._calc_cheatcode_gripper_pose(
                    task,
                    port_transform,
                    z_offset=z_offset,
                )
                self._send_relative_target(move_robot, target_pose)
            except TransformException as ex:
                self.get_logger().warn(f"Online CheatCode TF lookup failed: {ex}")
            self.sleep_for(dt)
        wait_started = self.time_now()
        wait_timeout = Duration(
            seconds=float(os.environ.get("AIC_OFFICIAL_TEACHER_FINAL_HOLD_SEC", "5.0"))
        )
        while (self.time_now() - wait_started) < wait_timeout:
            if self._task_completed_in_simulation(task):
                self.get_logger().info("Online CheatCode insertion event observed during hold.")
                return True
            self.sleep_for(dt)
        return True

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
            if self._action_mode == "relative_delta_gripper_tcp":
                try:
                    if (
                        self._online_cheatcode_final_insertion
                        and target.waypoint.phase == "final_insertion"
                    ):
                        return self._run_online_cheatcode_insertion(
                            task,
                            move_robot,
                            send_feedback,
                        )
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
                        self.set_pose_target(move_robot=move_robot, pose=pose, frame_id="base_link")
                    self.sleep_for(command_dt_sec)
                send_feedback("official_teacher_replay_finished")
                return True
            self.sleep_for(command_dt_sec)
