#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


import numpy as np

from aic_example_policies.controllers import PIDController
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

QuaternionTuple = tuple[float, float, float, float]


class CheatCodePIDController(Policy):
    def __init__(self, parent_node):
        self.pid_x = PIDController(kp=0.8, ki=0.0, kd=0.0)
        self.pid_y = PIDController(kp=0.8, ki=0.0, kd=0.0)
        self.xy_alignment_tolerance_m = 0.01
        self.xy_alignment_stable_cycles = 5
        self._task = None
        self._latest_insertion_event_namespace = ""
        super().__init__(parent_node)
        self._insertion_event_sub = self._parent_node.create_subscription(
            String,
            "/scoring/insertion_event",
            self._insertion_event_callback,
            10,
        )

    def _insertion_event_callback(self, msg: String) -> None:
        self._latest_insertion_event_namespace = msg.data.strip().strip("/")
        self.get_logger().info(
            f"Received insertion event for namespace: '{self._latest_insertion_event_namespace}'"
        )

    def _task_completed_in_simulation(self, task: Task) -> bool:
        namespace = self._latest_insertion_event_namespace
        if not namespace:
            return False
        tokens = [token for token in namespace.split("/") if token]
        if len(tokens) < 2:
            return False
        return tokens[-2] == task.target_module_name and tokens[-1] == task.port_name

    def _wait_for_tf(
        self, target_frame: str, source_frame: str, timeout_sec: float = 10.0
    ) -> bool:
        """Wait for a TF frame to become available."""
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                )
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"Waiting for transform '{source_frame}' -> '{target_frame}'... -- are you running eval with `ground_truth:=true`?"
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().error(
            f"Transform '{source_frame}' not available after {timeout_sec}s"
        )
        return False

    def calc_gripper_pose(
        self,
        port_transform: Transform,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        z_offset: float = 0.1,
        reset_pids: bool = False,
        dt: float = 0.05,
    ) -> Pose:
        """Find the gripper pose that results in plug alignment."""
        if reset_pids:
            self.pid_x.reset()
            self.pid_y.reset()

        q_port = (
            port_transform.rotation.w,
            port_transform.rotation.x,
            port_transform.rotation.y,
            port_transform.rotation.z,
        )
        plug_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            f"{self._task.cable_name}/{self._task.plug_name}_link",
            Time(),
        )
        q_plug = (
            plug_tf_stamped.transform.rotation.w,
            plug_tf_stamped.transform.rotation.x,
            plug_tf_stamped.transform.rotation.y,
            plug_tf_stamped.transform.rotation.z,
        )
        q_plug_inv = (
            -q_plug[0],
            q_plug[1],
            q_plug[2],
            q_plug[3],
        )
        q_diff = quaternion_multiply(q_port, q_plug_inv)
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
        q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_gripper_slerp = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)

        gripper_xyz = (
            gripper_tf_stamped.transform.translation.x,
            gripper_tf_stamped.transform.translation.y,
            gripper_tf_stamped.transform.translation.z,
        )
        port_xy = (
            port_transform.translation.x,
            port_transform.translation.y,
        )
        plug_xyz = (
            plug_tf_stamped.transform.translation.x,
            plug_tf_stamped.transform.translation.y,
            plug_tf_stamped.transform.translation.z,
        )

        plug_tip_gripper_offset = (
            gripper_xyz[0] - plug_xyz[0],
            gripper_xyz[1] - plug_xyz[1],
            gripper_xyz[2] - plug_xyz[2],
        )

        adj_x = self.pid_x.update(setpoint=port_xy[0], measurement=plug_xyz[0], dt=dt)
        adj_y = self.pid_y.update(setpoint=port_xy[1], measurement=plug_xyz[1], dt=dt)

        self.get_logger().info(
            f"[PID Log] Z-Offset: {z_offset:0.3f} | "
            f"ErrX: {self.pid_x.last_error:0.4f} (Adj: {adj_x:0.4f}) | "
            f"ErrY: {self.pid_y.last_error:0.4f} (Adj: {adj_y:0.4f})"
        )

        target_x = port_xy[0] + adj_x
        target_y = port_xy[1] + adj_y
        target_z = port_transform.translation.z + z_offset - plug_tip_gripper_offset[2]

        blend_xyz = (
            position_fraction * target_x + (1.0 - position_fraction) * gripper_xyz[0],
            position_fraction * target_y + (1.0 - position_fraction) * gripper_xyz[1],
            position_fraction * target_z + (1.0 - position_fraction) * gripper_xyz[2],
        )

        return Pose(
            position=Point(
                x=blend_xyz[0],
                y=blend_xyz[1],
                z=blend_xyz[2],
            ),
            orientation=Quaternion(
                w=q_gripper_slerp[0],
                x=q_gripper_slerp[1],
                y=q_gripper_slerp[2],
                z=q_gripper_slerp[3],
            ),
        )

    def _xy_error_is_aligned(self) -> bool:
        return (
            abs(self.pid_x.last_error) <= self.xy_alignment_tolerance_m
            and abs(self.pid_y.last_error) <= self.xy_alignment_tolerance_m
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"CheatCodePIDController.insert_cable() task: {task}")
        self._task = task
        self._latest_insertion_event_namespace = ""
        self.pid_x.reset()
        self.pid_y.reset()

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

        # Wait for both the port and cable tip TFs to become available.
        # These come via ground_truth and may not be immediate.
        for frame in [port_frame, cable_tip_frame]:
            if not self._wait_for_tf("base_link", frame):
                return False

        try:
            port_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                port_frame,
                Time(),
            )
        except TransformException as ex:
            self.get_logger().error(f"Could not look up port transform: {ex}")
            return False
        port_transform = port_tf_stamped.transform

        z_offset = 0.2

        duration = 5.0  # 20HZ TODO: check what good frequency should be
        dt = 0.05
        steps = int(duration / dt)

        for t in range(steps):
            if self._task_completed_in_simulation(task):
                self.get_logger().info(
                    "[CheatCodePID] Early exit: simulation reported task completion."
                )
                return True
            interp_fraction = t / float(steps)
            should_reset = t == 0
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(
                        port_transform,
                        slerp_fraction=interp_fraction,
                        position_fraction=interp_fraction,
                        z_offset=z_offset,
                        reset_pids=should_reset,
                        dt=dt,
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during interpolation: {ex}")
            self.sleep_for(0.05)

        # Descend until the cable is inserted into the port.
        aligned_cycles = 0
        while True:
            if self._task_completed_in_simulation(task):
                self.get_logger().info(
                    "[CheatCodePID] Early exit: simulation reported task completion."
                )
                return True
            if z_offset < -0.015:
                break

            try:
                pose = self.calc_gripper_pose(port_transform, z_offset=z_offset)
                self.set_pose_target(move_robot=move_robot, pose=pose)
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during insertion: {ex}")
                aligned_cycles = 0
                self.sleep_for(0.05)
                continue

            if self._xy_error_is_aligned():
                aligned_cycles += 1
            else:
                aligned_cycles = 0

            if aligned_cycles >= self.xy_alignment_stable_cycles:
                z_offset -= 0.0005
                aligned_cycles = 0
                self.get_logger().info(
                    f"[CheatCodePID] XY aligned within "
                    f"{self.xy_alignment_tolerance_m * 1000.0:0.1f} mm; "
                    f"advancing z_offset to {z_offset:0.5f}"
                )
            else:
                self.get_logger().info(
                    f"[CheatCodePID] Holding z_offset {z_offset:0.5f} until XY error "
                    f"is <= {self.xy_alignment_tolerance_m * 1000.0:0.1f} mm for "
                    f"{self.xy_alignment_stable_cycles} cycles "
                    f"(current: {aligned_cycles}/{self.xy_alignment_stable_cycles})"
                )
            self.sleep_for(0.05)

        self.get_logger().info("Waiting briefly for insertion event...")
        wait_started = self.time_now()
        wait_timeout = Duration(seconds=5.0)
        while (self.time_now() - wait_started) < wait_timeout:
            if self._task_completed_in_simulation(task):
                self.get_logger().info(
                    "[CheatCodePID] Insertion event observed before timeout."
                )
                return True
            self.sleep_for(0.05)

        self.get_logger().info("CheatCodePIDController.insert_cable() exiting...")
        return False


CheatCodePidController = CheatCodePIDController
