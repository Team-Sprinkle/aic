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

import os
from pathlib import Path
import re

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

QuaternionTuple = tuple[float, float, float, float]


class CheatCodePIDController(Policy):
    def __init__(self, parent_node):
        self.pid_x = PIDController(kp=17.0, ki=0.08, kd=2.0)
        self.pid_y = PIDController(kp=17.0, ki=0.08, kd=2.0)
        self.xy_alignment_tolerance_m = 0.01
        self.xy_alignment_stable_cycles = 5
        self._task = None
        self._latest_insertion_event_namespace = ""
        self._plot_output_dir = self._resolve_plot_output_dir()
        super().__init__(parent_node)

        # --- NEW: Initialize tracking attributes ---
        self.history_time = []
        self.history_err_x = []
        self.history_err_y = []
        self.start_time = None

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

    def _safe_plot_errors(self, success: bool = False):
        """Call plot_errors() without letting exceptions crash the caller."""
        try:
            self.plot_errors(success=success)
        except Exception as ex:
            self.get_logger().error(f"Failed to save PID telemetry plot: {ex}")

    def plot_errors(self, success: bool = False):
        """Helper to visualize PID performance."""
        if not self.history_time:
            self.get_logger().warn("No history recorded to plot.")
            return

        plt.figure(figsize=(10, 6))
        err_x_mm = [e * 1e3 for e in self.history_err_x]
        err_y_mm = [e * 1e3 for e in self.history_err_y]

        plt.plot(
            self.history_time,
            err_x_mm,
            label="Error X (mm)",
            color="red",
            linewidth=1.5,
        )
        plt.plot(
            self.history_time,
            err_y_mm,
            label="Error Y (mm)",
            color="blue",
            linewidth=1.5,
        )

        final_x = err_x_mm[-1]
        final_y = err_y_mm[-1]

        plt.axhline(0, color="black", linestyle="--", alpha=0.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Error (mm)")

        task_label = self._task.target_module_name if self._task else "Task"
        status = "SUCCESS" if success else "FAILURE"
        plt.title(
            f"PID Performance: {task_label} [{status}]\n"
            f"X: Kp={self.pid_x.kp} Ki={self.pid_x.ki} Kd={self.pid_x.kd}  |  "
            f"Y: Kp={self.pid_y.kp} Ki={self.pid_y.ki} Kd={self.pid_y.kd}\n"
            f"Final Error: X={final_x:.3f} mm, Y={final_y:.3f} mm"
        )

        if hasattr(self, "_pre_descent_index") and self._pre_descent_index < len(
            self.history_time
        ):
            plt.axvline(
                self.history_time[self._pre_descent_index],
                color="green",
                linestyle=":",
                label="Descent Start",
            )

        plt.legend(loc="upper right")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)

        plt.annotate(
            f"FINAL ERROR\nX: {final_x:.3f} mm\nY: {final_y:.3f} mm",
            xy=(self.history_time[-1], (final_x + final_y) / 2),
            xytext=(10, 20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.3),
        )

        plt.tight_layout()

        # Save to file (recommended for ROS/Sim environments)
        plot_path = self._build_plot_path()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        self.get_logger().info(f"Telemetry plot saved as {plot_path}")
        # plt.show() # Uncomment if you have a display/GUI

    def _resolve_plot_output_dir(self) -> Path:
        configured_dir = os.environ.get("AIC_PID_TUNING_PLOTS_DIR")
        if configured_dir:
            return Path(configured_dir).expanduser()
        return Path.cwd() / "outputs" / "pid_tuning_plots"

    def _build_plot_path(self) -> Path:
        task_name = "task"
        if self._task is not None:
            task_name = f"{self._task.target_module_name}_{self._task.port_name}"
        safe_task_name = re.sub(r"[^A-Za-z0-9._-]+", "_", task_name).strip("_")
        if not safe_task_name:
            safe_task_name = "task"
        timestamp_ns = self.time_now().nanoseconds
        return self._plot_output_dir / f"{safe_task_name}_{timestamp_ns}.png"

    def _record_telemetry(self):
        """Records current PID errors and time for plotting."""
        if self.start_time is None:
            self.start_time = self.time_now().nanoseconds / 1e9
        current_time = (self.time_now().nanoseconds / 1e9) - self.start_time
        self.history_time.append(current_time)
        self.history_err_x.append(self.pid_x.last_error)
        self.history_err_y.append(self.pid_y.last_error)

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

        self._record_telemetry()

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

        # Reset telemetry and PIDs
        self.history_time = []
        self.history_err_x = []
        self.history_err_y = []
        self.start_time = None
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

        duration = 5.5
        dt = 0.05
        steps = int(duration / dt)

        for t in range(steps):
            if self._task_completed_in_simulation(task):
                self.get_logger().info(
                    "[CheatCodePID] Early exit: simulation reported task completion."
                )
                self._safe_plot_errors(success=True)
                return True
            interp_fraction = t / float(steps)
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(
                        port_transform,
                        slerp_fraction=interp_fraction,
                        position_fraction=interp_fraction,
                        z_offset=z_offset,
                        reset_pids=False,
                        dt=dt,
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during interpolation: {ex}")
            self.sleep_for(0.05)

        self._pre_descent_index = len(self.history_err_x) - 1
        self.get_logger().info(
            f"[CheatCodePID] Pre-descent XY error: "
            f"ErrX={self.pid_x.last_error * 1e3:.3f} mm, "
            f"ErrY={self.pid_y.last_error * 1e3:.3f} mm"
        )

        # Descend until the cable is inserted into the port.
        # aligned_cycles = 0
        while True:
            if self._task_completed_in_simulation(task):
                self.get_logger().info(
                    "[CheatCodePID] Early exit: simulation reported task completion."
                )
                self._safe_plot_errors(success=True)
                return True
            if z_offset < -0.015:
                break

            z_offset -= 0.0005
            self.get_logger().info(f"z_offset: {z_offset:0.5}")
            try:
                pose = self.calc_gripper_pose(port_transform, z_offset=z_offset)
                self.set_pose_target(move_robot=move_robot, pose=pose)
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during insertion: {ex}")
                self.sleep_for(0.05)
                continue

            # if self._xy_error_is_aligned():
            #     aligned_cycles += 1
            # else:
            #     aligned_cycles = 0

            # if aligned_cycles >= self.xy_alignment_stable_cycles:
            #     z_offset -= 0.0005
            #     aligned_cycles = 0
            #     self.get_logger().info(
            #         f"[CheatCodePID] XY aligned within "
            #         f"{self.xy_alignment_tolerance_m * 1000.0:0.1f} mm; "
            #         f"advancing z_offset to {z_offset:0.5f}"
            #     )
            # else:
            #     self.get_logger().info(
            #         f"[CheatCodePID] Holding z_offset {z_offset:0.5f} until XY error "
            #         f"is <= {self.xy_alignment_tolerance_m * 1000.0:0.1f} mm for "
            #         f"{self.xy_alignment_stable_cycles} cycles "
            #         f"(current: {aligned_cycles}/{self.xy_alignment_stable_cycles})"
            #     )
            # self.sleep_for(0.05)

        self.get_logger().info("Waiting briefly for insertion event...")
        wait_started = self.time_now()
        wait_timeout = Duration(seconds=5.0)
        while (self.time_now() - wait_started) < wait_timeout:
            if self._task_completed_in_simulation(task):
                self.get_logger().info(
                    "[CheatCodePID] Insertion event observed before timeout."
                )
                self._safe_plot_errors(success=True)
                return True
            self.sleep_for(0.05)

        self.get_logger().info("CheatCodePIDController.insert_cable() exiting...")

        self._safe_plot_errors()

        return False


CheatCodePidController = CheatCodePIDController
