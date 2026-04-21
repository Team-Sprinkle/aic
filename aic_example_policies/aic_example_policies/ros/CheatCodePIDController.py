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
from geometry_msgs.msg import Point, Pose, Quaternion, Transform, Vector3
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

QuaternionTuple = tuple[float, float, float, float]


class CheatCodePIDController(Policy):
    def __init__(self, parent_node):
        self.pid_x = PIDController(kp=17.0, ki=0.08, kd=2.0)
        self.pid_y = PIDController(kp=17.0, ki=0.08, kd=2.0)
        self.xy_alignment_tolerance_m = 0.01
        self.xy_alignment_stable_cycles = 5
        self._task = None
        self._latest_insertion_event_namespace = ""
        self._pid_plots_enabled = (
            os.environ.get("AIC_PID_TUNING_PLOTS", "false").lower() == "true"
        )
        self._plot_output_dir = self._resolve_plot_output_dir()
        self._collision_injection_enabled = (
            os.environ.get("AIC_PID_COLLISION_INJECTION", "true").lower()
            not in ("0", "false", "no", "off")
        )
        self._collision_injection_start_z_offset_m = float(
            os.environ.get("AIC_PID_COLLISION_INJECTION_START_Z_OFFSET_M", "0.2")
        )
        self._collision_injection_xy_bias_m = float(
            os.environ.get("AIC_PID_COLLISION_INJECTION_XY_BIAS_M", "0.025")
        )
        self._collision_injection_xy_noise_m = float(
            os.environ.get("AIC_PID_COLLISION_INJECTION_XY_NOISE_M", "0.003")
        )
        self._collision_injection_ramp_duration_sec = float(
            os.environ.get("AIC_PID_COLLISION_INJECTION_RAMP_DURATION_SEC", "2.0")
        )
        self._collision_injection_seed = int(
            os.environ.get("AIC_PID_COLLISION_INJECTION_SEED", "7")
        )
        self._collision_injection_rng = np.random.default_rng(
            self._collision_injection_seed
        )
        self._collision_injection_target_xy_offset_m = None
        self._collision_injection_start_time_sec = None
        self._collision_injection_logged = False
        self._force_log_interval_sec = float(
            os.environ.get("AIC_PID_FORCE_LOG_INTERVAL_SEC", "0.25")
        )
        self._last_force_log_time_sec = None
        if self._pid_plots_enabled and not _HAS_MATPLOTLIB:
            raise RuntimeError(
                "PID tuning plots are enabled but matplotlib is not installed. "
                "Hint: pixi add matplotlib"
            )
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
        self._collision_force_threshold_n = 20.0
        self._collision_duration_threshold_sec = 1.0
        self._collision_detected = False
        # Contact/collision guideline:
        # - Soft contact: sustained tared force above the low contact threshold,
        #   below the scoring collision threshold. Useful for early "something is
        #   touching" diagnostics, but not sufficient to prove blocked descent.
        # - Lateral contact: sustained lateral force or high lateral-force ratio.
        #   This catches off-axis binding that may be visually obvious before the
        #   total force crosses 20 N.
        # - Blocked descent: the policy keeps commanding downward z_offset
        #   motion, but the plug-to-port Z progress stalls while contact force is
        #   present. This is the best diagnostic for "robot is naively pushing
        #   while blocked" even when total force is below 20 N.
        # - Insertion collision: scoring guideline only, tared |F| > 20 N for
        #   > 1 s. Keep this separate from contact diagnostics.
        self._contact_force_threshold_n = float(
            os.environ.get("AIC_PID_CONTACT_FORCE_THRESHOLD_N", "5.0")
        )
        self._contact_duration_threshold_sec = float(
            os.environ.get("AIC_PID_CONTACT_DURATION_THRESHOLD_SEC", "1.0")
        )
        self._contact_detected = False
        self._lateral_contact_force_threshold_n = float(
            os.environ.get("AIC_PID_LATERAL_CONTACT_FORCE_THRESHOLD_N", "10.0")
        )
        self._lateral_contact_ratio_threshold = float(
            os.environ.get("AIC_PID_LATERAL_CONTACT_RATIO_THRESHOLD", "0.65")
        )
        self._lateral_contact_duration_threshold_sec = float(
            os.environ.get("AIC_PID_LATERAL_CONTACT_DURATION_THRESHOLD_SEC", "0.3")
        )
        self._lateral_contact_detected = False
        self._lateral_contact_sec = 0.0
        self._blocked_descent_duration_threshold_sec = float(
            os.environ.get("AIC_PID_BLOCKED_DESCENT_DURATION_THRESHOLD_SEC", "0.5")
        )
        self._blocked_descent_progress_threshold_m = float(
            os.environ.get("AIC_PID_BLOCKED_DESCENT_PROGRESS_THRESHOLD_M", "0.0001")
        )
        self._blocked_descent_detected = False
        self._blocked_descent_sec = 0.0
        self._last_blocked_descent_check_time_sec = None
        self._last_plug_port_z_m = None
        self._latest_force_mag_n = 0.0
        self._latest_lateral_force_mag_n = 0.0
        self._latest_lateral_force_ratio = 0.0
        self._max_force_mag_n = 0.0
        self._force_above_threshold_sec = 0.0
        self._contact_force_sec = 0.0
        self._last_force_sample_time_sec = None
        self._obs_sub = self._parent_node.create_subscription(
            Observation,
            "/observations",
            self._observation_callback,
            qos_profile_sensor_data,
        )

    def _insertion_event_callback(self, msg: String) -> None:
        self._latest_insertion_event_namespace = msg.data.strip().strip("/")
        self.get_logger().info(
            f"Received insertion event for namespace: '{self._latest_insertion_event_namespace}'"
        )

    def _observation_callback(self, msg: Observation) -> None:
        raw_force = msg.wrist_wrench.wrench.force
        tare = msg.controller_state.fts_tare_offset.wrench.force
        tared_force = np.array(
            [
                raw_force.x - tare.x,
                raw_force.y - tare.y,
                raw_force.z - tare.z,
            ]
        )
        self._latest_force_mag_n = float(np.linalg.norm(tared_force))
        self._latest_lateral_force_mag_n = float(np.linalg.norm(tared_force[:2]))
        self._latest_lateral_force_ratio = self._latest_lateral_force_mag_n / max(
            self._latest_force_mag_n, 1e-6
        )
        self._max_force_mag_n = max(self._max_force_mag_n, self._latest_force_mag_n)

        stamp = msg.wrist_wrench.header.stamp
        sample_time_sec = stamp.sec + stamp.nanosec / 1e9
        if sample_time_sec <= 0.0:
            sample_time_sec = self.time_now().nanoseconds / 1e9

        dt = 0.0
        if self._last_force_sample_time_sec is not None:
            dt = max(0.0, sample_time_sec - self._last_force_sample_time_sec)
        self._last_force_sample_time_sec = sample_time_sec

        if self._latest_force_mag_n > self._collision_force_threshold_n:
            self._force_above_threshold_sec += dt
        elif self._latest_force_mag_n >= self._contact_force_threshold_n:
            self._contact_force_sec += dt

        lateral_contact_active = (
            self._latest_lateral_force_mag_n
            >= self._lateral_contact_force_threshold_n
            or (
                self._latest_force_mag_n >= self._contact_force_threshold_n
                and self._latest_lateral_force_ratio
                >= self._lateral_contact_ratio_threshold
            )
        )
        if lateral_contact_active:
            self._lateral_contact_sec += dt

        if (
            self._force_log_interval_sec > 0.0
            and (
                self._last_force_log_time_sec is None
                or sample_time_sec - self._last_force_log_time_sec
                >= self._force_log_interval_sec
            )
        ):
            self.get_logger().info(
                "Tared force: "
                f"|F|={self._latest_force_mag_n:.2f} N, "
                f"Fx={tared_force[0]:.2f} N, "
                f"Fy={tared_force[1]:.2f} N, "
                f"Fz={tared_force[2]:.2f} N, "
                f"Fxy={self._latest_lateral_force_mag_n:.2f} N, "
                f"lateral_ratio={self._latest_lateral_force_ratio:.2f}, "
                f"contact_time={self._contact_force_sec:.2f} s, "
                f"lateral_contact_time={self._lateral_contact_sec:.2f} s, "
                f"blocked_descent_time={self._blocked_descent_sec:.2f} s, "
                f"time_above_20N={self._force_above_threshold_sec:.2f} s"
            )
            self._last_force_log_time_sec = sample_time_sec

        if (
            not self._contact_detected
            and self._contact_force_sec > self._contact_duration_threshold_sec
        ):
            self.get_logger().warn(
                "Contact detected: sustained tared force between "
                f"{self._contact_force_threshold_n:.1f} N and "
                f"{self._collision_force_threshold_n:.1f} N for "
                f"{self._contact_force_sec:.2f} seconds. "
                "This is contact or blocked descent, not a scoring insertion collision."
            )
            self._contact_detected = True

        if (
            not self._lateral_contact_detected
            and self._lateral_contact_sec
            > self._lateral_contact_duration_threshold_sec
        ):
            self.get_logger().warn(
                "Lateral contact detected: sustained off-axis tared force. "
                f"Fxy={self._latest_lateral_force_mag_n:.2f} N, "
                f"|F|={self._latest_force_mag_n:.2f} N, "
                f"lateral_ratio={self._latest_lateral_force_ratio:.2f}, "
                f"duration={self._lateral_contact_sec:.2f} s. "
                "This indicates binding/contact even before scoring collision."
            )
            self._lateral_contact_detected = True

        if (
            not self._collision_detected
            and self._force_above_threshold_sec > self._collision_duration_threshold_sec
        ):
            self.get_logger().warn(
                "Collision detected: insertion force above "
                f"{self._collision_force_threshold_n:.1f} N for "
                f"{self._force_above_threshold_sec:.2f} seconds. "
                f"Max tared force: {self._max_force_mag_n:.2f} N"
            )
            self._collision_detected = True

    def _task_completed_in_simulation(self, task: Task) -> bool:
        namespace = self._latest_insertion_event_namespace
        if not namespace:
            return False
        tokens = [token for token in namespace.split("/") if token]
        if len(tokens) < 2:
            return False
        return tokens[-2] == task.target_module_name and tokens[-1] == task.port_name

    def _build_collision_test_port_transform(
        self, port_transform: Transform, z_offset: float
    ) -> Transform:
        """Artificially add smooth XY target noise during descent to create collision."""
        if (
            not self._collision_injection_enabled
            or z_offset > self._collision_injection_start_z_offset_m
        ):
            return port_transform

        if self._collision_injection_target_xy_offset_m is None:
            bias = self._collision_injection_xy_bias_m * np.array([1.0, 0.6])
            noise = self._collision_injection_rng.normal(
                loc=0.0, scale=self._collision_injection_xy_noise_m, size=2
            )
            self._collision_injection_target_xy_offset_m = bias + noise
            self._collision_injection_start_time_sec = (
                self.time_now().nanoseconds / 1e9
            )

        current_time_sec = self.time_now().nanoseconds / 1e9
        elapsed_sec = current_time_sec - self._collision_injection_start_time_sec
        if self._collision_injection_ramp_duration_sec <= 0.0:
            ramp_fraction = 1.0
        else:
            ramp_fraction = min(
                max(elapsed_sec / self._collision_injection_ramp_duration_sec, 0.0),
                1.0,
            )
        smooth_fraction = 3.0 * ramp_fraction**2 - 2.0 * ramp_fraction**3
        xy_offset = self._collision_injection_target_xy_offset_m * smooth_fraction

        if not self._collision_injection_logged:
            self.get_logger().warn(
                "Injecting smooth descent XY drift to induce insertion collision: "
                f"target_offset=({self._collision_injection_target_xy_offset_m[0] * 1e3:.1f}, "
                f"{self._collision_injection_target_xy_offset_m[1] * 1e3:.1f}) mm, "
                f"ramp_duration={self._collision_injection_ramp_duration_sec:.2f} s, "
                f"start_z_offset={self._collision_injection_start_z_offset_m:.3f} m"
            )
            self._collision_injection_logged = True

        return Transform(
            translation=Vector3(
                x=port_transform.translation.x + float(xy_offset[0]),
                y=port_transform.translation.y + float(xy_offset[1]),
                z=port_transform.translation.z,
            ),
            rotation=port_transform.rotation,
        )

    def _update_blocked_descent_detection(
        self, port_transform: Transform, cable_tip_frame: str
    ) -> None:
        try:
            plug_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                cable_tip_frame,
                Time(),
            )
        except TransformException:
            return

        current_time_sec = self.time_now().nanoseconds / 1e9
        plug_port_z_m = (
            plug_tf_stamped.transform.translation.z - port_transform.translation.z
        )

        if (
            self._last_plug_port_z_m is None
            or self._last_blocked_descent_check_time_sec is None
        ):
            self._last_plug_port_z_m = plug_port_z_m
            self._last_blocked_descent_check_time_sec = current_time_sec
            return

        dt = max(0.0, current_time_sec - self._last_blocked_descent_check_time_sec)
        z_progress_m = self._last_plug_port_z_m - plug_port_z_m
        self._last_plug_port_z_m = plug_port_z_m
        self._last_blocked_descent_check_time_sec = current_time_sec

        contact_present = self._latest_force_mag_n >= self._contact_force_threshold_n
        progress_stalled = z_progress_m < self._blocked_descent_progress_threshold_m
        if contact_present and progress_stalled:
            self._blocked_descent_sec += dt

        if (
            not self._blocked_descent_detected
            and self._blocked_descent_sec
            > self._blocked_descent_duration_threshold_sec
        ):
            self.get_logger().warn(
                "Blocked descent detected: z_offset is still decreasing, "
                "but plug-to-port Z progress is stalled under contact force. "
                f"plug_port_z={plug_port_z_m:.4f} m, "
                f"last_progress={z_progress_m * 1e3:.2f} mm, "
                f"|F|={self._latest_force_mag_n:.2f} N, "
                f"blocked_time={self._blocked_descent_sec:.2f} s"
            )
            self._blocked_descent_detected = True

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
        if not self._pid_plots_enabled:
            return
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
            desc_idx = self._pre_descent_index
            plt.axvline(
                self.history_time[desc_idx],
                color="green",
                linestyle=":",
                label="Descent Start",
            )
            desc_x = err_x_mm[desc_idx]
            desc_y = err_y_mm[desc_idx]
            plt.annotate(
                f"DESCENT START\nX: {desc_x:.3f} mm\nY: {desc_y:.3f} mm",
                xy=(self.history_time[desc_idx], max(desc_x, desc_y)),
                xytext=(-10, 20),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", alpha=0.3),
                ha="right",
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
        self._collision_detected = False
        self._contact_detected = False
        self._lateral_contact_detected = False
        self._blocked_descent_detected = False
        self._latest_force_mag_n = 0.0
        self._latest_lateral_force_mag_n = 0.0
        self._latest_lateral_force_ratio = 0.0
        self._max_force_mag_n = 0.0
        self._force_above_threshold_sec = 0.0
        self._contact_force_sec = 0.0
        self._lateral_contact_sec = 0.0
        self._blocked_descent_sec = 0.0
        self._last_force_sample_time_sec = None
        self._last_force_log_time_sec = None
        self._last_blocked_descent_check_time_sec = None
        self._last_plug_port_z_m = None
        self._collision_injection_rng = np.random.default_rng(
            self._collision_injection_seed
        )
        self._collision_injection_target_xy_offset_m = None
        self._collision_injection_start_time_sec = None
        self._collision_injection_logged = False

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

        duration = 5.0
        dt = 0.05
        steps = int(duration / dt)

        def cubic_polynomial_trajectory(t_frac: float) -> float:
            """3rd-order polynomial (C1 continuity): smooth acceleration/deceleration."""
            return 3 * (t_frac**2) - 2 * (t_frac**3)

        for t in range(steps):
            if self._task_completed_in_simulation(task):
                self.get_logger().info(
                    "[CheatCodePID] Early exit: simulation reported task completion."
                )
                self._safe_plot_errors(success=True)
                return True
            raw_fraction = t / float(steps)
            interp_fraction = cubic_polynomial_trajectory(raw_fraction)
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
                insertion_port_transform = self._build_collision_test_port_transform(
                    port_transform, z_offset
                )
                pose = self.calc_gripper_pose(
                    insertion_port_transform, z_offset=z_offset
                )
                self.set_pose_target(move_robot=move_robot, pose=pose)
                self._update_blocked_descent_detection(
                    port_transform, cable_tip_frame
                )
                self.sleep_for(0.05)
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during insertion: {ex}")
                self.sleep_for(0.05)
                continue

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
