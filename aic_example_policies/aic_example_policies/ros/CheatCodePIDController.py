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
from enum import Enum, auto

import numpy as np

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_example_policies.controllers import PIDController
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform, Vector3, Wrench
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from std_msgs.msg import Header
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


class InsertionState(Enum):
    TRANSPORT = auto()
    INSERT = auto()
    # PRE_INSERTION = auto()
    # SLIP = auto()
    RECOVER = auto()
    # FAILURE = auto()


class PIDControllerMode(Enum):
    ON = auto()
    OFF = auto()
    CHEATCODE_DEFAULT = auto()


class CheatCodePIDController(Policy):
    def __init__(self, parent_node):
        self.pid_x = PIDController(kp=17.0, ki=0.08, kd=2.0)
        self.pid_y = PIDController(kp=17.0, ki=0.08, kd=2.0)
        pid_mode_name = os.environ.get("AIC_PID_CONTROLLER_MODE", "ON").upper()
        try:
            self._pid_controller_mode = PIDControllerMode[pid_mode_name]
        except KeyError as ex:
            valid_modes = ", ".join(mode.name for mode in PIDControllerMode)
            raise RuntimeError(
                "AIC_PID_CONTROLLER_MODE must be one of "
                f"{valid_modes}; got '{pid_mode_name}'"
            ) from ex
        self._cheatcode_i_gain = 0.15
        self._cheatcode_max_integrator_windup = 0.05
        self._cheatcode_tip_x_error_integrator = 0.0
        self._cheatcode_tip_y_error_integrator = 0.0
        self.xy_alignment_tolerance_m = 0.01
        self.xy_alignment_stable_cycles = 5
        self._task = None
        self._latest_insertion_event_namespace = ""
        self._pid_plots_enabled = (
            os.environ.get("AIC_PID_TUNING_PLOTS", "false").lower() == "true"
        )
        self._plot_output_dir = self._resolve_plot_output_dir()
        self._transport_force_injection_enabled = os.environ.get(
            "AIC_PID_TRANSPORT_FORCE_INJECTION_ENABLED", "false"
        ).lower() not in ("0", "false", "no", "off")
        self._transport_force_injection_min_n = float(
            os.environ.get("AIC_PID_TRANSPORT_FORCE_INJECTION_MIN_N", "5.0")
        )
        self._transport_force_injection_max_n = float(
            os.environ.get("AIC_PID_TRANSPORT_FORCE_INJECTION_MAX_N", "20.0")
        )
        self._transport_force_injection_seed = int(
            os.environ.get("AIC_PID_TRANSPORT_FORCE_INJECTION_SEED", "7")
        )
        self._transport_force_injection_rng = np.random.default_rng(
            self._transport_force_injection_seed
        )
        self._transport_force_injection_time_sec = None
        self._transport_force_injection_applied = False
        self._transport_force_injection_wrench = None
        self._transport_force_injection_logged = False
        self._recover_lift_m = float(os.environ.get("AIC_PID_RECOVER_LIFT_M", "0.01"))
        self._recover_motion_dt_sec = float(
            os.environ.get("AIC_PID_RECOVER_MOTION_DT_SEC", "0.2")
        )
        if (
            self._transport_force_injection_enabled
            and self._transport_force_injection_min_n
            >= self._transport_force_injection_max_n
        ):
            raise RuntimeError(
                "AIC_PID_TRANSPORT_FORCE_INJECTION_MIN_N must be less than "
                "AIC_PID_TRANSPORT_FORCE_INJECTION_MAX_N"
            )
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
        self.get_logger().info(
            "[CheatCodePID] Controller mode: "
            f"{self._pid_controller_mode.name}; "
            f"PID X(kp={self.pid_x.kp}, ki={self.pid_x.ki}, kd={self.pid_x.kd}), "
            f"PID Y(kp={self.pid_y.kp}, ki={self.pid_y.ki}, kd={self.pid_y.kd}), "
            f"CheatCode i_gain={self._cheatcode_i_gain}, "
            f"CheatCode max_integrator_windup={self._cheatcode_max_integrator_windup}"
        )

        # --- NEW: Initialize tracking attributes ---
        self.history_time = []
        self.history_err_x = []
        self.history_err_y = []
        self.history_state = []
        self.history_collision_detected = []
        self.history_stuck_detected = []
        self._insertion_state = None
        self.start_time = None
        self._diagnostics_active = False

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
        self._stuck_fz_threshold_n = float(
            os.environ.get("AIC_PID_STUCK_FZ_THRESHOLD_N", "5.0")
        )
        self._stuck_duration_threshold_sec = float(
            os.environ.get("AIC_PID_STUCK_DURATION_THRESHOLD_SEC", "1.0")
        )
        self._stuck_z_stationary_threshold_m = float(
            os.environ.get("AIC_PID_STUCK_Z_STATIONARY_THRESHOLD_M", "0.0001")
        )
        self._stuck_force_increase_threshold_n = float(
            os.environ.get("AIC_PID_STUCK_FORCE_INCREASE_THRESHOLD_N", "0.05")
        )
        self._stuck_detected = False
        self._blocked_descent_detected = False
        self._stuck_sec = 0.0
        self._last_blocked_descent_check_time_sec = None
        self._last_plug_port_z_m = None
        self._last_stuck_abs_fz_n = None
        self._stuck_window_start_abs_fz_n = None
        self._latest_force_mag_n = 0.0
        self._latest_tared_force_z_n = 0.0
        self._latest_lateral_force_mag_n = 0.0
        self._latest_lateral_force_ratio = 0.0
        self._max_force_mag_n = 0.0
        self._force_above_threshold_sec = 0.0
        self._contact_force_sec = 0.0
        self._last_force_sample_time_sec = None
        self._last_stuck_log_time_sec = None
        self._stuck_debug_log_interval_sec = float(
            os.environ.get("AIC_PID_STUCK_DEBUG_LOG_INTERVAL_SEC", "0.5")
        )
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
        self._latest_tared_force_z_n = float(tared_force[2])
        self._latest_lateral_force_mag_n = float(np.linalg.norm(tared_force[:2]))
        self._latest_lateral_force_ratio = self._latest_lateral_force_mag_n / max(
            self._latest_force_mag_n, 1e-6
        )
        self._max_force_mag_n = max(self._max_force_mag_n, self._latest_force_mag_n)

        if not self._diagnostics_active:
            return

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
        else:
            self._force_above_threshold_sec = 0.0

        if self._latest_force_mag_n >= self._contact_force_threshold_n:
            self._contact_force_sec += dt
        else:
            self._contact_force_sec = 0.0

        lateral_contact_active = (
            self._latest_lateral_force_mag_n >= self._lateral_contact_force_threshold_n
            or (
                self._latest_force_mag_n >= self._contact_force_threshold_n
                and self._latest_lateral_force_ratio
                >= self._lateral_contact_ratio_threshold
            )
        )
        if lateral_contact_active:
            self._lateral_contact_sec += dt
        else:
            self._lateral_contact_sec = 0.0

        if self._force_log_interval_sec > 0.0 and (
            self._last_force_log_time_sec is None
            or sample_time_sec - self._last_force_log_time_sec
            >= self._force_log_interval_sec
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
                f"stuck_time={self._stuck_sec:.2f} s, "
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
            and self._lateral_contact_sec > self._lateral_contact_duration_threshold_sec
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

    def _reset_transport_force_injection(self, transport_duration_sec: float) -> None:
        self._transport_force_injection_time_sec = None
        self._transport_force_injection_applied = False
        self._transport_force_injection_wrench = None
        self._transport_force_injection_logged = False

        if not self._transport_force_injection_enabled or transport_duration_sec <= 0.0:
            return

        min_force_n = max(
            self._transport_force_injection_min_n,
            float(np.nextafter(5.0, 20.0)),
        )
        max_force_n = min(
            self._transport_force_injection_max_n,
            float(np.nextafter(20.0, 5.0)),
        )
        if min_force_n >= max_force_n:
            self.get_logger().warn(
                "Transport force injection disabled: configured force range does not "
                "overlap the required open interval 5 N < F < 20 N."
            )
            return

        self._transport_force_injection_time_sec = float(
            self._transport_force_injection_rng.uniform(0.0, transport_duration_sec)
        )
        force_mag_n = float(
            self._transport_force_injection_rng.uniform(min_force_n, max_force_n)
        )
        axis_index = int(self._transport_force_injection_rng.integers(0, 2))
        direction_sign = float(self._transport_force_injection_rng.choice([-1.0, 1.0]))
        force_x = direction_sign * force_mag_n if axis_index == 0 else 0.0
        force_y = direction_sign * force_mag_n if axis_index == 1 else 0.0

        self._transport_force_injection_wrench = Wrench(
            force=Vector3(x=force_x, y=force_y, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        self.get_logger().warn(
            "[CheatCodePID] Scheduled one-shot TRANSPORT TCP force injection: "
            f"time={self._transport_force_injection_time_sec:.3f}/"
            f"{transport_duration_sec:.3f} s, "
            f"force=({force_x:.2f}, {force_y:.2f}, 0.00) N, "
            f"magnitude={force_mag_n:.2f} N"
        )

    def _transport_force_injection_wrench_for_time(
        self, elapsed_sec: float
    ) -> Wrench | None:
        if (
            self._transport_force_injection_time_sec is None
            or self._transport_force_injection_wrench is None
            or self._transport_force_injection_applied
            or elapsed_sec < self._transport_force_injection_time_sec
        ):
            return None

        if not self._transport_force_injection_logged:
            now_sec = self.time_now().nanoseconds / 1e9
            force = self._transport_force_injection_wrench.force
            self.get_logger().warn(
                "[CheatCodePID] Applying one-shot TRANSPORT TCP force injection: "
                f"t={now_sec:.3f} s, transport_elapsed={elapsed_sec:.3f} s, "
                f"force=({force.x:.2f}, {force.y:.2f}, {force.z:.2f}) N"
            )
            self._transport_force_injection_logged = True
        self._transport_force_injection_applied = True

        return self._transport_force_injection_wrench

    def _set_pose_target_with_optional_wrench(
        self,
        move_robot: MoveRobotCallback,
        pose: Pose,
        feedforward_wrench_at_tip: Wrench | None = None,
        frame_id: str = "base_link",
        stiffness: list = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0],
        damping: list = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0],
    ) -> None:
        motion_update = MotionUpdate(
            header=Header(
                frame_id=frame_id,
                stamp=self._parent_node.get_clock().now().to_msg(),
            ),
            pose=pose,
            target_stiffness=np.diag(stiffness).flatten(),
            target_damping=np.diag(damping).flatten(),
            feedforward_wrench_at_tip=feedforward_wrench_at_tip
            or Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0),
                torque=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )
        try:
            move_robot(motion_update=motion_update)
        except Exception as ex:
            self.get_logger().info(f"move_robot exception: {ex}")

    def _current_tcp_pose(self) -> Pose:
        gripper_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            "gripper/tcp",
            Time(),
        )
        transform = gripper_tf_stamped.transform
        return Pose(
            position=Point(
                x=transform.translation.x,
                y=transform.translation.y,
                z=transform.translation.z,
            ),
            orientation=Quaternion(
                w=transform.rotation.w,
                x=transform.rotation.x,
                y=transform.rotation.y,
                z=transform.rotation.z,
            ),
        )

    def _plug_port_xy_error(
        self, port_transform: Transform, cable_tip_frame: str
    ) -> tuple[float, float]:
        plug_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            cable_tip_frame,
            Time(),
        )
        return (
            port_transform.translation.x - plug_tf_stamped.transform.translation.x,
            port_transform.translation.y - plug_tf_stamped.transform.translation.y,
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
            self._last_stuck_abs_fz_n = abs(self._latest_tared_force_z_n)
            return

        dt = max(0.0, current_time_sec - self._last_blocked_descent_check_time_sec)
        z_progress_m = self._last_plug_port_z_m - plug_port_z_m
        abs_fz_n = abs(self._latest_tared_force_z_n)
        self._last_plug_port_z_m = plug_port_z_m
        self._last_blocked_descent_check_time_sec = current_time_sec
        self._last_stuck_abs_fz_n = abs_fz_n

        fz_contact_present = abs_fz_n > self._stuck_fz_threshold_n
        z_stationary = abs(z_progress_m) <= self._stuck_z_stationary_threshold_m
        if fz_contact_present and z_stationary:
            if self._stuck_window_start_abs_fz_n is None:
                self._stuck_window_start_abs_fz_n = abs_fz_n
            self._stuck_sec += dt
        else:
            self._stuck_sec = 0.0
            self._stuck_window_start_abs_fz_n = None

        window_force_delta_n = (
            0.0
            if self._stuck_window_start_abs_fz_n is None
            else abs_fz_n - self._stuck_window_start_abs_fz_n
        )
        force_increased = window_force_delta_n >= self._stuck_force_increase_threshold_n

        if (
            self._stuck_debug_log_interval_sec > 0.0
            and fz_contact_present
            and (
                self._last_stuck_log_time_sec is None
                or current_time_sec - self._last_stuck_log_time_sec
                >= self._stuck_debug_log_interval_sec
            )
        ):
            self.get_logger().info(
                "[CheatCodePID] STUCK check: "
                f"|Fz|={abs_fz_n:.2f} N "
                f"(threshold>{self._stuck_fz_threshold_n:.2f}), "
                f"z_change={z_progress_m * 1e3:.3f} mm "
                f"(stationary<={self._stuck_z_stationary_threshold_m * 1e3:.3f} mm), "
                f"|Fz|_window_delta={window_force_delta_n:.2f} N "
                f"(threshold>={self._stuck_force_increase_threshold_n:.2f}), "
                f"stuck_time={self._stuck_sec:.2f}/"
                f"{self._stuck_duration_threshold_sec:.2f} s"
            )
            self._last_stuck_log_time_sec = current_time_sec

        if (
            not self._stuck_detected
            and self._stuck_sec > self._stuck_duration_threshold_sec
            and force_increased
        ):
            self.get_logger().warn(
                "[CheatCodePID] STUCK condition detected during INSERT: "
                "z_offset is decreasing, plug-to-port Z is stationary, "
                "and |Fz| is increasing above threshold. "
                f"plug_port_z={plug_port_z_m:.4f} m, "
                f"last_z_change={z_progress_m * 1e3:.2f} mm, "
                f"Fz={self._latest_tared_force_z_n:.2f} N, "
                f"|Fz|_window_delta={window_force_delta_n:.2f} N, "
                f"stuck_time={self._stuck_sec:.2f} s"
            )
            self._stuck_detected = True
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
            f"Mode={self._pid_controller_mode.name}  |  "
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
        if self._insertion_state is None:
            self.history_state.append("")
        else:
            self.history_state.append(self._insertion_state.name)
        self.history_collision_detected.append(self._collision_detected)
        self.history_stuck_detected.append(self._stuck_detected)

    def _reset_xy_controller_state(self) -> None:
        self.pid_x.reset()
        self.pid_y.reset()
        self._cheatcode_tip_x_error_integrator = 0.0
        self._cheatcode_tip_y_error_integrator = 0.0

    def _xy_controller_adjustments(
        self,
        port_xy: tuple[float, float],
        plug_xyz: tuple[float, float, float],
        reset_controller: bool,
        dt: float,
    ) -> tuple[float, float]:
        tip_x_error = port_xy[0] - plug_xyz[0]
        tip_y_error = port_xy[1] - plug_xyz[1]
        self.pid_x.last_error = tip_x_error
        self.pid_y.last_error = tip_y_error

        if reset_controller:
            self._reset_xy_controller_state()
            self.pid_x.last_error = tip_x_error
            self.pid_y.last_error = tip_y_error

        if self._pid_controller_mode == PIDControllerMode.OFF:
            return 0.0, 0.0

        if self._pid_controller_mode == PIDControllerMode.CHEATCODE_DEFAULT:
            if not reset_controller:
                self._cheatcode_tip_x_error_integrator = float(
                    np.clip(
                        self._cheatcode_tip_x_error_integrator + tip_x_error,
                        -self._cheatcode_max_integrator_windup,
                        self._cheatcode_max_integrator_windup,
                    )
                )
                self._cheatcode_tip_y_error_integrator = float(
                    np.clip(
                        self._cheatcode_tip_y_error_integrator + tip_y_error,
                        -self._cheatcode_max_integrator_windup,
                        self._cheatcode_max_integrator_windup,
                    )
                )
            return (
                self._cheatcode_i_gain * self._cheatcode_tip_x_error_integrator,
                self._cheatcode_i_gain * self._cheatcode_tip_y_error_integrator,
            )

        return (
            self.pid_x.update(setpoint=port_xy[0], measurement=plug_xyz[0], dt=dt),
            self.pid_y.update(setpoint=port_xy[1], measurement=plug_xyz[1], dt=dt),
        )

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

        adj_x, adj_y = self._xy_controller_adjustments(
            port_xy=port_xy,
            plug_xyz=plug_xyz,
            reset_controller=reset_pids,
            dt=dt,
        )

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

    def _transition_to_state(
        self, current_state: InsertionState, next_state: InsertionState
    ) -> InsertionState:
        if current_state != next_state:
            self.get_logger().info(
                f"[CheatCodePID] State transition: {current_state.name} -> {next_state.name}"
            )
        self._insertion_state = next_state
        return next_state

    def _run_transport_state(
        self,
        task: Task,
        move_robot: MoveRobotCallback,
        port_transform: Transform,
        z_offset: float,
        duration: float,
        dt: float,
    ) -> tuple[InsertionState | None, bool]:
        steps = int(duration / dt)
        self._reset_transport_force_injection(duration)

        def cubic_polynomial_trajectory(t_frac: float) -> float:
            """3rd-order polynomial (C1 continuity): smooth acceleration/deceleration."""
            return 3 * (t_frac**2) - 2 * (t_frac**3)

        for t in range(steps):
            if self._task_completed_in_simulation(task):
                self.get_logger().info(
                    "[CheatCodePID] Early exit: simulation reported task completion."
                )
                return None, True
            raw_fraction = t / float(steps)
            elapsed_sec = t * dt
            interp_fraction = cubic_polynomial_trajectory(raw_fraction)
            reset_xy_controller = (
                self._pid_controller_mode == PIDControllerMode.CHEATCODE_DEFAULT
            )
            try:
                self._set_pose_target_with_optional_wrench(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(
                        port_transform,
                        slerp_fraction=interp_fraction,
                        position_fraction=interp_fraction,
                        z_offset=z_offset,
                        reset_pids=reset_xy_controller,
                        dt=dt,
                    ),
                    feedforward_wrench_at_tip=self._transport_force_injection_wrench_for_time(
                        elapsed_sec
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during interpolation: {ex}")
            self.sleep_for(dt)

        self._pre_descent_index = len(self.history_err_x) - 1
        self.get_logger().info(
            f"[CheatCodePID] Pre-descent XY error: "
            f"ErrX={self.pid_x.last_error * 1e3:.3f} mm, "
            f"ErrY={self.pid_y.last_error * 1e3:.3f} mm"
        )
        return InsertionState.INSERT, False

    def _run_insert_state(
        self,
        task: Task,
        move_robot: MoveRobotCallback,
        port_transform: Transform,
        cable_tip_frame: str,
        z_offset: float,
        dt: float,
    ) -> tuple[InsertionState | None, bool, float]:
        while True:
            if self._task_completed_in_simulation(task):
                self.get_logger().info(
                    "[CheatCodePID] Early exit: simulation reported task completion."
                )
                return None, True, z_offset
            if z_offset < -0.015:
                break

            z_offset -= 0.0005
            self.get_logger().info(f"z_offset: {z_offset:0.5}")
            try:
                pose = self.calc_gripper_pose(port_transform, z_offset=z_offset)
                self.set_pose_target(move_robot=move_robot, pose=pose)
                self._update_blocked_descent_detection(port_transform, cable_tip_frame)
                if self._stuck_detected:
                    return InsertionState.RECOVER, False, z_offset
                self.sleep_for(dt)
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during insertion: {ex}")
                self.sleep_for(dt)

        self.get_logger().info("Waiting briefly for insertion event...")
        wait_started = self.time_now()
        wait_timeout = Duration(seconds=5.0)
        while (self.time_now() - wait_started) < wait_timeout:
            if self._task_completed_in_simulation(task):
                self.get_logger().info(
                    "[CheatCodePID] Insertion event observed before timeout."
                )
                return None, True, z_offset
            self.sleep_for(0.05)
        return None, False, z_offset

    def _run_recover_state(
        self,
        move_robot: MoveRobotCallback,
        port_transform: Transform,
        cable_tip_frame: str,
        z_offset: float,
    ) -> tuple[InsertionState, float]:
        self.get_logger().warn(
            "[CheatCodePID] Entering RECOVER: lifting TCP and reducing XY error by half."
        )

        tcp_pose = self._current_tcp_pose()
        lift_pose = Pose(
            position=Point(
                x=tcp_pose.position.x,
                y=tcp_pose.position.y,
                z=tcp_pose.position.z + self._recover_lift_m,
            ),
            orientation=tcp_pose.orientation,
        )
        self.set_pose_target(move_robot=move_robot, pose=lift_pose)
        self.sleep_for(self._recover_motion_dt_sec)

        error_x, error_y = self._plug_port_xy_error(port_transform, cable_tip_frame)
        lifted_pose = self._current_tcp_pose()

        # TODO: Use PID CONTROLLER mode to compute the correction instead of hardcoding half the error
        reduce_error_pose = Pose(
            position=Point(
                x=lifted_pose.position.x + 0.5 * error_x,
                y=lifted_pose.position.y + 0.5 * error_y,
                z=lifted_pose.position.z,
            ),
            orientation=lifted_pose.orientation,
        )
        self.get_logger().warn(
            "[CheatCodePID] RECOVER XY correction: "
            f"error=({error_x * 1e3:.2f}, {error_y * 1e3:.2f}) mm, "
            f"commanded_half=({0.5 * error_x * 1e3:.2f}, "
            f"{0.5 * error_y * 1e3:.2f}) mm"
        )
        self.set_pose_target(move_robot=move_robot, pose=reduce_error_pose)
        self.sleep_for(self._recover_motion_dt_sec)

        self._stuck_detected = False
        self._blocked_descent_detected = False
        self._stuck_sec = 0.0
        self._last_blocked_descent_check_time_sec = None
        self._last_plug_port_z_m = None
        self._last_stuck_abs_fz_n = None
        self._stuck_window_start_abs_fz_n = None

        return InsertionState.INSERT, z_offset + self._recover_lift_m

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
        self._stuck_detected = False
        self._blocked_descent_detected = False
        self._latest_force_mag_n = 0.0
        self._latest_tared_force_z_n = 0.0
        self._latest_lateral_force_mag_n = 0.0
        self._latest_lateral_force_ratio = 0.0
        self._max_force_mag_n = 0.0
        self._force_above_threshold_sec = 0.0
        self._contact_force_sec = 0.0
        self._lateral_contact_sec = 0.0
        self._stuck_sec = 0.0
        self._last_force_sample_time_sec = None
        self._last_force_log_time_sec = None
        self._last_blocked_descent_check_time_sec = None
        self._last_plug_port_z_m = None
        self._last_stuck_abs_fz_n = None
        self._stuck_window_start_abs_fz_n = None
        self._last_stuck_log_time_sec = None
        self._diagnostics_active = True
        self._transport_force_injection_rng = np.random.default_rng(
            self._transport_force_injection_seed
        )
        self._transport_force_injection_time_sec = None
        self._transport_force_injection_applied = False
        self._transport_force_injection_wrench = None
        self._transport_force_injection_logged = False

        # Reset telemetry and PIDs
        self.history_time = []
        self.history_err_x = []
        self.history_err_y = []
        self.history_state = []
        self.history_collision_detected = []
        self.history_stuck_detected = []
        self._insertion_state = None
        self.start_time = None
        self._reset_xy_controller_state()

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

        state = InsertionState.TRANSPORT
        self._insertion_state = state
        self.get_logger().info(f"[CheatCodePID] Starting state machine in {state.name}")

        success = False
        while True:
            if state == InsertionState.TRANSPORT:
                next_state, success = self._run_transport_state(
                    task=task,
                    move_robot=move_robot,
                    port_transform=port_transform,
                    z_offset=z_offset,
                    duration=duration,
                    dt=dt,
                )
                if success or next_state is None:
                    break
                state = self._transition_to_state(state, next_state)
            elif state == InsertionState.INSERT:
                next_state, success, z_offset = self._run_insert_state(
                    task=task,
                    move_robot=move_robot,
                    port_transform=port_transform,
                    cable_tip_frame=cable_tip_frame,
                    z_offset=z_offset,
                    dt=dt,
                )
                if next_state is not None:
                    state = self._transition_to_state(state, next_state)
                    continue
                break
            elif state == InsertionState.RECOVER:
                next_state, z_offset = self._run_recover_state(
                    move_robot=move_robot,
                    port_transform=port_transform,
                    cable_tip_frame=cable_tip_frame,
                    z_offset=z_offset,
                )
                state = self._transition_to_state(state, next_state)
                continue

        self.get_logger().info("CheatCodePIDController.insert_cable() exiting...")

        self._safe_plot_errors(success=success)
        self._diagnostics_active = False

        return success


CheatCodePidController = CheatCodePIDController
