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

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_example_policies.controllers import PIDController
from aic_example_policies.utils.cheatcode_pid_utils import (
    ConfigLoader,
    ForceMonitor,
    PIDControllerMode,
    TelemetryManager,
)
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
from std_msgs.msg import Header, String
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp


class InsertionState(Enum):
    TRANSPORT = auto()
    INSERT = auto()
    RECOVER = auto()


@dataclass
class InsertionContext:
    task: Task
    move_robot: MoveRobotCallback
    port_transform: Transform
    cable_tip_frame: str
    z_offset: float
    success: bool = False


@dataclass
class StateResult:
    next_state: InsertionState | None
    done: bool = False


class CheatCodePIDController(Policy):
    PRE_INSERTION_Z_OFFSET_M_DEFAULT = 0.08

    def __init__(self, parent_node):
        self._cfg = ConfigLoader.load(
            pre_insertion_default=self.PRE_INSERTION_Z_OFFSET_M_DEFAULT
        )
        self.pid_x = PIDController(
            kp=self._cfg.pid_x_gains[0],
            ki=self._cfg.pid_x_gains[1],
            kd=self._cfg.pid_x_gains[2],
        )
        self.pid_y = PIDController(
            kp=self._cfg.pid_y_gains[0],
            ki=self._cfg.pid_y_gains[1],
            kd=self._cfg.pid_y_gains[2],
        )

        self._cheatcode_tip_x_error_integrator = 0.0
        self._cheatcode_tip_y_error_integrator = 0.0

        self._task = None
        self._latest_insertion_event_namespace = ""
        self._insertion_state = None
        self._pre_insertion_pose = None

        self._transport_force_injection_rng = np.random.default_rng(
            self._cfg.transport_force_injection_seed
        )
        self._transport_force_injection_time_sec = None
        self._transport_force_injection_applied = False
        self._transport_force_injection_wrench = None
        self._transport_force_injection_logged = False

        super().__init__(parent_node)

        self._monitor = ForceMonitor(self._cfg, logger_getter=self.get_logger)
        self._telemetry = TelemetryManager(
            self._cfg,
            logger_getter=self.get_logger,
            now_ns_getter=lambda: self.time_now().nanoseconds,
        )

        self.get_logger().info(
            "[CheatCodePID] Controller mode: "
            f"{self._cfg.pid_mode.name}; "
            f"PID X(kp={self.pid_x.kp}, ki={self.pid_x.ki}, kd={self.pid_x.kd}), "
            f"PID Y(kp={self.pid_y.kp}, ki={self.pid_y.ki}, kd={self.pid_y.kd}), "
            f"Recover PID X(kp={self._cfg.recover_pid_x_gains[0]}, "
            f"ki={self._cfg.recover_pid_x_gains[1]}, "
            f"kd={self._cfg.recover_pid_x_gains[2]}), "
            f"Recover PID Y(kp={self._cfg.recover_pid_y_gains[0]}, "
            f"ki={self._cfg.recover_pid_y_gains[1]}, "
            f"kd={self._cfg.recover_pid_y_gains[2]}), "
            f"pre_insertion_z_offset={self._cfg.pre_insertion_z_offset_m:.3f} m, "
            f"config={self._cfg.config_path}, "
            f"CheatCode i_gain={self._cfg.cheatcode_i_gain}, "
            f"CheatCode max_integrator_windup={self._cfg.cheatcode_max_integrator_windup}"
        )

        self._insertion_event_sub = self._parent_node.create_subscription(
            String,
            "/scoring/insertion_event",
            self._insertion_event_callback,
            10,
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
        self._monitor.update_observation(msg, self.time_now().nanoseconds / 1e9)

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

        if (
            not self._cfg.transport_force_injection_enabled
            or transport_duration_sec <= 0.0
        ):
            return

        def _coerce_axis(raw_axis: str | None) -> int:
            if raw_axis is None:
                return 0
            axis = raw_axis.strip().lower()
            if axis in ("x", "0"):
                return 0
            if axis in ("y", "1"):
                return 1
            raise ValueError(
                "Transport force injection axis must be 'x' or 'y' when configured."
            )

        def _coerce_sign(raw_sign: float | None) -> float:
            if raw_sign is None:
                return 1.0
            return 1.0 if float(raw_sign) >= 0.0 else -1.0

        min_force_n = max(
            self._cfg.transport_force_injection_min_n,
            float(np.nextafter(5.0, 20.0)),
        )
        max_force_n = min(
            self._cfg.transport_force_injection_max_n,
            float(np.nextafter(20.0, 5.0)),
        )
        if min_force_n >= max_force_n:
            self.get_logger().warn(
                "Transport force injection disabled: configured force range does not "
                "overlap the required open interval 5 N < F < 20 N."
            )
            return

        if self._cfg.transport_force_injection_randomize_time:
            self._transport_force_injection_time_sec = float(
                self._transport_force_injection_rng.uniform(0.0, transport_duration_sec)
            )
        else:
            fallback_time_sec = transport_duration_sec / 2.0
            self._transport_force_injection_time_sec = float(
                np.clip(
                    (
                        self._cfg.transport_force_injection_config_time_sec
                        if self._cfg.transport_force_injection_config_time_sec
                        is not None
                        else fallback_time_sec
                    ),
                    0.0,
                    transport_duration_sec,
                )
            )

        if self._cfg.transport_force_injection_randomize_force:
            force_mag_n = float(
                self._transport_force_injection_rng.uniform(min_force_n, max_force_n)
            )
        else:
            fallback_force_n = (min_force_n + max_force_n) / 2.0
            force_mag_n = float(
                self._cfg.transport_force_injection_force_n
                if self._cfg.transport_force_injection_force_n is not None
                else fallback_force_n
            )
            force_mag_n = float(np.clip(force_mag_n, min_force_n, max_force_n))

        if self._cfg.transport_force_injection_randomize_axis:
            axis_index = int(self._transport_force_injection_rng.integers(0, 2))
        else:
            axis_index = _coerce_axis(self._cfg.transport_force_injection_axis)

        if self._cfg.transport_force_injection_randomize_direction:
            direction_sign = float(
                self._transport_force_injection_rng.choice([-1.0, 1.0])
            )
        else:
            direction_sign = _coerce_sign(
                self._cfg.transport_force_injection_direction_sign
            )

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

    @staticmethod
    def _cubic_progress(t_frac: float) -> float:
        return 3 * (t_frac**2) - 2 * (t_frac**3)

    def _iter_cubic_motion(self, duration: float, dt: float):
        steps = max(1, int(duration / dt))
        for step in range(steps + 1):
            raw_fraction = step / float(steps)
            yield step, steps, self._cubic_progress(raw_fraction), step * dt

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
        self._monitor.update_blocked_descent(plug_port_z_m, current_time_sec)

    def _wait_for_tf(
        self, target_frame: str, source_frame: str, timeout_sec: float = 10.0
    ) -> bool:
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

    def _record_telemetry(self):
        self._telemetry.record(
            pid_error_x=self.pid_x.last_error,
            pid_error_y=self.pid_y.last_error,
            state_name=self._insertion_state.name if self._insertion_state else "",
            collision_detected=self._monitor.collision_detected,
            stuck_detected=self._monitor.stuck_detected,
            now_sec=self.time_now().nanoseconds / 1e9,
        )

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

        if self._cfg.pid_mode == PIDControllerMode.OFF:
            return 0.0, 0.0

        if self._cfg.pid_mode == PIDControllerMode.CHEATCODE_DEFAULT:
            if not reset_controller:
                self._cheatcode_tip_x_error_integrator = float(
                    np.clip(
                        self._cheatcode_tip_x_error_integrator + tip_x_error,
                        -self._cfg.cheatcode_max_integrator_windup,
                        self._cfg.cheatcode_max_integrator_windup,
                    )
                )
                self._cheatcode_tip_y_error_integrator = float(
                    np.clip(
                        self._cheatcode_tip_y_error_integrator + tip_y_error,
                        -self._cfg.cheatcode_max_integrator_windup,
                        self._cfg.cheatcode_max_integrator_windup,
                    )
                )
            return (
                self._cfg.cheatcode_i_gain * self._cheatcode_tip_x_error_integrator,
                self._cfg.cheatcode_i_gain * self._cheatcode_tip_y_error_integrator,
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
            abs(self.pid_x.last_error) <= self._cfg.xy_alignment_tolerance_m
            and abs(self.pid_y.last_error) <= self._cfg.xy_alignment_tolerance_m
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

    def _set_pid_gains(
        self, pid: PIDController, gains: tuple[float, float, float]
    ) -> None:
        pid.kp, pid.ki, pid.kd = gains

    def _interpolate_pose(
        self, start_pose: Pose, end_pose: Pose, fraction: float
    ) -> Pose:
        frac = float(np.clip(fraction, 0.0, 1.0))
        q_start = (
            start_pose.orientation.w,
            start_pose.orientation.x,
            start_pose.orientation.y,
            start_pose.orientation.z,
        )
        q_end = (
            end_pose.orientation.w,
            end_pose.orientation.x,
            end_pose.orientation.y,
            end_pose.orientation.z,
        )
        q_interp = quaternion_slerp(q_start, q_end, frac)
        return Pose(
            position=Point(
                x=(1.0 - frac) * start_pose.position.x + frac * end_pose.position.x,
                y=(1.0 - frac) * start_pose.position.y + frac * end_pose.position.y,
                z=(1.0 - frac) * start_pose.position.z + frac * end_pose.position.z,
            ),
            orientation=Quaternion(
                w=q_interp[0],
                x=q_interp[1],
                y=q_interp[2],
                z=q_interp[3],
            ),
        )

    @staticmethod
    def _format_pose_for_log(pose: Pose) -> str:
        return (
            "pos=("
            f"{pose.position.x:.4f}, {pose.position.y:.4f}, {pose.position.z:.4f}), "
            "quat=("
            f"{pose.orientation.w:.4f}, {pose.orientation.x:.4f}, "
            f"{pose.orientation.y:.4f}, {pose.orientation.z:.4f})"
        )

    def _update_pre_insertion_pose(
        self,
        port_transform: Transform,
        z_offset: float,
        dt: float,
        reset_pids: bool,
    ) -> Pose:
        target_pose = self.calc_gripper_pose(
            port_transform=port_transform,
            slerp_fraction=1.0,
            position_fraction=1.0,
            z_offset=z_offset,
            reset_pids=reset_pids,
            dt=dt,
        )
        self._pre_insertion_pose = Pose(
            position=Point(
                x=target_pose.position.x,
                y=target_pose.position.y,
                z=target_pose.position.z + self._cfg.pre_insertion_z_offset_m,
            ),
            orientation=target_pose.orientation,
        )
        self.get_logger().info(
            "[CheatCodePID] Updated pre_insertion_pose: "
            f"target_z={target_pose.position.z:.4f} m, "
            f"pre_insertion_z={self._pre_insertion_pose.position.z:.4f} m, "
            f"z_offset={self._cfg.pre_insertion_z_offset_m:.4f} m"
        )
        return self._pre_insertion_pose

    def _execute_transport_state(self, ctx: InsertionContext) -> StateResult:
        self._reset_transport_force_injection(self._cfg.transport_duration_sec)
        for _, _, interp_fraction, elapsed_sec in self._iter_cubic_motion(
            self._cfg.transport_duration_sec,
            self._cfg.transport_dt_sec,
        ):
            if self._task_completed_in_simulation(ctx.task):
                self.get_logger().info(
                    "[CheatCodePID] Early exit: simulation reported task completion."
                )
                ctx.success = True
                return StateResult(next_state=None, done=True)

            reset_xy_controller = (
                self._cfg.pid_mode == PIDControllerMode.CHEATCODE_DEFAULT
            )
            try:
                self._set_pose_target_with_optional_wrench(
                    move_robot=ctx.move_robot,
                    pose=self.calc_gripper_pose(
                        ctx.port_transform,
                        slerp_fraction=interp_fraction,
                        position_fraction=interp_fraction,
                        z_offset=ctx.z_offset,
                        reset_pids=reset_xy_controller,
                        dt=self._cfg.transport_dt_sec,
                    ),
                    feedforward_wrench_at_tip=self._transport_force_injection_wrench_for_time(
                        elapsed_sec
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during interpolation: {ex}")
            self.sleep_for(self._cfg.transport_dt_sec)

        self._telemetry.mark_pre_descent()
        self.get_logger().info(
            f"[CheatCodePID] Pre-descent XY error: "
            f"ErrX={self.pid_x.last_error * 1e3:.3f} mm, "
            f"ErrY={self.pid_y.last_error * 1e3:.3f} mm"
        )
        return StateResult(next_state=InsertionState.INSERT)

    def _execute_insert_state(self, ctx: InsertionContext) -> StateResult:
        start_z_offset = ctx.z_offset
        target_z_offset = self._cfg.insert_min_z_offset_m

        for _, _, interp_fraction, _ in self._iter_cubic_motion(
            self._cfg.insert_duration_sec,
            self._cfg.insert_dt_sec,
        ):
            if self._task_completed_in_simulation(ctx.task):
                self.get_logger().info(
                    "[CheatCodePID] Early exit: simulation reported task completion."
                )
                ctx.success = True
                return StateResult(next_state=None, done=True)

            ctx.z_offset = start_z_offset + interp_fraction * (
                target_z_offset - start_z_offset
            )
            self.get_logger().info(f"z_offset: {ctx.z_offset:0.5}")
            try:
                pose = self.calc_gripper_pose(ctx.port_transform, z_offset=ctx.z_offset)
                self.set_pose_target(move_robot=ctx.move_robot, pose=pose)
                self._update_blocked_descent_detection(
                    ctx.port_transform, ctx.cable_tip_frame
                )
                if self._monitor.stuck_detected:
                    try:
                        stuck_tcp_pose = self._current_tcp_pose()
                        self.get_logger().warn(
                            "[CheatCodePID] STUCK TCP pose: "
                            f"{self._format_pose_for_log(stuck_tcp_pose)}"
                        )
                    except TransformException as ex:
                        self.get_logger().warn(
                            f"[CheatCodePID] Failed to log STUCK TCP pose: {ex}"
                        )
                    return StateResult(next_state=InsertionState.RECOVER)
                self.sleep_for(self._cfg.insert_dt_sec)
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during insertion: {ex}")
                self.sleep_for(self._cfg.insert_dt_sec)

        self.get_logger().info("Waiting briefly for insertion event...")
        wait_started = self.time_now()
        wait_timeout = Duration(seconds=self._cfg.insert_wait_timeout_sec)
        while (self.time_now() - wait_started) < wait_timeout:
            if self._task_completed_in_simulation(ctx.task):
                self.get_logger().info(
                    "[CheatCodePID] Insertion event observed before timeout."
                )
                ctx.success = True
                return StateResult(next_state=None, done=True)
            self.sleep_for(0.05)

        return StateResult(next_state=None, done=True)

    def _execute_recover_state(self, ctx: InsertionContext) -> StateResult:
        self.get_logger().warn(
            "[CheatCodePID] Entering RECOVER: moving to pre_insertion_pose."
        )
        original_x_gains = (self.pid_x.kp, self.pid_x.ki, self.pid_x.kd)
        original_y_gains = (self.pid_y.kp, self.pid_y.ki, self.pid_y.kd)
        self._set_pid_gains(self.pid_x, self._cfg.recover_pid_x_gains)
        self._set_pid_gains(self.pid_y, self._cfg.recover_pid_y_gains)
        self._reset_xy_controller_state()

        try:
            self._update_pre_insertion_pose(
                port_transform=ctx.port_transform,
                z_offset=ctx.z_offset,
                dt=self._cfg.recover_motion_dt_sec,
                reset_pids=True,
            )
        except TransformException as ex:
            self.get_logger().warn(
                f"[CheatCodePID] TF lookup failed while computing pre_insertion_pose: {ex}"
            )
            self._set_pid_gains(self.pid_x, original_x_gains)
            self._set_pid_gains(self.pid_y, original_y_gains)
            return StateResult(next_state=InsertionState.INSERT)

        start_pose = self._current_tcp_pose()
        self.get_logger().info(
            "[CheatCodePID] RECOVER start TCP pose: "
            f"{self._format_pose_for_log(start_pose)}"
        )
        self.get_logger().info(
            "[CheatCodePID] RECOVER target pre_insertion_pose: "
            f"{self._format_pose_for_log(self._pre_insertion_pose)}"
        )
        for step, _, interp_fraction, _ in self._iter_cubic_motion(
            self._cfg.recover_duration_sec,
            self._cfg.recover_motion_dt_sec,
        ):
            recover_pose = self._interpolate_pose(
                start_pose=start_pose,
                end_pose=self._pre_insertion_pose,
                fraction=interp_fraction,
            )
            try:
                plug_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                    "base_link",
                    ctx.cable_tip_frame,
                    Time(),
                )
                port_xy = (
                    ctx.port_transform.translation.x,
                    ctx.port_transform.translation.y,
                )
                plug_xyz = (
                    plug_tf_stamped.transform.translation.x,
                    plug_tf_stamped.transform.translation.y,
                    plug_tf_stamped.transform.translation.z,
                )
                adj_x, adj_y = self._xy_controller_adjustments(
                    port_xy=port_xy,
                    plug_xyz=plug_xyz,
                    reset_controller=step == 0,
                    dt=self._cfg.recover_motion_dt_sec,
                )
                recover_pose = Pose(
                    position=Point(
                        x=recover_pose.position.x + adj_x,
                        y=recover_pose.position.y + adj_y,
                        z=recover_pose.position.z,
                    ),
                    orientation=recover_pose.orientation,
                )
            except TransformException as ex:
                self.get_logger().warn(
                    f"[CheatCodePID] TF lookup failed during recover interpolation: {ex}"
                )
            self._set_pose_target_with_optional_wrench(
                move_robot=ctx.move_robot,
                pose=recover_pose,
            )
            if step == 0:
                self.get_logger().info(
                    "[CheatCodePID] RECOVER commanded pose (first step): "
                    f"{self._format_pose_for_log(recover_pose)}"
                )
            self._record_telemetry()
            self.sleep_for(self._cfg.recover_motion_dt_sec)

        self.get_logger().info(
            "[CheatCodePID] RECOVER commanded pose (last step): "
            f"{self._format_pose_for_log(recover_pose)}"
        )

        self._set_pid_gains(self.pid_x, original_x_gains)
        self._set_pid_gains(self.pid_y, original_y_gains)
        self._reset_xy_controller_state()
        self._monitor.reset_blocked_descent_state()

        return StateResult(next_state=InsertionState.INSERT)

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
        self._monitor.reset()
        self._monitor.set_active(True)

        self._transport_force_injection_rng = np.random.default_rng(
            self._cfg.transport_force_injection_seed
        )
        self._transport_force_injection_time_sec = None
        self._transport_force_injection_applied = False
        self._transport_force_injection_wrench = None
        self._transport_force_injection_logged = False
        self._pre_insertion_pose = None

        self._telemetry.reset()
        self._insertion_state = None
        self._reset_xy_controller_state()

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

        for frame in [port_frame, cable_tip_frame]:
            if not self._wait_for_tf("base_link", frame):
                self._monitor.set_active(False)
                return False

        try:
            port_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                port_frame,
                Time(),
            )
        except TransformException as ex:
            self.get_logger().error(f"Could not look up port transform: {ex}")
            self._monitor.set_active(False)
            return False

        ctx = InsertionContext(
            task=task,
            move_robot=move_robot,
            port_transform=port_tf_stamped.transform,
            cable_tip_frame=cable_tip_frame,
            z_offset=self._cfg.transport_initial_z_offset_m,
            success=False,
        )

        try:
            self._update_pre_insertion_pose(
                port_transform=ctx.port_transform,
                z_offset=ctx.z_offset,
                dt=self._cfg.transport_dt_sec,
                reset_pids=True,
            )
            self._reset_xy_controller_state()
        except TransformException as ex:
            self.get_logger().warn(
                f"[CheatCodePID] Could not initialize pre_insertion_pose at activation: {ex}"
            )

        state_handlers = {
            InsertionState.TRANSPORT: self._execute_transport_state,
            InsertionState.INSERT: self._execute_insert_state,
            InsertionState.RECOVER: self._execute_recover_state,
        }

        state = InsertionState.TRANSPORT
        self._insertion_state = state
        self.get_logger().info(f"[CheatCodePID] Starting state machine in {state.name}")

        while state is not None:
            result = state_handlers[state](ctx)
            if result.done:
                break
            if result.next_state is None:
                break
            state = self._transition_to_state(state, result.next_state)

        self.get_logger().info("CheatCodePIDController.insert_cable() exiting...")

        self._telemetry.safe_plot(
            success=ctx.success,
            task=self._task,
            pid_mode=self._cfg.pid_mode,
            pid_x=self.pid_x,
            pid_y=self.pid_y,
            recover_pid_x_gains=self._cfg.recover_pid_x_gains,
            recover_pid_y_gains=self._cfg.recover_pid_y_gains,
        )
        self._monitor.set_active(False)

        return ctx.success


CheatCodePidController = CheatCodePIDController
