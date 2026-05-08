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
import math
import os
from collections import deque

import numpy as np

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_example_policies.ros.CheatCode import CheatCode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import Header
from tf2_ros import TransformException


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_vector(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return list(default)
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) != len(default):
        raise ValueError(f"{name} must have {len(default)} comma-separated values")
    return values


class CheatCodeModified(CheatCode):
    """CheatCode variant for collision/backoff debugging.

    This intentionally offsets the ground-truth target laterally, then reacts to
    wrist z-force during insertion using only local observations.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._misalign_x_m = _env_float("AIC_CHEATCODE_MODIFIED_MISALIGN_X_M", 0.004)
        self._misalign_y_m = _env_float("AIC_CHEATCODE_MODIFIED_MISALIGN_Y_M", 0.0)
        self._force_z_abs_threshold_n = _env_float(
            "AIC_CHEATCODE_MODIFIED_FORCE_Z_ABS_THRESHOLD_N", 1.0e9
        )
        self._force_z_delta_threshold_n = _env_float(
            "AIC_CHEATCODE_MODIFIED_FORCE_Z_DELTA_THRESHOLD_N", 1.0e9
        )
        self._force_z_step_threshold_n = _env_float(
            "AIC_CHEATCODE_MODIFIED_FORCE_Z_STEP_THRESHOLD_N", 1.0e9
        )
        self._force_z_drop_threshold_n = _env_float(
            "AIC_CHEATCODE_MODIFIED_FORCE_Z_DROP_THRESHOLD_N", 1.0e9
        )
        self._force_z_window_threshold_n = _env_float(
            "AIC_CHEATCODE_MODIFIED_FORCE_Z_WINDOW_THRESHOLD_N", 1.0e9
        )
        self._force_z_window_drop_threshold_n = _env_float(
            "AIC_CHEATCODE_MODIFIED_FORCE_Z_WINDOW_DROP_THRESHOLD_N", 1.0e9
        )
        self._force_l2_drop_threshold_n = _env_float(
            "AIC_CHEATCODE_MODIFIED_FORCE_L2_DROP_THRESHOLD_N", 1.7
        )
        self._force_l2_horizon_sec = _env_float(
            "AIC_CHEATCODE_MODIFIED_FORCE_L2_HORIZON_SEC", 0.15
        )
        self._force_median_samples = max(
            1,
            int(_env_float("AIC_CHEATCODE_MODIFIED_FORCE_MEDIAN_SAMPLES", 3.0)),
        )
        self._force_direction_step_m = _env_float(
            "AIC_CHEATCODE_MODIFIED_FORCE_DIRECTION_STEP_M", 0.015
        )
        self._force_direction_max_step_m = _env_float(
            "AIC_CHEATCODE_MODIFIED_FORCE_DIRECTION_MAX_STEP_M", 0.015
        )
        self._force_direction_xy_scale = _env_float(
            "AIC_CHEATCODE_MODIFIED_FORCE_DIRECTION_XY_SCALE", 0.3
        )
        self._stall_window_sec = _env_float(
            "AIC_CHEATCODE_MODIFIED_STALL_WINDOW_SEC", 0.25
        )
        self._stall_max_descent_m = _env_float(
            "AIC_CHEATCODE_MODIFIED_STALL_MAX_DESCENT_M", 0.00015
        )
        self._stall_force_rise_n = _env_float(
            "AIC_CHEATCODE_MODIFIED_STALL_FORCE_RISE_N", 1.0e9
        )
        self._backoff_m = _env_float("AIC_CHEATCODE_MODIFIED_BACKOFF_M", 0.015)
        self._backoff_sec = _env_float("AIC_CHEATCODE_MODIFIED_BACKOFF_SEC", 0.7)
        self._backoff_dt = _env_float("AIC_CHEATCODE_MODIFIED_BACKOFF_DT", 0.02)
        self._cartesian_stiffness = _env_vector(
            "AIC_CHEATCODE_MODIFIED_CARTESIAN_STIFFNESS",
            [90.0, 90.0, 90.0, 50.0, 50.0, 50.0],
        )
        self._cartesian_damping = _env_vector(
            "AIC_CHEATCODE_MODIFIED_CARTESIAN_DAMPING",
            [50.0, 50.0, 50.0, 20.0, 20.0, 20.0],
        )
        self._backoff_stiffness = _env_vector(
            "AIC_CHEATCODE_MODIFIED_BACKOFF_STIFFNESS",
            [90.0, 90.0, 90.0, 50.0, 50.0, 50.0],
        )
        self._backoff_damping = _env_vector(
            "AIC_CHEATCODE_MODIFIED_BACKOFF_DAMPING",
            [50.0, 50.0, 50.0, 20.0, 20.0, 20.0],
        )
        self.get_logger().info(
            "[CheatCodeModified] "
            f"misalign=({self._misalign_x_m:.4f}, {self._misalign_y_m:.4f}) m, "
            f"force_abs={self._force_z_abs_threshold_n:.3f} N, "
            f"force_delta={self._force_z_delta_threshold_n:.3f} N, "
            f"force_step={self._force_z_step_threshold_n:.3f} N, "
            f"force_drop={self._force_z_drop_threshold_n:.3f} N, "
            f"force_window={self._force_z_window_threshold_n:.3f} N, "
            f"force_window_drop={self._force_z_window_drop_threshold_n:.3f} N, "
            f"force_l2_drop={self._force_l2_drop_threshold_n:.3f} N, "
            f"force_l2_horizon={self._force_l2_horizon_sec:.3f} s, "
            f"stall_window={self._stall_window_sec:.3f} s, "
            f"backoff={self._backoff_m:.4f} m"
        )

    def calc_gripper_pose(self, *args, **kwargs) -> Pose:
        pose = super().calc_gripper_pose(*args, **kwargs)
        pose.position.x += self._misalign_x_m
        pose.position.y += self._misalign_y_m
        return pose

    def _latest_force_z(self, get_observation: GetObservationCallback) -> float | None:
        observation = get_observation()
        return self._force_z_from_observation(observation)

    def _force_vector_from_observation(self, observation) -> np.ndarray | None:
        if observation is None:
            return None
        wrench = getattr(getattr(observation, "wrist_wrench", None), "wrench", None)
        force = getattr(wrench, "force", None)
        if force is None:
            return None
        return np.array(
            [float(force.x), float(force.y), float(force.z)],
            dtype=np.float64,
        )

    def _force_z_from_observation(self, observation) -> float | None:
        force_xyz = self._force_vector_from_observation(observation)
        if force_xyz is None:
            return None
        return float(force_xyz[2])

    def _median_force(self, samples) -> np.ndarray | None:
        force_samples = [
            entry[1]
            for entry in samples
            if entry[1] is not None
        ]
        if len(force_samples) < self._force_median_samples:
            return None
        return np.median(np.stack(force_samples[-self._force_median_samples :]), axis=0)

    def _force_l2_drop_delta(self, force_history) -> np.ndarray | None:
        if len(force_history) < 2 * self._force_median_samples:
            return None
        now_time = force_history[-1][0]
        current = self._median_force(force_history)
        if current is None:
            return None
        previous_candidates = [
            entry
            for entry in force_history
            if entry[0] <= now_time - self._force_l2_horizon_sec
        ]
        previous = self._median_force(previous_candidates)
        if previous is None:
            return None
        return previous - current

    def _force_triggered(
        self,
        force_z: float | None,
        baseline_force_z: float,
        previous_force_z: float | None,
        force_history,
    ) -> tuple[bool, np.ndarray | None]:
        delta_force = self._force_l2_drop_delta(force_history)
        if delta_force is not None:
            delta_norm = float(np.linalg.norm(delta_force))
            if delta_norm > self._force_l2_drop_threshold_n:
                return True, delta_force
        if force_z is None:
            return False, None
        delta_from_baseline = force_z - baseline_force_z
        step_delta = 0.0 if previous_force_z is None else force_z - previous_force_z
        history_forces = [
            float(entry[1][2]) for entry in force_history if entry[1] is not None
        ]
        window_rise = 0.0
        window_drop = 0.0
        if history_forces:
            window_rise = force_z - min(history_forces)
            window_drop = max(history_forces) - force_z
        absolute_crossing = (
            previous_force_z is not None
            and previous_force_z < self._force_z_abs_threshold_n <= force_z
        )
        triggered = (
            absolute_crossing
            or abs(delta_from_baseline) >= self._force_z_delta_threshold_n
            or step_delta >= self._force_z_step_threshold_n
            or -step_delta >= self._force_z_drop_threshold_n
            or window_rise >= self._force_z_window_threshold_n
            or window_drop >= self._force_z_window_drop_threshold_n
        )
        if not triggered:
            return False, None
        return True, np.array([0.0, 0.0, max(0.0, -step_delta)], dtype=np.float64)

    def _stall_triggered(self, force_history) -> bool:
        if len(force_history) < 2:
            return False
        latest_time, latest_force_xyz, latest_tcp_z = force_history[-1]
        latest_force_z = None if latest_force_xyz is None else float(latest_force_xyz[2])
        if latest_force_z is None or latest_tcp_z is None:
            return False
        window = [
            entry
            for entry in force_history
            if latest_time - entry[0] <= self._stall_window_sec
        ]
        if len(window) < 2:
            return False
        oldest_time, oldest_force_xyz, oldest_tcp_z = window[0]
        oldest_force_z = None if oldest_force_xyz is None else float(oldest_force_xyz[2])
        if oldest_force_z is None or oldest_tcp_z is None:
            return False
        if latest_time <= oldest_time:
            return False
        z_descent = oldest_tcp_z - latest_tcp_z
        force_rise = latest_force_z - min(
            float(entry[1][2]) for entry in window if entry[1] is not None
        )
        return (
            z_descent <= self._stall_max_descent_m
            and force_rise >= self._stall_force_rise_n
        )

    def _sample_force_history(
        self,
        get_observation: GetObservationCallback,
        force_history,
        *,
        fallback_time: float,
    ) -> np.ndarray | None:
        observation = get_observation()
        force_xyz = self._force_vector_from_observation(observation)
        try:
            tcp_z = self._current_tcp_pose_in_base().position.z
        except TransformException:
            tcp_z = None
        now = self.time_now()
        sample_time = (
            float(now.nanoseconds) * 1e-9
            if hasattr(now, "nanoseconds")
            else fallback_time
        )
        force_history.append((sample_time, force_xyz, tcp_z))
        return force_xyz

    def _guarded_sleep_after_command(
        self,
        *,
        move_robot: MoveRobotCallback,
        get_observation: GetObservationCallback,
        force_history,
        baseline_force_z: float,
        previous_force_z: float | None,
        dt: float,
        fallback_time: float,
    ) -> tuple[bool, float | None]:
        force_xyz = self._sample_force_history(
            get_observation,
            force_history,
            fallback_time=fallback_time,
        )
        force_z = None if force_xyz is None else float(force_xyz[2])
        triggered, delta_force = self._force_triggered(
            force_z,
            baseline_force_z,
            previous_force_z,
            force_history,
        )
        if not triggered and self._stall_triggered(force_history):
            triggered = True
            delta_force = np.array([0.0, 0.0, self._backoff_m], dtype=np.float64)
        if triggered and delta_force is not None:
            self._backoff_immediately(
                move_robot,
                get_observation,
                trigger_force_z=force_z,
                delta_force=delta_force,
            )
            return True, force_z
        self.sleep_for(dt)
        return False, force_z

    def _current_tcp_pose_in_base(self) -> Pose:
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
                x=transform.rotation.x,
                y=transform.rotation.y,
                z=transform.rotation.z,
                w=transform.rotation.w,
            ),
        )

    def _send_position_motion_update(
        self,
        move_robot: MoveRobotCallback,
        pose: Pose,
        *,
        stiffness: list[float],
        damping: list[float],
    ) -> None:
        motion_update = MotionUpdate(
            header=Header(
                frame_id="base_link",
                stamp=self._parent_node.get_clock().now().to_msg(),
            ),
            pose=pose,
            target_stiffness=np.diag(stiffness).flatten(),
            target_damping=np.diag(damping).flatten(),
            feedforward_wrench_at_tip=Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0),
                torque=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            wrench_feedback_gains_at_tip=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )
        move_robot(motion_update=motion_update)

    def _send_delta_pose_target(
        self,
        move_robot: MoveRobotCallback,
        target_pose: Pose,
    ) -> None:
        gripper_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            "gripper/tcp",
            Time(),
        )
        current_pose = Pose(
            position=Point(
                x=gripper_tf_stamped.transform.translation.x,
                y=gripper_tf_stamped.transform.translation.y,
                z=gripper_tf_stamped.transform.translation.z,
            ),
            orientation=Quaternion(
                x=gripper_tf_stamped.transform.rotation.x,
                y=gripper_tf_stamped.transform.rotation.y,
                z=gripper_tf_stamped.transform.rotation.z,
                w=gripper_tf_stamped.transform.rotation.w,
            ),
        )
        from aic_model.policy import compute_delta_pose

        delta_pose = compute_delta_pose(current_pose, target_pose)
        self.set_delta_pose_target(
            move_robot=move_robot,
            delta_pose=delta_pose,
            stiffness=self._cartesian_stiffness,
            damping=self._cartesian_damping,
        )

    def _backoff_immediately(
        self,
        move_robot: MoveRobotCallback,
        get_observation: GetObservationCallback,
        *,
        trigger_force_z: float | None,
        delta_force: np.ndarray,
    ) -> None:
        try:
            target_pose = self._current_tcp_pose_in_base()
        except TransformException as ex:
            self.get_logger().warn(f"[CheatCodeModified] TF lookup failed for backoff: {ex}")
            return
        direction = np.asarray(delta_force, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-9:
            direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            direction = direction / (norm + 1.0e-6)
        delta_xyz = self._force_direction_step_m * direction
        delta_xyz[:2] *= self._force_direction_xy_scale
        delta_norm = float(np.linalg.norm(delta_xyz))
        if delta_norm > self._force_direction_max_step_m:
            delta_xyz *= self._force_direction_max_step_m / delta_norm

        gripper_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            "gripper/tcp",
            Time(),
        )
        rotation = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        qx = float(gripper_tf_stamped.transform.rotation.x)
        qy = float(gripper_tf_stamped.transform.rotation.y)
        qz = float(gripper_tf_stamped.transform.rotation.z)
        qw = float(gripper_tf_stamped.transform.rotation.w)
        rotation[0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
        rotation[0, 1] = 2.0 * (qx * qy - qz * qw)
        rotation[0, 2] = 2.0 * (qx * qz + qy * qw)
        rotation[1, 0] = 2.0 * (qx * qy + qz * qw)
        rotation[1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
        rotation[1, 2] = 2.0 * (qy * qz - qx * qw)
        rotation[2, 0] = 2.0 * (qx * qz - qy * qw)
        rotation[2, 1] = 2.0 * (qy * qz + qx * qw)
        rotation[2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)
        delta_base = rotation @ delta_xyz
        if delta_base[2] < 0.0:
            delta_base[2] = abs(delta_base[2])

        target_pose.position.x += float(delta_base[0])
        target_pose.position.y += float(delta_base[1])
        target_pose.position.z += float(delta_base[2])
        steps = max(1, int(math.ceil(self._backoff_sec / self._backoff_dt)))
        self.get_logger().warn(
            "[CheatCodeModified] Force trigger: "
            f"fz={trigger_force_z}, delta_force={delta_force.tolist()}, "
            f"delta_tcp={delta_xyz.tolist()}, "
            f"delta_base={delta_base.tolist()}, "
            f"sending backoff target for {steps} cycles"
        )
        for _ in range(steps):
            if self._task_completed_in_simulation(self._task):
                return
            self._send_position_motion_update(
                move_robot,
                target_pose,
                stiffness=self._backoff_stiffness,
                damping=self._backoff_damping,
            )
            self._latest_force_z(get_observation)
            self.sleep_for(self._backoff_dt)

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"CheatCodeModified.insert_cable() task: {task}")
        self._task = task
        self._latest_insertion_event_namespace = ""

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

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
        interpolation_duration_sec = 5.5
        dt = 0.05
        steps = int(interpolation_duration_sec / dt)
        self.get_logger().info(
            f"[CheatCodeModified] Interpolating for {interpolation_duration_sec} seconds"
        )

        for t in range(steps + 1):
            if self._task_completed_in_simulation(task):
                return True
            interp_fraction = t / steps
            try:
                self._send_delta_pose_target(
                    move_robot=move_robot,
                    target_pose=self.calc_gripper_pose(
                        port_transform,
                        slerp_fraction=interp_fraction,
                        position_fraction=interp_fraction,
                        z_offset=z_offset,
                        reset_xy_integrator=True,
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during interpolation: {ex}")
            self.sleep_for(dt)

        self.get_logger().info("[CheatCodeModified] Descent")
        baseline_force_z = self._latest_force_z(get_observation) or 0.0
        previous_force_z: float | None = baseline_force_z
        force_history = deque(maxlen=max(2, int(math.ceil(1.0 / dt)) + 1))
        self.get_logger().info(
            "[CheatCodeModified] guarded descent force baseline "
            f"z={baseline_force_z:.3f} N"
        )

        handoff_blend_sec = 2.0
        handoff_speed_mps = 0.02
        start_descent_z_offset = 0.005
        rate_limited_handoff_sec = abs(z_offset - start_descent_z_offset) / handoff_speed_mps
        blend_steps = max(1, int(max(handoff_blend_sec, rate_limited_handoff_sec) / dt))
        for t in range(blend_steps):
            if self._task_completed_in_simulation(task):
                return True
            fraction = (t + 1) / blend_steps
            try:
                self._send_delta_pose_target(
                    move_robot=move_robot,
                    target_pose=self.calc_gripper_pose(
                        port_transform,
                        slerp_fraction=1.0,
                        position_fraction=1.0,
                        z_offset=z_offset + fraction * (start_descent_z_offset - z_offset),
                        reset_xy_integrator=True,
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during insertion handoff blend: {ex}")
            triggered, previous_force_z = self._guarded_sleep_after_command(
                move_robot=move_robot,
                get_observation=get_observation,
                force_history=force_history,
                baseline_force_z=baseline_force_z,
                previous_force_z=previous_force_z,
                dt=dt,
                fallback_time=float(t) * dt,
            )
            if triggered:
                return True

        z_offset = start_descent_z_offset
        settle_duration_sec = 1.0
        settle_steps = max(1, int(settle_duration_sec / dt))
        for _ in range(settle_steps):
            if self._task_completed_in_simulation(task):
                return True
            try:
                self._send_delta_pose_target(
                    move_robot=move_robot,
                    target_pose=self.calc_gripper_pose(
                        port_transform,
                        z_offset=z_offset,
                        reset_xy_integrator=True,
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during insertion settle: {ex}")
            triggered, previous_force_z = self._guarded_sleep_after_command(
                move_robot=move_robot,
                get_observation=get_observation,
                force_history=force_history,
                baseline_force_z=baseline_force_z,
                previous_force_z=previous_force_z,
                dt=dt,
                fallback_time=handoff_blend_sec + float(_) * dt,
            )
            if triggered:
                return True

        self.get_logger().info(
            f"[CheatCodeModified] insertion force baseline remains z={baseline_force_z:.3f} N"
        )

        insertion_speed_mps = 0.0009
        end_z_offset = -0.015
        insertion_distance = max(0.0, z_offset - end_z_offset)
        insertion_duration = max(dt, insertion_distance / insertion_speed_mps)
        insertion_steps = max(1, int(math.ceil(insertion_duration / dt)))
        for step in range(1, insertion_steps + 1):
            if self._task_completed_in_simulation(task):
                return True

            force_xyz = self._sample_force_history(
                get_observation,
                force_history,
                fallback_time=handoff_blend_sec + settle_duration_sec + float(step) * dt,
            )
            force_z = None if force_xyz is None else float(force_xyz[2])
            triggered, delta_force = self._force_triggered(
                force_z,
                baseline_force_z,
                previous_force_z,
                force_history,
            )
            if not triggered and self._stall_triggered(force_history):
                triggered = True
                delta_force = np.array([0.0, 0.0, self._backoff_m], dtype=np.float64)
            if triggered and delta_force is not None:
                self._backoff_immediately(
                    move_robot,
                    get_observation,
                    trigger_force_z=force_z,
                    delta_force=delta_force,
                )
                return True
            previous_force_z = force_z

            fraction = (
                10.0 * (step / insertion_steps) ** 3
                - 15.0 * (step / insertion_steps) ** 4
                + 6.0 * (step / insertion_steps) ** 5
            )
            z_offset = start_descent_z_offset - fraction * insertion_distance
            try:
                self._send_delta_pose_target(
                    move_robot=move_robot,
                    target_pose=self.calc_gripper_pose(port_transform, z_offset=z_offset),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during insertion: {ex}")
            self.sleep_for(dt)

        self.get_logger().info("Waiting briefly for insertion event...")
        wait_started = self.time_now()
        wait_timeout = Duration(seconds=5.0)
        while (self.time_now() - wait_started) < wait_timeout:
            if self._task_completed_in_simulation(task):
                break
            self.sleep_for(0.05)

        self.get_logger().info("CheatCodeModified.insert_cable() exiting...")
        return True
