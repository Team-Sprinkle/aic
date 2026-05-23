from __future__ import annotations

import os
import base64
import subprocess
import time
from typing import Any

import numpy as np
import yaml

from aic_model.policy import (
    DEFAULT_CARTESIAN_DAMPING,
    DEFAULT_CARTESIAN_STIFFNESS,
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
    build_pose_from_vectors,
    pose_to_position_motion_update,
)
from aic_task_interfaces.msg import Task
from std_msgs.msg import String

from gazebo_rl.action import DEFAULT_MAX_ROTATION_RAD, DEFAULT_MAX_TRANSLATION_M, delta_tcp_action_from_array
from gazebo_rl.ipc import IPCError, JsonLineConnection, connect_with_retry
from gazebo_rl.observation import observation_to_dict


class GazeboRLBridgePolicy(Policy):
    """AIC policy that delegates each motion tick to an external RL trainer."""

    def __init__(self, parent_node):
        self._connection: JsonLineConnection | None = None
        self._task: Task | None = None
        self._step_count = 0
        self._latest_insertion_event_namespace = ""
        super().__init__(parent_node)
        self._insertion_event_sub = self._parent_node.create_subscription(
            String,
            "/scoring/insertion_event",
            self._insertion_event_callback,
            10,
        )

    def _env_float(self, name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)))
        except ValueError:
            self.get_logger().warn(f"Invalid {name}; using default {default}")
            return default

    def _env_int(self, name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)))
        except ValueError:
            self.get_logger().warn(f"Invalid {name}; using default {default}")
            return default

    def _env_bool(self, name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _episode_start_near_gate(self) -> dict[str, Any] | None:
        if not self._env_bool("AIC_GAZEBO_RL_PREPOSITION_START_NEAR_GATE", False):
            return None
        path = os.environ.get("AIC_GAZEBO_RL_EPISODE_CONFIG")
        if not path:
            return None
        try:
            data = yaml.safe_load(open(path, encoding="utf-8"))
        except Exception as ex:
            raise RuntimeError(f"Could not load Gazebo episode config for pre-positioning: {path}: {ex}") from ex
        start = ((data or {}).get("scene") or {}).get("start_near_gate")
        return start if isinstance(start, dict) else None

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

    def _connect(self, task: Task) -> JsonLineConnection:
        host = os.environ.get("AIC_GAZEBO_RL_HOST", "127.0.0.1")
        port = self._env_int("AIC_GAZEBO_RL_PORT", 8765)
        timeout = self._env_float("AIC_GAZEBO_RL_CONNECT_TIMEOUT_SEC", 60.0)
        conn = connect_with_retry(host, port, timeout_sec=timeout)
        conn.send(
            "hello",
            {
                "policy": "gazebo_rl.bridge_policy.GazeboRLBridgePolicy",
                "action_space": "relative_tcp_delta",
                "action_shape": [6],
                "max_translation_m": DEFAULT_MAX_TRANSLATION_M,
                "max_rotation_rad": DEFAULT_MAX_ROTATION_RAD,
                "task": {
                    "id": task.id,
                    "cable_name": task.cable_name,
                    "plug_name": task.plug_name,
                    "target_module_name": task.target_module_name,
                    "port_name": task.port_name,
                },
            },
        )
        return conn

    def _send_error(self, message: str) -> None:
        if self._connection is None:
            return
        try:
            self._connection.send("error", {"message": message, "step_count": self._step_count})
        except Exception:
            pass

    def _send_delta_action(self, move_robot: MoveRobotCallback, raw_action: Any) -> None:
        delta = delta_tcp_action_from_array(raw_action)
        motion_update = pose_to_position_motion_update(
            build_pose_from_vectors(delta.delta_position_xyz, delta.delta_quaternion_xyzw),
            stamp=self._parent_node.get_clock().now().to_msg(),
            frame_id="gripper/tcp",
            stiffness=DEFAULT_CARTESIAN_STIFFNESS,
            damping=DEFAULT_CARTESIAN_DAMPING,
        )
        result = move_robot(motion_update=motion_update)
        if result is False:
            raise RuntimeError("move_robot rejected the Gazebo RL motion update")

    def _send_absolute_tcp_pose(
        self,
        move_robot: MoveRobotCallback,
        position_xyz: np.ndarray,
        orientation_xyzw: np.ndarray,
    ) -> None:
        motion_update = pose_to_position_motion_update(
            build_pose_from_vectors(position_xyz, orientation_xyzw),
            stamp=self._parent_node.get_clock().now().to_msg(),
            frame_id="base_link",
            stiffness=DEFAULT_CARTESIAN_STIFFNESS,
            damping=DEFAULT_CARTESIAN_DAMPING,
        )
        result = move_robot(motion_update=motion_update)
        if result is False:
            raise RuntimeError("move_robot rejected the Gazebo RL pre-position motion update")

    def _live_start_near_gate_target(
        self,
        task: Task,
        start: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        tf_buffer = getattr(self._parent_node, "_tf_buffer", None)
        if tf_buffer is None:
            raise RuntimeError("Cannot pre-position start_near_gate: parent node has no TF buffer")
        from rclpy.time import Time

        entrance_frame = f"task_board/{task.target_module_name}/{task.port_name}_link_entrance"
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        entrance_tf = tf_buffer.lookup_transform("base_link", entrance_frame, Time()).transform
        port_tf = tf_buffer.lookup_transform("base_link", port_frame, Time()).transform
        entrance = np.asarray(
            [entrance_tf.translation.x, entrance_tf.translation.y, entrance_tf.translation.z],
            dtype=float,
        )
        port = np.asarray([port_tf.translation.x, port_tf.translation.y, port_tf.translation.z], dtype=float)
        axis = port - entrance
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-9:
            raise RuntimeError(f"Cannot infer insertion axis from {entrance_frame} and {port_frame}")
        axis = axis / axis_norm

        quat = np.asarray(
            [
                entrance_tf.rotation.x,
                entrance_tf.rotation.y,
                entrance_tf.rotation.z,
                entrance_tf.rotation.w,
            ],
            dtype=float,
        )
        quat_norm = float(np.linalg.norm(quat))
        if quat_norm <= 1e-9:
            quat = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
        else:
            quat = quat / quat_norm
        x, y, z, w = quat
        rot = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )
        lateral = rot[:, 1]
        lateral = lateral - float(np.dot(lateral, axis)) * axis
        lateral_norm = float(np.linalg.norm(lateral))
        if lateral_norm <= 1e-9:
            lateral = np.asarray([0.0, 1.0, 0.0], dtype=float)
            lateral = lateral - float(np.dot(lateral, axis)) * axis
            lateral_norm = float(np.linalg.norm(lateral))
        if lateral_norm <= 1e-9:
            lateral = np.asarray([1.0, 0.0, 0.0], dtype=float)
            lateral = lateral - float(np.dot(lateral, axis)) * axis
            lateral_norm = float(np.linalg.norm(lateral))
        lateral = lateral / lateral_norm

        axial_distance = float(start.get("axial_distance_m", start.get("distance", 0.0)) or 0.0)
        lateral_distance = float(start.get("lateral_distance_m", 0.0) or 0.0)
        target = entrance - axis * axial_distance + lateral * lateral_distance
        return target, entrance, axis, lateral, axial_distance, lateral_distance

    def _lookup_pose(
        self,
        target_frame: str,
        source_frame: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        tf_buffer = getattr(self._parent_node, "_tf_buffer", None)
        if tf_buffer is None:
            raise RuntimeError("Cannot apply start_near_gate: parent node has no TF buffer")
        from rclpy.time import Time

        tf = tf_buffer.lookup_transform(target_frame, source_frame, Time()).transform
        position = np.asarray([tf.translation.x, tf.translation.y, tf.translation.z], dtype=float)
        quat_xyzw = np.asarray([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w], dtype=float)
        norm = float(np.linalg.norm(quat_xyzw))
        if norm <= 1e-9:
            quat_xyzw = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
        else:
            quat_xyzw = quat_xyzw / norm
        return position, quat_xyzw

    @staticmethod
    def _rotation_matrix_from_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
        x, y, z, w = quat_xyzw
        return np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )

    def _live_start_near_gate_target_world(
        self,
        task: Task,
        start: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        entrance_frame = f"task_board/{task.target_module_name}/{task.port_name}_link_entrance"
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        entrance, entrance_quat = self._lookup_pose("world", entrance_frame)
        port, _ = self._lookup_pose("world", port_frame)
        axis = port - entrance
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-9:
            raise RuntimeError(f"Cannot infer insertion axis from {entrance_frame} and {port_frame}")
        axis = axis / axis_norm
        lateral = self._rotation_matrix_from_xyzw(entrance_quat)[:, 1]
        lateral = lateral - float(np.dot(lateral, axis)) * axis
        lateral_norm = float(np.linalg.norm(lateral))
        if lateral_norm <= 1e-9:
            lateral = np.asarray([0.0, 1.0, 0.0], dtype=float)
            lateral = lateral - float(np.dot(lateral, axis)) * axis
            lateral_norm = float(np.linalg.norm(lateral))
        if lateral_norm <= 1e-9:
            lateral = np.asarray([1.0, 0.0, 0.0], dtype=float)
            lateral = lateral - float(np.dot(lateral, axis)) * axis
            lateral_norm = float(np.linalg.norm(lateral))
        lateral = lateral / lateral_norm
        axial_distance = float(start.get("axial_distance_m", start.get("distance", 0.0)) or 0.0)
        lateral_distance = float(start.get("lateral_distance_m", 0.0) or 0.0)
        target = entrance - axis * axial_distance + lateral * lateral_distance
        return target, entrance, axis, lateral, axial_distance, lateral_distance

    def _set_task_board_pose_world(self, position: np.ndarray, orientation_xyzw: np.ndarray) -> None:
        request = (
            'name: "task_board" '
            f"position {{ x: {position[0]:.9f} y: {position[1]:.9f} z: {position[2]:.9f} }} "
            "orientation { "
            f"x: {orientation_xyzw[0]:.9f} y: {orientation_xyzw[1]:.9f} "
            f"z: {orientation_xyzw[2]:.9f} w: {orientation_xyzw[3]:.9f} "
            "}"
        )
        request_b64 = base64.b64encode(request.encode("utf-8")).decode("ascii")
        helper = "\n".join(
            [
                "import base64",
                "import subprocess",
                "import sys",
                "req = base64.b64decode(sys.argv[1]).decode('utf-8')",
                "cmd = [",
                "    'gz', 'service',",
                "    '-s', '/world/aic_world/set_pose',",
                "    '--reqtype', 'gz.msgs.Pose',",
                "    '--reptype', 'gz.msgs.Boolean',",
                "    '--timeout', '2000',",
                "    '--req', req,",
                "]",
                "p = subprocess.run(cmd)",
                "sys.exit(p.returncode)",
            ]
        )
        distrobox = os.environ.get("AIC_GAZEBO_RL_SIM_DISTROBOX", "").strip()
        if distrobox:
            cmd = [
                "distrobox",
                "enter",
                "-r",
                "--no-tty",
                distrobox,
                "--",
                "bash",
                "-lc",
                "source /ws_aic/install/setup.bash; exec python3 - \"$1\"",
                "gz-set-pose",
                request_b64,
            ]
            env = os.environ.copy()
            env["DBX_CONTAINER_MANAGER"] = env.get("DBX_CONTAINER_MANAGER", "docker")
        else:
            cmd = [
                "bash",
                "-lc",
                "source /ws_aic/install/setup.bash 2>/dev/null || true; exec python3 - \"$1\"",
                "gz-set-pose",
                request_b64,
            ]
            env = os.environ.copy()
        result = subprocess.run(
            cmd,
            env=env,
            input=helper,
            text=True,
            capture_output=True,
            timeout=8.0,
            check=False,
        )
        combined = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0 or "data: false" in combined:
            raise RuntimeError(
                "Failed to move task_board through Gazebo set_pose service; "
                f"returncode={result.returncode}, output={combined.strip()}"
            )

    @staticmethod
    def _start_near_gate_measurement(
        *,
        tip_center: np.ndarray,
        entrance: np.ndarray,
        axis: np.ndarray,
        lateral: np.ndarray,
        requested_axial: float,
        requested_lateral: float,
        target: np.ndarray,
    ) -> dict[str, float]:
        delta_from_gate = tip_center - entrance
        actual_axial = -float(np.dot(delta_from_gate, axis))
        actual_lateral_signed = float(np.dot(delta_from_gate, lateral))
        lateral_vector = delta_from_gate - float(np.dot(delta_from_gate, axis)) * axis
        actual_lateral = float(np.linalg.norm(lateral_vector))
        target_error = float(np.linalg.norm(target - tip_center))
        return {
            "error_m": target_error,
            "tip_to_gate_distance_m": float(np.linalg.norm(delta_from_gate)),
            "requested_axial_distance_m": float(requested_axial),
            "requested_lateral_distance_m": float(requested_lateral),
            "actual_axial_distance_m": actual_axial,
            "actual_lateral_distance_m": actual_lateral,
            "actual_lateral_signed_m": actual_lateral_signed,
            "axial_error_m": abs(actual_axial - requested_axial),
            "lateral_error_m": abs(actual_lateral - requested_lateral),
        }

    def _preposition_start_near_gate(
        self,
        *,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        command_dt_sec: float,
        ground_truth: bool,
        include_images: bool,
        send_feedback: SendFeedbackCallback,
    ) -> dict[str, Any] | None:
        start = self._episode_start_near_gate()
        if not start:
            return None
        mode = os.environ.get("AIC_GAZEBO_RL_START_NEAR_GATE_MODE", "task_board").strip().lower()
        if mode in {"task_board", "board", "scene"}:
            return self._apply_start_near_gate_by_moving_task_board(
                task=task,
                get_observation=get_observation,
                ground_truth=ground_truth,
                include_images=include_images,
            )
        if mode not in {"tcp", "preposition"}:
            raise RuntimeError(f"Unsupported AIC_GAZEBO_RL_START_NEAR_GATE_MODE={mode!r}")
        target, live_entrance, live_axis, live_lateral, requested_axial, requested_lateral = (
            self._live_start_near_gate_target(task, start)
        )
        if target.shape != (3,):
            raise RuntimeError("start_near_gate pre-positioning requires a 3D reference/body start position")
        max_steps = self._env_int("AIC_GAZEBO_RL_PREPOSITION_MAX_STEPS", 240)
        tolerance = self._env_float("AIC_GAZEBO_RL_PREPOSITION_TOLERANCE_M", 0.003)
        max_translation = self._env_float("AIC_GAZEBO_RL_PREPOSITION_MAX_TRANSLATION_M", DEFAULT_MAX_TRANSLATION_M)
        last_error = float("inf")
        last_measurement: dict[str, float] | None = None
        for step in range(max_steps):
            obs_msg = get_observation()
            obs = observation_to_dict(
                obs_msg,
                task=task,
                step_count=-max_steps + step,
                tf_buffer=getattr(self._parent_node, "_tf_buffer", None),
                ground_truth=ground_truth,
                include_images=include_images,
                logger=lambda msg: self.get_logger().warn(msg),
            )
            pose = ((obs.get("oracle") or {}).get("plug_pose_base_link") or {}).get("position")
            if pose is None:
                raise RuntimeError(
                    "Cannot pre-position start_near_gate: live tip-center frame is unavailable. "
                    "For SFP tasks this must be cable/<plug_name>_link, e.g. cable_0/sfp_tip_link; "
                    "TCP fallback is disabled because start_near_gate is defined as tip center to port entrance center."
                )
            tcp_pose = ((obs.get("controller") or {}).get("current_tcp_pose") or {})
            tcp_position = tcp_pose.get("position")
            tcp_orientation = tcp_pose.get("orientation_xyzw")
            if tcp_position is None or tcp_orientation is None:
                raise RuntimeError("Cannot pre-position start_near_gate: current TCP pose is unavailable")
            current = np.asarray(pose, dtype=float)
            delta_world = target - current
            measurement = self._start_near_gate_measurement(
                tip_center=current,
                entrance=live_entrance,
                axis=live_axis,
                lateral=live_lateral,
                requested_axial=requested_axial,
                requested_lateral=requested_lateral,
                target=target,
            )
            last_measurement = measurement
            last_error = measurement["error_m"]
            if last_error <= tolerance:
                self.get_logger().info(
                    "start_near_gate pre-position reached tolerance using actual tip center: "
                    f"error_m={last_error:.6f}, actual_axial_m={measurement['actual_axial_distance_m']:.6f}, "
                    f"actual_lateral_m={measurement['actual_lateral_distance_m']:.6f}, steps={step}"
                )
                return {
                    "mode": "preposition",
                    "steps": step,
                    **measurement,
                    "source": "plug_pose_base_link",
                    "measurement": "tip_center_to_port_entrance_center",
                    "target_position_world": target.astype(float).tolist(),
                    "current_tip_center_position_world": current.astype(float).tolist(),
                    "live_entrance_position_world": live_entrance.astype(float).tolist(),
                    "live_insertion_axis_world": live_axis.astype(float).tolist(),
                    "live_lateral_direction_world": live_lateral.astype(float).tolist(),
                }
            step_norm = float(np.linalg.norm(delta_world))
            if step_norm > max_translation:
                step_delta = delta_world / step_norm * max_translation
            else:
                step_delta = delta_world
            next_tcp_position = np.asarray(tcp_position, dtype=float) + step_delta
            next_tcp_orientation = np.asarray(tcp_orientation, dtype=float)
            try:
                self._send_absolute_tcp_pose(move_robot, next_tcp_position, next_tcp_orientation)
            except Exception as ex:
                send_feedback(f"start_near_gate pre-position failed at step {step}: {ex}")
                raise
            self.sleep_for(command_dt_sec)
        raise RuntimeError(
            "start_near_gate pre-position did not reach target within "
            f"{max_steps} steps; final_error_m={last_error:.6f}, tolerance_m={tolerance:.6f}, "
            f"last_measurement={last_measurement}"
        )

    def _apply_start_near_gate_by_moving_task_board(
        self,
        *,
        task: Task,
        get_observation: GetObservationCallback,
        ground_truth: bool,
        include_images: bool,
    ) -> dict[str, Any]:
        start = self._episode_start_near_gate()
        if not start:
            raise RuntimeError("start_near_gate task-board reset requested without start metadata")
        tip_frame = f"{task.cable_name}/{task.plug_name}_link"
        tolerance = self._env_float("AIC_GAZEBO_RL_PREPOSITION_TOLERANCE_M", 0.003)
        max_iterations = int(self._env_float("AIC_GAZEBO_RL_TASK_BOARD_RESET_MAX_ITERATIONS", 6))
        applied_deltas: list[list[float]] = []
        measurement: dict[str, float] | None = None
        tip_center_world: np.ndarray | None = None
        entrance_world: np.ndarray | None = None
        axis_world: np.ndarray | None = None
        lateral_world: np.ndarray | None = None
        target_world: np.ndarray | None = None
        requested_axial = 0.0
        requested_lateral = 0.0

        for iteration in range(max(1, max_iterations)):
            target_world, entrance_world, axis_world, lateral_world, requested_axial, requested_lateral = (
                self._live_start_near_gate_target_world(task, start)
            )
            tip_center_world, _ = self._lookup_pose("world", tip_frame)
            measurement = self._start_near_gate_measurement(
                tip_center=tip_center_world,
                entrance=entrance_world,
                axis=axis_world,
                lateral=lateral_world,
                requested_axial=requested_axial,
                requested_lateral=requested_lateral,
                target=target_world,
            )
            if measurement["error_m"] <= tolerance:
                break
            board_position, board_orientation = self._lookup_pose("world", "task_board")
            desired_entrance = tip_center_world + axis_world * requested_axial - lateral_world * requested_lateral
            board_delta = desired_entrance - entrance_world
            applied_deltas.append(board_delta.astype(float).tolist())
            self._set_task_board_pose_world(board_position + board_delta, board_orientation)
            time.sleep(0.25)
        else:
            pass

        # Verify through the same live observation path used by reward calculation.
        obs_msg = get_observation()
        obs = observation_to_dict(
            obs_msg,
            task=task,
            step_count=-1,
            tf_buffer=getattr(self._parent_node, "_tf_buffer", None),
            ground_truth=ground_truth,
            include_images=include_images,
            logger=lambda msg: self.get_logger().warn(msg),
        )
        tip_pose = ((obs.get("oracle") or {}).get("plug_pose_base_link") or {}).get("position")
        if tip_pose is None:
            raise RuntimeError(f"Cannot verify start_near_gate: live tip-center frame {tip_frame} is unavailable")
        target_base, entrance_base, axis_base, lateral_base, requested_axial, requested_lateral = (
            self._live_start_near_gate_target(task, start)
        )
        current = np.asarray(tip_pose, dtype=float)
        measurement = self._start_near_gate_measurement(
            tip_center=current,
            entrance=entrance_base,
            axis=axis_base,
            lateral=lateral_base,
            requested_axial=requested_axial,
            requested_lateral=requested_lateral,
            target=target_base,
        )
        if measurement["error_m"] > tolerance:
            raise RuntimeError(
                "start_near_gate task-board reset did not verify against live tip center; "
                f"tolerance_m={tolerance:.6f}, measurement={measurement}, applied_deltas={applied_deltas}"
            )
        self.get_logger().info(
            "start_near_gate task-board reset verified using actual tip center: "
            f"error_m={measurement['error_m']:.6f}, actual_axial_m={measurement['actual_axial_distance_m']:.6f}, "
            f"actual_lateral_m={measurement['actual_lateral_distance_m']:.6f}, iterations={len(applied_deltas)}"
        )
        assert target_world is not None
        assert entrance_world is not None
        assert axis_world is not None
        assert lateral_world is not None
        assert tip_center_world is not None
        return {
            "mode": "task_board_set_pose",
            "steps": len(applied_deltas),
            **measurement,
            "source": "plug_pose_base_link",
            "measurement": "tip_center_to_port_entrance_center",
            "task_board_deltas_world": applied_deltas,
            "target_position_world": target_world.astype(float).tolist(),
            "current_tip_center_position_world": tip_center_world.astype(float).tolist(),
            "live_entrance_position_world": entrance_world.astype(float).tolist(),
            "live_insertion_axis_world": axis_world.astype(float).tolist(),
            "live_lateral_direction_world": lateral_world.astype(float).tolist(),
        }

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self._task = task
        self._step_count = 0
        self._latest_insertion_event_namespace = ""
        command_dt_sec = self._env_float("AIC_GAZEBO_RL_COMMAND_DT_SEC", 0.05)
        max_steps = self._env_int("AIC_GAZEBO_RL_MAX_STEPS", 50)
        action_timeout_sec = self._env_float("AIC_GAZEBO_RL_ACTION_TIMEOUT_SEC", 30.0)
        ground_truth = self._env_bool("AIC_GAZEBO_RL_GROUND_TRUTH", False)
        include_images = self._env_bool("AIC_GAZEBO_RL_INCLUDE_IMAGES", False)

        self.get_logger().info(
            f"GazeboRLBridgePolicy starting task {task.id}; max_steps={max_steps}, "
            f"dt={command_dt_sec}, include_images={include_images}"
        )

        try:
            self._connection = self._connect(task)
            preposition_report = self._preposition_start_near_gate(
                task=task,
                get_observation=get_observation,
                move_robot=move_robot,
                command_dt_sec=command_dt_sec,
                ground_truth=ground_truth,
                include_images=include_images,
                send_feedback=send_feedback,
            )
            if preposition_report is not None:
                self._connection.send("preposition", preposition_report)
        except Exception as ex:
            send_feedback(f"Gazebo RL IPC connection failed: {ex}")
            self.get_logger().error(f"Gazebo RL IPC connection failed: {ex}")
            return False

        succeeded = False
        try:
            while self._step_count < max_steps:
                if self._task_completed_in_simulation(task):
                    succeeded = True
                    self._connection.send("done", {"reason": "insertion_event", "step_count": self._step_count})
                    break

                try:
                    obs_msg = get_observation()
                    obs = observation_to_dict(
                        obs_msg,
                        task=task,
                        step_count=self._step_count,
                        tf_buffer=getattr(self._parent_node, "_tf_buffer", None),
                        ground_truth=ground_truth,
                        include_images=include_images,
                        logger=lambda msg: self.get_logger().warn(msg),
                    )
                except Exception as ex:
                    send_feedback(f"Observation conversion failed at step {self._step_count}: {ex}")
                    self.get_logger().warn(f"Observation conversion failed, skipping tick: {ex}")
                    self.sleep_for(command_dt_sec)
                    continue

                self._connection.send(
                    "observation",
                    {
                        "observation": obs,
                        "step_count": self._step_count,
                        "terminated": False,
                    },
                )

                try:
                    message = self._connection.recv(timeout_sec=action_timeout_sec)
                except TimeoutError as ex:
                    send_feedback(str(ex))
                    self._send_error(str(ex))
                    break

                if message.type == "done":
                    self.get_logger().info(f"Trainer ended rollout: {message.payload}")
                    break
                if message.type != "action":
                    raise IPCError(f"Expected action message, received {message.type}")

                try:
                    self._send_delta_action(move_robot, message.payload.get("action"))
                except Exception as ex:
                    send_feedback(f"Invalid Gazebo RL action at step {self._step_count}: {ex}")
                    self._send_error(str(ex))
                    break

                self._step_count += 1
                self.sleep_for(command_dt_sec)

            else:
                self._connection.send("done", {"reason": "max_steps", "step_count": self._step_count})
        except (IPCError, OSError) as ex:
            send_feedback(f"Gazebo RL IPC disconnected: {ex}")
            self.get_logger().warn(f"Gazebo RL IPC disconnected: {ex}")
        except Exception as ex:
            send_feedback(f"Gazebo RL bridge error: {ex}")
            self.get_logger().error(f"Gazebo RL bridge error: {ex}")
            self._send_error(str(ex))
        finally:
            if self._connection is not None:
                try:
                    self._connection.close()
                except Exception:
                    pass
                self._connection = None

        self.get_logger().info(
            f"GazeboRLBridgePolicy exiting task {task.id}; steps={self._step_count}, succeeded={succeeded}"
        )
        return True
