"""Official ROS-path trace capture with native Gazebo-side parity snapshots.

This module is intended to run inside the official AIC ROS / Gazebo container.
It drives the official controller with participant-facing MotionUpdate commands
while recording:

- official ROS observations / controller state / joint state / wrench / contacts
- native Gazebo observations via the training-side Gazebo transport client

The result is a fixed-rollout parity artifact that compares the official ROS
surface to the new Gazebo-native extraction path against the same live world.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


@dataclass(frozen=True)
class FixedVelocityAction:
    linear_xyz: tuple[float, float, float]
    angular_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    frame_id: str = "base_link"
    sim_steps: int = 25


DEFAULT_FIXED_ACTIONS: tuple[FixedVelocityAction, ...] = (
    FixedVelocityAction((0.0, 0.0, 0.0), sim_steps=10),
    FixedVelocityAction((0.01, 0.0, 0.0)),
    FixedVelocityAction((0.0, -0.01, 0.0)),
    FixedVelocityAction((0.0, 0.0, -0.01)),
)


class CameraBridgeSidecar:
    """Dedicated non-lazy camera bridge for image-mode trace capture."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._process = subprocess.Popen(
            [
                "ros2",
                "run",
                "ros_gz_bridge",
                "parameter_bridge",
                "/left_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
                "/center_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
                "/right_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        time.sleep(1.0)

    def close(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3.0)
        self._process = None


def _ensure_repo_local_gazebo_env_on_path() -> None:
    repo_gazebo_env = Path(__file__).resolve().parents[1] / "aic_utils" / "aic_gazebo_env"
    if repo_gazebo_env.exists():
        sys.path.insert(0, str(repo_gazebo_env))


def capture_official_and_native_trace(
    *,
    output_json: str | None = None,
    actions: tuple[FixedVelocityAction, ...] = DEFAULT_FIXED_ACTIONS,
    include_images: bool = False,
    image_settle_ticks: int = 10,
) -> dict[str, Any]:
    _ensure_repo_local_gazebo_env_on_path()
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from rclpy.qos import ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import Twist, Vector3, Wrench
    from aic_control_interfaces.msg import (
        ControllerState,
        MotionUpdate,
        TargetMode,
        TrajectoryGenerationMode,
    )
    from aic_control_interfaces.srv import ChangeTargetMode
    from aic_gazebo_env.gazebo_client import GazeboCliClient, GazeboCliClientConfig  # type: ignore
    from aic_gazebo_env.protocol import GetObservationRequest  # type: ignore
    if include_images:
        from sensor_msgs.msg import Image
        from aic_gym_gz.io import _ros_image_to_array, summarize_image_batch

    class Recorder(Node):
        def __init__(self) -> None:
            super().__init__("aic_official_native_trace_recorder")
            self._change_target_mode = self.create_client(
                ChangeTargetMode,
                "/aic_controller/change_target_mode",
            )
            self.motion_pub = self.create_publisher(
                MotionUpdate,
                "/aic_controller/pose_commands",
                10,
            )
            self.latest_controller_state: ControllerState | None = None
            controller_state_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )
            for topic in (
                "/aic_controller/controller_state",
                "/controller_manager/aic_controller/controller_state",
                "/controller_manager/controller_state",
            ):
                self.create_subscription(
                    ControllerState,
                    topic,
                    self._controller_state_callback,
                    controller_state_qos,
                )
            self.latest_images: dict[str, np.ndarray] = {}
            self.latest_image_timestamps: dict[str, float] = {}
            if include_images:
                for name, topic in {
                    "left": "/left_camera/image",
                    "center": "/center_camera/image",
                    "right": "/right_camera/image",
                }.items():
                    self.create_subscription(
                        Image,
                        topic,
                        lambda message, camera_name=name: self._image_callback(camera_name, message),
                        qos_profile_sensor_data,
                    )

        def _controller_state_callback(self, message: ControllerState) -> None:
            self.latest_controller_state = message

        def _image_callback(self, camera_name: str, message: Any) -> None:
            image = _ros_image_to_array(message, expected_shape=(64, 64, 3))
            timestamp = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
            self.latest_images[camera_name] = image
            self.latest_image_timestamps[camera_name] = timestamp

        def set_cartesian_target_mode(self) -> None:
            request = ChangeTargetMode.Request()
            request.target_mode.mode = TargetMode.MODE_CARTESIAN
            self._call_service(
                self._change_target_mode,
                request,
                service_name="/aic_controller/change_target_mode",
            )

        def wait_for_pose_command_subscriber(self, *, timeout_s: float) -> None:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if self.motion_pub.get_subscription_count() > 0:
                    return
                rclpy.spin_once(self, timeout_sec=0.1)
            raise TimeoutError(
                "Timed out waiting for /aic_controller/pose_commands subscriber."
            )

        def _call_service(
            self,
            client,
            request,
            *,
            service_name: str,
        ) -> None:
            if not client.wait_for_service(timeout_sec=10.0):
                raise RuntimeError(
                    f"Timed out waiting for service {service_name}."
                )
            future = client.call_async(request)
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.1)
                if future.done():
                    exception = future.exception()
                    if exception is not None:
                        raise RuntimeError(
                            f"Service call failed for {service_name}: {exception}"
                        ) from exception
                    return
            raise TimeoutError(f"Timed out waiting for service response from {service_name}.")

        def publish_velocity_command(self, action: FixedVelocityAction) -> None:
            message = MotionUpdate()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = action.frame_id
            message.velocity = Twist(
                linear=Vector3(
                    x=float(action.linear_xyz[0]),
                    y=float(action.linear_xyz[1]),
                    z=float(action.linear_xyz[2]),
                ),
                angular=Vector3(
                    x=float(action.angular_xyz[0]),
                    y=float(action.angular_xyz[1]),
                    z=float(action.angular_xyz[2]),
                ),
            )
            message.target_stiffness = np.diag([75.0] * 6).flatten().tolist()
            message.target_damping = np.diag([35.0] * 6).flatten().tolist()
            message.feedforward_wrench_at_tip = Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0),
                torque=Vector3(x=0.0, y=0.0, z=0.0),
            )
            message.wrench_feedback_gains_at_tip = [0.0] * 6
            message.trajectory_generation_mode.mode = (
                TrajectoryGenerationMode.MODE_VELOCITY
            )
            self.motion_pub.publish(message)

        def wait_for_camera_images(self, *, timeout_s: float) -> None:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.1)
                if all(name in self.latest_images for name in ("left", "center", "right")):
                    return
            raise TimeoutError("Timed out waiting for wrist camera images for official trace.")

        def image_summary(self) -> dict[str, dict[str, Any]]:
            return summarize_image_batch(self.latest_images, self.latest_image_timestamps)

    def _find_first_entity(
        observation: dict[str, Any],
        *candidates: str,
    ) -> tuple[str | None, dict[str, Any] | None]:
        entities = observation.get("entities_by_name", {})
        for candidate in candidates:
            entity = entities.get(candidate)
            if entity is not None:
                return candidate, entity
        return None, None

    def _extract_trace_fields(
        observation: dict[str, Any],
        *,
        controller_state: ControllerState | None = None,
    ) -> dict[str, Any]:
        tcp_name, tcp_entity = _find_first_entity(
            observation,
            "ati/tool_link",
            "wrist_3_link",
        )
        plug_name, plug_entity = _find_first_entity(
            observation,
            "lc_plug_link",
            "sc_plug_link",
            "sfp_module_link",
            "cable_0",
        )
        target_name, target_entity = _find_first_entity(
            observation,
            "tabletop",
            "task_board",
            "task_board_base_link",
        )

        def _pose_position(entity: dict[str, Any] | None) -> list[float] | None:
            if entity is None:
                return None
            pose = entity.get("pose", {})
            position = pose.get("position")
            if isinstance(position, list):
                return [float(value) for value in position]
            return None

        def _pose_orientation(entity: dict[str, Any] | None) -> list[float] | None:
            if entity is None:
                return None
            pose = entity.get("pose", {})
            orientation = pose.get("orientation")
            if isinstance(orientation, list):
                return [float(value) for value in orientation]
            return None

        tracked_pair = observation.get("task_geometry", {}).get("tracked_entity_pair", {})
        controller_tcp_pose = None
        controller_ref_pose = None
        controller_tcp_velocity = None
        controller_target_mode = None
        if controller_state is not None:
            controller_tcp_pose = [
                float(controller_state.tcp_pose.position.x),
                float(controller_state.tcp_pose.position.y),
                float(controller_state.tcp_pose.position.z),
            ]
            controller_ref_pose = [
                float(controller_state.reference_tcp_pose.position.x),
                float(controller_state.reference_tcp_pose.position.y),
                float(controller_state.reference_tcp_pose.position.z),
            ]
            controller_tcp_velocity = [
                float(controller_state.tcp_velocity.linear.x),
                float(controller_state.tcp_velocity.linear.y),
                float(controller_state.tcp_velocity.linear.z),
                float(controller_state.tcp_velocity.angular.x),
                float(controller_state.tcp_velocity.angular.y),
                float(controller_state.tcp_velocity.angular.z),
            ]
            controller_target_mode = int(controller_state.target_mode.mode)
        return {
            "joint_positions": list(observation.get("joint_positions") or []),
            "joint_velocities": list(observation.get("joint_velocities") or []),
            "sim_time": (
                float(observation.get("sim_time"))
                if isinstance(observation.get("sim_time"), (int, float))
                else None
            ),
            "tcp_entity_name": tcp_name,
            "tcp_position": _pose_position(tcp_entity),
            "tcp_orientation": _pose_orientation(tcp_entity),
            "plug_entity_name": plug_name,
            "plug_position": _pose_position(plug_entity),
            "plug_orientation": _pose_orientation(plug_entity),
            "target_entity_name": target_name,
            "target_position": _pose_position(target_entity),
            "target_orientation": _pose_orientation(target_entity),
            "relative_position": list(tracked_pair.get("relative_position") or []),
            "distance_to_target": tracked_pair.get("distance"),
            "orientation_error": tracked_pair.get("orientation_error"),
            "success_like": bool(tracked_pair.get("success", False)),
            "force_magnitude": 0.0,
            "off_limit_contact": False,
            "entity_count": observation.get("entity_count"),
            "joint_count": observation.get("joint_count"),
            "step_count_raw": observation.get("step_count"),
            "controller_tcp_position": controller_tcp_pose,
            "controller_reference_tcp_position": controller_ref_pose,
            "controller_tcp_velocity": controller_tcp_velocity,
            "controller_target_mode": controller_target_mode,
        }

    last_state_generation: int | None = None
    last_native_observation: dict[str, Any] | None = None

    def _position_delta(
        current: list[float] | None,
        previous: list[float] | None,
    ) -> float:
        if current is None or previous is None or len(current) != 3 or len(previous) != 3:
            return 0.0
        return float(
            np.linalg.norm(
                np.asarray(current, dtype=np.float64) - np.asarray(previous, dtype=np.float64)
            )
        )

    def _observation_is_sane(observation: dict[str, Any]) -> bool:
        fields = _extract_trace_fields(observation)
        return bool(
            int(observation.get("entity_count") or 0) >= 100
            and int(observation.get("joint_count") or 0) >= 6
            and fields.get("tcp_position") is not None
            and fields.get("plug_position") is not None
            and fields.get("target_position") is not None
        )

    def _transport_sample_looks_stale(
        observation: dict[str, Any],
        *,
        controller_state: ControllerState | None,
        previous_observation: dict[str, Any] | None,
        action: FixedVelocityAction | None,
    ) -> bool:
        if controller_state is None or previous_observation is None or action is None:
            return False
        action_norm = float(
            np.linalg.norm(
                np.asarray(action.linear_xyz + action.angular_xyz, dtype=np.float64)
            )
        )
        if action_norm <= 1e-6:
            return False
        current_fields = _extract_trace_fields(observation, controller_state=controller_state)
        previous_fields = _extract_trace_fields(previous_observation)
        controller_delta = _position_delta(
            current_fields.get("controller_tcp_position"),
            previous_fields.get("controller_tcp_position"),
        )
        native_delta = _position_delta(
            current_fields.get("tcp_position"),
            previous_fields.get("tcp_position"),
        )
        return controller_delta >= 5e-4 and native_delta <= 5e-5

    def _fetch_sane_cli_fallback(*, timeout_s: float) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            response = cli_fallback_client.get_observation(GetObservationRequest())
            observation = response.observation
            if _observation_is_sane(observation):
                return observation
            time.sleep(0.05)
        return None

    def _try_promote_to_sane_observation(
        observation: dict[str, Any],
        *,
        action: FixedVelocityAction | None,
        allow_cli_fallback: bool,
        timeout_s: float,
    ) -> tuple[dict[str, Any], str] | None:
        source = "transport"
        if _observation_is_sane(observation) and not _transport_sample_looks_stale(
            observation,
            controller_state=node.latest_controller_state,
            previous_observation=last_native_observation,
            action=action,
        ):
            return observation, source

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if allow_cli_fallback:
                fallback = _fetch_sane_cli_fallback(timeout_s=0.25)
                if fallback is not None:
                    return fallback, "cli_fallback"
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(0.02)
            candidate = native_client.get_observation(GetObservationRequest()).observation
            if _observation_is_sane(candidate) and not _transport_sample_looks_stale(
                candidate,
                controller_state=node.latest_controller_state,
                previous_observation=last_native_observation,
                action=action,
            ):
                return candidate, "transport"
        return None

    def _world_control_cli(*, request: str, timeout_s: float) -> str:
        completed = subprocess.run(
            [
                "gz",
                "service",
                "-s",
                "/world/aic_world/control",
                "--reqtype",
                "gz.msgs.WorldControl",
                "--reptype",
                "gz.msgs.Boolean",
                "--timeout",
                str(int(timeout_s * 1000)),
                "--req",
                request,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(timeout_s, 1.0),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Gazebo world control CLI failed: "
                f"stdout={completed.stdout.strip()} stderr={completed.stderr.strip()}"
            )
        return completed.stdout.strip()

    def _pause_world() -> str:
        return _world_control_cli(request="pause: true", timeout_s=10.0)

    def _world_step(steps: int) -> str:
        if steps <= 0:
            raise ValueError("steps must be positive")
        try:
            response = native_client._bridge.request(  # type: ignore[attr-defined]
                {
                    "op": "world_control",
                    "service": "/world/aic_world/control",
                    "multi_step": int(steps),
                    "timeout_ms": 10000,
                }
            )
            return str(response.get("reply_text", ""))
        except Exception:
            return _world_control_cli(request=f"multi_step: {int(steps)}", timeout_s=10.0)

    def _wait_for_camera_images_with_steps(*, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                node.wait_for_camera_images(timeout_s=0.5)
                return
            except TimeoutError:
                _world_step(2)
                for _ in range(4):
                    rclpy.spin_once(node, timeout_sec=0.05)
                    time.sleep(0.01)
        raise TimeoutError("Timed out waiting for wrist camera images for official trace.")

    def _capture_native_observation_after_step(
        *,
        ticks: int,
        action: FixedVelocityAction | None = None,
    ) -> tuple[dict[str, Any], str]:
        nonlocal last_state_generation, last_native_observation
        _world_step(ticks)
        # Allow the controller_state subscriber and the Gazebo CLI readers to
        # observe the post-step world state before sampling.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(0.01)
            response = native_client.get_observation(GetObservationRequest())
            state_generation = response.info.get("state_generation")
            if not isinstance(state_generation, int):
                observation = response.observation
                promoted = _try_promote_to_sane_observation(
                    observation,
                    action=action,
                    allow_cli_fallback=True,
                    timeout_s=1.0,
                )
                if promoted is not None:
                    observation, source = promoted
                    last_native_observation = observation
                    return observation, source
                continue
            if last_state_generation is None or state_generation > last_state_generation:
                observation = response.observation
                promoted = _try_promote_to_sane_observation(
                    observation,
                    action=action,
                    allow_cli_fallback=True,
                    timeout_s=1.0,
                )
                if promoted is not None:
                    last_state_generation = state_generation
                    observation, source = promoted
                    last_native_observation = observation
                    return observation, source
        raise TimeoutError("Timed out waiting for a fresh native Gazebo state sample after stepping.")

    def _wait_for_native_world_ready(*, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        last_observation: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            _world_step(1)
            observation = native_client.get_observation(GetObservationRequest()).observation
            last_observation = observation
            entities = observation.get("entities_by_name", {})
            if (
                "ati/tool_link" in entities
                and "tabletop" in entities
                and int(observation.get("joint_count") or 0) >= 6
            ):
                return observation
            time.sleep(0.2)
        raise TimeoutError(
            "Timed out waiting for robot/task entities to appear in Gazebo observation. "
            f"Last entity_count={None if last_observation is None else last_observation.get('entity_count')}"
        )

    rclpy.init()
    camera_bridge = CameraBridgeSidecar() if include_images else None
    if camera_bridge is not None:
        camera_bridge.start()
    node = Recorder()
    native_client = GazeboCliClient(
        GazeboCliClientConfig(
            executable="gz",
            world_path="/home/ubuntu/ws_aic/src/aic/aic_description/world/aic.sdf",
            timeout=10.0,
            world_name="aic_world",
            source_entity_name="ati/tool_link",
            target_entity_name="tabletop",
            transport_backend="transport",
            observation_transport="auto",
        )
    )
    cli_fallback_client = GazeboCliClient(
        GazeboCliClientConfig(
            executable="gz",
            world_path="/home/ubuntu/ws_aic/src/aic/aic_description/world/aic.sdf",
            timeout=10.0,
            world_name="aic_world",
            source_entity_name="ati/tool_link",
            target_entity_name="tabletop",
            transport_backend="cli",
            observation_transport="one_shot",
        )
    )
    target_mode_note: str | None = None
    pose_command_note: str | None = None
    start_wall = time.perf_counter()
    reset_wall_s: float | None = None
    step_wall_s: list[float] = []
    try:
        try:
            node.set_cartesian_target_mode()
        except (RuntimeError, TimeoutError) as exc:
            target_mode_note = (
                "Failed to call /aic_controller/change_target_mode; continuing under the "
                "assumption that the official controller is already in Cartesian target mode. "
                f"Error: {exc}"
            )
        try:
            node.wait_for_pose_command_subscriber(timeout_s=20.0)
        except TimeoutError as exc:
            pose_command_note = (
                "Timed out waiting for /aic_controller/pose_commands subscriber count to become "
                "visible; continuing because the official controller may still be active even when "
                f"publisher-side discovery is lagging. Error: {exc}"
            )
        pause_note: str | None = None
        try:
            _pause_world()
        except Exception as exc:
            pause_note = (
                "Failed to pause the world through direct Gazebo world_control before exact stepping. "
                f"Error: {exc}"
            )
        _wait_for_native_world_ready(timeout_s=20.0)
        initial_observation, initial_observation_source = _capture_native_observation_after_step(
            ticks=10 + (image_settle_ticks if include_images else 0)
        )
        initial_images = None
        if include_images:
            _wait_for_camera_images_with_steps(timeout_s=20.0)
            initial_images = node.image_summary()
        reset_wall_s = time.perf_counter() - start_wall

        records: list[dict[str, Any]] = []
        for step_idx, action in enumerate(actions):
            step_start = time.perf_counter()
            effective_ticks = action.sim_steps + (image_settle_ticks if include_images else 0)
            node.publish_velocity_command(action)
            for _ in range(8):
                rclpy.spin_once(node, timeout_sec=0.05)
                time.sleep(0.01)
                node.publish_velocity_command(action)
            native_observation, native_observation_source = _capture_native_observation_after_step(
                ticks=effective_ticks,
                action=action,
            )
            records.append(
                {
                    "step_idx": step_idx,
                    "action": {
                        "linear_xyz": list(action.linear_xyz),
                        "angular_xyz": list(action.angular_xyz),
                        "frame_id": action.frame_id,
                        "sim_steps": effective_ticks,
                        "command_sim_steps": action.sim_steps,
                    },
                    "native_observation_source": native_observation_source,
                    "native": _extract_trace_fields(
                        native_observation,
                        controller_state=node.latest_controller_state,
                    ),
                    "images": node.image_summary() if include_images else None,
                }
            )
            step_wall_s.append(time.perf_counter() - step_start)

        total_wall_s = time.perf_counter() - start_wall
        initial_sim_time = _extract_trace_fields(
            initial_observation,
            controller_state=node.latest_controller_state,
        ).get("sim_time")
        final_sim_time = records[-1]["native"].get("sim_time") if records else initial_sim_time
        simulated_seconds = None
        if isinstance(initial_sim_time, (int, float)) and isinstance(final_sim_time, (int, float)):
            simulated_seconds = max(0.0, float(final_sim_time) - float(initial_sim_time))
        if simulated_seconds is None or simulated_seconds == 0.0:
            simulated_seconds = sum(
                float(record["action"].get("sim_steps", 0)) * 0.001
                for record in records
            )

        report = {
            "mode": "official_ros_control_vs_native_gazebo_observation_same_world",
            "control_surface": "official_ros_motion_update",
            "observation_surface": "gazebo_native_post_step_readout",
            "image_surface": "ros_sidecar_camera_fallback" if include_images else None,
            "initial_native_observation_source": initial_observation_source,
            "initial_native": _extract_trace_fields(
                initial_observation,
                controller_state=node.latest_controller_state,
            ),
            "initial_images": initial_images,
            "approximation_notes": [
                "Commands are published through the official aic_controller ROS interface.",
                "Simulation control uses exact stepped progression through /gz_server/step_simulation while paused.",
                "Native Gazebo state is read immediately after each exact step window through the training-side Gazebo client.",
                "If the transport-backed native sample is stale relative to official controller motion, a sane one-shot CLI observation is used for that step and recorded in native_observation_source.",
                "Controller-state snapshots are captured from /aic_controller/controller_state to prove that the official controller advanced its references during the same stepped rollout.",
                "This harness intentionally avoids /gz_server/reset_simulation because the current official bringup crashes under reset on this machine.",
                "Image capture currently uses a dedicated non-lazy ROS parameter_bridge sidecar because the composed lazy bridge is unreliable in this container.",
            ]
            + ([target_mode_note] if target_mode_note is not None else [])
            + ([pose_command_note] if pose_command_note is not None else []),
            "timing": {
                "ready_to_first_sane_state_latency_s": reset_wall_s,
                "step_latency_s": step_wall_s,
                "mean_step_latency_s": (
                    sum(step_wall_s) / len(step_wall_s) if step_wall_s else None
                ),
                "total_wall_s": total_wall_s,
                "simulated_seconds": simulated_seconds,
                "simulated_seconds_per_wall_second": (
                    simulated_seconds / total_wall_s if total_wall_s > 0.0 else None
                ),
                "samples_per_second": (
                    len(records) / total_wall_s if total_wall_s > 0.0 else None
                ),
            },
        }
        if pause_note is not None:
            report["approximation_notes"].append(pause_note)
        report.update(
            {
            "num_steps": len(records),
            "records": records,
            }
        )
    finally:
        node.destroy_node()
        if camera_bridge is not None:
            camera_bridge.close()
        rclpy.shutdown()

    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--include-images", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            capture_official_and_native_trace(
                output_json=args.output,
                include_images=args.include_images,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
