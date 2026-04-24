"""Observation extraction and command application interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import subprocess
import threading
import time
from typing import Any

import numpy as np

from .runtime import RuntimeState


class AicGazeboIO(ABC):
    """Gazebo-native observation / actuation interface."""

    @abstractmethod
    def observation_from_state(
        self,
        state: RuntimeState,
        *,
        include_images: bool,
        step_count: int,
    ) -> dict[str, Any]:
        """Convert runtime state to the public env observation."""

    @abstractmethod
    def sanitize_action(self, action: np.ndarray) -> np.ndarray:
        """Clamp and normalize an env action before it reaches the backend."""

    def close(self) -> None:
        """Release any side resources held by the IO layer."""


def _base_observation(
    state: RuntimeState,
    *,
    step_count: int,
) -> dict[str, Any]:
    relative = state.target_port_pose[:3] - state.plug_pose[:3]
    return {
        "step_count": step_count,
        "sim_tick": int(state.sim_tick),
        "sim_time": float(state.sim_time),
        "joint_positions": state.joint_positions.astype(np.float32).copy(),
        "joint_velocities": state.joint_velocities.astype(np.float32).copy(),
        "gripper_state": np.array([state.gripper_position], dtype=np.float32),
        "tcp_pose": state.tcp_pose.astype(np.float32).copy(),
        "tcp_velocity": state.tcp_velocity.astype(np.float32).copy(),
        "plug_pose": state.plug_pose.astype(np.float32).copy(),
        "target_port_pose": state.target_port_pose.astype(np.float32).copy(),
        "target_port_entrance_pose": (
            np.zeros(7, dtype=np.float32)
            if state.target_port_entrance_pose is None
            else state.target_port_entrance_pose.astype(np.float32).copy()
        ),
        "plug_to_port_relative": np.concatenate(
            [relative, np.array([np.linalg.norm(relative)], dtype=np.float64)]
        ).astype(np.float32),
        "wrench": state.wrench.astype(np.float32).copy(),
        "wrench_timestamp": np.array([state.wrench_timestamp], dtype=np.float32),
        "off_limit_contact": np.array([float(state.off_limit_contact)], dtype=np.float32),
        "controller_tcp_pose": _controller_array(
            state.controller_state.get("tcp_pose"),
            size=7,
        ),
        "controller_reference_tcp_pose": _controller_array(
            state.controller_state.get("reference_tcp_pose"),
            size=7,
        ),
        "controller_tcp_velocity": _controller_array(
            state.controller_state.get("tcp_velocity"),
            size=6,
        ),
        "controller_tcp_error": _controller_array(
            state.controller_state.get("tcp_error"),
            size=6,
        ),
        "controller_reference_joint_state": _controller_array(
            state.controller_state.get("reference_joint_state"),
            size=6,
        ),
        "controller_target_mode": np.array(
            [float(state.controller_state.get("target_mode", 0))],
            dtype=np.float32,
        ),
        "fts_tare_wrench": _controller_array(
            state.controller_state.get("fts_tare_offset"),
            size=6,
        ),
        "score_geometry": {
            "distance_to_target": np.array(
                [float(state.score_geometry.get("distance_to_target", 0.0))],
                dtype=np.float32,
            ),
            "distance_threshold": np.array(
                [float(state.score_geometry.get("distance_threshold", 0.0))],
                dtype=np.float32,
            ),
            "plug_to_port_depth": np.array(
                [float(state.score_geometry.get("plug_to_port_depth", 0.0))],
                dtype=np.float32,
            ),
            "port_to_entrance_depth": np.array(
                [float(state.score_geometry.get("port_to_entrance_depth", 0.0))],
                dtype=np.float32,
            ),
            "distance_to_entrance": np.array(
                [float(state.score_geometry.get("distance_to_entrance", 0.0))],
                dtype=np.float32,
            ),
            "lateral_misalignment": np.array(
                [float(state.score_geometry.get("lateral_misalignment", 0.0))],
                dtype=np.float32,
            ),
            "orientation_error": np.array(
                [float(state.score_geometry.get("orientation_error", 0.0) or 0.0)],
                dtype=np.float32,
            ),
            "insertion_progress": np.array(
                [float(state.score_geometry.get("insertion_progress", 0.0))],
                dtype=np.float32,
            ),
            "partial_insertion": np.array(
                [1.0 if state.score_geometry.get("partial_insertion", False) else 0.0],
                dtype=np.float32,
            ),
        },
    }


def _sanitize_action(action: np.ndarray) -> np.ndarray:
    array = np.asarray(action, dtype=np.float64)
    if array.shape != (6,):
        raise ValueError(f"Expected action with shape (6,), got {array.shape}.")
    array[:3] = np.clip(array[:3], -0.25, 0.25)
    array[3:] = np.clip(array[3:], -2.0, 2.0)
    return array


def summarize_image_batch(
    images: dict[str, np.ndarray],
    timestamps: dict[str, float],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for name, image in images.items():
        summary[name] = {
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "pixel_sum": int(image.sum()),
            "mean": float(image.mean()),
            "std": float(image.std()),
            "timestamp": float(timestamps.get(name, 0.0)),
            "present": bool(image.size > 0 and int(image.sum()) > 0),
        }
    return summary


def _controller_array(value: Any, *, size: int) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value.astype(np.float32, copy=True)
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value, dtype=np.float32)
    else:
        array = np.zeros(size, dtype=np.float32)
    if array.shape[0] < size:
        padded = np.zeros(size, dtype=np.float32)
        padded[: array.shape[0]] = array
        return padded
    return array[:size]


@dataclass(frozen=True)
class GazeboNativeIOPlaceholder(AicGazeboIO):
    """Reserved slot for a future pure Gazebo-transport image path."""

    def observation_from_state(
        self,
        state: RuntimeState,
        *,
        include_images: bool,
        step_count: int,
    ) -> dict[str, Any]:
        del state, include_images, step_count
        raise NotImplementedError(
            "Gazebo-native IO is not wired in this shell. "
            "Use MockGazeboIO for tests or RosCameraSidecarIO for live image fallback."
        )

    def sanitize_action(self, action: np.ndarray) -> np.ndarray:
        return _sanitize_action(action)


@dataclass(frozen=True)
class MockGazeboIO(AicGazeboIO):
    """State-only IO used for deterministic testing."""

    image_shape: tuple[int, int, int] = (256, 256, 3)

    def observation_from_state(
        self,
        state: RuntimeState,
        *,
        include_images: bool,
        step_count: int,
    ) -> dict[str, Any]:
        observation = _base_observation(state, step_count=step_count)
        if include_images:
            blank = np.zeros((3, *self.image_shape), dtype=np.uint8)
            observation["images"] = {
                "left": blank[0],
                "center": blank[1],
                "right": blank[2],
            }
            observation["image_timestamps"] = np.zeros(3, dtype=np.float32)
            observation["camera_info"] = {
                name: _default_camera_info()
                for name in ("left", "center", "right")
            }
        return observation

    def sanitize_action(self, action: np.ndarray) -> np.ndarray:
        return _sanitize_action(action)


class RosCameraSubscriber:
    """ROS sidecar subscriber for wrist RGB cameras.

    This is intentionally isolated from the state hot loop. State extraction
    stays Gazebo-native; only image ingestion currently falls back to ROS
    because direct Gazebo topic access is unstable in the current container.
    """

    def __init__(
        self,
        *,
        image_shape: tuple[int, int, int] = (256, 256, 3),
        topic_map: dict[str, str] | None = None,
        node_name: str = "aic_gym_gz_camera_sidecar",
    ) -> None:
        self._image_shape = image_shape
        self._topic_map = topic_map or {
            "left": "/left_camera/image",
            "center": "/center_camera/image",
            "right": "/right_camera/image",
        }
        self._node_name = str(node_name)
        self._latest_images: dict[str, np.ndarray] = {
            name: np.zeros(image_shape, dtype=np.uint8) for name in self._topic_map
        }
        self._latest_timestamps: dict[str, float] = {name: 0.0 for name in self._topic_map}
        self._latest_camera_info: dict[str, dict[str, np.ndarray]] = {
            name: _default_camera_info() for name in self._topic_map
        }
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._node = None
        self._executor = None
        self._rclpy = None
        self._context = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._spin, name="aic-camera-sidecar", daemon=True)
        self._thread.start()

    def wait_until_ready(self, *, timeout_s: float = 10.0) -> bool:
        self.start()
        return self._ready_event.wait(timeout_s)

    def latest_images(self) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, dict[str, np.ndarray]]]:
        with self._lock:
            return (
                {name: image.copy() for name, image in self._latest_images.items()},
                dict(self._latest_timestamps),
                {
                    name: {key: value.copy() for key, value in info.items()}
                    for name, info in self._latest_camera_info.items()
                },
            )

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._executor is not None and self._node is not None:
            try:
                self._executor.remove_node(self._node)
            except Exception:
                pass
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
        if self._rclpy is not None:
            try:
                if self._context is not None:
                    self._rclpy.shutdown(context=self._context)
            except Exception:
                pass
        self._node = None
        self._executor = None
        self._rclpy = None
        self._context = None

    def _spin(self) -> None:
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        context = Context()
        rclpy.init(context=context)
        self._rclpy = rclpy
        self._context = context
        node = Node(self._node_name, context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        self._node = node
        self._executor = executor

        for name, topic in self._topic_map.items():
            node.create_subscription(
                Image,
                topic,
                lambda message, camera_name=name: self._image_callback(camera_name, message),
                qos_profile_sensor_data,
            )
            node.create_subscription(
                CameraInfo,
                topic.replace("/image", "/camera_info"),
                lambda message, camera_name=name: self._camera_info_callback(camera_name, message),
                qos_profile_sensor_data,
            )

        while rclpy.ok(context=context) and not self._stop_event.is_set():
            executor.spin_once(timeout_sec=0.1)

    def _image_callback(self, camera_name: str, message: Any) -> None:
        image = _ros_image_to_array(message, expected_shape=self._image_shape)
        timestamp = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
        with self._lock:
            self._latest_images[camera_name] = image
            self._latest_timestamps[camera_name] = timestamp
            if all(value > 0.0 for value in self._latest_timestamps.values()):
                self._ready_event.set()

    def _camera_info_callback(self, camera_name: str, message: Any) -> None:
        with self._lock:
            self._latest_camera_info[camera_name] = {
                "size": np.array([float(message.width), float(message.height)], dtype=np.float32),
                "k": np.asarray(message.k, dtype=np.float32),
                "p": np.asarray(message.p, dtype=np.float32),
            }


class CameraBridgeSidecar:
    """Dedicated non-lazy bridge for wrist camera topics."""

    def __init__(self, *, topic_map: dict[str, str] | None = None) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._topic_map = topic_map or {
            "left": "/left_camera/image",
            "center": "/center_camera/image",
            "right": "/right_camera/image",
        }

    def _bridge_arguments(self) -> list[str]:
        arguments: list[str] = []
        for topic in self._topic_map.values():
            arguments.extend(
                [
                    f"{topic}@sensor_msgs/msg/Image[gz.msgs.Image",
                    f"{topic.replace('/image', '/camera_info')}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
                ]
            )
        return arguments

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._process = subprocess.Popen(
            [
                "ros2",
                "run",
                "ros_gz_bridge",
                "parameter_bridge",
                *self._bridge_arguments(),
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


@dataclass
class RosCameraSidecarIO(AicGazeboIO):
    """Live IO path with ROS-only image fallback."""

    camera_subscriber: RosCameraSubscriber | None = None
    camera_bridge: CameraBridgeSidecar = field(default_factory=CameraBridgeSidecar)
    ready_timeout_s: float = 10.0
    image_shape: tuple[int, int, int] = (256, 256, 3)

    def __post_init__(self) -> None:
        if self.camera_subscriber is None:
            self.camera_subscriber = RosCameraSubscriber(image_shape=self.image_shape)
        self.camera_bridge.start()
        self.camera_subscriber.start()

    def observation_from_state(
        self,
        state: RuntimeState,
        *,
        include_images: bool,
        step_count: int,
    ) -> dict[str, Any]:
        observation = _base_observation(state, step_count=step_count)
        if not include_images:
            return observation
        if not self.camera_subscriber.wait_until_ready(timeout_s=self.ready_timeout_s):
            raise TimeoutError("Timed out waiting for wrist camera images.")
        images, timestamps, camera_info = self.camera_subscriber.latest_images()
        observation["images"] = images
        observation["image_timestamps"] = np.array(
            [timestamps["left"], timestamps["center"], timestamps["right"]],
            dtype=np.float32,
        )
        observation["camera_info"] = camera_info
        return observation

    def sanitize_action(self, action: np.ndarray) -> np.ndarray:
        return _sanitize_action(action)

    def close(self) -> None:
        self.camera_subscriber.close()
        self.camera_bridge.close()


def _ros_image_to_array(message: Any, *, expected_shape: tuple[int, int, int]) -> np.ndarray:
    height = int(message.height)
    width = int(message.width)
    channels = int(expected_shape[2])
    buffer = np.frombuffer(message.data, dtype=np.uint8)
    if message.encoding == "rgb8":
        array = buffer.reshape(height, width, channels)
    elif message.encoding == "bgr8":
        array = buffer.reshape(height, width, channels)[:, :, ::-1]
    else:
        raise ValueError(f"Unsupported image encoding: {message.encoding}")
    expected_height, expected_width, _ = expected_shape
    if (height, width) != (expected_height, expected_width):
        row_index = np.linspace(0, height - 1, expected_height, dtype=np.int64)
        col_index = np.linspace(0, width - 1, expected_width, dtype=np.int64)
        array = array[row_index][:, col_index]
    return array.copy()


def _default_camera_info() -> dict[str, np.ndarray]:
    return {
        "size": np.zeros(2, dtype=np.float32),
        "k": np.zeros(9, dtype=np.float32),
        "p": np.zeros(12, dtype=np.float32),
    }
