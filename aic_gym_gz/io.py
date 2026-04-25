"""Observation extraction and command application interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
import ast
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
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


SCENE_PROBE_TOPIC = "/scene_probe/image"
SCENE_PROBE_MODEL_NAME = "scene_probe_camera"
SCENE_PROBE_VIEW_POSES: dict[str, tuple[float, float, float, float, float, float]] = {
    "top_down_xy": (0.95, -0.10, 1.30, 0.0, 0.0, 3.14),
    "front_xz": (0.15, -1.05, 1.22, 1.5708, 0.0, 1.5708),
    "side_yz": (-0.95, -0.15, 1.18, 0.0, 0.0, 0.0),
    "oblique_xy": (0.95, -0.95, 1.55, 0.0, 0.0, 2.35),
}


def capture_scene_probe_images(
    *,
    view_names: tuple[str, ...] | list[str] | None = None,
    world_name: str = "aic_world",
    model_name: str = SCENE_PROBE_MODEL_NAME,
    image_topic: str = SCENE_PROBE_TOPIC,
    timeout_s: float = 4.0,
    expected_shape: tuple[int, int, int] = (512, 512, 3),
) -> dict[str, np.ndarray]:
    requested_views = tuple(view_names or tuple(SCENE_PROBE_VIEW_POSES))
    captured: dict[str, np.ndarray] = {}
    for view_name in requested_views:
        pose = SCENE_PROBE_VIEW_POSES.get(view_name)
        if pose is None:
            continue
        if not set_gazebo_entity_pose(
            model_name,
            pose_xyz_rpy=pose,
            world_name=world_name,
            timeout_s=max(timeout_s, 2.0),
        ):
            continue
        time.sleep(0.15)
        frame = fetch_gazebo_topic_image(
            image_topic,
            timeout_s=timeout_s,
            expected_shape=expected_shape,
        )
        if frame is None or frame.size == 0 or int(frame.sum()) == 0:
            continue
        captured[view_name] = np.asarray(frame, dtype=np.uint8)
    return captured


def set_gazebo_entity_pose(
    entity_name: str,
    *,
    pose_xyz_rpy: tuple[float, float, float, float, float, float],
    world_name: str = "aic_world",
    timeout_s: float = 5.0,
) -> bool:
    x, y, z, roll, pitch, yaw = (float(value) for value in pose_xyz_rpy)
    qx, qy, qz, qw = _quaternion_from_rpy(roll, pitch, yaw)
    request = (
        f'name: "{entity_name}" '
        f'position {{ x: {x} y: {y} z: {z} }} '
        f'orientation {{ x: {qx} y: {qy} z: {qz} w: {qw} }}'
    )
    try:
        completed = subprocess.run(
            [
                "gz",
                "service",
                "-s",
                f"/world/{world_name}/set_pose",
                "--reqtype",
                "gz.msgs.Pose",
                "--reptype",
                "gz.msgs.Boolean",
                "--timeout",
                str(int(max(float(timeout_s), 0.1) * 1000.0)),
                "--req",
                request,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(float(timeout_s), 0.1),
            env=_gazebo_subprocess_env(),
        )
    except Exception:
        return False
    if completed.returncode != 0:
        return False
    return "data: true" in completed.stdout


def fetch_gazebo_topic_image(
    topic: str,
    *,
    timeout_s: float = 5.0,
    expected_shape: tuple[int, int, int] | None = None,
) -> np.ndarray | None:
    try:
        completed = subprocess.run(
            ["gz", "topic", "-e", "-n", "1", "-t", topic],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(float(timeout_s), 0.1),
            env=_gazebo_subprocess_env(),
        )
    except Exception:
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    return _gazebo_image_text_to_array(
        completed.stdout,
        expected_shape=expected_shape,
    )


def fetch_ros_topic_image(
    topic: str,
    *,
    timeout_s: float = 5.0,
    expected_shape: tuple[int, int, int] | None = None,
) -> np.ndarray | None:
    try:
        completed = subprocess.run(
            [
                "ros2",
                "topic",
                "echo",
                "--once",
                "--qos-profile",
                "sensor_data",
                topic,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(float(timeout_s), 0.1),
        )
    except Exception:
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    return _ros_topic_text_to_array(
        completed.stdout,
        expected_shape=expected_shape,
    )


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


def _gazebo_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("GZ_IP", "127.0.0.1")
    return env


def _quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


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
        reliable_qos: bool = False,
    ) -> None:
        self._image_shape = image_shape
        self._topic_map = topic_map or {
            "left": "/left_camera/image",
            "center": "/center_camera/image",
            "right": "/right_camera/image",
        }
        self._node_name = str(node_name)
        self._reliable_qos = bool(reliable_qos)
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
        if os.environ.get("AIC_GYM_GZ_FORCE_CYCLONEDDS", "").strip() in {"1", "true", "TRUE"}:
            os.environ["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
            os.environ.pop("ZENOH_SESSION_CONFIG_URI", None)
            os.environ.pop("ZENOH_CONFIG_OVERRIDE", None)
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
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
        image_qos = qos_profile_sensor_data
        if self._reliable_qos:
            image_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)

        for name, topic in self._topic_map.items():
            node.create_subscription(
                Image,
                topic,
                lambda message, camera_name=name: self._image_callback(camera_name, message),
                image_qos,
            )
            node.create_subscription(
                CameraInfo,
                topic.replace("/image", "/camera_info"),
                lambda message, camera_name=name: self._camera_info_callback(camera_name, message),
                image_qos,
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


class GazeboCameraSubscriber:
    """Persistent Gazebo-topic image subscriber for paused exact-step worlds."""

    def __init__(
        self,
        *,
        image_shape: tuple[int, int, int] = (256, 256, 3),
        topic_map: dict[str, str] | None = None,
        quiet_period_s: float = 0.03,
    ) -> None:
        from aic_utils.aic_gazebo_env.aic_gazebo_env.gazebo_client import (
            PersistentGazeboTopicReader,
        )

        self._image_shape = image_shape
        self._topic_map = topic_map or {
            "left": "/left_camera/image",
            "center": "/center_camera/image",
            "right": "/right_camera/image",
        }
        executable = _resolve_executable_from_active_env("gz")
        self._readers = (
            {}
            if executable is None
            else {
                name: PersistentGazeboTopicReader(
                    executable=executable,
                    topic=topic,
                    quiet_period_s=quiet_period_s,
                )
                for name, topic in self._topic_map.items()
            }
        )
        self._latest_images: dict[str, np.ndarray] = {
            name: np.zeros(image_shape, dtype=np.uint8) for name in self._topic_map
        }
        self._latest_timestamps: dict[str, float] = {name: 0.0 for name in self._topic_map}
        self._latest_generations: dict[str, int] = {name: 0 for name in self._topic_map}

    def start(self) -> None:
        for reader in self._readers.values():
            reader.start()

    def latest_images(self, *, timeout_s: float = 0.01) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        for name, reader in self._readers.items():
            try:
                sample, generation = reader.get_sample(
                    after_generation=self._latest_generations.get(name, 0),
                    timeout=max(float(timeout_s), 0.001),
                )
            except TimeoutError:
                continue
            frame = _gazebo_image_text_to_array(sample, expected_shape=self._image_shape)
            if not _is_nonblank_image(frame):
                continue
            self._latest_images[name] = np.asarray(frame, dtype=np.uint8)
            self._latest_timestamps[name] = time.time()
            self._latest_generations[name] = int(generation)
        return (
            {name: image.copy() for name, image in self._latest_images.items()},
            dict(self._latest_timestamps),
        )

    def close(self) -> None:
        for reader in self._readers.values():
            reader.stop()


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
        env = _ros_gz_subprocess_env()
        ros2_executable = _resolve_executable_from_active_env("ros2")
        if ros2_executable is None:
            raise FileNotFoundError(
                "Could not find 'ros2' for the wrist camera bridge. Run from an "
                "activated ROS/Pixi environment or put ros2 on PATH."
            )
        self._process = subprocess.Popen(
            [
                ros2_executable,
                "run",
                "ros_gz_bridge",
                "parameter_bridge",
                *self._bridge_arguments(),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
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


class RosImageFileSubscriber:
    """System-Python ROS image subscriber with file-backed frame handoff."""

    def __init__(
        self,
        *,
        image_shape: tuple[int, int, int],
        topic_map: dict[str, str] | None = None,
    ) -> None:
        self._image_shape = image_shape
        self._topic_map = topic_map or {
            "left": "/left_camera/image",
            "center": "/center_camera/image",
            "right": "/right_camera/image",
        }
        self._process: subprocess.Popen[str] | None = None
        self._frame_dir = Path(tempfile.mkdtemp(prefix="aic_ros_image_frames_"))
        self._latest_images: dict[str, np.ndarray] = {
            name: np.zeros(image_shape, dtype=np.uint8) for name in self._topic_map
        }
        self._latest_timestamps: dict[str, float] = {name: 0.0 for name in self._topic_map}

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        script = Path(__file__).with_name("ros_image_file_subscriber.py")
        if not script.exists():
            return
        executable = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable
        self._process = subprocess.Popen(
            [
                executable,
                str(script),
                "--topics-json",
                json.dumps(self._topic_map, sort_keys=True),
                "--output-dir",
                str(self._frame_dir),
                "--height",
                str(int(self._image_shape[0])),
                "--width",
                str(int(self._image_shape[1])),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def latest_images(self) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        for name in self._topic_map:
            meta_path = self._frame_dir / f"{name}.json"
            frame_path = self._frame_dir / f"{name}.npy"
            if not meta_path.exists() or not frame_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                frame = np.load(frame_path)
            except Exception:
                continue
            if not _is_nonblank_image(frame):
                continue
            self._latest_images[name] = np.asarray(frame, dtype=np.uint8)
            self._latest_timestamps[name] = max(float(meta.get("timestamp", 0.0)), time.time())
        return (
            {name: image.copy() for name, image in self._latest_images.items()},
            dict(self._latest_timestamps),
        )

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3.0)
        self._process = None
        shutil.rmtree(self._frame_dir, ignore_errors=True)


@dataclass
class RosCameraSidecarIO(AicGazeboIO):
    """Live IO path with ROS-only image fallback."""

    camera_subscriber: RosCameraSubscriber | None = None
    gazebo_camera_subscriber: GazeboCameraSubscriber | None = None
    file_camera_subscriber: RosImageFileSubscriber | None = None
    camera_bridge: CameraBridgeSidecar = field(default_factory=CameraBridgeSidecar)
    ready_timeout_s: float = 10.0
    image_shape: tuple[int, int, int] = (256, 256, 3)
    start_bridge: bool = True
    blocking_bootstrap: bool = True
    allow_direct_fetch_fallback: bool = True
    background_direct_fetch: bool = False
    _wrist_bootstrapped: bool = field(default=False, init=False, repr=False)
    _wrist_ready_wait_attempted: bool = field(default=False, init=False, repr=False)
    _last_good_images: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _last_good_timestamps: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _last_good_camera_info: dict[str, dict[str, np.ndarray]] = field(default_factory=dict, init=False, repr=False)
    _background_fetch_stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _background_fetch_thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.camera_subscriber is None:
            self.camera_subscriber = RosCameraSubscriber(image_shape=self.image_shape)
        if self.gazebo_camera_subscriber is None:
            self.gazebo_camera_subscriber = GazeboCameraSubscriber(image_shape=self.image_shape)
        if self.file_camera_subscriber is None:
            self.file_camera_subscriber = RosImageFileSubscriber(image_shape=self.image_shape)
        self._ensure_camera_pipeline_started()

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
        self._ensure_camera_pipeline_started()
        if self.background_direct_fetch and self._background_fetch_thread is None:
            self._background_fetch_thread = threading.Thread(
                target=self._background_fetch_loop,
                name="aic-camera-background-fetch",
                daemon=True,
            )
            self._background_fetch_thread.start()
        bootstrap = not self._wrist_bootstrapped
        if bootstrap and self.blocking_bootstrap and not self._wrist_ready_wait_attempted:
            self._wrist_ready_wait_attempted = True
            print('{"io_stage":"wait_until_ready_begin"}', flush=True)
            ready = self.camera_subscriber.wait_until_ready(timeout_s=self.ready_timeout_s)
            print(f'{{"io_stage":"wait_until_ready_done","ready":{str(bool(ready)).lower()}}}', flush=True)
        images, timestamps, camera_info = self._collect_live_wrist_frames(
            settle_timeout_s=(
                18.0
                if (bootstrap and self.blocking_bootstrap and self.allow_direct_fetch_fallback)
                else 0.0
            ),
            direct_fetch_timeout_s=(
                8.0
                if (bootstrap and self.allow_direct_fetch_fallback and self.blocking_bootstrap)
                else (
                    0.0
                    if self.background_direct_fetch
                    else 2.0
                    if (bootstrap and self.allow_direct_fetch_fallback)
                    else (
                        0.1
                        if self.allow_direct_fetch_fallback
                    and any(float(value) <= 0.0 for value in self._last_good_timestamps.values())
                        else 0.0
                    )
                )
            ),
        )
        if bootstrap:
            self._wrist_bootstrapped = _has_all_wrist_frames(images, timestamps)
        observation["images"] = images
        observation["image_timestamps"] = np.array(
            [
                float(timestamps.get("left", 0.0)),
                float(timestamps.get("center", 0.0)),
                float(timestamps.get("right", 0.0)),
            ],
            dtype=np.float32,
        )
        observation["camera_info"] = camera_info
        return observation

    def sanitize_action(self, action: np.ndarray) -> np.ndarray:
        return _sanitize_action(action)

    def close(self) -> None:
        self._background_fetch_stop.set()
        if self._background_fetch_thread is not None:
            self._background_fetch_thread.join(timeout=2.0)
        self._background_fetch_thread = None
        if self.gazebo_camera_subscriber is not None:
            self.gazebo_camera_subscriber.close()
        if self.file_camera_subscriber is not None:
            self.file_camera_subscriber.close()
        self.camera_subscriber.close()
        self.camera_bridge.close()

    def _ensure_camera_pipeline_started(self) -> None:
        if self.gazebo_camera_subscriber is not None:
            self.gazebo_camera_subscriber.start()
        if self.start_bridge:
            self.camera_bridge.start()
        if self.file_camera_subscriber is not None:
            self.file_camera_subscriber.start()
        self.camera_subscriber.start()

    def _collect_live_wrist_frames(
        self,
        *,
        settle_timeout_s: float = 3.0,
        direct_fetch_timeout_s: float = 1.5,
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, dict[str, np.ndarray]]]:
        deadline = time.time() + max(float(settle_timeout_s), 0.0)
        camera_info: dict[str, dict[str, np.ndarray]] = {}
        images: dict[str, np.ndarray] = {}
        timestamps: dict[str, float] = {}
        printed_direct_begin = False
        any_direct_success = False
        while True:
            if self.gazebo_camera_subscriber is not None:
                gazebo_images, gazebo_timestamps = self.gazebo_camera_subscriber.latest_images()
                for name, image in gazebo_images.items():
                    timestamp = float(gazebo_timestamps.get(name, 0.0))
                    if _is_nonblank_image(image) and timestamp > 0.0:
                        images[name] = image
                        timestamps[name] = timestamp
            if self.file_camera_subscriber is not None:
                file_images, file_timestamps = self.file_camera_subscriber.latest_images()
                for name, image in file_images.items():
                    timestamp = float(file_timestamps.get(name, 0.0))
                    if _is_nonblank_image(image) and timestamp > 0.0:
                        images[name] = image
                        timestamps[name] = timestamp
                        camera_info.setdefault(name, _default_camera_info())
            subscriber_images, subscriber_timestamps, camera_info = self.camera_subscriber.latest_images()
            for name, image in subscriber_images.items():
                timestamp = float(subscriber_timestamps.get(name, 0.0))
                if _is_nonblank_image(image) and timestamp > 0.0:
                    images[name] = image
                    timestamps[name] = timestamp
            missing = [name for name in ("left", "center", "right") if name not in images]
            if not missing:
                break
            if direct_fetch_timeout_s <= 0.0:
                break
            if not printed_direct_begin:
                print('{"io_stage":"direct_gazebo_wrist_fetch_begin"}', flush=True)
                printed_direct_begin = True
            for name in missing:
                direct_image = fetch_gazebo_topic_image(
                    f"/{name}_camera/image",
                    timeout_s=direct_fetch_timeout_s,
                    expected_shape=self.image_shape,
                )
                if not _is_nonblank_image(direct_image):
                    direct_image = fetch_ros_topic_image(
                        f"/{name}_camera/image",
                        timeout_s=max(direct_fetch_timeout_s, 0.1),
                        expected_shape=self.image_shape,
                    )
                if not _is_nonblank_image(direct_image):
                    continue
                images[name] = np.asarray(direct_image, dtype=np.uint8)
                timestamps[name] = max(float(subscriber_timestamps.get(name, 0.0)), time.time())
                any_direct_success = True
            if len(images) == 3 or time.time() >= deadline:
                break
            time.sleep(0.1)
        for name in ("left", "center", "right"):
            if name in images:
                self._last_good_images[name] = np.asarray(images[name], dtype=np.uint8).copy()
                self._last_good_timestamps[name] = float(timestamps.get(name, time.time()))
                if name in camera_info:
                    self._last_good_camera_info[name] = {
                        key: np.asarray(value, dtype=np.float32).copy()
                        for key, value in camera_info[name].items()
                    }
            elif name in self._last_good_images:
                images[name] = self._last_good_images[name].copy()
                timestamps[name] = float(self._last_good_timestamps.get(name, 0.0))
                if name not in camera_info and name in self._last_good_camera_info:
                    camera_info[name] = {
                        key: value.copy() for key, value in self._last_good_camera_info[name].items()
                    }
        if printed_direct_begin:
            print(
                f'{{"io_stage":"direct_gazebo_wrist_fetch_done","ok":{str(any_direct_success or len(images) == 3).lower()},"frame_count":{len(images)}}}',
                flush=True,
            )
        return images, timestamps, camera_info

    def _background_fetch_loop(self) -> None:
        while not self._background_fetch_stop.is_set():
            try:
                subscriber_images, subscriber_timestamps, camera_info = self.camera_subscriber.latest_images()
                updated_any = False
                if self.file_camera_subscriber is not None:
                    file_images, file_timestamps = self.file_camera_subscriber.latest_images()
                    for name in ("left", "center", "right"):
                        timestamp = float(file_timestamps.get(name, 0.0))
                        image = file_images.get(name)
                        if _is_nonblank_image(image) and timestamp > 0.0:
                            self._last_good_images[name] = np.asarray(image, dtype=np.uint8).copy()
                            self._last_good_timestamps[name] = timestamp
                            self._last_good_camera_info.setdefault(name, _default_camera_info())
                            updated_any = True
                for name in ("left", "center", "right"):
                    timestamp = float(subscriber_timestamps.get(name, 0.0))
                    image = subscriber_images.get(name)
                    if _is_nonblank_image(image) and timestamp > 0.0:
                        self._last_good_images[name] = np.asarray(image, dtype=np.uint8).copy()
                        self._last_good_timestamps[name] = timestamp
                        if name in camera_info:
                            self._last_good_camera_info[name] = {
                                key: np.asarray(value, dtype=np.float32).copy()
                                for key, value in camera_info[name].items()
                            }
                        updated_any = True
                missing = [
                    name
                    for name in ("left", "center", "right")
                    if float(self._last_good_timestamps.get(name, 0.0)) <= 0.0
                ]
                if not missing:
                    self._wrist_bootstrapped = True
                    self._background_fetch_stop.wait(0.5)
                    continue
                for name in missing:
                    direct_image = fetch_gazebo_topic_image(
                        f"/{name}_camera/image",
                        timeout_s=2.0,
                        expected_shape=self.image_shape,
                    )
                    if not _is_nonblank_image(direct_image):
                        direct_image = fetch_ros_topic_image(
                            f"/{name}_camera/image",
                            timeout_s=2.0,
                            expected_shape=self.image_shape,
                        )
                    if not _is_nonblank_image(direct_image):
                        continue
                    self._last_good_images[name] = np.asarray(direct_image, dtype=np.uint8).copy()
                    self._last_good_timestamps[name] = time.time()
                    self._last_good_camera_info.setdefault(name, _default_camera_info())
                    updated_any = True
                if updated_any and all(float(self._last_good_timestamps.get(name, 0.0)) > 0.0 for name in ("left", "center", "right")):
                    self._wrist_bootstrapped = True
            except Exception:
                pass
            self._background_fetch_stop.wait(0.5)


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


def _gazebo_image_text_to_array(
    text: str,
    *,
    expected_shape: tuple[int, int, int] | None,
) -> np.ndarray | None:
    width_match = re.search(r"\bwidth:\s*(\d+)", text)
    height_match = re.search(r"\bheight:\s*(\d+)", text)
    data_match = re.search(r'\bdata:\s*"((?:\\.|[^"])*)"', text, re.DOTALL)
    if width_match is None or height_match is None or data_match is None:
        return None
    try:
        width = int(width_match.group(1))
        height = int(height_match.group(1))
        decoded = ast.literal_eval('"' + data_match.group(1) + '"')
    except Exception:
        return None
    if isinstance(decoded, str):
        payload = decoded.encode("latin1", errors="ignore")
    elif isinstance(decoded, bytes):
        payload = decoded
    else:
        return None
    buffer = np.frombuffer(payload, dtype=np.uint8)
    if buffer.size < width * height * 3:
        return None
    frame = buffer[: width * height * 3].reshape(height, width, 3)
    if expected_shape is not None:
        expected_height, expected_width, _ = expected_shape
        if (height, width) != (expected_height, expected_width):
            row_index = np.linspace(0, height - 1, expected_height, dtype=np.int64)
            col_index = np.linspace(0, width - 1, expected_width, dtype=np.int64)
            frame = frame[row_index][:, col_index]
    return frame.copy()


def _ros_topic_text_to_array(
    text: str,
    *,
    expected_shape: tuple[int, int, int] | None,
) -> np.ndarray | None:
    width_match = re.search(r"^\s*width:\s*(\d+)\s*$", text, re.M)
    height_match = re.search(r"^\s*height:\s*(\d+)\s*$", text, re.M)
    if width_match is None or height_match is None:
        return None
    width = int(width_match.group(1))
    height = int(height_match.group(1))
    data_match = re.search(r"^\s*data:\s*$([\s\S]+)", text, re.M)
    if data_match is None:
        return None
    values = [int(match) for match in re.findall(r"^\s*-\s*(\d+)\s*$", data_match.group(1), re.M)]
    if len(values) < width * height * 3:
        return None
    frame = np.asarray(values[: width * height * 3], dtype=np.uint8).reshape(height, width, 3)
    if expected_shape is not None:
        expected_height, expected_width, _ = expected_shape
        if (height, width) != (expected_height, expected_width):
            row_index = np.linspace(0, height - 1, expected_height, dtype=np.int64)
            col_index = np.linspace(0, width - 1, expected_width, dtype=np.int64)
            frame = frame[row_index][:, col_index]
    return frame.copy()


def _default_camera_info() -> dict[str, np.ndarray]:
    return {
        "size": np.zeros(2, dtype=np.float32),
        "k": np.zeros(9, dtype=np.float32),
        "p": np.zeros(12, dtype=np.float32),
    }


def _is_nonblank_image(image: np.ndarray | None) -> bool:
    return bool(
        image is not None
        and getattr(image, "size", 0) > 0
        and int(np.asarray(image, dtype=np.uint8).sum()) > 0
    )


def _has_all_wrist_frames(
    images: dict[str, np.ndarray],
    timestamps: dict[str, float],
) -> bool:
    return all(
        _is_nonblank_image(images.get(name)) and float(timestamps.get(name, 0.0)) > 0.0
        for name in ("left", "center", "right")
    )


def _resolve_executable_from_active_env(name: str) -> str | None:
    resolved = shutil.which(name)
    if resolved is not None:
        return resolved
    prefix_bin = Path(sys.prefix) / "bin" / name
    if prefix_bin.exists():
        return str(prefix_bin)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_bin = Path(conda_prefix) / "bin" / name
        if conda_bin.exists():
            return str(conda_bin)
    return None


def _ros_gz_subprocess_env() -> dict[str, str]:
    env = _gazebo_subprocess_env()
    prefix = Path(os.environ.get("CONDA_PREFIX") or sys.prefix)
    prefix_bin = prefix / "bin"
    if prefix_bin.exists():
        env["PATH"] = f"{prefix_bin}{os.pathsep}{env.get('PATH', '')}"
    if prefix.exists():
        env.setdefault("AMENT_PREFIX_PATH", str(prefix))
        env.setdefault("COLCON_PREFIX_PATH", str(prefix))
    env.setdefault("ROS_DISTRO", "kilted")
    return env
