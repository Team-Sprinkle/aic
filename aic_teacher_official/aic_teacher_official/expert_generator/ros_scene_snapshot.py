"""Live ROS/Gazebo scene snapshot capture."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aic_teacher_official.debug_recorder import validate_image, write_json
from aic_teacher_official.expert_generator.collision_scene import object_geometries_from_engine_config
from aic_teacher_official.expert_generator.scene_snapshot import SceneSnapshot, SerializablePose


class LiveSceneSnapshotProvider:
    """Capture `SceneSnapshot` from an active official policy node."""

    def __init__(
        self,
        parent_node: Any,
        *,
        output_dir: str | Path,
        run_id: str,
        seed: int | None = None,
        engine_config: str | Path | None = None,
        image_sample_period_sec: float = 0.5,
        image_capture_duration_sec: float = 2.0,
        max_images: int = 8,
    ):
        self.parent_node = parent_node
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.seed = seed
        self.engine_config = Path(engine_config) if engine_config else None
        self.image_sample_period_sec = image_sample_period_sec
        self.image_capture_duration_sec = image_capture_duration_sec
        self.max_images = max_images

    def capture_from_policy(
        self,
        *,
        task: Any,
        get_observation: Any,
        mode: str,
        scene_id: str,
    ) -> SceneSnapshot:
        tcp_frame = "gripper/tcp"
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        plug_frame = f"{task.cable_name}/{task.plug_name}_link"
        for frame in (tcp_frame, port_frame, plug_frame):
            self._wait_for_tf("base_link", frame)
        tcp = self._lookup_pose("base_link", tcp_frame)
        port = self._lookup_pose("base_link", port_frame)
        plug = self._lookup_pose("base_link", plug_frame)
        image_paths = self._capture_images(get_observation)
        observation = get_observation()
        joint_state = _joint_state_from_observation(observation)
        ft_baseline = _ft_from_observation(observation)
        collision_objects = []
        unavailable: dict[str, str] = {}
        if self.engine_config is not None:
            try:
                collision_objects = object_geometries_from_engine_config(self.engine_config)
            except Exception as ex:
                unavailable["collision_objects"] = f"{type(ex).__name__}: {ex}"
        else:
            unavailable["collision_objects"] = "engine_config_not_provided"
        snapshot = SceneSnapshot(
            run_id=self.run_id,
            seed=self.seed,
            scene_id=scene_id,
            mode=mode,
            joint_state=joint_state,
            tcp_pose=tcp,
            target_port_pose=port,
            plug_pose=plug,
            plug_tip_pose=plug,
            tf_frames={
                "base": "base_link",
                "tcp": tcp_frame,
                "target_port": port_frame,
                "plug": plug_frame,
            },
            camera_images=[str(path) for path in image_paths],
            ft_baseline=ft_baseline,
            task_config=_task_to_dict(task),
            collision_objects=collision_objects,
            unavailable_reasons=unavailable,
            metadata={"source": "LiveSceneSnapshotProvider"},
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.output_dir / "scene_snapshot.json", snapshot.to_dict())
        return snapshot

    def _wait_for_tf(self, target_frame: str, source_frame: str, timeout_sec: float = 15.0) -> None:
        from rclpy.duration import Duration
        from rclpy.time import Time
        from tf2_ros import TransformException

        start = self.parent_node.get_clock().now()
        timeout = Duration(seconds=timeout_sec)
        while (self.parent_node.get_clock().now() - start) < timeout:
            try:
                self.parent_node._tf_buffer.lookup_transform(target_frame, source_frame, Time())
                return
            except TransformException:
                self.parent_node.get_clock()
                import time

                time.sleep(0.1)
        raise TimeoutError(f"Timed out waiting for TF frame {source_frame!r} -> {target_frame!r}")

    def _lookup_pose(self, target_frame: str, source_frame: str) -> SerializablePose:
        from rclpy.time import Time

        transform = self.parent_node._tf_buffer.lookup_transform(target_frame, source_frame, Time()).transform
        return SerializablePose(
            position=[
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
            ],
            orientation_xyzw=[
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w,
            ],
            frame_id=target_frame,
        )

    def _capture_images(self, get_observation: Any) -> list[Path]:
        image_dir = self.output_dir / "live_strategy_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        captured: list[Path] = []
        import time

        start = time.monotonic()
        sample_index = 0
        while True:
            observation = get_observation()
            if observation is not None:
                for name in ("left", "center", "right"):
                    image = getattr(observation, f"{name}_image", None)
                    if image is None:
                        continue
                    path = image_dir / f"{name}_{sample_index:03d}.png"
                    _write_png(image, path)
                    captured.append(path)
            if time.monotonic() - start >= self.image_capture_duration_sec:
                break
            sample_index += 1
            time.sleep(self.image_sample_period_sec)
        manifest = []
        selected: list[Path] = []
        for path in captured:
            validation = validate_image(path)
            manifest.append({"path": str(path), "validation": validation})
            if validation.get("valid") and not validation.get("near_constant"):
                selected.append(path)
        write_json(
            image_dir / "image_validation.json",
            {
                "captured": len(captured),
                "selected": len(selected[: self.max_images]),
                "entries": manifest,
            },
        )
        if not selected:
            raise RuntimeError("No valid nonblank live observation images were captured for VLM strategy.")
        return selected[: self.max_images]


def _write_png(image: Any, path: Path) -> None:
    import cv2
    import numpy as np

    encoding = image.encoding.lower()
    if encoding not in {"rgb8", "bgr8", "rgba8", "bgra8", "mono8"}:
        raise ValueError(f"Unsupported image encoding: {image.encoding}")
    channels = 1 if encoding == "mono8" else 4 if encoding in {"rgba8", "bgra8"} else 3
    height = int(image.height)
    width = int(image.width)
    step = int(image.step)
    row_bytes = width * channels
    data = bytes(image.data)
    rows = [data[row_start : row_start + row_bytes] for row_start in range(0, height * step, step)]
    array = np.frombuffer(b"".join(rows), dtype=np.uint8)
    if encoding == "mono8":
        array = array.reshape((height, width))
    else:
        array = array.reshape((height, width, channels))
        if encoding == "rgb8":
            array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        elif encoding == "rgba8":
            array = cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
        elif encoding == "bgra8":
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    success, encoded = cv2.imencode(".png", array)
    if not success:
        raise RuntimeError("OpenCV failed to encode PNG")
    path.write_bytes(encoded.tobytes())


def _joint_state_from_observation(observation: Any) -> dict[str, list[float]] | None:
    if observation is None or not hasattr(observation, "joint_states"):
        return None
    joints = observation.joint_states
    return {
        "name": list(getattr(joints, "name", [])),
        "position": [float(v) for v in getattr(joints, "position", [])],
        "velocity": [float(v) for v in getattr(joints, "velocity", [])],
        "effort": [float(v) for v in getattr(joints, "effort", [])],
    }


def _ft_from_observation(observation: Any) -> dict[str, Any] | None:
    if observation is None or not hasattr(observation, "wrist_wrench"):
        return None
    wrench = observation.wrist_wrench.wrench
    return {
        "force": {
            "x": float(wrench.force.x),
            "y": float(wrench.force.y),
            "z": float(wrench.force.z),
        },
        "torque": {
            "x": float(wrench.torque.x),
            "y": float(wrench.torque.y),
            "z": float(wrench.torque.z),
        },
    }


def _task_to_dict(task: Any) -> dict[str, Any]:
    fields = [
        "id",
        "cable_type",
        "cable_name",
        "plug_type",
        "plug_name",
        "port_type",
        "port_name",
        "target_module_name",
        "time_limit",
    ]
    return {field: getattr(task, field) for field in fields if hasattr(task, field)}
