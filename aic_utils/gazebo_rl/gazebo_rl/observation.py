from __future__ import annotations

import base64
from typing import Any, Callable

import numpy as np


LoggerFn = Callable[[str], None]


def _warn(logger: LoggerFn | None, message: str) -> None:
    if logger is not None:
        logger(message)


def _seq(values: Any) -> list[float]:
    if values is None:
        return []
    return [float(v) for v in values]


def _stamp_to_sec(stamp: Any) -> float | None:
    if stamp is None:
        return None
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    if sec == 0 and nanosec == 0:
        return None
    return float(sec) + float(nanosec) * 1e-9


def _xyz(obj: Any) -> list[float]:
    return [
        float(getattr(obj, "x", 0.0)),
        float(getattr(obj, "y", 0.0)),
        float(getattr(obj, "z", 0.0)),
    ]


def _quat_xyzw(obj: Any) -> list[float]:
    return [
        float(getattr(obj, "x", 0.0)),
        float(getattr(obj, "y", 0.0)),
        float(getattr(obj, "z", 0.0)),
        float(getattr(obj, "w", 1.0)),
    ]


def pose_to_dict(pose: Any | None) -> dict[str, list[float]] | None:
    if pose is None:
        return None
    return {
        "position": _xyz(getattr(pose, "position", None)),
        "orientation_xyzw": _quat_xyzw(getattr(pose, "orientation", None)),
    }


def twist_to_dict(twist: Any | None) -> dict[str, list[float]] | None:
    if twist is None:
        return None
    return {
        "linear": _xyz(getattr(twist, "linear", None)),
        "angular": _xyz(getattr(twist, "angular", None)),
    }


def wrench_to_dict(wrench: Any | None) -> dict[str, list[float]]:
    if wrench is None:
        return {"force": [0.0, 0.0, 0.0], "torque": [0.0, 0.0, 0.0]}
    return {
        "force": _xyz(getattr(wrench, "force", None)),
        "torque": _xyz(getattr(wrench, "torque", None)),
    }


def task_to_dict(task: Any | None) -> dict[str, Any]:
    if task is None:
        return {}
    names = [
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
    return {name: getattr(task, name, None) for name in names}


def _rgb_array_from_image(image: Any, data: bytes) -> np.ndarray:
    height = int(getattr(image, "height", 0))
    width = int(getattr(image, "width", 0))
    encoding = str(getattr(image, "encoding", "rgb8")).lower()
    channels = 4 if encoding in {"rgba8", "bgra8"} else 1 if encoding in {"mono8", "8uc1"} else 3
    expected = height * width * channels
    array = np.frombuffer(data, dtype=np.uint8)
    if height <= 0 or width <= 0 or array.size < expected:
        raise ValueError(f"Invalid image payload for {height}x{width} {encoding}")
    array = array[:expected].reshape(height, width, channels)
    if encoding in {"bgr8", "bgra8"}:
        return array[..., :3][..., ::-1].copy()
    if encoding in {"rgba8", "rgb8"}:
        return array[..., :3].copy()
    if encoding in {"mono8", "8uc1"}:
        return np.repeat(array, 3, axis=2)
    return array[..., :3].copy()


def image_to_dict(
    image: Any | None,
    *,
    target_size: tuple[int, int] = (288, 256),
    jpeg_quality: int = 80,
) -> dict[str, Any] | None:
    if image is None:
        return None
    data = bytes(getattr(image, "data", b"") or b"")
    if not data:
        return None
    import cv2

    rgb = _rgb_array_from_image(image, data)
    resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise ValueError("Could not JPEG-encode Gazebo observation image")
    return {
        "height": int(target_size[1]),
        "width": int(target_size[0]),
        "encoding": "jpeg_rgb8",
        "is_bigendian": int(getattr(image, "is_bigendian", 0)),
        "step": int(target_size[0] * 3),
        "stamp": _stamp_to_sec(getattr(getattr(image, "header", None), "stamp", None)),
        "data_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }


def _tf_pose(tf_buffer: Any, target_frame: str, source_frame: str) -> dict[str, Any]:
    from rclpy.time import Time

    tf = tf_buffer.lookup_transform(target_frame, source_frame, Time())
    transform = tf.transform
    return {
        "frame_id": target_frame,
        "child_frame_id": source_frame,
        "stamp": _stamp_to_sec(getattr(getattr(tf, "header", None), "stamp", None)),
        "position": _xyz(transform.translation),
        "orientation_xyzw": _quat_xyzw(transform.rotation),
    }


def _relative_vector(a: dict[str, Any] | None, b: dict[str, Any] | None) -> list[float] | None:
    if a is None or b is None:
        return None
    pa = a.get("position")
    pb = b.get("position")
    if pa is None or pb is None:
        return None
    return [float(pb[i]) - float(pa[i]) for i in range(3)]


def observation_to_dict(
    observation: Any,
    *,
    task: Any | None = None,
    step_count: int = 0,
    tf_buffer: Any | None = None,
    ground_truth: bool = False,
    include_images: bool = False,
    logger: LoggerFn | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "step_count": int(step_count),
        "sim_time": None,
        "task": task_to_dict(task),
        "joints": {"name": [], "position": [], "velocity": [], "effort": []},
        "gripper": {"position": None, "velocity": None},
        "wrist_wrench": wrench_to_dict(None),
        "controller": {
            "current_tcp_pose": None,
            "reference_tcp_pose": None,
            "tcp_velocity": None,
            "tcp_error": [],
            "target_mode": None,
        },
        "images": {},
        "oracle": {},
    }

    if observation is None:
        _warn(logger, "Missing Observation message; returning zero-like observation")
        return out

    if include_images:
        for field, key in (
            ("center_image", "observation.images.center_camera"),
            ("left_image", "observation.images.left_camera"),
            ("right_image", "observation.images.right_camera"),
        ):
            image_dict = image_to_dict(getattr(observation, field, None))
            if image_dict is not None:
                out["images"][key] = image_dict

    joint_states = getattr(observation, "joint_states", None)
    if joint_states is not None:
        out["sim_time"] = _stamp_to_sec(getattr(getattr(joint_states, "header", None), "stamp", None))
        out["joints"] = {
            "name": list(getattr(joint_states, "name", []) or []),
            "position": _seq(getattr(joint_states, "position", [])),
            "velocity": _seq(getattr(joint_states, "velocity", [])),
            "effort": _seq(getattr(joint_states, "effort", [])),
        }
        names = out["joints"]["name"]
        positions = out["joints"]["position"]
        velocities = out["joints"]["velocity"]
        gripper_indices = [i for i, name in enumerate(names) if "gripper" in name or "finger" in name]
        if gripper_indices:
            idx = gripper_indices[0]
            out["gripper"]["position"] = positions[idx] if idx < len(positions) else None
            out["gripper"]["velocity"] = velocities[idx] if idx < len(velocities) else None
    else:
        _warn(logger, "Observation has no joint_states field")

    wrist = getattr(observation, "wrist_wrench", None)
    if wrist is not None:
        out["wrist_wrench"] = wrench_to_dict(getattr(wrist, "wrench", None))
        if out["sim_time"] is None:
            out["sim_time"] = _stamp_to_sec(getattr(getattr(wrist, "header", None), "stamp", None))

    controller = getattr(observation, "controller_state", None)
    if controller is not None:
        target_mode = getattr(controller, "target_mode", None)
        out["controller"] = {
            "current_tcp_pose": pose_to_dict(getattr(controller, "tcp_pose", None)),
            "reference_tcp_pose": pose_to_dict(getattr(controller, "reference_tcp_pose", None)),
            "tcp_velocity": twist_to_dict(getattr(controller, "tcp_velocity", None)),
            "tcp_error": _seq(getattr(controller, "tcp_error", [])),
            "target_mode": getattr(target_mode, "mode", None),
        }

    if ground_truth and tf_buffer is not None:
        task_dict = out["task"]
        target_module = task_dict.get("target_module_name")
        port_name = task_dict.get("port_name")
        cable_name = task_dict.get("cable_name")
        plug_name = task_dict.get("plug_name")
        frames = {
            "tcp_pose_base_link": "gripper/tcp",
            "target_port_pose_base_link": (
                f"task_board/{target_module}/{port_name}_link"
                if target_module and port_name
                else None
            ),
            "target_port_entrance_pose_base_link": (
                f"task_board/{target_module}/{port_name}_link_entrance"
                if target_module and port_name
                else None
            ),
            "plug_pose_base_link": (
                f"{cable_name}/{plug_name}_link" if cable_name and plug_name else None
            ),
        }
        oracle: dict[str, Any] = {}
        for key, frame in frames.items():
            if frame is None:
                continue
            try:
                oracle[key] = _tf_pose(tf_buffer, "base_link", frame)
            except Exception as ex:
                _warn(logger, f"TF lookup failed for {frame} -> base_link: {ex}")
                oracle[key] = None
        oracle["relative_plug_to_port_vector"] = _relative_vector(
            oracle.get("plug_pose_base_link"),
            oracle.get("target_port_pose_base_link"),
        )
        out["oracle"] = oracle

    return out
