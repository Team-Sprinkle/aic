"""Slow official oracle planner policy.

This policy runs in the official ROS/Gazebo environment after AIC Engine spawns
the task. It captures real task/TF context, optionally saves wrist camera images,
calls GPT-5 mini for Cartesian delta approach/alignment waypoints, and writes a
PiecewiseTrajectory JSON. It intentionally does not try to complete insertion;
the postprocessed replay policy handles the second run.
"""

from __future__ import annotations

import os
from pathlib import Path

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException

from aic_teacher_official.context import OfficialTeacherContext
from aic_teacher_official.generate_piecewise import (
    PiecewiseGeneratorConfig,
    generate_piecewise_file,
)
from aic_teacher_official.vlm_planner import call_gpt5_mini_delta_planner


class OfficialTeacherOraclePlanner(Policy):
    """Capture a real official task and write a VLM piecewise trajectory."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._output = os.environ.get(
            "AIC_OFFICIAL_TEACHER_PIECEWISE_OUTPUT",
            "artifacts/piecewise_trajectory.json",
        )
        self._image_dir = Path(
            os.environ.get(
                "AIC_OFFICIAL_TEACHER_IMAGE_DIR",
                str(Path(self._output).with_suffix("") / "images"),
            )
        )
        self._max_vlm_calls = int(os.environ.get("AIC_OFFICIAL_TEACHER_MAX_VLM_CALLS", "20"))
        self._use_vlm = os.environ.get("AIC_OFFICIAL_TEACHER_USE_VLM", "true").lower() == "true"

    def _wait_for_tf(self, target_frame: str, source_frame: str, timeout_sec: float = 15.0) -> bool:
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(target_frame, source_frame, Time())
                return True
            except TransformException:
                self.sleep_for(0.1)
        return False

    def _capture_context(self, task: Task) -> OfficialTeacherContext:
        tcp_frame = "gripper/tcp"
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        plug_frame = f"{task.cable_name}/{task.plug_name}_link"
        for frame in (tcp_frame, port_frame, plug_frame):
            if not self._wait_for_tf("base_link", frame):
                raise RuntimeError(f"Timed out waiting for TF frame: {frame}")
        tcp = self._parent_node._tf_buffer.lookup_transform("base_link", tcp_frame, Time()).transform
        port = self._parent_node._tf_buffer.lookup_transform("base_link", port_frame, Time()).transform
        plug = self._parent_node._tf_buffer.lookup_transform("base_link", plug_frame, Time()).transform
        return OfficialTeacherContext(
            start_position=[tcp.translation.x, tcp.translation.y, tcp.translation.z],
            port_position=[port.translation.x, port.translation.y, port.translation.z],
            orientation_xyzw=[tcp.rotation.x, tcp.rotation.y, tcp.rotation.z, tcp.rotation.w],
            port_orientation_xyzw=[
                port.rotation.x,
                port.rotation.y,
                port.rotation.z,
                port.rotation.w,
            ],
            plug_position=[plug.translation.x, plug.translation.y, plug.translation.z],
            plug_orientation_xyzw=[
                plug.rotation.x,
                plug.rotation.y,
                plug.rotation.z,
                plug.rotation.w,
            ],
            target_module_name=task.target_module_name,
            port_name=task.port_name,
            cable_name=task.cable_name,
            plug_name=task.plug_name,
            diagnostics={
                "source": "OfficialTeacherOraclePlanner",
                "task_id": task.id,
                "tcp_frame": tcp_frame,
                "port_frame": port_frame,
                "plug_frame": plug_frame,
                "port_type": task.port_type,
                "plug_type": task.plug_type,
            },
        )

    def _save_observation_images(self, get_observation: GetObservationCallback) -> list[Path]:
        observation = get_observation()
        if observation is None:
            return []
        self._image_dir.mkdir(parents=True, exist_ok=True)
        image_paths: list[Path] = []
        for name in ("left", "center", "right"):
            image = getattr(observation, f"{name}_image")
            path = self._image_dir / f"{name}.png"
            try:
                self._write_png(image, path)
                image_paths.append(path)
            except Exception as ex:
                self.get_logger().warn(f"Could not save {name} image for VLM context: {ex}")
        return image_paths

    @staticmethod
    def _write_png(image, path: Path) -> None:
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

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(f"OfficialTeacherOraclePlanner task: {task}")
        send_feedback("official_teacher_oracle_planning_started")
        context = self._capture_context(task)
        image_paths = self._save_observation_images(get_observation)
        vlm_plan = None
        if self._use_vlm:
            vlm_plan = call_gpt5_mini_delta_planner(
                context,
                image_paths=image_paths,
                max_calls=self._max_vlm_calls,
                model="gpt-5-mini",
            )
        generate_piecewise_file(
            PiecewiseGeneratorConfig(
                start_position=context.start_position,
                port_position=context.port_position,
                orientation_xyzw=context.orientation_xyzw,
                approach_offset=[-0.08, -0.08, 0.22],
                task_name="Insert cable into target port",
                context=context,
                vlm_delta_plan=vlm_plan,
            ),
            self._output,
        )
        context.save_json(Path(self._output).with_suffix(".context.json"))
        send_feedback(f"official_teacher_oracle_piecewise_written:{self._output}")
        self.get_logger().info(f"Wrote official teacher piecewise trajectory: {self._output}")
        return False
