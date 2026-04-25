"""Official ROS/Gazebo context capture for slow teacher planning."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OfficialTeacherContext:
    start_position: list[float]
    port_position: list[float]
    orientation_xyzw: list[float]
    port_orientation_xyzw: list[float] | None = None
    plug_position: list[float] | None = None
    plug_orientation_xyzw: list[float] | None = None
    target_module_name: str = ""
    port_name: str = ""
    cable_name: str = ""
    plug_name: str = ""
    diagnostics: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OfficialTeacherContext":
        return cls(
            start_position=[float(v) for v in data["start_position"]],
            port_position=[float(v) for v in data["port_position"]],
            orientation_xyzw=[float(v) for v in data.get("orientation_xyzw", [1.0, 0.0, 0.0, 0.0])],
            port_orientation_xyzw=(
                [float(v) for v in data["port_orientation_xyzw"]]
                if data.get("port_orientation_xyzw") is not None
                else None
            ),
            plug_position=(
                [float(v) for v in data["plug_position"]]
                if data.get("plug_position") is not None
                else None
            ),
            plug_orientation_xyzw=(
                [float(v) for v in data["plug_orientation_xyzw"]]
                if data.get("plug_orientation_xyzw") is not None
                else None
            ),
            target_module_name=str(data.get("target_module_name", "")),
            port_name=str(data.get("port_name", "")),
            cable_name=str(data.get("cable_name", "")),
            plug_name=str(data.get("plug_name", "")),
            diagnostics=dict(data.get("diagnostics", {})),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "OfficialTeacherContext":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_position": [float(v) for v in self.start_position],
            "port_position": [float(v) for v in self.port_position],
            "orientation_xyzw": [float(v) for v in self.orientation_xyzw],
            "port_orientation_xyzw": (
                [float(v) for v in self.port_orientation_xyzw]
                if self.port_orientation_xyzw is not None
                else None
            ),
            "plug_position": (
                [float(v) for v in self.plug_position] if self.plug_position is not None else None
            ),
            "plug_orientation_xyzw": (
                [float(v) for v in self.plug_orientation_xyzw]
                if self.plug_orientation_xyzw is not None
                else None
            ),
            "target_module_name": self.target_module_name,
            "port_name": self.port_name,
            "cable_name": self.cable_name,
            "plug_name": self.plug_name,
            "diagnostics": dict(self.diagnostics or {}),
        }

    def save_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def capture_context_from_ros(
    *,
    target_module_name: str,
    port_name: str,
    timeout_sec: float = 10.0,
) -> OfficialTeacherContext:
    """Capture current TCP and target port from TF in a running official sim."""
    try:
        import rclpy
        from rclpy.duration import Duration
        from rclpy.node import Node
        from rclpy.time import Time
        from tf2_ros import Buffer, TransformException, TransformListener
    except Exception as ex:  # pragma: no cover - ROS import availability is environment-specific.
        raise RuntimeError(
            "ROS context capture requires rclpy/tf2_ros in the official environment. "
            "Run inside pixi with the official simulation already started, or pass --context-json."
        ) from ex

    rclpy.init(args=None)
    node = Node("official_teacher_context_capture")
    tf_buffer = Buffer()
    _listener = TransformListener(tf_buffer, node, spin_thread=True)
    deadline = node.get_clock().now() + Duration(seconds=timeout_sec)
    frames = {
        "tcp": "gripper/tcp",
        "port": f"task_board/{target_module_name}/{port_name}_link",
    }
    try:
        while node.get_clock().now() < deadline:
            try:
                tcp = tf_buffer.lookup_transform("base_link", frames["tcp"], Time()).transform
                port = tf_buffer.lookup_transform("base_link", frames["port"], Time()).transform
                context = OfficialTeacherContext(
                    start_position=[
                        tcp.translation.x,
                        tcp.translation.y,
                        tcp.translation.z,
                    ],
                    port_position=[
                        port.translation.x,
                        port.translation.y,
                        port.translation.z,
                    ],
                    orientation_xyzw=[
                        tcp.rotation.x,
                        tcp.rotation.y,
                        tcp.rotation.z,
                        tcp.rotation.w,
                    ],
                    port_orientation_xyzw=[
                        port.rotation.x,
                        port.rotation.y,
                        port.rotation.z,
                        port.rotation.w,
                    ],
                    target_module_name=target_module_name,
                    port_name=port_name,
                    diagnostics={
                        "source": "official_ros_tf",
                        "tcp_frame": frames["tcp"],
                        "port_frame": frames["port"],
                    },
                )
                return context
            except TransformException:
                rclpy.spin_once(node, timeout_sec=0.1)
        raise TimeoutError(
            f"Timed out waiting for TF frames: {frames['tcp']} and {frames['port']}"
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()
