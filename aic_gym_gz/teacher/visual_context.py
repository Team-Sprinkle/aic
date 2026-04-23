"""Teacher-side visual context helpers for VLM planning."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def encode_image_data_url(image: np.ndarray, *, format: str = "PNG") -> str:
    clipped = np.asarray(image, dtype=np.uint8)
    with BytesIO() as buffer:
        Image.fromarray(clipped).save(buffer, format=format)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{format.lower()};base64,{encoded}"


def build_recent_visual_observations(
    *,
    frames: list[dict[str, Any]],
    max_frames: int = 2,
    camera_names: tuple[str, ...] = ("left", "center", "right"),
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames[-max_frames:]):
        images = dict(frame.get("images", {}))
        timestamps = dict(frame.get("image_timestamps", {}))
        for camera_name in camera_names:
            image = images.get(camera_name)
            if image is None:
                continue
            observations.append(
                {
                    "label": f"recent_{frame_index}_{camera_name}",
                    "camera_name": camera_name,
                    "sim_tick": frame.get("sim_tick"),
                    "sim_time": frame.get("sim_time"),
                    "timestamp": timestamps.get(camera_name),
                    "source": "official_wrist_camera_history",
                    "image_data_url": encode_image_data_url(np.asarray(image, dtype=np.uint8)),
                }
            )
    return observations


def build_scene_overview_images(
    *,
    scenario,
    state,
    image_size: tuple[int, int] = (256, 256),
) -> list[dict[str, Any]]:
    views = [
        ("top_down_xy", "x", "y"),
        ("front_xz", "x", "z"),
        ("side_yz", "y", "z"),
    ]
    return [
        {
            "label": f"scene_overview_{view_name}",
            "view_name": view_name,
            "source": "teacher_schematic_scene_overview",
            "image_data_url": encode_image_data_url(
                _render_scene_view(
                    scenario=scenario,
                    state=state,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    image_size=image_size,
                    title=view_name,
                )
            ),
        }
        for view_name, x_axis, y_axis in views
    ]


def _render_scene_view(
    *,
    scenario,
    state,
    x_axis: str,
    y_axis: str,
    image_size: tuple[int, int],
    title: str,
) -> np.ndarray:
    width, height = image_size
    image = Image.new("RGB", (width, height), color=(246, 246, 242))
    draw = ImageDraw.Draw(image)
    margin = 20

    def axis_value(pose: np.ndarray | list[float], axis: str) -> float:
        index = {"x": 0, "y": 1, "z": 2}[axis]
        return float(pose[index])

    board_pose = np.asarray(scenario.task_board.pose_xyz_rpy[:3], dtype=np.float64)
    plug_pose = np.asarray(state.plug_pose[:3], dtype=np.float64)
    target_pose = np.asarray(state.target_port_pose[:3], dtype=np.float64)
    entrance_pose = (
        target_pose
        if state.target_port_entrance_pose is None
        else np.asarray(state.target_port_entrance_pose[:3], dtype=np.float64)
    )

    obstacle_points: list[tuple[float, float, str]] = []
    for category_index, mapping in enumerate(
        (
            scenario.task_board.nic_rails,
            scenario.task_board.sc_rails,
            scenario.task_board.mount_rails,
        )
    ):
        lane_offset = (-0.06, 0.0, 0.06)[category_index]
        for entity in mapping.values():
            if not entity.present:
                continue
            if x_axis == "x":
                x_value = board_pose[0] + float(entity.translation)
            elif x_axis == "y":
                x_value = board_pose[1] + lane_offset
            else:
                x_value = board_pose[2]
            if y_axis == "z":
                y_value = board_pose[2] + 0.03
            elif y_axis == "y":
                y_value = board_pose[1] + lane_offset
            else:
                y_value = board_pose[0] + float(entity.translation)
            obstacle_points.append((x_value, y_value, entity.name or "rail"))

    xs = [axis_value(board_pose, x_axis), axis_value(plug_pose, x_axis), axis_value(target_pose, x_axis), axis_value(entrance_pose, x_axis)]
    ys = [axis_value(board_pose, y_axis), axis_value(plug_pose, y_axis), axis_value(target_pose, y_axis), axis_value(entrance_pose, y_axis)]
    for x_value, y_value, _ in obstacle_points:
        xs.append(x_value)
        ys.append(y_value)
    x_min, x_max = min(xs) - 0.12, max(xs) + 0.12
    y_min, y_max = min(ys) - 0.12, max(ys) + 0.12

    def to_px(x_value: float, y_value: float) -> tuple[int, int]:
        x_norm = 0.5 if abs(x_max - x_min) <= 1e-6 else (x_value - x_min) / (x_max - x_min)
        y_norm = 0.5 if abs(y_max - y_min) <= 1e-6 else (y_value - y_min) / (y_max - y_min)
        px = int(margin + x_norm * (width - 2 * margin))
        py = int(height - margin - y_norm * (height - 2 * margin))
        return px, py

    board_center = to_px(axis_value(board_pose, x_axis), axis_value(board_pose, y_axis))
    draw.rectangle(
        [
            (board_center[0] - 60, board_center[1] - 50),
            (board_center[0] + 60, board_center[1] + 50),
        ],
        outline=(90, 90, 90),
        width=2,
    )
    for x_value, y_value, _ in obstacle_points:
        px, py = to_px(x_value, y_value)
        draw.rectangle([(px - 6, py - 6), (px + 6, py + 6)], fill=(180, 60, 60), outline=(120, 20, 20))

    plug_px = to_px(axis_value(plug_pose, x_axis), axis_value(plug_pose, y_axis))
    target_px = to_px(axis_value(target_pose, x_axis), axis_value(target_pose, y_axis))
    entrance_px = to_px(axis_value(entrance_pose, x_axis), axis_value(entrance_pose, y_axis))
    draw.ellipse([(plug_px[0] - 6, plug_px[1] - 6), (plug_px[0] + 6, plug_px[1] + 6)], fill=(30, 150, 30))
    draw.ellipse([(target_px[0] - 6, target_px[1] - 6), (target_px[0] + 6, target_px[1] + 6)], fill=(40, 90, 210))
    draw.ellipse([(entrance_px[0] - 5, entrance_px[1] - 5), (entrance_px[0] + 5, entrance_px[1] + 5)], fill=(20, 180, 190))
    draw.line([plug_px, entrance_px], fill=(80, 80, 80), width=2)
    draw.text((10, 8), f"teacher scene overview: {title}", fill=(20, 20, 20))
    draw.text((10, height - 18), "green=plug blue=target cyan=entrance red=obstacles", fill=(40, 40, 40))
    return np.asarray(image, dtype=np.uint8)
