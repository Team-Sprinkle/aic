"""Teacher-side visual context helpers for VLM planning and offline visualization."""

from __future__ import annotations

import atexit
import base64
from dataclasses import dataclass
from io import BytesIO
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..io import RosCameraSubscriber, capture_scene_probe_images, fetch_gazebo_topic_image


SCENE_OVERVIEW_VIEWS: tuple[tuple[str, str, str], ...] = (
    ("top_down_xy", "x", "y"),
    ("front_xz", "x", "z"),
    ("side_yz", "y", "z"),
    ("oblique_xy", "x", "y"),
)


@dataclass(frozen=True)
class VisualSelectionSummary:
    recent_labels: list[str]
    scene_labels: list[str]
    selected_recent_count: int
    selected_scene_count: int
    remaining_episode_budget: int


@dataclass
class LiveOverviewCameraProvider:
    topic_map: dict[str, str] = None  # type: ignore[assignment]
    image_shape: tuple[int, int, int] = (512, 512, 3)

    def __post_init__(self) -> None:
        if self.topic_map is None:
            self.topic_map = {
                "top_down_xy": "/overview_camera/image",
                "front_xz": "/overview_front_camera/image",
                "side_yz": "/overview_side_camera/image",
                "oblique_xy": "/overview_oblique_camera/image",
            }
        self._subscriber = RosCameraSubscriber(
            image_shape=self.image_shape,
            topic_map=self.topic_map,
            node_name="aic_gym_gz_live_overview_camera_provider",
        )
        self._started = False

    def latest_images(
        self,
        *,
        timeout_s: float = 20.0,
        image_size: tuple[int, int] = (256, 256),
    ) -> dict[str, np.ndarray]:
        self._ensure_started()
        deadline = time.monotonic() + max(float(timeout_s), 0.5)
        self._subscriber.wait_until_ready(timeout_s=max(float(timeout_s), 0.5))
        while time.monotonic() < deadline:
            images, timestamps, _ = self._subscriber.latest_images()
            resized: dict[str, np.ndarray] = {}
            if any(float(value) > 0.0 for value in timestamps.values()):
                for view_name, image in images.items():
                    if image is None:
                        continue
                    frame = np.asarray(image, dtype=np.uint8)
                    if frame.size == 0 or int(frame.sum()) == 0:
                        continue
                    if frame.shape[:2] != image_size:
                        frame = np.asarray(
                            Image.fromarray(frame).resize((image_size[1], image_size[0]), Image.BILINEAR),
                            dtype=np.uint8,
                        )
                    resized[view_name] = frame
            for view_name, topic in self.topic_map.items():
                if view_name in resized:
                    continue
                frame = fetch_gazebo_topic_image(
                    topic,
                    timeout_s=min(timeout_s, 2.0),
                    expected_shape=(image_size[0], image_size[1], 3),
                )
                if frame is None or frame.size == 0 or int(frame.sum()) == 0:
                    continue
                resized[view_name] = np.asarray(frame, dtype=np.uint8)
            missing_views = tuple(
                view_name
                for view_name in self.topic_map
                if view_name not in resized
            )
            if missing_views:
                probe_images = capture_scene_probe_images(
                    view_names=missing_views,
                    expected_shape=(image_size[0], image_size[1], 3),
                )
                for view_name, frame in probe_images.items():
                    if frame is None or frame.size == 0 or int(frame.sum()) == 0:
                        continue
                    resized[view_name] = np.asarray(frame, dtype=np.uint8)
            if len(resized) == len(self.topic_map):
                return resized
            time.sleep(0.2)
        return resized if "resized" in locals() else {}

    def close(self) -> None:
        if not self._started:
            return
        self._subscriber.close()
        self._started = False

    def _ensure_started(self) -> None:
        if self._started:
            return
        self._subscriber.start()
        self._started = True


_LIVE_OVERVIEW_PROVIDER: LiveOverviewCameraProvider | None = None


def latest_live_overview_images(
    *,
    timeout_s: float = 20.0,
    image_size: tuple[int, int] = (256, 256),
) -> dict[str, np.ndarray]:
    global _LIVE_OVERVIEW_PROVIDER
    if _LIVE_OVERVIEW_PROVIDER is None:
        _LIVE_OVERVIEW_PROVIDER = LiveOverviewCameraProvider()
    return _LIVE_OVERVIEW_PROVIDER.latest_images(timeout_s=timeout_s, image_size=image_size)


def _close_live_overview_provider() -> None:
    global _LIVE_OVERVIEW_PROVIDER
    if _LIVE_OVERVIEW_PROVIDER is None:
        return
    _LIVE_OVERVIEW_PROVIDER.close()
    _LIVE_OVERVIEW_PROVIDER = None


atexit.register(_close_live_overview_provider)


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
    selected_frames = frames[-max_frames:]
    latest_sim_time = None
    if selected_frames:
        latest_sim_time = max(
            float(frame.get("sim_time", 0.0) or 0.0) for frame in selected_frames
        )
    for frame_index, frame in enumerate(selected_frames):
        images = dict(frame.get("images", {}))
        timestamps = dict(frame.get("image_timestamps", {}))
        for camera_name in camera_names:
            image = images.get(camera_name)
            if image is None:
                continue
            sim_time = frame.get("sim_time")
            sim_tick = frame.get("sim_tick")
            age_s = None
            if latest_sim_time is not None and sim_time is not None:
                age_s = float(latest_sim_time - float(sim_time))
            observations.append(
                {
                    "label": f"recent_{frame_index}_{camera_name}",
                    "camera_name": camera_name,
                    "sim_tick": sim_tick,
                    "sim_time": sim_time,
                    "timestamp": timestamps.get(camera_name),
                    "age_from_latest_s": age_s,
                    "age_from_latest_steps": frame.get("age_from_latest_steps"),
                    "timepoint_label": (
                        "current"
                        if age_s is not None and age_s <= 1e-6
                        else f"{age_s:.3f}s_before_current"
                        if age_s is not None
                        else "historical"
                    ),
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
    live_images_by_view: dict[str, np.ndarray] | None = None,
    require_live_images: bool = False,
) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    for view_name, _, _ in SCENE_OVERVIEW_VIEWS:
        source = "teacher_schematic_scene_overview"
        live_image = None if live_images_by_view is None else live_images_by_view.get(view_name)
        if live_image is not None:
            image = np.asarray(live_image, dtype=np.uint8)
            source = "live_overview_topic"
        elif require_live_images:
            image = None
            source = "missing_live_overview"
        else:
            image = render_scene_overview_frame(
                scenario=scenario,
                state=state,
                image_size=image_size,
                view_name=view_name,
            )
        images.append(
            {
                "label": f"scene_overview_{view_name}",
                "view_name": view_name,
                "source": source,
                "timestamp": None,
                "image_data_url": None if image is None else encode_image_data_url(image),
            }
        )
    return images


def render_scene_overview_frame(
    *,
    scenario,
    state,
    view_name: str = "top_down_xy",
    image_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    for candidate_name, x_axis, y_axis in SCENE_OVERVIEW_VIEWS:
        if candidate_name == view_name:
            return _render_scene_view(
                scenario=scenario,
                state=state,
                x_axis=x_axis,
                y_axis=y_axis,
                image_size=image_size,
                title=view_name,
            )
    raise KeyError(f"Unknown scene overview view {view_name!r}.")


def select_visual_context_items(
    *,
    recent_visual_observations: list[dict[str, Any]],
    scene_overview_images: list[dict[str, Any]],
    max_recent_visual_images: int,
    max_scene_overview_images: int,
    episode_remaining_budget: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], VisualSelectionSummary]:
    remaining_budget = (
        max(0, int(episode_remaining_budget))
        if episode_remaining_budget is not None
        else max_recent_visual_images + max_scene_overview_images
    )
    recent_limit = min(max_recent_visual_images, remaining_budget)
    selected_recent = _sample_recent_visual_observations(
        recent_visual_observations,
        limit=recent_limit,
    )
    remaining_budget = max(remaining_budget - len(selected_recent), 0)
    scene_limit = min(max_scene_overview_images, remaining_budget)
    selected_scene = scene_overview_images[:scene_limit]
    remaining_budget = max(remaining_budget - len(selected_scene), 0)
    summary = VisualSelectionSummary(
        recent_labels=[str(item.get("label", "")) for item in selected_recent],
        scene_labels=[str(item.get("label", "")) for item in selected_scene],
        selected_recent_count=len(selected_recent),
        selected_scene_count=len(selected_scene),
        remaining_episode_budget=remaining_budget,
    )
    return selected_recent, selected_scene, summary


def _sample_recent_visual_observations(
    observations: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or not observations:
        return []
    if len(observations) <= limit:
        return list(observations)
    current_by_camera: dict[str, dict[str, Any]] = {}
    for item in reversed(observations):
        camera_name = str(item.get("camera_name", "unknown"))
        if camera_name not in current_by_camera:
            current_by_camera[camera_name] = item
    selected: list[dict[str, Any]] = list(reversed(list(current_by_camera.values())))
    if len(selected) >= limit:
        return selected[:limit]
    remaining_candidates = [item for item in observations if item not in selected]
    slots = limit - len(selected)
    if slots <= 0 or not remaining_candidates:
        return selected
    indices = np.linspace(0, len(remaining_candidates) - 1, slots, dtype=np.int64)
    for index in indices.tolist():
        candidate = remaining_candidates[index]
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected[:limit]


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
