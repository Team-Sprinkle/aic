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
            image = annotate_live_overview_frame(
                live_image=np.asarray(live_image, dtype=np.uint8),
                scenario=scenario,
                state=state,
                view_name=view_name,
                image_size=image_size,
            )
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
                "diagnostic_overlay": bool(live_image is not None),
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


def annotate_live_overview_frame(
    *,
    live_image: np.ndarray,
    scenario,
    state,
    view_name: str,
    image_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Add planner-visible port-frame diagnostics to a real Gazebo overview."""

    width, height = image_size
    image = Image.fromarray(np.asarray(live_image, dtype=np.uint8)).resize((width, height), Image.BILINEAR).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    schematic = Image.fromarray(
        render_scene_overview_frame(
            scenario=scenario,
            state=state,
            view_name=view_name,
            image_size=(96, 96),
        )
    )
    image.paste(schematic, (width - 100, 38))

    score_geometry = getattr(state, "score_geometry", {}) or {}
    plug = np.asarray(state.plug_pose[:3], dtype=np.float64)
    target = np.asarray(state.target_port_pose[:3], dtype=np.float64)
    entrance = (
        target
        if state.target_port_entrance_pose is None
        else np.asarray(state.target_port_entrance_pose[:3], dtype=np.float64)
    )
    axis = target - entrance
    axis_norm = float(np.linalg.norm(axis))
    axis_unit = axis / axis_norm if axis_norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    offset = plug - entrance
    axial_depth = float(np.dot(offset, axis_unit))
    axial_residual = float(axis_norm - axial_depth)
    lateral_vec = offset - axial_depth * axis_unit
    lateral = float(np.linalg.norm(lateral_vec))
    wrench = np.asarray(getattr(state, "wrench", np.zeros(6, dtype=np.float64)), dtype=np.float64).reshape(-1)
    force_xyz = np.zeros(3, dtype=np.float64) if wrench.size < 3 else wrench[:3]
    axial_force = float(np.dot(force_xyz, axis_unit))
    pre_insert = entrance - 0.025 * axis_unit
    distance_to_entrance = float(score_geometry.get("distance_to_entrance", np.linalg.norm(plug - entrance)) or 0.0)
    distance_to_target = float(score_geometry.get("distance_to_target", np.linalg.norm(plug - target)) or 0.0)
    progress = float(score_geometry.get("insertion_progress", 0.0) or 0.0)

    draw.rectangle([(0, 0), (width, 34)], fill=(20, 24, 28, 220))
    draw.text(
        (8, 8),
        f"live {view_name} | sim {float(getattr(state, 'sim_time', 0.0)):.2f}s tick {int(getattr(state, 'sim_tick', 0))}",
        fill=(250, 250, 250, 255),
    )
    draw.rectangle([(0, height - 72), (width, height)], fill=(20, 24, 28, 225))
    lines = [
        f"port frame: depth={axial_depth:+.3f}m residual={axial_residual:+.3f}m lateral={lateral:.3f}m progress={progress:.2f}",
        f"dist entrance={distance_to_entrance:.3f}m target={distance_to_target:.3f}m",
        f"axis_world=[{axis_unit[0]:+.2f},{axis_unit[1]:+.2f},{axis_unit[2]:+.2f}] F_axis={axial_force:+.1f}N pre_xyz=[{pre_insert[0]:.3f},{pre_insert[1]:.3f},{pre_insert[2]:.3f}]",
    ]
    for index, line in enumerate(lines):
        draw.text((8, height - 66 + 18 * index), line[:96], fill=(245, 245, 245, 255))
    draw.rectangle([(width - 101, 37), (width - 3, 136)], outline=(245, 245, 245, 220), width=1)
    draw.text((width - 98, 138), "schematic + axis", fill=(245, 245, 245, 255))
    return np.asarray(image, dtype=np.uint8)


def render_wrist_diagnostic_frame(
    *,
    scenario,
    state,
    camera_name: str,
    image_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    view_name = {
        "left": "oblique_xy",
        "center": "front_xz",
        "right": "side_yz",
    }.get(camera_name, "front_xz")
    image = Image.fromarray(
        render_scene_overview_frame(
            scenario=scenario,
            state=state,
            view_name=view_name,
            image_size=image_size,
        )
    )
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image_size
    draw.rectangle([(0, 0), (width - 1, height - 1)], outline=(40, 40, 40, 255), width=2)
    draw.rectangle([(6, 6), (width - 7, 28)], fill=(20, 24, 28, 210))
    draw.text(
        (12, 11),
        f"synthetic wrist {camera_name} | diagnostic projection",
        fill=(245, 245, 245, 255),
    )
    draw.line([(width // 2 - 18, height // 2), (width // 2 + 18, height // 2)], fill=(40, 40, 40, 170), width=1)
    draw.line([(width // 2, height // 2 - 18), (width // 2, height // 2 + 18)], fill=(40, 40, 40, 170), width=1)
    return np.asarray(image, dtype=np.uint8)


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
    image = Image.new("RGB", (width, height), color=(242, 244, 241))
    draw = ImageDraw.Draw(image)
    margin = 48

    def axis_value(pose: np.ndarray | list[float], axis: str) -> float:
        index = {"x": 0, "y": 1, "z": 2}[axis]
        return float(pose[index])

    board_pose = np.asarray(scenario.task_board.pose_xyz_rpy[:3], dtype=np.float64)
    tcp_pose = np.asarray(state.tcp_pose[:3], dtype=np.float64)
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

    cable_tail = plug_pose + np.array([-0.12, 0.05, 0.04], dtype=np.float64)
    xs = [
        axis_value(board_pose, x_axis),
        axis_value(tcp_pose, x_axis),
        axis_value(plug_pose, x_axis),
        axis_value(target_pose, x_axis),
        axis_value(entrance_pose, x_axis),
        axis_value(cable_tail, x_axis),
    ]
    ys = [
        axis_value(board_pose, y_axis),
        axis_value(tcp_pose, y_axis),
        axis_value(plug_pose, y_axis),
        axis_value(target_pose, y_axis),
        axis_value(entrance_pose, y_axis),
        axis_value(cable_tail, y_axis),
    ]
    for x_value, y_value, _ in obstacle_points:
        xs.append(x_value)
        ys.append(y_value)
    x_min, x_max = min(xs) - 0.16, max(xs) + 0.16
    y_min, y_max = min(ys) - 0.14, max(ys) + 0.14

    def to_px(x_value: float, y_value: float) -> tuple[int, int]:
        x_norm = 0.5 if abs(x_max - x_min) <= 1e-6 else (x_value - x_min) / (x_max - x_min)
        y_norm = 0.5 if abs(y_max - y_min) <= 1e-6 else (y_value - y_min) / (y_max - y_min)
        px = int(margin + x_norm * (width - 2 * margin))
        py = int(height - margin - y_norm * (height - 2 * margin))
        return px, py

    def marker(
        pose: np.ndarray,
        *,
        radius: int,
        fill: tuple[int, int, int],
        outline: tuple[int, int, int],
        label: str,
        label_offset: tuple[int, int] = (8, -18),
    ) -> tuple[int, int]:
        px, py = to_px(axis_value(pose, x_axis), axis_value(pose, y_axis))
        draw.ellipse([(px - radius, py - radius), (px + radius, py + radius)], fill=fill, outline=outline, width=2)
        draw.text((px + label_offset[0], py + label_offset[1]), label, fill=outline)
        return px, py

    board_center = to_px(axis_value(board_pose, x_axis), axis_value(board_pose, y_axis))
    board_w = max(90, min(width - 2 * margin, 180))
    board_h = max(70, min(height - 2 * margin, 130))
    draw.rectangle(
        [
            (board_center[0] - board_w // 2, board_center[1] - board_h // 2),
            (board_center[0] + board_w // 2, board_center[1] + board_h // 2),
        ],
        fill=(226, 228, 220),
        outline=(80, 80, 76),
        width=2,
    )
    draw.text((board_center[0] - board_w // 2 + 8, board_center[1] - board_h // 2 + 6), "task board", fill=(60, 60, 56))

    entrance_px = to_px(axis_value(entrance_pose, x_axis), axis_value(entrance_pose, y_axis))
    target_px = to_px(axis_value(target_pose, x_axis), axis_value(target_pose, y_axis))
    axis_vector = target_pose - entrance_pose
    axis_norm = float(np.linalg.norm(axis_vector))
    axis_unit = axis_vector / axis_norm if axis_norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    pre_insert_pose = entrance_pose - 0.025 * axis_unit
    pre_insert_px = to_px(axis_value(pre_insert_pose, x_axis), axis_value(pre_insert_pose, y_axis))
    corridor_half = 10
    draw.line([pre_insert_px, entrance_px], fill=(70, 170, 80), width=3)
    draw.line([entrance_px, target_px], fill=(20, 140, 180), width=5)
    draw.line(
        [(entrance_px[0], entrance_px[1] - corridor_half), (target_px[0], target_px[1] - corridor_half)],
        fill=(140, 210, 225),
        width=1,
    )
    draw.line(
        [(entrance_px[0], entrance_px[1] + corridor_half), (target_px[0], target_px[1] + corridor_half)],
        fill=(140, 210, 225),
        width=1,
    )

    for x_value, y_value, label in obstacle_points:
        px, py = to_px(x_value, y_value)
        draw.rectangle([(px - 6, py - 6), (px + 6, py + 6)], fill=(180, 60, 60), outline=(120, 20, 20))
        draw.text((px + 7, py + 4), label[:12], fill=(120, 20, 20))

    cable_tail_px = to_px(axis_value(cable_tail, x_axis), axis_value(cable_tail, y_axis))
    tcp_px = marker(tcp_pose, radius=5, fill=(100, 100, 110), outline=(55, 55, 65), label="tcp")
    plug_px = marker(plug_pose, radius=7, fill=(30, 150, 30), outline=(10, 90, 10), label="plug")
    target_px = marker(target_pose, radius=7, fill=(40, 90, 210), outline=(20, 45, 140), label="target")
    entrance_px = marker(entrance_pose, radius=6, fill=(20, 180, 190), outline=(0, 105, 120), label="entrance")
    marker(pre_insert_pose, radius=5, fill=(245, 190, 50), outline=(140, 90, 10), label="pre")
    draw.line([cable_tail_px, plug_px, tcp_px], fill=(70, 70, 70), width=3)
    draw.line([plug_px, entrance_px], fill=(80, 80, 80), width=2)

    axis_origin = (width - 74, height - 48)
    draw.line([axis_origin, (axis_origin[0] + 34, axis_origin[1])], fill=(210, 60, 60), width=3)
    draw.line([axis_origin, (axis_origin[0], axis_origin[1] - 34)], fill=(60, 120, 210), width=3)
    draw.text((axis_origin[0] + 38, axis_origin[1] - 7), x_axis, fill=(150, 30, 30))
    draw.text((axis_origin[0] - 5, axis_origin[1] - 48), y_axis, fill=(30, 70, 160))

    score_geometry = getattr(state, "score_geometry", {}) or {}
    distance_to_target = float(score_geometry.get("distance_to_target", np.linalg.norm(plug_pose - target_pose)) or 0.0)
    distance_to_entrance = float(score_geometry.get("distance_to_entrance", np.linalg.norm(plug_pose - entrance_pose)) or 0.0)
    lateral = float(score_geometry.get("lateral_misalignment", 0.0) or 0.0)
    progress = float(score_geometry.get("insertion_progress", 0.0) or 0.0)
    yaw_error = float(score_geometry.get("orientation_error", abs(float(state.plug_pose[5] - state.target_port_pose[5]))) or 0.0)
    offset = plug_pose - entrance_pose
    axial_depth = float(np.dot(offset, axis_unit))
    axial_residual = float(axis_norm - axial_depth)
    lateral_vector = offset - axial_depth * axis_unit
    port_lateral = float(np.linalg.norm(lateral_vector))
    wrench = np.asarray(getattr(state, "wrench", np.zeros(6, dtype=np.float64)), dtype=np.float64).reshape(-1)
    force_xyz = np.zeros(3, dtype=np.float64) if wrench.size < 3 else wrench[:3]
    axial_force = float(np.dot(force_xyz, axis_unit))
    header = f"{title} | sim {float(getattr(state, 'sim_time', 0.0)):.3f}s tick {int(getattr(state, 'sim_tick', 0))}"
    draw.rectangle([(0, 0), (width, 36)], fill=(35, 40, 44))
    draw.text((10, 10), header, fill=(245, 245, 245))
    details = (
        f"target {distance_to_target:.3f}m entrance {distance_to_entrance:.3f}m "
        f"depth_res {axial_residual:+.3f}m lat {port_lateral:.3f}m F_axis {axial_force:+.1f}N yaw {yaw_error:.3f}"
    )
    draw.rectangle([(0, height - 34), (width, height)], fill=(35, 40, 44))
    draw.text((10, height - 24), details[:92], fill=(245, 245, 245))
    return np.asarray(image, dtype=np.uint8)
