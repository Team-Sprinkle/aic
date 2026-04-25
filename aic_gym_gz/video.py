"""Headless-safe trajectory video capture helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .io import (
    CameraBridgeSidecar,
    RosCameraSubscriber,
    capture_scene_probe_images,
    fetch_gazebo_topic_image,
    fetch_ros_topic_image,
)
def default_video_output_dir(*, run_name: str) -> Path:
    return Path("aic_gym_gz/artifacts/inspect_runs") / run_name / "videos"


def build_run_name(*, prefix: str, seed: int | None, trial_id: str | None) -> str:
    safe_trial = "default" if not trial_id else str(trial_id).replace("/", "_")
    safe_seed = "none" if seed is None else str(seed)
    return f"{prefix}_seed{safe_seed}_{safe_trial}"


@dataclass
class _StreamWriter:
    path: Path
    fps: float
    frame_count: int = 0
    sim_times: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(
            self.path,
            fps=self.fps,
            codec="libx264",
            quality=7,
            macro_block_size=None,
        )

    def append(self, frame: np.ndarray, *, sim_time: float | None = None) -> None:
        image = np.asarray(frame, dtype=np.uint8)
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        if image.shape[-1] != 3:
            raise ValueError(f"Expected RGB frame with shape (*, *, 3), got {image.shape}.")
        self._writer.append_data(image)
        self.frame_count += 1
        if sim_time is not None:
            self.sim_times.append(float(sim_time))

    def close(self) -> dict[str, Any]:
        self._writer.close()
        return {
            "path": str(self.path),
            "frame_count": self.frame_count,
            "fps": self.fps,
            "first_sim_time": None if not self.sim_times else float(self.sim_times[0]),
            "last_sim_time": None if not self.sim_times else float(self.sim_times[-1]),
        }


@dataclass
class HeadlessTrajectoryVideoRecorder:
    output_dir: Path
    fps: float = 12.5
    overview_view_name: str = "top_down_xy"
    enabled: bool = True
    require_real_wrist_images: bool = True
    require_live_overview: bool = True
    prefer_live_overview_camera: bool = True
    live_overview_topic: str = "/overview_camera/image"
    live_overview_shape: tuple[int, int, int] = (512, 512, 3)
    overview_capture_stride: int = 5

    OVERVIEW_TOPIC_MAP = {
        "top_down_xy": "/overview_camera/image",
        "front_xz": "/overview_front_camera/image",
        "side_yz": "/overview_side_camera/image",
        "oblique_xy": "/overview_oblique_camera/image",
    }

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self._writers: dict[str, _StreamWriter] = {}
        self._overview_camera_bridge: CameraBridgeSidecar | None = None
        self._overview_camera_subscriber: RosCameraSubscriber | None = None
        self._overview_sources: dict[str, dict[str, int]] = {
            view_name: {
                "live_overview_topic": 0,
                "gz_overview_topic": 0,
                "scene_probe_camera": 0,
                "missing_live_overview": 0,
            }
            for view_name in self.OVERVIEW_TOPIC_MAP
        }
        self._metadata: dict[str, Any] = {
            "output_dir": str(self.output_dir),
            "fps": self.fps,
            "overview_view_name": self.overview_view_name,
            "require_real_wrist_images": self.require_real_wrist_images,
            "require_live_overview": self.require_live_overview,
            "prefer_live_overview_camera": self.prefer_live_overview_camera,
            "live_overview_topic": self.live_overview_topic,
            "streams": {},
        }
        self._seen_live_overview = False
        self._capture_count = 0
        self._cached_overview_frames: dict[str, tuple[np.ndarray | None, str]] = {}
        self._wrist_missing_counts = {"left": 0, "center": 0, "right": 0}
        self._last_sim_time: float | None = None
        if not self.enabled:
            return
        self._writers = {
            "camera_left": _StreamWriter(self.output_dir / "camera_left.mp4", fps=self.fps),
            "camera_center": _StreamWriter(self.output_dir / "camera_center.mp4", fps=self.fps),
            "camera_right": _StreamWriter(self.output_dir / "camera_right.mp4", fps=self.fps),
        }
        if self.prefer_live_overview_camera:
            self._overview_camera_bridge = CameraBridgeSidecar(topic_map=dict(self.OVERVIEW_TOPIC_MAP))
            self._overview_camera_subscriber = RosCameraSubscriber(
                image_shape=self.live_overview_shape,
                topic_map=dict(self.OVERVIEW_TOPIC_MAP),
                node_name="aic_gym_gz_video_overview_subscriber",
            )
            self._overview_camera_bridge.start()
            self._overview_camera_subscriber.start()

    def capture(
        self,
        *,
        observation: dict[str, Any],
        scenario: Any,
        state: Any,
    ) -> None:
        if not self.enabled:
            return
        self._capture_count += 1
        sim_time = float(observation.get("sim_time", getattr(state, "sim_time", 0.0)))
        self._last_sim_time = sim_time
        images = observation.get("images") or {}
        for camera_name in ("left", "center", "right"):
            frame = images.get(camera_name)
            if frame is None or np.asarray(frame).size == 0 or int(np.asarray(frame).sum()) == 0:
                self._wrist_missing_counts[camera_name] += 1
                continue
            self._writers[f"camera_{camera_name}"].append(
                _camera_frame(
                    frame,
                    label=camera_name,
                    require_real=False,
                ),
                sim_time=sim_time,
            )
        overview_frames = self._overview_frames(scenario=scenario, state=state)
        for view_name, (overview, overview_source) in overview_frames.items():
            self._overview_sources[view_name][overview_source] += 1
            if overview is None:
                continue
            self._ensure_writer(name=f"overview_{view_name}", path=self.output_dir / f"overview_{view_name}.mp4").append(
                overview,
                sim_time=sim_time,
            )

    def close(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "output_dir": str(self.output_dir)}
        if self._overview_camera_subscriber is not None:
            self._overview_camera_subscriber.close()
        if self._overview_camera_bridge is not None:
            self._overview_camera_bridge.close()
        if self.require_real_wrist_images:
            self._backfill_missing_wrist_streams()
        if self.require_live_overview:
            self._backfill_missing_overview_streams()
        streams = {
            name: writer.close()
            for name, writer in self._writers.items()
        }
        if self.require_real_wrist_images:
            missing_wrist = [
                camera_name
                for camera_name in ("left", "center", "right")
                if streams.get(f"camera_{camera_name}", {}).get("frame_count", 0) <= 0
            ]
            if missing_wrist:
                raise RuntimeError(
                    "Real wrist camera frames were required for video export, but "
                    f"no nonblank frames were recorded for cameras {missing_wrist}."
                )
        if self.require_live_overview:
            missing_views = [
                view_name
                for view_name in self.OVERVIEW_TOPIC_MAP
                if streams.get(f"overview_{view_name}", {}).get("frame_count", 0) <= 0
            ]
            if missing_views:
                raise RuntimeError(
                    "Live overview camera frames were required for video export, but "
                    f"no nonblank frames were recorded for views {missing_views}."
                )
        self._metadata["overview_frame_sources"] = self._overview_sources
        self._metadata["streams"] = streams
        metadata_path = self.output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(self._metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "enabled": True,
            "output_dir": str(self.output_dir),
            "metadata_path": str(metadata_path),
            "streams": streams,
        }

    def _overview_frames(self, *, scenario: Any, state: Any) -> dict[str, tuple[np.ndarray, str]]:
        del scenario, state
        should_refresh = (
            not self._cached_overview_frames
            or self._capture_count <= 1
            or max(int(self.overview_capture_stride), 1) == 1
            or (self._capture_count - 1) % max(int(self.overview_capture_stride), 1) == 0
        )
        if not should_refresh:
            return dict(self._cached_overview_frames)
        deadline = time.monotonic() + (20.0 if self.require_live_overview and not self._seen_live_overview else 0.5)
        frames: dict[str, tuple[np.ndarray, str]] = {}
        while time.monotonic() < deadline:
            images: dict[str, np.ndarray] = {}
            if self._overview_camera_subscriber is not None:
                images, _, _ = self._overview_camera_subscriber.latest_images()
            refreshed: dict[str, tuple[np.ndarray, str]] = {}
            for view_name, topic in self.OVERVIEW_TOPIC_MAP.items():
                overview_image = images.get(view_name)
                if overview_image is not None and overview_image.size > 0 and int(overview_image.sum()) > 0:
                    refreshed[view_name] = (np.asarray(overview_image, dtype=np.uint8), "live_overview_topic")
                    continue
                overview_image = fetch_gazebo_topic_image(
                    topic,
                    timeout_s=0.5,
                    expected_shape=self.live_overview_shape,
                )
                if overview_image is not None and overview_image.size > 0 and int(overview_image.sum()) > 0:
                    refreshed[view_name] = (np.asarray(overview_image, dtype=np.uint8), "gz_overview_topic")
            missing_views = tuple(
                view_name
                for view_name in self.OVERVIEW_TOPIC_MAP
                if view_name not in refreshed
            )
            if missing_views:
                probe_frames = capture_scene_probe_images(
                    view_names=missing_views,
                    expected_shape=self.live_overview_shape,
                )
                for view_name in missing_views:
                    probe_image = probe_frames.get(view_name)
                    if probe_image is None or probe_image.size == 0 or int(probe_image.sum()) == 0:
                        continue
                    refreshed[view_name] = (np.asarray(probe_image, dtype=np.uint8), "scene_probe_camera")
            if len(refreshed) == len(self.OVERVIEW_TOPIC_MAP):
                self._seen_live_overview = True
                frames = refreshed
                break
            if not self.require_live_overview:
                frames = refreshed
                break
            time.sleep(0.2)
        for view_name in self.OVERVIEW_TOPIC_MAP:
            if view_name in frames:
                continue
            frames[view_name] = (None, "missing_live_overview")
        self._cached_overview_frames = dict(frames)
        return frames

    def _ensure_writer(self, *, name: str, path: Path) -> _StreamWriter:
        writer = self._writers.get(name)
        if writer is None:
            writer = _StreamWriter(path, fps=self.fps)
            self._writers[name] = writer
        return writer

    def _backfill_missing_wrist_streams(self) -> None:
        missing = [
            camera_name
            for camera_name in ("left", "center", "right")
            if self._writers.get(f"camera_{camera_name}") is not None
            and self._writers[f"camera_{camera_name}"].frame_count <= 0
        ]
        if not missing:
            return
        for camera_name in missing:
            frame = None
            for _attempt in range(3):
                frame = fetch_gazebo_topic_image(
                    f"/{camera_name}_camera/image",
                    timeout_s=15.0 if camera_name == "left" else 8.0,
                    expected_shape=(256, 256, 3),
                )
                if frame is None or frame.size == 0 or int(frame.sum()) == 0:
                    frame = fetch_ros_topic_image(
                        f"/{camera_name}_camera/image",
                        timeout_s=10.0,
                        expected_shape=(256, 256, 3),
                    )
                if frame is not None and frame.size > 0 and int(frame.sum()) > 0:
                    break
            if frame is None or frame.size == 0 or int(frame.sum()) == 0:
                continue
            self._ensure_writer(
                name=f"camera_{camera_name}",
                path=self.output_dir / f"camera_{camera_name}.mp4",
            ).append(
                np.asarray(frame, dtype=np.uint8),
                sim_time=self._last_sim_time,
            )

    def _backfill_missing_overview_streams(self) -> None:
        missing_views = [
            view_name
            for view_name in self.OVERVIEW_TOPIC_MAP
            if self._writers.get(f"overview_{view_name}") is None
            or self._writers[f"overview_{view_name}"].frame_count <= 0
        ]
        if not missing_views:
            return
        refreshed: dict[str, tuple[np.ndarray, str]] = {}
        for view_name in missing_views:
            frame = None
            for _attempt in range(3):
                frame = fetch_gazebo_topic_image(
                    self.OVERVIEW_TOPIC_MAP[view_name],
                    timeout_s=10.0,
                    expected_shape=self.live_overview_shape,
                )
                if frame is not None and frame.size > 0 and int(frame.sum()) > 0:
                    refreshed[view_name] = (np.asarray(frame, dtype=np.uint8), "gz_overview_topic")
                    break
        still_missing = [view_name for view_name in missing_views if view_name not in refreshed]
        if still_missing:
            for _attempt in range(3):
                probe_frames = capture_scene_probe_images(
                    view_names=tuple(view_name for view_name in still_missing if view_name not in refreshed),
                    expected_shape=self.live_overview_shape,
                    timeout_s=10.0,
                )
                for view_name in still_missing:
                    if view_name in refreshed:
                        continue
                    frame = probe_frames.get(view_name)
                    if frame is None or frame.size == 0 or int(frame.sum()) == 0:
                        continue
                    refreshed[view_name] = (np.asarray(frame, dtype=np.uint8), "scene_probe_camera")
                if all(view_name in refreshed for view_name in still_missing):
                    break
        for view_name, (frame, source) in refreshed.items():
            self._overview_sources[view_name][source] += 1
            self._ensure_writer(
                name=f"overview_{view_name}",
                path=self.output_dir / f"overview_{view_name}.mp4",
            ).append(
                frame,
                sim_time=self._last_sim_time,
            )


def _camera_frame(frame: np.ndarray | None, *, label: str, require_real: bool = False) -> np.ndarray:
    if frame is None:
        if require_real:
            raise RuntimeError(f"Real wrist camera frame required for {label!r}, but no frame was provided.")
        return _placeholder_frame(label=f"{label}: unavailable")
    image = np.asarray(frame, dtype=np.uint8)
    if image.size == 0 or int(image.sum()) == 0:
        if require_real:
            raise RuntimeError(f"Real wrist camera frame required for {label!r}, but the frame was blank.")
        return _placeholder_frame(label=f"{label}: blank")
    return image


def _placeholder_frame(*, label: str, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    image = Image.new("RGB", size, color=(24, 28, 36))
    draw = ImageDraw.Draw(image)
    draw.text((16, 16), label, fill=(230, 230, 230))
    draw.rectangle([(12, 12), (size[0] - 12, size[1] - 12)], outline=(120, 140, 180), width=2)
    return np.asarray(image, dtype=np.uint8)


def record_teacher_artifact_replay(
    *,
    env: Any,
    artifact: Any,
    recorder: HeadlessTrajectoryVideoRecorder,
    seed: int | None,
    trial_id: str | None,
) -> dict[str, Any]:
    observation, info = env.reset(seed=seed, options={"trial_id": trial_id} if trial_id else {})
    scenario = env._scenario
    state = env._state
    if scenario is None or state is None:
        raise RuntimeError("Environment did not expose scenario/state after reset for replay recording.")
    recorder.capture(observation=observation, scenario=scenario, state=state)
    final_info = dict(info)
    for segment in artifact.trajectory_segments:
        for point in segment.get("points", []):
            action = np.asarray(point["action"], dtype=np.float32)
            observation, _, terminated, truncated, final_info = env.step(action)
            state = env._state
            assert state is not None
            recorder.capture(observation=observation, scenario=scenario, state=state)
            if terminated or truncated:
                return dict(final_info)
    return dict(final_info)
