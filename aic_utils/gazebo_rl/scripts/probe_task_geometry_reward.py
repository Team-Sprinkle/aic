#!/usr/bin/env python3
"""Probe Gazebo online task-geometry reward with a privileged TCP controller."""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from gazebo_rl.gym_env import GazeboRLEnv
from gazebo_rl.task_geometry_reward import dense_task_geometry_reward

CAMERA_KEYS = {
    "center": "observation.images.center_camera",
    "left": "observation.images.left_camera",
    "right": "observation.images.right_camera",
}


class VideoRecorder:
    def __init__(self, output_dir: Path, fps: float):
        self.output_dir = output_dir
        self.fps = float(fps)
        self._writers: dict[str, Any] = {}
        self._paths: dict[str, Path] = {}

    def add_observation(self, obs: dict[str, Any]) -> None:
        images = obs.get("images") or {}
        for camera_name, key in CAMERA_KEYS.items():
            image = images.get(key)
            if isinstance(image, dict):
                frame = self._decode_image(image)
                if frame is not None:
                    self.add_frame(camera_name, frame)

    def add_frame(self, camera_name: str, rgb: np.ndarray) -> None:
        import cv2

        frame = np.asarray(rgb)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB frame for {camera_name}, got {frame.shape}")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        height, width = frame.shape[:2]
        writer = self._writers.get(camera_name)
        if writer is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"gazebo_{camera_name}_camera.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {path}")
            self._writers[camera_name] = writer
            self._paths[camera_name] = path
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self) -> dict[str, str]:
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()
        return {name: str(path) for name, path in sorted(self._paths.items())}

    @staticmethod
    def _decode_image(image: dict[str, Any]) -> np.ndarray | None:
        data_b64 = image.get("data_b64")
        if not data_b64:
            return None
        import cv2

        encoded = np.frombuffer(base64.b64decode(data_b64), dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    q = q / norm
    return -q if q[3] < 0.0 else q


def _quat_inv(q: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = _quat_normalize(q1)
    x2, y2, z2, w2 = _quat_normalize(q2)
    return _quat_normalize(
        np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=np.float64,
        )
    )


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    vq = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
    return _quat_mul(_quat_mul(q, vq), _quat_inv(q))[:3]


def _axis_angle_from_quat(q: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    xyz = q[:3]
    sin_half = float(np.linalg.norm(xyz))
    if sin_half <= 1.0e-12:
        return 2.0 * xyz
    angle = 2.0 * math.atan2(sin_half, float(np.clip(q[3], -1.0, 1.0)))
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return xyz / sin_half * angle


def _pose(oracle: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray] | None:
    pose = oracle.get(key)
    if not isinstance(pose, dict):
        return None
    pos = np.asarray(pose.get("position"), dtype=np.float64).reshape(-1)
    quat = np.asarray(pose.get("orientation_xyzw"), dtype=np.float64).reshape(-1)
    if pos.shape[0] < 3 or quat.shape[0] < 4:
        return None
    return pos[:3], _quat_normalize(quat[:4])


def _clip_by_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if max_norm <= 0.0 or norm <= 1.0e-12:
        return np.zeros_like(vec)
    return vec if norm <= max_norm else vec * (max_norm / norm)


def cheatcode_tcp_action(
    obs: dict[str, Any],
    *,
    max_translation_m: float,
    max_rotation_rad: float,
    z_offset_m: float,
    position_fraction: float,
    orientation_fraction: float,
) -> tuple[list[float], dict[str, float]]:
    oracle = obs.get("oracle") or {}
    tcp = _pose(oracle, "tcp_pose_base_link")
    plug = _pose(oracle, "plug_pose_base_link")
    port = _pose(oracle, "target_port_pose_base_link")
    if tcp is None or plug is None or port is None:
        raise RuntimeError("Ground-truth tcp/plug/port poses are required for the probe")

    tcp_pos, tcp_quat = tcp
    plug_pos, plug_quat = plug
    port_pos, port_quat = port

    q_diff = _quat_mul(port_quat, _quat_inv(plug_quat))
    target_tcp_quat = _quat_mul(q_diff, tcp_quat)
    target_tcp_pos = tcp_pos.copy()
    target_tcp_pos[0] = position_fraction * port_pos[0] + (1.0 - position_fraction) * tcp_pos[0]
    target_tcp_pos[1] = position_fraction * port_pos[1] + (1.0 - position_fraction) * tcp_pos[1]
    target_tcp_pos[2] = (
        position_fraction * (port_pos[2] + z_offset_m + (tcp_pos[2] - plug_pos[2]))
        + (1.0 - position_fraction) * tcp_pos[2]
    )

    delta_pos_tcp = _quat_apply(_quat_inv(tcp_quat), target_tcp_pos - tcp_pos)
    delta_rot_tcp = (
        orientation_fraction
        * _axis_angle_from_quat(_quat_mul(_quat_inv(tcp_quat), target_tcp_quat))
    )
    cmd_pos = _clip_by_norm(delta_pos_tcp, max_translation_m)
    cmd_rot = _clip_by_norm(delta_rot_tcp, max_rotation_rad)
    diagnostics = {
        "tcp_delta_pos_norm_m": float(np.linalg.norm(delta_pos_tcp)),
        "tcp_delta_rot_norm_rad": float(np.linalg.norm(delta_rot_tcp)),
        "cmd_tcp_pos_norm_m": float(np.linalg.norm(cmd_pos)),
        "cmd_tcp_rot_norm_rad": float(np.linalg.norm(cmd_rot)),
        "target_tcp_x_base": float(target_tcp_pos[0]),
        "target_tcp_y_base": float(target_tcp_pos[1]),
        "target_tcp_z_base": float(target_tcp_pos[2]),
        "plug_x_error_base": float(port_pos[0] - plug_pos[0]),
        "plug_y_error_base": float(port_pos[1] - plug_pos[1]),
        "plug_z_error_base": float(port_pos[2] - plug_pos[2]),
        "z_offset_m": float(z_offset_m),
    }
    return [*cmd_pos.tolist(), *cmd_rot.tolist()], diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-dir", type=Path, default=Path.cwd())
    parser.add_argument("--engine-config")
    parser.add_argument("--workspace-container", default="/home/chmin/yj/ws_aic/src/aic")
    parser.add_argument("--sim-docker-container", default="aic_eval")
    parser.add_argument("--docker-host", default="unix:///run/user/1000/docker.sock")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--max-translation", type=float, default=0.003)
    parser.add_argument("--max-rotation", type=float, default=0.03)
    parser.add_argument("--command-dt-sec", type=float, default=0.05)
    parser.add_argument("--per-trial-timeout-sec", type=float, default=300.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--record-cameras", action="store_true")
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--approach-z-offset", type=float, default=0.005)
    parser.add_argument("--insert-z-offset", type=float, default=-0.015)
    parser.add_argument("--descent-start-step", type=int, default=90)
    parser.add_argument("--position-fraction", type=float, default=1.0)
    parser.add_argument("--orientation-fraction", type=float, default=1.0)
    args = parser.parse_args()
    video_recorder = (
        VideoRecorder(args.video_dir or args.output.parent / "gazebo_camera_videos", args.video_fps)
        if args.record_cameras
        else None
    )

    env = GazeboRLEnv(
        workspace_dir=args.workspace_dir,
        engine_config=args.engine_config,
        sim_docker_container=args.sim_docker_container,
        docker_host=args.docker_host,
        workspace_container=args.workspace_container,
        ground_truth=True,
        gazebo_gui=False,
        launch_rviz=False,
        max_steps=args.steps,
        per_trial_timeout_sec=args.per_trial_timeout_sec,
        host=args.host,
        command_dt_sec=args.command_dt_sec,
        results_dir=args.results_dir,
        include_images=args.record_cameras,
    )
    rows: list[dict[str, Any]] = []
    video_paths: dict[str, str] = {}
    try:
        obs, info = env.reset()
        if video_recorder is not None:
            video_recorder.add_observation(obs)
        prev_obs = None
        for step in range(1, args.steps + 1):
            if step < args.descent_start_step:
                z_offset = args.approach_z_offset
            else:
                denom = max(1, args.steps - args.descent_start_step)
                phase = min(1.0, (step - args.descent_start_step) / denom)
                fraction = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
                z_offset = args.approach_z_offset + fraction * (
                    args.insert_z_offset - args.approach_z_offset
                )
            action, diagnostics = cheatcode_tcp_action(
                obs,
                max_translation_m=args.max_translation,
                max_rotation_rad=args.max_rotation,
                z_offset_m=z_offset,
                position_fraction=args.position_fraction,
                orientation_fraction=args.orientation_fraction,
            )
            obs, reward, terminated, truncated, step_info = env.step(action)
            if video_recorder is not None:
                video_recorder.add_observation(obs)
            probe_reward, probe_info = dense_task_geometry_reward(prev_obs=prev_obs, obs=obs)
            row = {
                "step": step,
                "reward": reward,
                "probe_reward": probe_reward,
                "terminated": terminated,
                "truncated": truncated,
                **diagnostics,
                **probe_info,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            prev_obs = obs
            if terminated or truncated:
                break
    finally:
        if video_recorder is not None:
            video_paths = video_recorder.close()
        env.close()

    summary = {
        "rows": rows,
        "best": max(rows, key=lambda row: float(row.get("reward", -1.0e9))) if rows else None,
        "max_abs_live_probe_diff": (
            max(abs(float(row["reward"]) - float(row["probe_reward"])) for row in rows) if rows else None
        ),
        "video_paths": video_paths,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("SUMMARY", json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
