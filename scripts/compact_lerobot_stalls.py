#!/usr/bin/env python3
"""Compact low-motion intervals in a recorded LeRobot AIC dataset.

This is for post-replay nominalrecovery data: it removes long controller holds
from the parquet frame table and recomputes delta-pose actions from consecutive
TCP states so a policy can learn the intended smooth motion.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import cv2
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dataset", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--stall-translation-m", type=float, default=0.00005)
    parser.add_argument("--stall-action-norm", type=float, default=0.00005)
    parser.add_argument("--min-stall-frames", type=int, default=5)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--trim-videos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if q[3] < 0.0:
        q = -q
    return q / norm


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.asarray([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.asarray(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _quat_delta_rotvec(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    delta = _quat_normalize(_quat_multiply(_quat_normalize(q1), _quat_conjugate(_quat_normalize(q0))))
    vector = delta[:3]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(delta[3]))
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return vector / vector_norm * angle


def _state_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _action_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _keep_mask(df: pd.DataFrame, *, stall_translation_m: float, stall_action_norm: float, min_stall_frames: int) -> np.ndarray:
    states = np.stack([_state_array(value) for value in df["observation.state"]])
    actions = np.stack([_action_array(value) for value in df["action"]])
    positions = states[:, :3]
    motion = np.zeros(len(df), dtype=np.float64)
    motion[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    action_norm = np.linalg.norm(actions[:, :3], axis=1)
    low_motion = (motion < stall_translation_m) & (action_norm < stall_action_norm)

    keep = np.ones(len(df), dtype=bool)
    start = None
    for index, is_low in enumerate(low_motion):
        if is_low and start is None:
            start = index
        if (not is_low or index == len(low_motion) - 1) and start is not None:
            end = index if not is_low else index + 1
            if end - start >= min_stall_frames:
                keep[start + 1 : end] = False
            start = None
    keep[0] = True
    keep[-1] = True
    return keep


def _recompute_actions(df: pd.DataFrame, *, fps: float) -> pd.DataFrame:
    rows = df.copy().reset_index(drop=True)
    states = np.stack([_state_array(value) for value in rows["observation.state"]])
    actions: list[np.ndarray] = []
    for index in range(len(rows)):
        if index >= len(rows) - 1:
            actions.append(np.zeros(6, dtype=np.float32))
            continue
        current = states[index]
        nxt = states[index + 1]
        delta_position = nxt[:3] - current[:3]
        delta_rotation = _quat_delta_rotvec(current[3:7], nxt[3:7])
        actions.append(np.concatenate([delta_position, delta_rotation]).astype(np.float32))
    rows["action"] = actions
    rows["frame_index"] = np.arange(len(rows), dtype=np.int64)
    rows["index"] = np.arange(len(rows), dtype=np.int64)
    rows["timestamp"] = (np.arange(len(rows), dtype=np.float32) / np.float32(fps)).astype(np.float32)
    return rows


def _copy_sidecars(input_root: Path, output_root: Path, *, copy_videos: bool) -> None:
    for child in input_root.iterdir():
        if child.name == "data":
            continue
        if child.name == "videos" and not copy_videos:
            continue
        destination = output_root / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, destination)


def _trim_video_file(input_path: Path, output_path: Path, keep_indices: np.ndarray, *, fps: float) -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python is required to trim videos")
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"unable to open video {input_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    keep_set = set(int(index) for index in keep_indices.tolist())
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index in keep_set:
            writer.write(frame)
        frame_index += 1
    cap.release()
    writer.release()


def _trim_videos(input_root: Path, output_root: Path, keep_indices: np.ndarray, *, fps: float) -> int:
    count = 0
    for input_path in sorted((input_root / "videos").glob("**/*.mp4")):
        relative = input_path.relative_to(input_root)
        _trim_video_file(input_path, output_root / relative, keep_indices, fps=fps)
        count += 1
    return count


def _update_episode_metadata(output_root: Path, *, frame_count: int, fps: float) -> None:
    episode_files = sorted((output_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        return
    episodes = pd.read_parquet(episode_files[0])
    duration = frame_count / fps
    if len(episodes) > 0:
        episodes.loc[0, "length"] = int(frame_count)
        episodes.loc[0, "dataset_from_index"] = 0
        episodes.loc[0, "dataset_to_index"] = int(frame_count)
        for column in episodes.columns:
            if column.endswith("/to_timestamp"):
                episodes.loc[0, column] = float(duration)
    episodes.to_parquet(episode_files[0], index=False)


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_dataset)
    output_root = Path(args.output_dataset)
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{output_root} exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    _copy_sidecars(input_root, output_root, copy_videos=not args.trim_videos)

    input_files = sorted((input_root / "data").glob("chunk-*/*.parquet"))
    if len(input_files) != 1:
        raise SystemExit(f"expected exactly one data parquet for now, found {len(input_files)}")
    df = pd.read_parquet(input_files[0])
    keep = _keep_mask(
        df,
        stall_translation_m=args.stall_translation_m,
        stall_action_norm=args.stall_action_norm,
        min_stall_frames=args.min_stall_frames,
    )
    compacted = _recompute_actions(df.loc[keep].copy(), fps=args.fps)
    output_file = output_root / "data" / "chunk-000" / "file-000.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    compacted.to_parquet(output_file, index=False)
    trimmed_videos = 0
    if args.trim_videos and (input_root / "videos").exists():
        trimmed_videos = _trim_videos(input_root, output_root, np.flatnonzero(keep), fps=args.fps)
    _update_episode_metadata(output_root, frame_count=len(compacted), fps=args.fps)

    info_path = output_root / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["total_frames"] = int(len(compacted))
        info["fps"] = int(args.fps) if float(args.fps).is_integer() else float(args.fps)
        info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")

    report = {
        "schema_version": "aic_lerobot_stall_compaction/v1",
        "input_dataset": str(input_root),
        "output_dataset": str(output_root),
        "input_frames": int(len(df)),
        "output_frames": int(len(compacted)),
        "removed_frames": int(len(df) - len(compacted)),
        "trimmed_videos": trimmed_videos,
        "stall_translation_m": args.stall_translation_m,
        "stall_action_norm": args.stall_action_norm,
        "min_stall_frames": args.min_stall_frames,
        "actions": "recomputed_from_consecutive_tcp_state_pose_delta",
        "video_note": (
            "videos were trimmed to the kept frame indices"
            if args.trim_videos
            else "video sidecars copied unchanged; use state/action only or trim videos separately"
        ),
    }
    (output_root / "stall_compaction_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
