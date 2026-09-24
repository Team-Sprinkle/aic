#!/usr/bin/env python3
"""Render synchronized native-rate wrist and fixed wide Gazebo audit videos."""

from __future__ import annotations

import bisect
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys

import cv2
import numpy as np
import yaml


CAMERAS = ("left", "center", "right", "overhead", "side")
OUTPUTS = {
    "wrist_triptych_20fps.mp4": (("left", "center", "right"), (480, 270)),
    "overhead_20fps.mp4": (("overhead",), (1280, 720)),
    "side_20fps.mp4": (("side",), (1280, 720)),
}
FPS = 20


def nearest(rows, stamps, sim_ns):
    index = bisect.bisect_left(stamps, sim_ns)
    candidates = rows[max(0, index - 1):min(len(rows), index + 1)]
    return min(candidates, key=lambda row: abs(row[0] - sim_ns))


def open_encoder(path: Path, size: tuple[int, int]):
    return subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size",
        f"{size[0]}x{size[1]}", "-framerate", str(FPS), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-profile:v", "baseline", "-level:v", "3.1", "-pix_fmt", "yuv420p",
        "-threads", "2", "-movflags", "+faststart", str(path),
    ], stdin=subprocess.PIPE)


def main(audit: Path, bag: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    metadata = yaml.safe_load((bag / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    start_wall = metadata["starting_time"]["nanoseconds_since_epoch"]
    end_wall = start_wall + metadata["duration"]["nanoseconds"]
    rows = {camera: [] for camera in CAMERAS}
    with (audit / "frame_manifest.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            wall_ns = int(row["wall_ns"])
            if start_wall <= wall_ns <= end_wall and row["camera"] in rows:
                rows[row["camera"]].append((int(row["sim_ns"]), audit / "frames" / row["file"]))
    for camera in CAMERAS:
        rows[camera].sort()
        if not rows[camera]:
            raise RuntimeError(f"No in-trial frames for {camera}")
    first = max(camera_rows[0][0] for camera_rows in rows.values())
    last = min(camera_rows[-1][0] for camera_rows in rows.values())
    if last - first < 10e9:
        raise RuntimeError(f"Too little common camera coverage: {(last-first)/1e9:.1f} s")
    frames = int((last - first) / (1e9 / FPS)) + 1
    stamps = {camera: [row[0] for row in camera_rows] for camera, camera_rows in rows.items()}
    coverage = {}
    for camera in CAMERAS:
        diffs = np.diff(np.asarray(stamps[camera], dtype=np.int64)) / 1e9
        coverage[camera] = {
            "captured_frames": len(rows[camera]),
            "median_interval_s": float(statistics.median(diffs)) if len(diffs) else None,
            "p95_interval_s": float(np.percentile(diffs, 95)) if len(diffs) else None,
            "max_interval_s": float(max(diffs)) if len(diffs) else None,
        }
    largest_source_gap = {camera: 0.0 for camera in CAMERAS}
    selected_panels = {}
    for filename, (cameras, (width, height)) in OUTPUTS.items():
        width_out = width * len(cameras)
        process = open_encoder(destination / filename, (width_out, height))
        cache = {}
        try:
            assert process.stdin is not None
            for index in range(frames):
                sim_ns = first + round(index * 1e9 / FPS)
                panels = []
                for camera in cameras:
                    nearest_ns, path = nearest(rows[camera], stamps[camera], sim_ns)
                    lag = abs(nearest_ns - sim_ns) / 1e9
                    largest_source_gap[camera] = max(largest_source_gap[camera], lag)
                    if path != cache.get(camera, (None, None))[0]:
                        image = cv2.imread(str(path))
                        if image is None:
                            raise RuntimeError(f"Cannot read {path}")
                        if image.shape[:2] != (height, width):
                            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                        cache[camera] = (path, image)
                    panel = cache[camera][1].copy()
                    cv2.putText(panel, f"{camera}  {(sim_ns-first)/1e9:.2f}s",
                                (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    panels.append(panel)
                combined = cv2.hconcat(panels)
                process.stdin.write(combined.tobytes())
                if filename != "wrist_triptych_20fps.mp4" and index in {
                    0, frames // 4, frames // 2, 3 * frames // 4, frames - 1
                }:
                    selected_panels.setdefault(filename, []).append(combined)
        finally:
            if process.stdin is not None:
                process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"ffmpeg failed for {filename}")
    for filename, panels in selected_panels.items():
        cv2.imwrite(str(destination / filename.replace(".mp4", "_contact_sheet.jpg")),
                    cv2.vconcat([cv2.resize(panel, (640, 360)) for panel in panels]))
    combined = destination / "all_views_20fps.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(destination / "wrist_triptych_20fps.mp4"),
        "-i", str(destination / "overhead_20fps.mp4"),
        "-i", str(destination / "side_20fps.mp4"),
        "-filter_complex",
        "[1:v]scale=960:540[o];[2:v]scale=960:540[s];"
        "[o][s]hstack=inputs=2[top];[0:v]scale=1920:360[bottom];"
        "[top][bottom]vstack=inputs=2[out]",
        "-map", "[out]", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "23", "-profile:v", "baseline", "-level:v", "4.0",
        "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart",
        str(combined),
    ], check=True)
    summary = {
        "schema": "aic_smooth_wide_review/v1",
        "source_bag": str(bag), "source_manifest": str(audit / "frame_manifest.csv"),
        "bag_start_wall_ns": start_wall, "bag_end_wall_ns": end_wall,
        "first_common_sim_ns": first, "last_common_sim_ns": last,
        "video_fps": FPS, "video_frame_count": frames,
        "duration_s": frames / FPS, "source_coverage": coverage,
        "max_nearest_source_offset_s": largest_source_gap,
        "outputs": [*OUTPUTS, combined.name],
        "note": "The wide views are fixed visual-only Gazebo cameras, not policy observations. Each encoded frame uses the nearest captured camera image in simulation time.",
    }
    (destination / "render_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
