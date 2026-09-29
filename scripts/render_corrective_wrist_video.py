#!/usr/bin/env python3
"""Render synchronized, simulator-time wrist RGB from a corrective episode."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


FPS = 20
CAMERAS = ("left", "center", "right")
PANEL = (432, 384)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.episode / "frames.jsonl").open()]
    rows = [row for row in rows if all((args.episode / row["images"][camera]).is_file()
                                   for camera in CAMERAS)]
    if len(rows) < 2:
        raise RuntimeError("Need at least two complete wrist-image rows")
    rows.sort(key=lambda row: row["sim_time"])
    times = [float(row["sim_time"]) for row in rows]
    first, last = times[0], times[-1]
    count = int((last - first) * FPS) + 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size",
        f"{PANEL[0] * len(CAMERAS)}x{PANEL[1]}", "-framerate", str(FPS),
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(args.output),
    ], stdin=subprocess.PIPE)
    cached = {}
    max_offset = 0.0
    try:
        assert process.stdin is not None
        for frame in range(count):
            now = first + frame / FPS
            index = bisect.bisect_left(times, now)
            index = min((max(0, index - 1), min(index, len(rows) - 1)),
                        key=lambda candidate: abs(times[candidate] - now))
            max_offset = max(max_offset, abs(times[index] - now))
            panels = []
            for camera in CAMERAS:
                path = args.episode / rows[index]["images"][camera]
                if path != cached.get(camera, (None, None))[0]:
                    image = cv2.imread(str(path))
                    if image is None:
                        raise RuntimeError(f"Could not read {path}")
                    cached[camera] = (path, cv2.resize(image, PANEL,
                                                       interpolation=cv2.INTER_LINEAR))
                panel = cached[camera][1].copy()
                cv2.putText(panel, f"{camera} {now-first:.2f}s", (12, 27),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2, cv2.LINE_AA)
                panels.append(panel)
            process.stdin.write(np.concatenate(panels, axis=1).tobytes())
    finally:
        if process.stdin is not None:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    summary = {"schema": "corrective_wrist_sim_time_video/v1",
               "episode": str(args.episode), "output": str(args.output),
               "source_rows": len(rows), "fps": FPS, "frames": count,
               "duration_s": count / FPS, "max_source_time_offset_s": max_offset,
               "note": "Three recorded wrist cameras; no fixed full-cable view."}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
