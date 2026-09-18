#!/usr/bin/env python3
"""Render AIC_POLICY_RECORD_DIR snapshots as a timestamped three-camera MP4."""
from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path

import av
import cv2
import numpy as np


def render(log: Path, output: Path, *, fps: int = 10, title: str = "Policy rollout"):
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    if not rows or fps <= 0:
        raise ValueError("Need at least one recorded frame and a positive FPS")
    stamps = [float(row["sim_time"]) for row in rows]
    if not all(math.isfinite(t) for t in stamps) or any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("Render one trial at a time with strictly increasing simulation timestamps")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def canvas(index):
        row = rows[index]
        panels = []
        for camera in ("left", "center", "right"):
            panel = np.zeros((272, 400, 3), np.uint8)
            filename = row["images"].get(camera)
            if filename:
                image = cv2.imread(str(log.parent / filename))
                if image is None:
                    raise ValueError(f"Cannot read {filename}")
                scale = min(400 / image.shape[1], 240 / image.shape[0])
                width, height = round(image.shape[1] * scale), round(image.shape[0] * scale)
                x, y = (400 - width) // 2, 32 + (240 - height) // 2
                panel[y:y + height, x:x + width] = cv2.resize(image, (width, height))
            cv2.putText(panel, camera, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (240, 240, 240), 1)
            panels.append(panel)
        frame = np.vstack((np.zeros((60, 1200, 3), np.uint8), np.hstack(panels)))
        cv2.putText(frame, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
        label = f"Snapshot at sim {stamps[index]:.3f}s | elapsed {stamps[index] - stamps[0]:.3f}s | held until next capture"
        cv2.putText(frame, label, (12, 47), cv2.FONT_HERSHEY_SIMPLEX, .5, (200, 210, 220), 1)
        return frame

    # Hold each still until the next timestamp, then hold the last for one second.
    # This preserves elapsed simulation time without inventing intermediate motion.
    count = math.ceil((stamps[-1] - stamps[0] + 1.0) * fps)
    with av.open(str(output), "w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 1200, 332, "yuv420p"
        stream.options = {"crf": "20", "preset": "fast", "threads": "2"}
        previous, pixels = None, None
        for tick in range(count):
            index = min(len(rows) - 1, bisect.bisect_right(stamps, stamps[0] + tick / fps) - 1)
            if index != previous:
                pixels, previous = canvas(index), index
            frame = av.VideoFrame.from_ndarray(pixels, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    sheet = output.with_suffix(".jpg")
    if not cv2.imwrite(str(sheet), np.vstack([canvas(i) for i in (0, len(rows) // 2, len(rows) - 1)])):
        raise RuntimeError(f"Cannot write {sheet}")
    metadata = {"source": str(log.resolve()), "snapshots": len(rows), "encoded_frames": count,
                "fps": fps, "sim_start": stamps[0], "sim_end": stamps[-1],
                "timing": "Recorded stills held to next simulation timestamp; final still held one second"}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames_jsonl", type=Path)
    parser.add_argument("output_mp4", type=Path)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--title", default="Policy rollout")
    args = parser.parse_args()
    render(args.frames_jsonl, args.output_mp4, fps=args.fps, title=args.title)


if __name__ == "__main__":
    main()
