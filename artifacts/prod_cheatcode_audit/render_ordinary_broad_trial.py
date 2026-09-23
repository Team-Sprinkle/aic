#!/usr/bin/env python3
"""Render one-second wrist-camera triptychs from the broad audit capture."""

import bisect
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

from transcode_compat_mp4 import transcode


CAMERAS = ("left", "center", "right")


def main(audit, bag, output):
    metadata = yaml.safe_load((bag / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    start = metadata["starting_time"]["nanoseconds_since_epoch"]
    duration = metadata["duration"]["nanoseconds"]
    end = start + duration
    raw = {camera: [] for camera in CAMERAS}
    for path in (audit / "frames").glob("*.jpg"):
        try:
            stamp, _, camera = path.stem.split("_", 2)
            ns = int(stamp)
        except (ValueError, TypeError):
            continue
        if start <= ns <= end and camera in raw:
            raw[camera].append((ns, path))
    for camera in CAMERAS:
        raw[camera].sort()
    if any(not raw[camera] for camera in CAMERAS):
        raise RuntimeError(f"Missing camera capture for {bag.name}")

    output.mkdir(parents=True, exist_ok=True)
    panels = []
    seconds = range(0, max(1, round(duration / 1e9)) + 1)
    for second in seconds:
        target = start + int(second * 1e9)
        camera_panels = []
        for camera in CAMERAS:
            items = raw[camera]
            times = [item[0] for item in items]
            pos = bisect.bisect_left(times, target)
            candidates = items[max(0, pos - 1):min(len(items), pos + 1)]
            _, path = min(candidates, key=lambda item: abs(item[0] - target))
            frame = cv2.imread(str(path))
            frame = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
            cv2.putText(frame, f"{bag.name.split('_')[2]} {camera} {second}s", (7, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 1, cv2.LINE_AA)
            camera_panels.append(frame)
        panels.append(cv2.hconcat(camera_panels))

    temporary_video = output / "all_cameras_mp4v_temp.mp4"
    writer = cv2.VideoWriter(str(temporary_video),
                             cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (1440, 270))
    for panel in panels:
        writer.write(panel)
    writer.release()
    try:
        transcode(temporary_video, output / "all_cameras_1hz.mp4")
    finally:
        temporary_video.unlink(missing_ok=True)
    indices = sorted({0, len(panels) // 4, len(panels) // 2,
                      3 * len(panels) // 4, len(panels) - 1})
    sheet = cv2.vconcat([panels[i] for i in indices])
    cv2.imwrite(str(output / "five_timepoints.jpg"), sheet)
    print(output, len(panels), {camera: len(raw[camera]) for camera in CAMERAS})


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
