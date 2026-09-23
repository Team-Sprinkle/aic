#!/usr/bin/env python3
"""Review a saved agent/VLM replay at force-relevant times across three cameras."""

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq
import yaml


def frame_at(video, index):
    reader = cv2.VideoCapture(str(video))
    reader.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = reader.read()
    reader.release()
    if not ok:
        raise RuntimeError(f"Could not read {video} frame {index}")
    return cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)


def main(attempt, output):
    parquet = attempt / "dataset/data/chunk-000/file-000.parquet"
    table = pq.read_table(parquet, columns=["observation.state", "timestamp"])
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    times = np.asarray(table["timestamp"].to_pylist(), dtype=float)
    force = np.linalg.norm(state[:, 26:29], axis=1)
    peak = int(force.argmax())
    indices = sorted({0, len(state) // 4, peak, 3 * len(state) // 4, len(state) - 1})
    cameras = ("left", "center", "right")
    panels = []
    for index in indices:
        row = []
        for camera in cameras:
            video = attempt / f"dataset/videos/observation.images.{camera}_camera/chunk-000/file-000.mp4"
            frame = frame_at(video, index)
            label = f"{camera} {times[index]:.1f}s force {force[index]:.1f}N"
            cv2.putText(frame, label, (7, 23), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (0, 255, 0), 1, cv2.LINE_AA)
            row.append(frame)
        panels.append(cv2.hconcat(row))
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "force_review.jpg"), cv2.vconcat(panels))
    writer = cv2.VideoWriter(str(output / "all_cameras_1hz.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (1440, 270))
    readers = {
        camera: cv2.VideoCapture(str(
            attempt / f"dataset/videos/observation.images.{camera}_camera/chunk-000/file-000.mp4"
        )) for camera in cameras
    }
    for index in range(0, len(state), 20):
        row = []
        for camera in cameras:
            reader = readers[camera]
            reader.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = reader.read()
            if not ok:
                raise RuntimeError(f"Could not read {camera} frame {index}")
            frame = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
            cv2.putText(frame, f"{camera} {times[index]:.1f}s {force[index]:.1f}N",
                        (7, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 1, cv2.LINE_AA)
            row.append(frame)
        writer.write(cv2.hconcat(row))
    for reader in readers.values():
        reader.release()
    writer.release()
    scoring = next((attempt / "results").glob("*/scoring.yaml"))
    scores = yaml.safe_load(scoring.read_text())
    record = {
        "attempt": str(attempt), "score": scores.get("total"),
        "frames": len(state), "peak_force_n": float(force[peak]),
        "peak_force_time_s": float(times[peak]),
        "review_indices": indices,
        "review_times_s": [float(times[i]) for i in indices],
        "measured_tcp_at_peak_m": state[peak, :3].tolist(),
        "measured_tcp_at_end_m": state[-1, :3].tolist(),
        "contact_message": next(
            (value["tier_2"]["categories"]["contacts"]["message"]
             for key, value in scores.items() if key != "total"), None),
    }
    (output / "review.json").write_text(json.dumps(record, indent=2) + "\n")
    print(output, record["score"], round(record["peak_force_n"], 2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
