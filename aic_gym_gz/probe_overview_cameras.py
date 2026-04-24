"""Probe fixed overview camera topics and optionally save snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from aic_gym_gz.io import RosCameraSubscriber, summarize_image_batch
from aic_gym_gz.utils import to_jsonable


TOPIC_MAP = {
    "top_down_xy": "/overview_camera/image",
    "front_xz": "/overview_front_camera/image",
    "side_yz": "/overview_side_camera/image",
    "oblique_xy": "/overview_oblique_camera/image",
}


def _image_summary(images: dict[str, np.ndarray]) -> dict[str, Any]:
    timestamps = {name: 0.0 for name in images}
    return summarize_image_batch(images, timestamps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--image-shape", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=(512, 512))
    parser.add_argument("--output", default="aic_gym_gz/artifacts/context_audit/overview_camera_probe.json")
    parser.add_argument("--snapshot-dir", default=None)
    parser.add_argument("--start-bridge", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    image_shape = (int(args.image_shape[0]), int(args.image_shape[1]), 3)
    subscriber = RosCameraSubscriber(
        image_shape=image_shape,
        topic_map=TOPIC_MAP,
        node_name="aic_gym_gz_overview_probe",
    )
    subscriber.start()
    try:
        ready = subscriber.wait_until_ready(timeout_s=float(args.timeout))
        images, timestamps, camera_info = subscriber.latest_images()
        payload = {
            "ready": bool(ready),
            "topic_map": dict(TOPIC_MAP),
            "timeout_s": float(args.timeout),
            "image_shape": list(image_shape),
            "timestamps": timestamps,
            "image_summary": _image_summary(images),
            "camera_info_available": {
                name: bool(info)
                for name, info in camera_info.items()
            },
        }
        if args.snapshot_dir:
            snapshot_dir = Path(args.snapshot_dir)
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snapshots: dict[str, str] = {}
            for name, image in images.items():
                frame = np.asarray(image, dtype=np.uint8)
                if frame.size == 0 or int(frame.sum()) == 0:
                    continue
                path = snapshot_dir / f"{name}.png"
                Image.fromarray(frame).save(path)
                snapshots[name] = str(path)
            payload["snapshots"] = snapshots
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(to_jsonable(payload), indent=2 if args.pretty else None, sort_keys=True)
        output_path.write_text(rendered + ("\n" if not rendered.endswith("\n") else ""), encoding="utf-8")
        print(rendered)
    finally:
        subscriber.close()


if __name__ == "__main__":
    main()
