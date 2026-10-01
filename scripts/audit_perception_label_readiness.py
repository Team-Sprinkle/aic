#!/usr/bin/env python3
"""Audit native SC projection labels without treating projections as visibility."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenes = defaultdict(set)
    episodes = defaultdict(set)
    counts = Counter()
    per_camera = defaultdict(Counter)
    missing_images = []
    actual_visibility_rows = 0
    for line in args.labels.open():
        row = json.loads(line)
        split = row["split"]
        port = int(row["task_vector"][3] > row["task_vector"][2])
        phase = "near" if abs(row["observed_sc_tip_pose_opening_frame"][2]) < 0.03 else "far"
        scenes[split].add(row["scene_sha256"])
        episodes[split].add((row["scene_sha256"], row["episode_id"]))
        counts[(split, port, phase)] += 1
        for cam, values in row["projected_training_pixels"].items():
            per_camera[cam][(split, port, phase, "in_frame")] += int(bool(values["opening"][3]))
            per_camera[cam][(split, port, phase, "frames")] += 1
            if not Path(row["images"][cam]).is_file():
                missing_images.append(row["images"][cam])
        actual_visibility_rows += int("actual_visibility" in row or "identifiable_visibility" in row)
    output = {
        "schema": "aic_perception_label_readiness/v1",
        "source": str(args.labels),
        "task_family": "sc_to_sc_only",
        "rows": sum(counts.values()),
        "scene_counts": {k: len(v) for k, v in scenes.items()},
        "episode_counts": {k: len(v) for k, v in episodes.items()},
        "scene_overlap": sorted(scenes["train"] & scenes["validation"]),
        "by_split_port_phase": [
            {"split": split, "port": port, "phase": phase, "frames": count}
            for (split, port, phase), count in sorted(counts.items())
        ],
        "per_camera": {
            cam: [
                {"split": split, "port": port, "phase": phase, "metric": metric, "count": count}
                for (split, port, phase, metric), count in sorted(counter.items())
            ]
            for cam, counter in per_camera.items()
        },
        "missing_native_image_count": len(missing_images),
        "missing_native_image_examples": missing_images[:10],
        "true_identifiable_visibility_label_rows": actual_visibility_rows,
        "occlusion_tested": False,
        "wrong_port_negative_labels": False,
        "sfp_to_nic_rows": 0,
        "gate": "Not ready for the two-family calibrated visibility handoff",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in (
        "rows", "scene_counts", "episode_counts", "missing_native_image_count", "gate",
    )}))


if __name__ == "__main__":
    main()
