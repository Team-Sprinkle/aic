#!/usr/bin/env python3
"""Audit native SC camera capture and scored provenance for an eval batch."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    scoring_path = args.batch / "eval_collection/collection_config/attempt_0001/scoring.yaml"
    scores = yaml.safe_load(scoring_path.read_text()) if scoring_path.exists() else {}
    episodes = []
    for ep in sorted((args.batch / "episodes").glob("episode_*")):
        meta_path = ep / "episode.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        frames = [json.loads(line) for line in (ep / "frames.jsonl").open()]
        native = [row for row in frames if row.get("native_images")]
        shape_counts = Counter()
        missing = []
        for row in native:
            for camera in ("center", "left", "right"):
                path = ep / row["native_images"][camera]
                if not path.exists():
                    missing.append(str(path))
                    continue
                with Image.open(path) as im:
                    shape_counts[f"{camera}:{im.width}x{im.height}:{im.mode}"] += 1
        first = native[0] if native else None
        episodes.append({
            "directory": str(ep), "metadata_sha256": sha256(meta_path),
            "task": meta.get("task"), "frames": len(frames),
            "native_frames": len(native), "native_frame_indices": [r["frame"] for r in native],
            "native_shapes": dict(shape_counts), "missing_native_paths": missing,
            "first_native_triplet": {camera: str(ep / first["native_images"][camera])
                                     for camera in ("center", "left", "right")} if first else None,
        })
    result = {"batch": str(args.batch), "engine_config_sha256": sha256(args.batch / "engine_config.yaml"),
              "scoring_sha256": sha256(scoring_path) if scoring_path.exists() else None,
              "trial_tier_3_scores": {key: value.get("tier_3", {}).get("score")
                                     for key, value in scores.items() if key != "total"},
              "episodes": episodes,
              "limitations": "Privileged teacher collection; native images are not autonomous evidence. "
                             "Episode-to-trial mapping needs scored rollout provenance before training."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
