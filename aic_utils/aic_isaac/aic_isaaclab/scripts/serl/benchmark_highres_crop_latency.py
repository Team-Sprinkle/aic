#!/usr/bin/env python3
"""Benchmark the saved crop representation with measured live world-trunk timings."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "crop_probe", HERE / "train_highres_crop_pose_probe.py"
)
crop_probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(crop_probe)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-replay", type=Path, required=True)
    parser.add_argument("--timing-replay", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    manifest = json.loads(args.scene_manifest.read_text())
    episode_ids = {row["episode_id"] for row in manifest["development"]}
    image_rows, image_audit = crop_probe.world_probe.load([args.image_replay], episode_ids)
    image_rows, rejected = crop_probe.attach_highres(image_rows, args.image_replay)
    timing_rows, timing_audit = crop_probe.world_probe.load([args.timing_replay], episode_ids)
    timing_rows = [row for row in timing_rows if row.get("model_inference_s") is not None]
    if not timing_rows:
        raise RuntimeError("Timing replay contains no model_inference_s values")
    image_rows = image_rows[: len(timing_rows)]
    if len(image_rows) != len(timing_rows):
        raise RuntimeError("Not enough image rows for the measured timing rows")

    episode_matches = [a["episode_id"] == b["episode_id"] for a, b in zip(image_rows, timing_rows)]
    translation_deltas = [
        float(torch.linalg.vector_norm(a["translation_mm"] - b["translation_mm"]))
        for a, b in zip(image_rows, timing_rows)
    ]
    if not all(episode_matches) or max(translation_deltas) > 0.05:
        raise RuntimeError(
            "Image and timing trajectories do not align causally: "
            f"episode_matches={sum(episode_matches)}/{len(episode_matches)}, "
            f"max_translation_delta_mm={max(translation_deltas):.6f}"
        )
    for image_row, timing_row in zip(image_rows, timing_rows):
        image_row["model_inference_s"] = timing_row["model_inference_s"]

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    locator = crop_probe.Locator()
    locator.load_state_dict(checkpoint["locator"])
    pose = crop_probe.CropPoseProbe()
    pose.load_state_dict(checkpoint["pose_members"][0])
    latency = crop_probe.benchmark_live_representation(
        locator, pose, image_rows, device, args.crop_size
    )
    latency["provenance"] = {
        "image_replay": crop_probe.world_probe.file_id(args.image_replay),
        "timing_replay": crop_probe.world_probe.file_id(args.timing_replay),
        "checkpoint": crop_probe.world_probe.file_id(args.checkpoint),
        "causal_alignment": {
            "episode_matches": sum(episode_matches),
            "row_count": len(episode_matches),
            "max_translation_delta_mm": max(translation_deltas),
        },
        "image_replay_audit": image_audit,
        "timing_replay_audit": timing_audit,
        "highres_rejected": rejected,
    }
    args.output.write_text(json.dumps(latency, indent=2) + "\n")
    metrics = json.loads(args.metrics.read_text())
    metrics["live_latency"] = latency
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(latency, indent=2))


if __name__ == "__main__":
    main()
