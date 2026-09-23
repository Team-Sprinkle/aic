#!/usr/bin/env python3
"""Benchmark the deployable coarse/refined multiview pose representation."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tri = load_module("triangulated_probe", HERE / "train_triangulated_pose_probe.py")
crop = tri.crop
world = tri.world


def quantiles(values):
    return {"p50": float(np.quantile(values, .50)), "p95": float(np.quantile(values, .95)),
            "p99": float(np.quantile(values, .99))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-replay", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--device", default="cuda")
    a = parser.parse_args(); device = torch.device(a.device)
    manifest = json.loads(a.scene_manifest.read_text())
    ids = {row["episode_id"] for row in manifest["development"]}
    rows, audit = world.load([a.evaluation_replay], ids)
    rows, rejected = crop.attach_highres(rows, a.evaluation_replay)
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    locator = crop.Locator().to(device).eval(); locator.load_state_dict(checkpoint["locator"])
    refiner = tri.KeypointRefiner().to(device).eval(); refiner.load_state_dict(checkpoint["refiner"])
    images = []
    for row in rows:
        images.append(torch.stack([torch.from_numpy(np.asarray(Image.open(item["path"]).convert("RGB")).copy()).permute(2, 0, 1)
                                   for item in row["highres"]]))

    def run(full_cpu):
        full = full_cpu.to(device).float() / 255.0
        coarse = locator(F.interpolate(full, size=(256, 288), mode="bilinear", align_corners=False))
        size = a.crop_size; height, width = full.shape[-2:]
        centers = coarse.reshape(3, 2, 2).mean(1)
        gx = torch.linspace(-(size - 1) / (width - 1), (size - 1) / (width - 1), size, device=device)
        gy = torch.linspace(-(size - 1) / (height - 1), (size - 1) / (height - 1), size, device=device)
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")
        grids = torch.stack([torch.stack(((center[0] * 2 - 1) + xx,
                                         (center[1] * 2 - 1) + yy), -1) for center in centers])
        refined = refiner(F.grid_sample(full, grids, mode="bilinear", padding_mode="zeros", align_corners=True))
        return coarse, refined

    with torch.inference_mode():
        for _ in range(10): run(images[0])
        elapsed = []
        for image in images:
            torch.cuda.synchronize(); start = time.perf_counter(); run(image); torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - start) * 1000)
    trunk = np.asarray([float(row["model_inference_s"]) * 1000 for row in rows])
    representation = np.asarray(elapsed); complete = representation + trunk
    report = {
        "sample_count": len(rows),
        "scope": "three-view coarse locator, three native 160px crops, shared refiner, plus measured frozen world trunk; sensor acquisition excluded",
        "representation_ms": quantiles(representation),
        "world_trunk_ms": quantiles(trunk),
        "complete_inference_ms": quantiles(complete),
        "p95_below_300ms": bool(np.quantile(complete, .95) < 300),
        "audit": {"causal": audit, "highres_rejected": rejected},
        "sources": {"checkpoint": world.file_id(a.checkpoint),
                    "evaluation_replay": world.file_id(a.evaluation_replay)},
    }
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    metrics = json.loads(a.metrics.read_text()); metrics["live_latency"] = report
    a.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
