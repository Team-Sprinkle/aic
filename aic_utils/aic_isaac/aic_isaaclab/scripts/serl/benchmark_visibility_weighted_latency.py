#!/usr/bin/env python3
"""Benchmark the complete observation-only visibility-weighted pose path."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("visibility_runtime", HERE / "train_visibility_weighted_pose_ablation.py")
runtime = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(runtime)


def quantiles(value):
    return {name: float(np.quantile(value, q)) for name, q in (("p50", .5), ("p95", .95), ("p99", .99))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); device = torch.device(args.device)

    manifest = json.loads(args.dataset_manifest.read_text())
    ids = set(manifest["splits"]["development"]["episode_ids"])
    rows, audit = runtime.opening.world.load([args.replay], ids)
    rows, rejected = runtime.opening.attach(rows, args.replay)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    locator = runtime.opening.crop.Locator().to(device).eval(); locator.load_state_dict(saved["locator"])
    landmark = runtime.pretrained.MobileNetLandmarks().to(device).eval(); landmark.load_state_dict(saved["landmark"])
    visibility = runtime.VisibilityHead(saved["visibility_input_dim"]).to(device).eval()
    visibility.load_state_dict(saved["visibility_head"])
    residual_cfg = saved["current"]
    residuals = []
    for state in residual_cfg["state_dicts"]:
        model = runtime.ResidualPose(len(residual_cfg["input_mean"])).to(device).eval()
        model.load_state_dict(state); residuals.append(model)
    affine = saved["affines"]["predicted_visibility"]
    selected = saved["selected_landmarks"]
    images = [torch.stack([
        torch.from_numpy(np.asarray(Image.open(view["path"]).convert("RGB")).copy()).permute(2, 0, 1)
        for view in row["highres"]
    ]) for row in rows]

    def run(image_cpu, row):
        full = image_cpu.to(device).float() / 255.0
        coarse = locator(F.interpolate(full, size=(256, 288), mode="bilinear", align_corners=False))
        height, width = full.shape[-2:]
        centers = coarse.reshape(3, 2, 2).mean(1)
        size = args.crop_size
        gx = torch.linspace(-(size-1)/(width-1), (size-1)/(width-1), size, device=device)
        gy = torch.linspace(-(size-1)/(height-1), (size-1)/(height-1), size, device=device)
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")
        grids = torch.stack([torch.stack(((center[0]*2-1)+xx, (center[1]*2-1)+yy), -1) for center in centers])
        crops = F.grid_sample(full, grids, mode="bilinear", padding_mode="zeros", align_corners=True)
        value = (crops - landmark.mean) / landmark.std
        taps = {}
        for index, layer in enumerate(landmark.features):
            value = layer(value)
            if index in (1, 3, 8): taps[index] = value
        fused = landmark.lateral40(taps[1])
        fused = fused + F.interpolate(landmark.lateral20(taps[3]), fused.shape[-2:], mode="bilinear", align_corners=False)
        fused = fused + F.interpolate(landmark.lateral10(taps[8]), fused.shape[-2:], mode="bilinear", align_corners=False)
        logits = landmark.head(fused)
        local = runtime.opening.decode(logits)
        probability = torch.softmax(logits.flatten(-2), -1)
        reliability_feature = torch.cat((taps[8].mean((-2, -1)), probability.max(-1).values,
            -(probability * probability.clamp_min(1e-9).log()).sum(-1) / math.log(probability.shape[-1])), 1)
        view_weights = torch.sigmoid(visibility(reliability_feature)).cpu()
        center_px = centers * torch.tensor([width-1, height-1], device=device)
        left_top = torch.round(center_px - size/2)
        pixels = local * (size-1) + left_top[:, None, :]
        landmarks = pixels / torch.tensor([width-1, height-1], device=device)
        coarse_cpu = coarse.cpu().reshape(1, 3, 4)
        landmark_cpu = landmarks.cpu().reshape(1, 3, 6, 2)
        points = runtime.opening.pair_points(coarse_cpu, landmark_cpu, selected["plug"], selected["port"])
        raw = runtime.relative_weighted(points, [row], view_weights.reshape(1, 3, 2))
        translated = np.c_[raw, np.ones(1)] @ affine
        feature = runtime.residual_features(coarse_cpu, landmark_cpu, translated, view_weights.reshape(1, 3, 2))
        normalized = ((feature[0] - residual_cfg["input_mean"]) / residual_cfg["input_std"]).to(device)
        residual = torch.stack([model(normalized) for model in residuals]).mean(0)
        return torch.tensor(translated[0], device=device) + residual * residual_cfg["target_std"].to(device) + residual_cfg["target_mean"].to(device)

    with torch.inference_mode():
        for _ in range(10): run(images[0], rows[0])
        elapsed = []
        for image, row in zip(images, rows):
            torch.cuda.synchronize(); start = time.perf_counter(); run(image, row); torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - start) * 1000)
    perception = np.asarray(elapsed)
    trunk = np.asarray([float(row["model_inference_s"]) * 1000 for row in rows])
    complete = perception + trunk
    report = {
        "schema_version": 1,
        "sample_count": len(rows),
        "scope": "three in-memory 576x512 RGB views; resize, observation-only locator, native 160x160 crops, shared landmark/visibility encoder, weighted triangulation, current residual ensemble; sensor acquisition excluded",
        "perception_ms": quantiles(perception),
        "existing_frozen_policy_trunk_ms": quantiles(trunk),
        "complete_if_paired_with_existing_trunk_ms": quantiles(complete),
        "complete_p95_below_300ms": bool(np.quantile(complete, .95) < 300),
        "audit": {"load": audit, "image_rejected": rejected},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    metrics = json.loads(args.metrics.read_text()); metrics["live_latency"] = report
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
