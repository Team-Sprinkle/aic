#!/usr/bin/env python3
"""Offline latency for three native RGB views through locator and keypoints.

The loop reads only image paths. It excludes ROS transport, actor, and command
conversion; the tiny ray solve is included with fixed representative cameras.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluate_sc_native_triangulation import triangulate
from train_sc_native_keypoints import PretrainedKeypointRefiner
from train_sc_native_locator import CAMERAS, HEIGHT, WIDTH, Locator
from train_sc_native_orientation import CropOrientation
from torchvision import transforms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--locator", type=Path, required=True)
    parser.add_argument("--keypoints", type=Path, required=True)
    parser.add_argument("--orientation-checkpoint", type=Path)
    parser.add_argument("--landmark-checkpoint", type=Path,
                        help="Include five-landmark port orientation inference and triangulation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(8)
    rows = [r for r in map(json.loads, args.labels.open()) if r["split"] == "validation"][:args.samples]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    locator_saved = torch.load(args.locator, map_location="cpu", weights_only=False)
    spatial_locator = locator_saved.get("architecture") == "task_conditioned_resnet18_global_locator"
    locator = (PretrainedKeypointRefiner(None, True) if spatial_locator else Locator()).to(device).eval()
    locator.load_state_dict(locator_saved["model"])
    saved = torch.load(args.keypoints, map_location="cpu", weights_only=False)
    if saved.get("architecture") != "pretrained_resnet18":
        raise ValueError("Benchmark expects the pretrained ResNet-18 refiner")
    conditioned = bool(saved.get("condition_target_port", False))
    keypoints = PretrainedKeypointRefiner(None, conditioned).to(device).eval()
    keypoints.load_state_dict(saved["model"])
    orientation_saved = (torch.load(args.orientation_checkpoint, map_location="cpu", weights_only=False)
                         if args.orientation_checkpoint else None)
    orientation = None
    if orientation_saved:
        orientation = CropOrientation(None).to(device).eval()
        orientation.load_state_dict(orientation_saved["model"])
        normalize = transforms.Normalize((.485,.456,.406),(.229,.224,.225))
    landmark_saved = (torch.load(args.landmark_checkpoint,map_location="cpu",weights_only=False)
                      if args.landmark_checkpoint else None)
    landmark_model = None
    if landmark_saved:
        landmark_model=PretrainedKeypointRefiner(None,True,point_count=len(landmark_saved["points"])).to(device).eval()
        landmark_model.load_state_dict(landmark_saved["model"])
    size = int(saved["crop_size"])
    # Fixed, nonprivileged representative camera geometry measures the ray solve.
    cameras = {}
    for index, cam in enumerate(CAMERAS):
        transform = np.eye(4)
        transform[0, 3] = (index - 1) * 0.06
        cameras[cam] = transform
    measured = []
    with torch.inference_mode():
        for row in rows:
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            raw, coarse = [], []
            for cam in CAMERAS:
                with Image.open(row["images"][cam]) as image:
                    rgb = image.convert("RGB")
                    if rgb.size != (WIDTH, HEIGHT):
                        raise ValueError("Unexpected native image shape")
                    raw.append(rgb.copy())
                    resized = rgb.resize((288, 256), Image.Resampling.BILINEAR)
                    coarse.append(torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1))
            coarse_input = torch.stack(coarse).to(device).float() / 255
            if spatial_locator:
                locator_task = torch.tensor(row["task_vector"][2:4], dtype=torch.float32,
                                            device=device).repeat(len(CAMERAS), 1)
                centers = locator(coarse_input, locator_task).cpu().numpy()
            else:
                centers = locator(coarse_input).cpu().numpy()
            crop_tensors, corners = [], []
            for image, center in zip(raw, centers):
                points = center.reshape(2, 2) * np.array([WIDTH - 1, HEIGHT - 1])
                cx, cy = points.mean(axis=0)
                left, top = round(float(cx - size / 2)), round(float(cy - size / 2))
                corners.append((left, top))
                cropped = image.crop((left, top, left + size, top + size))
                crop_tensors.append(torch.from_numpy(np.asarray(cropped).copy()).permute(2, 0, 1))
            stacked = torch.stack(crop_tensors).to(device).float() / 255
            onehot = torch.tensor(row["task_vector"][2:4], dtype=torch.float32,
                                  device=device).repeat(len(CAMERAS), 1)
            if conditioned:
                pixels = keypoints(stacked, onehot).cpu().numpy()
            else:
                pixels = keypoints(stacked).cpu().numpy()
            opening = {cam: [float(point[2] * (size - 1) + corner[0]),
                             float(point[3] * (size - 1) + corner[1])]
                       for cam, point, corner in zip(CAMERAS, pixels, corners)}
            triangulate(opening, cameras)
            if landmark_model is not None:
                landmark_pixels=landmark_model(stacked,onehot).cpu().numpy().reshape(len(CAMERAS),-1,2)
                for index,_ in enumerate(landmark_saved["points"]):
                    rays={cam:[float(landmark_pixels[j,index,0]*(size-1)+corners[j][0]),
                               float(landmark_pixels[j,index,1]*(size-1)+corners[j][1])]
                          for j,cam in enumerate(CAMERAS)}
                    triangulate(rays,cameras)
            if orientation is not None:
                state = ((torch.tensor(row["state"], dtype=torch.float32)-orientation_saved["state_mean"])
                         /orientation_saved["state_std"])
                state = torch.cat((state,torch.tensor(row["task_vector"],dtype=torch.float32)))
                orientation(torch.stack([normalize(crop.float()/255) for crop in crop_tensors])[None].to(device),
                            state[None].to(device))
            if device.type == "cuda":
                torch.cuda.synchronize()
            measured.append((time.perf_counter() - start) * 1000)
    timings = measured[10:] if len(measured) > 20 else measured
    result = {"schema": "sc_native_keypoint_offline_latency/v1",
              "scope": "three native JPEG decodes, resize, RGB locator, native crops, pretrained keypoints, ray solve"
                       + (", crop orientation model" if orientation is not None else "")
                       + (", five-landmark orientation and ray solves" if landmark_model is not None else "")
                       + "; excludes ROS acquisition, actor, and command conversion",
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "samples": len(measured), "warmup_excluded": len(measured) - len(timings),
              "p50_ms": float(np.percentile(timings, 50)),
              "p95_ms": float(np.percentile(timings, 95)),
              "p99_ms": float(np.percentile(timings, 99)),
              "max_ms": float(max(timings))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
