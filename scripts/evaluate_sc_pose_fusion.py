#!/usr/bin/env python3
"""Compare paired low-res and locator-crop SC pose predictions on scene groups."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from train_sc_pose_probe import CAMERAS, Probe


def summarize(rows):
    out = {}
    for split in ("train", "validation"):
        chosen = [row for row in rows if row["split"] == split and row["near"]]
        out[split] = {"count": len(chosen), "episodes": len({row["episode"] for row in chosen})}
        for key in ("lateral_mm", "axial_mm", "translation_mm", "orientation_deg"):
            values = [row[key] for row in chosen]
            out[split][key] = {"median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
        out[split]["lateral_sign_accuracy"] = float(np.mean([row["sign_correct"] for row in chosen]))
        out[split]["per_episode_lateral_p95_mm"] = {
            ep: float(np.percentile([r["lateral_mm"] for r in chosen if r["episode"] == ep], 95))
            for ep in sorted({r["episode"] for r in chosen})}
        out[split]["by_phase"] = {}
        for phase in ("far_approach", "preopening_30mm", "inside_opening"):
            phase_rows = [r for r in rows if r["split"] == split and r["phase"] == phase]
            if phase_rows:
                out[split]["by_phase"][phase] = {
                    "count": len(phase_rows), "episodes": len({r["episode"] for r in phase_rows}),
                    "lateral_median_mm": float(np.median([r["lateral_mm"] for r in phase_rows])),
                    "lateral_p95_mm": float(np.percentile([r["lateral_mm"] for r in phase_rows], 95)),
                    "axial_p95_mm": float(np.percentile([r["axial_mm"] for r in phase_rows], 95)),
                }
    return out


def residuals(labels, prediction, name):
    rows = []
    for item, (xyz, quat) in zip(labels, prediction):
        truth = np.asarray(item["observed_sc_tip_pose_opening_frame"], np.float64)
        delta = 1000 * (xyz - truth[:3])
        angle = math.degrees(2 * math.acos(float(np.clip(abs(np.dot(quat, truth[3:])), 0, 1))))
        rows.append({"split": item["split"], "episode": item["trial"],
                     "near": abs(truth[2]) < .03,
                     "phase": "far_approach" if truth[2] < -.03 else (
                         "preopening_30mm" if truth[2] < 0 else "inside_opening"),
                     "lateral_mm": float(np.linalg.norm(delta[:2])),
                     "axial_mm": float(abs(delta[2])),
                     "translation_mm": float(np.linalg.norm(delta)),
                     "orientation_deg": angle,
                     "sign_correct": bool(np.dot(xyz[:2], truth[:2]) >= 0)})
    return summarize(rows)


def read_checkpoint(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = Probe(images=True, pretrained_path=None).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return model, saved


def predict(labels, checkpoint, device):
    model, saved = read_checkpoint(checkpoint, device)
    transform = transforms.Compose([
        transforms.Resize((saved["image_size"], saved["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])
    results = []
    with torch.inference_mode():
        for row in labels:
            images = []
            for camera in CAMERAS:
                with Image.open(row["images"][camera]) as im:
                    images.append(transform(im.convert("RGB")))
            state = (torch.tensor(row["state"], dtype=torch.float32) - saved["state_mean"]) / saved["state_std"]
            state = torch.cat((state, torch.tensor(row["task_vector"], dtype=torch.float32)))
            xyz, quat = model(torch.stack(images)[None].to(device), state[None].to(device))
            results.append((xyz[0].cpu().numpy(), quat[0].cpu().numpy()))
    return results


def mix(a, b, weight_xy, weight_z, weight_q):
    out = []
    for (xyz_a, q_a), (xyz_b, q_b) in zip(a, b):
        xyz = np.array([(1-weight_xy)*xyz_a[0]+weight_xy*xyz_b[0],
                        (1-weight_xy)*xyz_a[1]+weight_xy*xyz_b[1],
                        (1-weight_z)*xyz_a[2]+weight_z*xyz_b[2]])
        if np.dot(q_a, q_b) < 0: q_b = -q_b
        q = (1-weight_q)*q_a+weight_q*q_b
        out.append((xyz, q / np.linalg.norm(q)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lowres-labels", type=Path, required=True)
    p.add_argument("--crop-labels", type=Path, required=True)
    p.add_argument("--lowres-checkpoint", type=Path, required=True)
    p.add_argument("--crop-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    low = [json.loads(line) for line in args.lowres_labels.open()]
    crop = [json.loads(line) for line in args.crop_labels.open()]
    if [(r["trial"], r["frame"], r["split"]) for r in low] != [
            (r["trial"], r["frame"], r["split"]) for r in crop]:
        raise ValueError("Unmatched low-res/native-crop rows")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_low = predict(low, args.lowres_checkpoint, device)
    pred_crop = predict(crop, args.crop_checkpoint, device)
    candidates = np.linspace(0, 1, 21)
    def train_p95(weight, dim):
        predictions = mix(pred_low, pred_crop,
                          weight if dim == "xy" else 0,
                          weight if dim == "z" else 0,
                          weight if dim == "q" else 0)
        train = residuals(low, predictions, "train")["train"]
        return train[{"xy":"lateral_mm", "z":"axial_mm", "q":"orientation_deg"}[dim]]["p95"]
    weights = {dim: float(min(candidates, key=lambda w: train_p95(w, dim)))
               for dim in ("xy", "z", "q")}
    fused = mix(pred_low, pred_crop, weights["xy"], weights["z"], weights["q"])
    result = {"schema": "sc_pose_fusion/v1", "episode_split": "verified manifest; grouped by reset scene",
              "training_only_choice": "three scalar weights chosen on training rows only",
              "weights_native_crop": weights,
              "lowres": residuals(low, pred_low, "lowres"),
              "native_crop": residuals(low, pred_crop, "native_crop"),
              "fused": residuals(low, fused, "fused"),
              "checkpoints": {"lowres": str(args.lowres_checkpoint), "native_crop": str(args.crop_checkpoint)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"weights": weights,
                      "validation_lateral_p95_mm": {key: result[key]["validation"]["lateral_mm"]["p95"]
                                                     for key in ("lowres", "native_crop", "fused")}}))


if __name__ == "__main__":
    main()
