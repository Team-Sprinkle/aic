#!/usr/bin/env python3
"""Bounded calibration-only check of pretrained spatial features for opening landmarks."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("opening", HERE / "train_opening_landmark_pose_probe.py")
opening = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(opening)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-replay", type=Path, required=True)
    p.add_argument("--scene-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--locator-updates", type=int, default=5000)
    p.add_argument("--landmark-updates", type=int, default=4000)
    p.add_argument("--patience", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--crop-size", type=int, default=160)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


class MobileNetLandmarks(nn.Module):
    """Small ImageNet backbone with a 40x40 feature pyramid heatmap head."""
    def __init__(self):
        super().__init__()
        self.features = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).features[:9]
        self.lateral40 = nn.Conv2d(16, 64, 1)
        self.lateral20 = nn.Conv2d(24, 64, 1)
        self.lateral10 = nn.Conv2d(48, 64, 1)
        self.head = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 6, 1))
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, image):
        value = (image - self.mean) / self.std
        taps = {}
        for index, layer in enumerate(self.features):
            value = layer(value)
            if index in (1, 3, 8): taps[index] = value
        value = self.lateral40(taps[1])
        value = value + nn.functional.interpolate(self.lateral20(taps[3]), size=value.shape[-2:], mode="bilinear", align_corners=False)
        value = value + nn.functional.interpolate(self.lateral10(taps[8]), size=value.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(value)


def train(model, fit, calibration, fit_coarse, cal_coarse, args, device):
    x, y, _ = opening.native_arrays(fit, fit_coarse, args.crop_size, jitter=True, seed=args.seed)
    xc, yc, _ = opening.native_arrays(calibration, cal_coarse, args.crop_size, jitter=False)
    model = model.to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed); torch.manual_seed(args.seed)
    best, best_loss, best_step, history = None, float("inf"), 0, []
    for step in range(1, args.landmark_updates + 1):
        index = torch.randint(len(x), (args.batch_size,), generator=generator)
        loss = opening.heatmap_loss(model(x[index].to(device).float()/255), y[index].to(device))
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step == 1 or step % 100 == 0:
            model.eval()
            with torch.no_grad():
                values = [opening.heatmap_loss(model(xc[s:s+96].to(device).float()/255), yc[s:s+96].to(device))
                          for s in range(0, len(xc), 96)]
                value = float(torch.stack(values).mean())
            model.train(); history.append({"step": step, "fit_loss": float(loss), "calibration_loss": value})
            if value < best_loss:
                best_loss, best_step, best = value, step, copy.deepcopy(model.state_dict())
            elif step - best_step >= args.patience:
                break
    model.load_state_dict(best); return model.cpu(), {"best_step": best_step, "completed_step": step,
                                                       "best_calibration_loss": best_loss, "history": history}


def main():
    a = arguments(); device = torch.device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(a.scene_manifest.read_text()); calibration_indices = {3, 7, 12, 16}
    fit_ids = [x["episode_id"] for i, x in enumerate(manifest["train"]) if i not in calibration_indices]
    cal_ids = [x["episode_id"] for i, x in enumerate(manifest["train"]) if i in calibration_indices]
    rows, audit = opening.world.load([a.train_replay], set(fit_ids + cal_ids)); rows, rejected = opening.attach(rows, a.train_replay)
    fit = [r for r in rows if r["episode_id"] in fit_ids]; calibration = [r for r in rows if r["episode_id"] in cal_ids]
    fit_c, cal_c = opening.coarse_rows(fit), opening.coarse_rows(calibration)
    locator, locator_training = opening.crop.train_locator(fit_c, cal_c, a, device)
    fit_coarse, locator_fit = opening.crop.locator_predict(locator, fit_c, device)
    cal_coarse, locator_cal = opening.crop.locator_predict(locator, cal_c, device)
    landmark, training = train(MobileNetLandmarks(), fit, calibration, fit_coarse, cal_coarse, a, device)
    fit_land, landmark_fit = opening.landmark_predict(landmark, fit, fit_coarse, a.crop_size, device)
    cal_land, landmark_cal = opening.landmark_predict(landmark, calibration, cal_coarse, a.crop_size, device)
    target_fit = np.stack([r["translation_mm"].numpy() for r in fit]); candidates = []
    for plug_source in ("coarse", "landmark"):
        for port_source in ("coarse", "entrance", "corners"):
            fp = opening.pair_points(fit_coarse, fit_land, plug_source, port_source)
            cp = opening.pair_points(cal_coarse, cal_land, plug_source, port_source)
            affine = opening.tri.affine_fit(opening.relative_world(fp, fit), target_fit)
            translation = np.c_[opening.relative_world(cp, calibration), np.ones(len(calibration))] @ affine
            prediction = torch.zeros(len(calibration), 6); prediction[:, :3] = torch.tensor(translation, dtype=torch.float32)
            near = [i for i, r in enumerate(calibration) if r["signed_depth_m"] >= -.003]
            metric = opening.world.metrics([calibration[i] for i in near], prediction[near], torch.zeros(len(near), 2),
                    torch.full((len(near), 3), 1/3), torch.zeros(len(near), 6), np.ones(6))[0]
            candidates.append({"plug": plug_source, "port": port_source,
                "score": metric["lateral_error_mm"]["median"] + metric["lateral_error_mm"]["p95"], "metrics": metric})
    selected = min(candidates, key=lambda x: x["score"])
    params = sum(p.numel() for p in landmark.parameters())
    summary = {"schema_version": 1, "status": "calibration_only", "development_set_opened": False,
        "model": {"name": "ImageNet MobileNetV3-small FPN", "parameters": params},
        "split": {"fit_episode_ids": fit_ids, "calibration_episode_ids": cal_ids, "fit_rows": len(fit), "calibration_rows": len(calibration)},
        "audit": audit, "rejected": rejected, "locator": {"training": locator_training, "fit": locator_fit, "calibration": locator_cal},
        "landmarks": {"training": training, "fit": landmark_fit, "calibration": landmark_cal},
        "selection": {"rule": "minimum calibration near-port lateral median+p95", "selected": selected, "candidates": candidates},
        "continuation_rule": "collect a fresh development split only if calibration indicates a material path toward median <=0.25 mm and p95 <=0.5 mm",
        "sources": {"train_replay": opening.world.file_id(a.train_replay), "scene_manifest": opening.world.file_id(a.scene_manifest)}}
    torch.save({"locator": locator.state_dict(), "landmark": landmark.state_dict(), "summary": summary}, a.output_dir/"calibration_checkpoint.pt")
    (a.output_dir/"calibration_metrics.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"output": str(a.output_dir), "parameters": params, "selected": selected}, indent=2))


if __name__ == "__main__": main()
