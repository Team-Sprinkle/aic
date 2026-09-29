#!/usr/bin/env python3
"""Diagnostic SC pose estimator using full views plus observation-selected crops.

The frozen full-view estimator supplies global context. A small learned
perception correction uses native crops selected by an RGB-only locator.
This predicts pose, never an action or TCP command.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from train_sc_pose_probe import CAMERAS, Probe


class PairedFrames(Dataset):
    def __init__(self, full, crops, mean, std, size):
        if [(r["trial"], r["frame"], r["split"]) for r in full] != [
                (r["trial"], r["frame"], r["split"]) for r in crops]:
            raise ValueError("Unmatched full and crop rows")
        self.full, self.crops, self.mean, self.std = full, crops, mean, std
        self.transform = transforms.Compose([
            transforms.Resize((size, size)), transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225))])

    def __len__(self): return len(self.full)

    def __getitem__(self, index):
        row = self.full[index]
        def views(paths):
            result = []
            for camera in CAMERAS:
                with Image.open(paths[camera]) as im:
                    result.append(self.transform(im.convert("RGB")))
            return torch.stack(result)
        state = (torch.tensor(row["state"], dtype=torch.float32) - self.mean) / self.std
        state = torch.cat((state, torch.tensor(row["task_vector"], dtype=torch.float32)))
        target = torch.tensor(row["observed_sc_tip_pose_opening_frame"], dtype=torch.float32)
        return views(row["images"]), views(self.crops[index]["images"]), state, target, row["trial"]


class FullCropFusion(nn.Module):
    def __init__(self, base_checkpoint, resnet_path):
        super().__init__()
        self.base = Probe(images=True, pretrained_path=None)
        self.base.load_state_dict(base_checkpoint["model"], strict=True)
        for p in self.base.parameters(): p.requires_grad_(False)
        self.base.eval()
        self.crop_backbone = models.resnet18(weights=None)
        self.crop_backbone.load_state_dict(torch.load(resnet_path, map_location="cpu", weights_only=True))
        self.crop_backbone.fc = nn.Identity()
        for name, p in self.crop_backbone.named_parameters():
            p.requires_grad_(name.startswith("layer4."))
        self.head = nn.Sequential(nn.LayerNorm(1536+128+3), nn.Linear(1536+128+3, 256),
                                  nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, full_images, crop_images, state):
        with torch.no_grad():
            base_xyz, base_q = self.base(full_images, state)
            context = self.base.state_net(state)
        batch, views = crop_images.shape[:2]
        local = self.crop_backbone(crop_images.reshape(batch*views, *crop_images.shape[2:]))
        local = local.reshape(batch, views*512)
        correction_mm = self.head(torch.cat((local, context, base_xyz), dim=-1)).clamp(-20, 20)
        return base_xyz + correction_mm / 1000, base_q


def evaluate(model, loader, device):
    model.eval()
    rows = []
    with torch.inference_mode():
        for full, crop, state, target, trials in loader:
            xyz, quat = model(full.to(device), crop.to(device), state.to(device))
            delta = (xyz.cpu() - target[:, :3]).numpy()*1000
            dots = (quat.cpu()*target[:, 3:]).sum(-1).abs().clamp(max=1).numpy()
            for i, trial in enumerate(trials):
                z = float(target[i, 2])
                rows.append({"trial": trial, "near": abs(z) < .03,
                             "phase": "far_approach" if z < -.03 else (
                                 "preopening_30mm" if z < 0 else "inside_opening"),
                             "lateral_mm": float(np.linalg.norm(delta[i, :2])),
                             "axial_mm": float(abs(delta[i, 2])),
                             "orientation_deg": math.degrees(2*math.acos(float(dots[i])))})
    near_rows = [r for r in rows if r["near"]]
    result = {"count": len(near_rows), "episodes": len({r['trial'] for r in near_rows})}
    for key in ("lateral_mm", "axial_mm", "orientation_deg"):
        values = [r[key] for r in near_rows]
        result[key] = {"median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
    result["per_episode_lateral_p95_mm"] = {
        trial: float(np.percentile([r["lateral_mm"] for r in near_rows if r["trial"] == trial], 95))
        for trial in sorted({r["trial"] for r in near_rows})}
    result["by_phase"] = {}
    for phase in ("far_approach", "preopening_30mm", "inside_opening"):
        subset = [r for r in rows if r["phase"] == phase]
        if subset:
            result["by_phase"][phase] = {"count": len(subset),
                "lateral_median_mm": float(np.median([r["lateral_mm"] for r in subset])),
                "lateral_p95_mm": float(np.percentile([r["lateral_mm"] for r in subset],95)),
                "axial_p95_mm": float(np.percentile([r["axial_mm"] for r in subset],95))}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-labels", type=Path, required=True)
    p.add_argument("--crop-labels", type=Path, required=True)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--pretrained-resnet", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260924)
    p.add_argument("--eval-checkpoint", type=Path,
                   help="Evaluate a saved fusion checkpoint with the current phase metrics and exit")
    args = p.parse_args()
    torch.set_num_threads(8);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    full = [json.loads(x) for x in args.full_labels.open()]
    crop = [json.loads(x) for x in args.crop_labels.open()]
    if len(full) != len(crop): raise ValueError("Unmatched paired rows")
    fit_idx = [i for i,r in enumerate(full) if r["split"] == "train"]
    val_idx = [i for i,r in enumerate(full) if r["split"] == "validation"]
    if {full[i]["scene_sha256"] for i in fit_idx} & {full[i]["scene_sha256"] for i in val_idx}:
        raise ValueError("Scene leakage")
    saved = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    ds_fit = PairedFrames([full[i] for i in fit_idx], [crop[i] for i in fit_idx],
                          saved["state_mean"], saved["state_std"], saved["image_size"])
    ds_val = PairedFrames([full[i] for i in val_idx], [crop[i] for i in val_idx],
                          saved["state_mean"], saved["state_std"], saved["image_size"])
    fit = DataLoader(ds_fit, batch_size=args.batch_size, shuffle=bool(len(ds_fit)), num_workers=args.workers,
                     pin_memory=True, persistent_workers=args.workers>0)
    val = DataLoader(ds_val, batch_size=args.batch_size, num_workers=args.workers,
                     pin_memory=True, persistent_workers=args.workers>0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullCropFusion(saved, args.pretrained_resnet).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_checkpoint:
        model.load_state_dict(torch.load(args.eval_checkpoint, map_location="cpu", weights_only=False)["model"], strict=True)
        result = {"train": evaluate(model, fit, device) if len(ds_fit) else None,
                  "validation": evaluate(model, val, device)}
        (args.output_dir/"phase_metrics.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({"validation":result["validation"]["lateral_mm"]}))
        return
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=1e-4)
    initial = evaluate(model, val, device)
    history = [{"epoch": 0, "validation": initial}]
    best = initial["lateral_mm"]["p95"]
    torch.save({"model": model.state_dict(), "epoch": 0, "validation": initial}, args.output_dir/"best.pt")
    print(json.dumps({"epoch": 0, "near_lateral_p95_mm": best}), flush=True)
    for epoch in range(1, args.epochs+1):
        model.train();start=time.time();losses=[]
        for full_images, crop_images, state, target, _ in fit:
            xyz, _ = model(full_images.to(device), crop_images.to(device), state.to(device))
            target = target.to(device)
            delta = (xyz-target[:, :3])*1000
            near = (target[:, 2].abs()<.03).float()
            loss = (((1+3*near)*(delta[:, :2]**2).sum(-1)+delta[:, 2]**2)).mean()/100
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1)
            optimizer.step();losses.append(float(loss.detach()))
        result = evaluate(model, val, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "validation": result, "elapsed_s": time.time()-start})
        score=result["lateral_mm"]["p95"]
        if score<best:
            best=score
            torch.save({"model":model.state_dict(), "epoch":epoch,"validation":result}, args.output_dir/"best.pt")
        print(json.dumps({"epoch":epoch,"near_lateral_p95_mm":score,
                          "best_mm":best,"elapsed_s":history[-1]["elapsed_s"]}),flush=True)
    (args.output_dir/"metrics.json").write_text(json.dumps({"schema":"sc_full_native_crop_fusion/v1",
        "full_labels":str(args.full_labels),"crop_labels":str(args.crop_labels),
        "base_checkpoint":str(args.base_checkpoint),"pretrained_resnet":str(args.pretrained_resnet),
        "train_episodes":len({full[i]['trial'] for i in fit_idx}),
        "validation_episodes":len({full[i]['trial'] for i in val_idx}),
        "best_near_lateral_p95_mm":best,"history":history},indent=2)+"\n")


if __name__ == "__main__": main()
