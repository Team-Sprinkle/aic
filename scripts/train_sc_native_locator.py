#!/usr/bin/env python3
"""Train an image-only SC plug/opening locator and save its predicted crops."""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


CAMERAS = ("center", "left", "right")
WIDTH, HEIGHT = 1152, 1024


class Locator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.GELU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 96, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d((3, 4)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(96 * 12, 128), nn.GELU(), nn.Linear(128, 4))

    def forward(self, images):
        return torch.sigmoid(self.head(self.conv(images)))


def percentile(values, p):
    if not values:
        return None
    return float(np.percentile(values, p))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--checkpoint", type=Path,
                        help="Frozen locator inference only; no training or checkpoint overwrite")
    args = parser.parse_args()
    torch.set_num_threads(8)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows = [json.loads(line) for line in args.labels.open()]
    train_scenes = {r["scene_sha256"] for r in rows if r["split"] == "train"}
    val_scenes = {r["scene_sha256"] for r in rows if r["split"] == "validation"}
    if train_scenes & val_scenes:
        raise ValueError("Scene leakage")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs, targets, masks, splits, keys = [], [], [], [], []
    for i, row in enumerate(rows):
        for camera in CAMERAS:
            with Image.open(row["images"][camera]) as im:
                rgb = im.convert("RGB")
                if rgb.size != (WIDTH, HEIGHT):
                    raise ValueError(f"Non-native image {row['images'][camera]} {rgb.size}")
                coarse = rgb.resize((288, 256), Image.Resampling.BILINEAR)
                inputs.append(torch.from_numpy(np.asarray(coarse).copy()).permute(2, 0, 1))
            points = row["projected_training_pixels"][camera]
            targets.append([(points["tip"][0] or 0) / (WIDTH - 1),
                            (points["tip"][1] or 0) / (HEIGHT - 1),
                            (points["opening"][0] or 0) / (WIDTH - 1),
                            (points["opening"][1] or 0) / (HEIGHT - 1)])
            masks.append([float(points["tip"][3])] * 2 + [float(points["opening"][3])] * 2)
            splits.append(row["split"]); keys.append((i, camera))
    x = torch.stack(inputs)
    y = torch.tensor(targets, dtype=torch.float32)
    mask = torch.tensor(masks, dtype=torch.float32)
    fit = torch.tensor([i for i, split in enumerate(splits) if split == "train"])
    val = torch.tensor([i for i, split in enumerate(splits) if split == "validation"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Locator().to(device)
    best_step, history, step = None, [], 0
    if args.checkpoint:
        frozen = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if frozen["crop_size"] != args.crop_size:
            raise ValueError("Frozen locator crop-size mismatch")
        model.load_state_dict(frozen["model"], strict=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(args.seed)
        best, best_loss, best_step = None, float("inf"), 0
        for step in range(1, args.updates + 1):
            indices = fit[torch.randint(len(fit), (args.batch_size,), generator=generator)]
            pred = model(x[indices].to(device).float() / 255)
            loss = (F.smooth_l1_loss(pred, y[indices].to(device), beta=.01, reduction="none")
                    * mask[indices].to(device)).sum() / mask[indices].sum().clamp_min(1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if step == 1 or step % 100 == 0:
                model.eval()
                with torch.inference_mode():
                    losses = []
                    for chunk in val.split(96):
                        estimate = model(x[chunk].to(device).float() / 255)
                        part = (F.smooth_l1_loss(estimate, y[chunk].to(device), beta=.01, reduction="none")
                                * mask[chunk].to(device)).sum()
                        losses.append(float(part))
                    metric = sum(losses) / mask[val].sum().item()
                history.append({"step": step, "train_loss": float(loss), "validation_loss": metric})
                if metric < best_loss:
                    best, best_loss, best_step = copy.deepcopy(model.state_dict()), metric, step
                print(json.dumps(history[-1]), flush=True)
                model.train()
                if step - best_step >= 500:
                    break
        model.load_state_dict(best)
    model.eval()
    predicted = []
    with torch.inference_mode():
        for chunk in torch.arange(len(x)).split(96):
            predicted.append(model(x[chunk].to(device).float() / 255).cpu())
    predicted = torch.cat(predicted)
    errors = {split: {point: [] for point in ("tip", "opening")} for split in ("train", "validation")}
    for i, (prediction, truth, visible, split) in enumerate(zip(predicted, y, mask, splits)):
        for point, start in (("tip", 0), ("opening", 2)):
            if visible[start]:
                pixel = (prediction[start:start+2] - truth[start:start+2]) * torch.tensor([WIDTH-1, HEIGHT-1])
                errors[split][point].append(float(pixel.norm()))
    crop_root = args.output_dir / "predicted_crops"
    crop_root.mkdir(exist_ok=True)
    crop_rows = [{**r, "images": dict(r["images"])} for r in rows]
    for i, (row_index, camera) in enumerate(keys):
        row = rows[row_index]
        pixel = predicted[i] * torch.tensor([WIDTH-1, HEIGHT-1] * 2)
        center = pixel.reshape(2, 2).mean(0)
        left = round(float(center[0]) - args.crop_size / 2)
        top = round(float(center[1]) - args.crop_size / 2)
        with Image.open(row["images"][camera]) as im:
            crop = im.convert("RGB").crop((left, top, left + args.crop_size, top + args.crop_size))
            dest = crop_root / f"{row['trial']}_{row['frame']:06d}_{camera}.jpg"
            crop.save(dest, quality=93)
        crop_rows[row_index]["images"][camera] = str(dest)
        crop_rows[row_index].setdefault("predicted_crop_centers", {})[camera] = center.tolist()
    labels_out = args.output_dir / "pose_labels_predicted_crops.jsonl"
    with labels_out.open("w") as out:
        for row in crop_rows:
            row.pop("projected_training_pixels")
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
    if not args.checkpoint:
        torch.save({"model": model.cpu().state_dict(), "seed": args.seed,
                    "crop_size": args.crop_size}, args.output_dir / "locator.pt")
    report = {"schema": "sc_native_locator/v1", "train_episodes": len({r['episode_id'] for r in rows if r['split']=='train'}),
              "validation_episodes": len({r['episode_id'] for r in rows if r['split']=='validation'}),
              "training_views": len(fit), "validation_views": len(val),
              "best_step": best_step, "completed_step": step, "history": history,
              "heldout_error_px": {split: {point: {"count": len(values), "median": percentile(values, 50),
                                                  "p95": percentile(values, 95)}
                                         for point, values in section.items()} for split, section in errors.items()},
              "crop_labels": str(labels_out),
              "frozen_checkpoint": str(args.checkpoint) if args.checkpoint else None,
              "runtime_crop_selection": "locator RGB predictions only; no TF or projected labels"}
    (args.output_dir / "locator_metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
