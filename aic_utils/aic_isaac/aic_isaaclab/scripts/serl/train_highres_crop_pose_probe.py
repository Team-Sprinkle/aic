#!/usr/bin/env python3
"""Train the bounded native-resolution locator and crop pose-probe ablation."""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("world_probe", HERE / "train_world_feature_pose_probe.py")
world_probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(world_probe)

CAMERAS = ("center_camera", "left_camera", "right_camera")
PHASES = ("approach", "alignment", "contact")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-replay", type=Path, required=True)
    p.add_argument("--evaluation-replay", type=Path, required=True)
    p.add_argument("--scene-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--locator-updates", type=int, default=4000)
    p.add_argument("--probe-updates", type=int, default=5000)
    p.add_argument("--patience", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--crop-size", type=int, default=160)
    p.add_argument("--ensemble-size", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


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

    def forward(self, image):
        return torch.sigmoid(self.head(self.conv(image)))


class CropPoseProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.crop_encoder = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2, 2), nn.GELU(),
            nn.Conv2d(24, 48, 3, 2, 1), nn.GELU(),
            nn.Conv2d(48, 96, 3, 2, 1), nn.GELU(),
            nn.Conv2d(96, 96, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, 64), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(384 + 3 * 64), nn.Linear(384 + 3 * 64, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 11),
        )

    def forward(self, feature, crops):
        encoded = [self.crop_encoder(crops[:, view]) for view in range(3)]
        return self.head(torch.cat([feature, *encoded], dim=1))


def attach_highres(rows, replay_path):
    transitions = torch.load(replay_path, map_location="cpu", weights_only=False)["transitions"]
    accepted = []
    rejected = Counter()
    for row in rows:
        t = transitions[row["transition_index"]]
        highres = (t.get("metadata") or {}).get("highres_observation") or {}
        cameras = highres.get("cameras") or {}
        parsed = []
        for camera in CAMERAS:
            item = cameras.get(camera) or {}
            labels = item.get("locator_supervision_xy") or {}
            opening = labels.get("entrance") or labels.get("target")
            if not item.get("path") or labels.get("plug") is None or opening is None:
                break
            parsed.append({"path": item["path"], "plug": labels["plug"], "target": opening,
                           "width": item["width"], "height": item["height"],
                           "intrinsic_matrix": item.get("intrinsic_matrix"),
                           "camera_position_world": item.get("camera_position_world"),
                           "camera_orientation_wxyz_ros": item.get("camera_orientation_wxyz_ros"),
                           "locator_label_method": item.get("locator_label_method")})
        if len(parsed) != 3:
            rejected["missing_camera_or_locator_label"] += 1
            continue
        row = dict(row)
        row["highres"] = parsed
        accepted.append(row)
    return accepted, dict(rejected)


def locator_arrays(rows):
    images, labels, episode_ids, camera_ids = [], [], [], []
    for row in rows:
        for camera_id, item in enumerate(row["highres"]):
            image = Image.open(item["path"]).convert("RGB").resize((288, 256), Image.Resampling.BILINEAR)
            images.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).to(torch.uint8))
            w, h = float(item["width"] - 1), float(item["height"] - 1)
            labels.append(torch.tensor([item["plug"][0] / w, item["plug"][1] / h,
                                        item["target"][0] / w, item["target"][1] / h]))
            episode_ids.append(row["episode_id"])
            camera_ids.append(camera_id)
    return torch.stack(images), torch.stack(labels), episode_ids, camera_ids


def train_locator(fit_rows, calibration_rows, args, device):
    x, y, episodes, _ = locator_arrays(fit_rows)
    xc, yc, _, _ = locator_arrays(calibration_rows)
    counts = Counter(episodes)
    weights = torch.tensor([1.0 / counts[episode] for episode in episodes])
    torch.manual_seed(args.seed)
    model = Locator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    best, best_loss, best_step = None, float("inf"), 0
    history = []
    for step in range(1, args.locator_updates + 1):
        index = torch.multinomial(weights, args.batch_size, replacement=True, generator=generator)
        prediction = model(x[index].to(device).float() / 255.0)
        loss = F.smooth_l1_loss(prediction, y[index].to(device), beta=0.01)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step == 1 or step % 100 == 0:
            with torch.no_grad():
                values = []
                for start in range(0, len(xc), 128):
                    values.append(model(xc[start:start + 128].to(device).float() / 255.0).cpu())
                validation = F.smooth_l1_loss(torch.cat(values), yc, beta=0.01)
            value = float(validation)
            history.append({"step": step, "fit_loss": float(loss), "calibration_loss": value})
            if value < best_loss:
                best_loss, best_step = value, step
                best = copy.deepcopy(model.state_dict())
            elif step - best_step >= args.patience:
                break
    model.load_state_dict(best)
    return model.cpu(), {"best_step": best_step, "completed_step": step,
                         "best_calibration_loss": best_loss, "history": history}


def locator_predict(model, rows, device):
    x, y, _, camera_ids = locator_arrays(rows)
    model = model.to(device).eval()
    predictions = []
    latencies = []
    with torch.no_grad():
        for start in range(0, len(x), 128):
            batch = x[start:start + 128].to(device).float() / 255.0
            if device.type == "cuda": torch.cuda.synchronize()
            tick = time.perf_counter()
            pred = model(batch)
            if device.type == "cuda": torch.cuda.synchronize()
            latencies.append((time.perf_counter() - tick) * 1000 / len(batch))
            predictions.append(pred.cpu())
    prediction = torch.cat(predictions)
    errors = []
    for i, (pred, true) in enumerate(zip(prediction, y)):
        row = rows[i // 3]; item = row["highres"][camera_ids[i]]
        scale = torch.tensor([item["width"] - 1, item["height"] - 1] * 2)
        errors.append(((pred - true) * scale).reshape(2, 2).norm(dim=1))
    errors = torch.stack(errors)
    report = {
        "view_count": len(errors), "episode_count": len(set(r["episode_id"] for r in rows)),
        "plug_error_px": {"median": float(errors[:, 0].median()), "p95": float(torch.quantile(errors[:, 0], .95))},
        "target_error_px": {"median": float(errors[:, 1].median()), "p95": float(torch.quantile(errors[:, 1], .95))},
        "batched_compute_ms_per_view": {"mean": float(np.mean(latencies)), "max_batch_mean": float(np.max(latencies))},
    }
    return prediction.reshape(len(rows), 3, 4), report


def make_predicted_crops(rows, normalized_keypoints, crop_size):
    result = []
    for row, all_points in zip(rows, normalized_keypoints):
        views = []
        for item, points in zip(row["highres"], all_points):
            image = Image.open(item["path"]).convert("RGB")
            scale = torch.tensor([item["width"] - 1, item["height"] - 1] * 2)
            pixels = (points * scale).reshape(2, 2)
            center = pixels.mean(0)
            left = int(round(float(center[0]) - crop_size / 2)); top = int(round(float(center[1]) - crop_size / 2))
            crop = image.crop((left, top, left + crop_size, top + crop_size)).resize((128, 128), Image.Resampling.BILINEAR)
            views.append(torch.from_numpy(np.asarray(crop).copy()).permute(2, 0, 1).to(torch.uint8))
        result.append(torch.stack(views))
    return torch.stack(result)


def target_tensors(rows):
    regression = torch.stack([torch.cat((r["translation_mm"], r["rotation_deg"])) for r in rows])
    classes = torch.tensor([[r["contact"], r["blocked"]] for r in rows])
    phases = torch.tensor([PHASES.index(r["phase"]) for r in rows], dtype=torch.long)
    return regression, classes, phases


def train_crop_ensemble(fit, calib, fit_crops, calib_crops, args, device):
    feature = torch.stack([r["feature"] for r in fit]); y, classes, phases = target_tensors(fit)
    cfeature = torch.stack([r["feature"] for r in calib]); cy, cclasses, cphases = target_tensors(calib)
    counts = Counter(r["episode_id"] for r in fit)
    weights = torch.tensor([1.0 / counts[r["episode_id"]] for r in fit])
    normalized_weights = weights / weights.sum()
    mean = (normalized_weights[:, None] * y).sum(0)
    std = torch.sqrt((normalized_weights[:, None] * (y - mean).square()).sum(0)).clamp_min(1e-3)
    models, histories = [], []
    for member in range(args.ensemble_size):
        seed = args.seed + member
        torch.manual_seed(seed); random.seed(seed)
        model = CropPoseProbe().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(seed)
        best, best_loss, best_step, history = None, float("inf"), 0, []
        for step in range(1, args.probe_updates + 1):
            index = torch.multinomial(weights, args.batch_size, replacement=True, generator=generator)
            out = model(feature[index].to(device), fit_crops[index].to(device).float() / 255.0)
            loss = (F.smooth_l1_loss(out[:, :6], ((y[index] - mean) / std).to(device))
                    + .25 * F.binary_cross_entropy_with_logits(out[:, 6:8], classes[index].to(device))
                    + .15 * F.cross_entropy(out[:, 8:11], phases[index].to(device)))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if step == 1 or step % 100 == 0:
                with torch.no_grad():
                    values = []
                    for start in range(0, len(calib), 128):
                        values.append(model(cfeature[start:start + 128].to(device),
                                            calib_crops[start:start + 128].to(device).float() / 255.0).cpu())
                    co = torch.cat(values)
                    validation = (F.smooth_l1_loss(co[:, :6], (cy - mean) / std)
                                  + .25 * F.binary_cross_entropy_with_logits(co[:, 6:8], cclasses)
                                  + .15 * F.cross_entropy(co[:, 8:11], cphases))
                value = float(validation); history.append({"step": step, "calibration_loss": value})
                if value < best_loss:
                    best_loss, best_step, best = value, step, copy.deepcopy(model.state_dict())
                elif step - best_step >= args.patience: break
        model.load_state_dict(best); models.append(model.cpu())
        histories.append({"member": member, "seed": seed, "best_step": best_step,
                          "completed_step": step, "best_calibration_loss": best_loss, "history": history})
    return models, mean, std, histories


def crop_predict(models, rows, crops, mean, std, device):
    feature = torch.stack([r["feature"] for r in rows]); outputs = []; latencies = []
    for model in models:
        model = model.to(device).eval(); member = []
        with torch.no_grad():
            for start in range(0, len(rows), 128):
                f = feature[start:start + 128].to(device); c = crops[start:start + 128].to(device).float() / 255.0
                if device.type == "cuda": torch.cuda.synchronize()
                tick = time.perf_counter(); value = model(f, c)
                if device.type == "cuda": torch.cuda.synchronize()
                latencies.append((time.perf_counter() - tick) * 1000 / len(f))
                member.append(value.cpu())
        outputs.append(torch.cat(member))
    stack = torch.stack(outputs); regression = stack[:, :, :6] * std + mean
    return (regression.mean(0), regression.var(0, unbiased=False),
            torch.sigmoid(stack[:, :, 6:8]).mean(0), torch.softmax(stack[:, :, 8:11], -1).mean(0), latencies)


def save_montage(rows, keypoints, output, crop_size):
    chosen = np.linspace(0, len(rows) - 1, min(9, len(rows)), dtype=int)
    tiles = []
    for index in chosen:
        row = rows[index]; per_view = []
        for item, points in zip(row["highres"], keypoints[index]):
            image = Image.open(item["path"]).convert("RGB")
            scale = torch.tensor([item["width"] - 1, item["height"] - 1] * 2)
            pred = (points * scale).reshape(2, 2); center = pred.mean(0)
            truth = np.array([item["plug"], item["target"]])
            draw = ImageDraw.Draw(image)
            for x, y in truth: draw.ellipse((x-4, y-4, x+4, y+4), outline="lime", width=2)
            for x, y in pred: draw.ellipse((float(x)-4, float(y)-4, float(x)+4, float(y)+4), outline="red", width=2)
            left, top = float(center[0])-crop_size/2, float(center[1])-crop_size/2
            draw.rectangle((left, top, left+crop_size, top+crop_size), outline="yellow", width=2)
            per_view.append(image.resize((288, 256)))
        strip = Image.new("RGB", (288 * 3, 256));
        for j, image in enumerate(per_view): strip.paste(image, (288*j, 0))
        tiles.append(strip)
    sheet = Image.new("RGB", (288 * 3, 256 * len(tiles)))
    for i, tile in enumerate(tiles): sheet.paste(tile, (0, 256*i))
    sheet.save(output)


def benchmark_live_representation(locator, pose_model, rows, device, crop_size):
    """Measure the deployable tensor path; simulator sensor acquisition is outside inference."""
    locator = locator.to(device).eval(); pose_model = pose_model.to(device).eval()

    preloaded = []
    for row in rows:
        views = []
        for item in row["highres"]:
            array = np.asarray(Image.open(item["path"]).convert("RGB")).copy()
            views.append(torch.from_numpy(array).permute(2, 0, 1))
        preloaded.append(torch.stack(views))

    def run(row, full_cpu):
        full = full_cpu.to(device).float() / 255.0
        locator_input = F.interpolate(full, size=(256, 288), mode="bilinear", align_corners=False)
        points = locator(locator_input).reshape(3, 2, 2)
        centers = points.mean(1)
        half_x = crop_size / (576 - 1); half_y = crop_size / (512 - 1)
        gx = torch.linspace(-half_x, half_x, 128, device=device)
        gy = torch.linspace(-half_y, half_y, 128, device=device)
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")
        grids = []
        for center in centers:
            grids.append(torch.stack(((center[0] * 2 - 1) + xx,
                                      (center[1] * 2 - 1) + yy), dim=-1))
        crops = F.grid_sample(full, torch.stack(grids), mode="bilinear", padding_mode="zeros",
                              align_corners=True)
        pose_model(row["feature"].to(device).reshape(1, -1), crops.unsqueeze(0))

    with torch.no_grad():
        for _ in range(5): run(rows[0], preloaded[0])
        values = []
        for row, full in zip(rows, preloaded):
            if device.type == "cuda": torch.cuda.synchronize()
            start = time.perf_counter(); run(row, full)
            if device.type == "cuda": torch.cuda.synchronize()
            values.append((time.perf_counter() - start) * 1000)
    representation = np.asarray(values)
    trunk = np.asarray([float(row["model_inference_s"]) * 1000 for row in rows
                        if row.get("model_inference_s") is not None])
    result = {
        "sample_count": len(values),
        "locator_crop_pose_ms": {"p50": float(np.quantile(representation, .5)),
                                  "p95": float(np.quantile(representation, .95)),
                                  "p99": float(np.quantile(representation, .99))},
        "scope": "world trunk plus locator, pre-resize crop extraction, shared crop encoder, and pose head; sensor acquisition excluded",
    }
    if len(trunk) == len(representation):
        total = trunk + representation
        result["world_trunk_ms"] = {"p50": float(np.quantile(trunk, .5)),
                                    "p95": float(np.quantile(trunk, .95)),
                                    "p99": float(np.quantile(trunk, .99))}
        result["complete_inference_ms"] = {"p50": float(np.quantile(total, .5)),
                                           "p95": float(np.quantile(total, .95)),
                                           "p99": float(np.quantile(total, .99))}
        result["p95_below_300ms"] = bool(np.quantile(total, .95) < 300)
    else:
        result["complete_inference_unavailable_reason"] = (
            f"world trunk timing present for {len(trunk)}/{len(representation)} heldout rows"
        )
    return result


def main():
    args = parse_args(); device = torch.device(args.device); args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.scene_manifest.read_text()); calibration_indices = {3, 5, 8, 14}
    fit_ids = [x["episode_id"] for i, x in enumerate(manifest["train"]) if i not in calibration_indices]
    calibration_ids = [x["episode_id"] for i, x in enumerate(manifest["train"]) if i in calibration_indices]
    evaluation_ids = [x["episode_id"] for x in manifest["development"]]
    train_rows, train_audit = world_probe.load([args.train_replay], set(fit_ids + calibration_ids))
    evaluation, evaluation_audit = world_probe.load([args.evaluation_replay], set(evaluation_ids))
    train_rows, rejected_train = attach_highres(train_rows, args.train_replay)
    evaluation, rejected_evaluation = attach_highres(evaluation, args.evaluation_replay)
    fit = [r for r in train_rows if r["episode_id"] in fit_ids]
    calibration = [r for r in train_rows if r["episode_id"] in calibration_ids]
    if not fit or not calibration or not evaluation: raise RuntimeError("High-resolution split is empty")

    locator, locator_training = train_locator(fit, calibration, args, device)
    fit_points, locator_fit = locator_predict(locator, fit, device)
    calibration_points, locator_calibration = locator_predict(locator, calibration, device)
    evaluation_points, locator_evaluation = locator_predict(locator, evaluation, device)
    fit_crops = make_predicted_crops(fit, fit_points, args.crop_size)
    calibration_crops = make_predicted_crops(calibration, calibration_points, args.crop_size)
    evaluation_crops = make_predicted_crops(evaluation, evaluation_points, args.crop_size)
    models, mean, std, histories = train_crop_ensemble(
        fit, calibration, fit_crops, calibration_crops, args, device)
    cpred, cvar, cprob, cphase, _ = crop_predict(models, calibration, calibration_crops, mean, std, device)
    cy, _, _ = target_tensors(calibration); residual_variance = ((cpred - cy) ** 2).mean(0).numpy().clip(1e-6)
    pred, variance, probability, phase_probability, compute_latency = crop_predict(
        models, evaluation, evaluation_crops, mean, std, device)
    overall, raw = world_probe.metrics(evaluation, pred, probability, phase_probability, variance, residual_variance)
    phase_slices = world_probe.sliced(evaluation, pred, probability, phase_probability, variance,
                                      residual_variance, "phase")
    motion_slices = world_probe.sliced(evaluation, pred, probability, phase_probability, variance,
                                       residual_variance, "motion_bin")
    near = [i for i, row in enumerate(evaluation) if row["signed_depth_m"] >= -.003]
    near_metrics = world_probe.metrics([evaluation[i] for i in near], pred[near], probability[near],
                                       phase_probability[near], variance[near], residual_variance)[0]
    gate = bool(near_metrics["lateral_error_mm"]["median"] <= .25
                and near_metrics["lateral_error_mm"]["p95"] <= .5
                and (near_metrics["lateral_correction_sign_accuracy"] or 0) >= .9)
    latency = np.asarray(compute_latency)
    live_latency = benchmark_live_representation(locator, models[0], evaluation, device, args.crop_size)
    summary = {
        "schema_version": 1,
        "representation": "frozen 384D full-image feature plus three locator-selected native-resolution crops",
        "crop_selection": "learned RGB locator only; simulator instance masks used only as locator supervision",
        "native_resolution": [512, 576], "locator_input_resolution": [256, 288],
        "crop_native_pixels": args.crop_size, "encoded_crop_resolution": [128, 128],
        "split": {"unit": "complete reset configuration / episode_id", "fit_episode_ids": fit_ids,
                  "calibration_episode_ids": calibration_ids, "evaluation_episode_ids": evaluation_ids,
                  "fit_rows": len(fit), "calibration_rows": len(calibration), "evaluation_rows": len(evaluation)},
        "sources": {"train_replay": world_probe.file_id(args.train_replay),
                    "evaluation_replay": world_probe.file_id(args.evaluation_replay),
                    "scene_manifest": world_probe.file_id(args.scene_manifest)},
        "training_config": vars(args) | {"train_replay": str(args.train_replay),
                                          "evaluation_replay": str(args.evaluation_replay),
                                          "scene_manifest": str(args.scene_manifest),
                                          "output_dir": str(args.output_dir)},
        "audit": {"causal_train": train_audit, "causal_evaluation": evaluation_audit,
                  "highres_rejected_train": rejected_train, "highres_rejected_evaluation": rejected_evaluation},
        "locator": {"training": locator_training, "fit": locator_fit,
                    "calibration": locator_calibration, "evaluation": locator_evaluation},
        "pose_probe": {"overall": overall, "near_port": near_metrics,
                       "phase": phase_slices, "motion_size": motion_slices, "training": histories,
                       "calibration_residual_variance": residual_variance.tolist()},
        "gate": {"required": "near-port lateral median <=0.25 mm, p95 <=0.5 mm, sign >=0.9",
                 "passed": gate},
        "offline_crop_probe_compute_ms_per_row": {"mean": float(latency.mean()),
                                                    "max_batch_mean": float(latency.max())},
        "live_latency": live_latency,
        "parameter_counts": {"locator": sum(p.numel() for p in locator.parameters()),
                             "crop_pose_probe_each": sum(p.numel() for p in models[0].parameters())},
    }
    torch.save({"locator": locator.state_dict(), "pose_members": [m.state_dict() for m in models],
                "target_mean": mean, "target_std": std, "summary": summary}, args.output_dir / "crop_pose_probe.pt")
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode_id", "phase", "motion_bin", "signed_depth_mm", "lateral_mm",
                         "translation_error_mm", "axial_error_mm", "lateral_error_mm", "orientation_error_deg"])
        for i, row in enumerate(evaluation):
            writer.writerow([row["episode_id"], row["phase"], row["motion_bin"], row["signed_depth_m"]*1000,
                             row["lateral_m"]*1000, raw["translation_error"][i], raw["axial_error"][i],
                             raw["lateral_error"][i], raw["rotation_error"][i]])
    try:
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        axes[0].hist(raw["translation_error"], bins=30); axes[0].set_xlabel("3D translation error (mm)")
        axes[1].hist(raw["lateral_error"], bins=30); axes[1].axvline(.5, color="r"); axes[1].set_xlabel("lateral error (mm)")
        axes[2].hist(raw["rotation_error"], bins=30); axes[2].set_xlabel("orientation error (degrees)")
        figure.tight_layout(); figure.savefig(args.output_dir / "heldout_error_histograms.png", dpi=160); plt.close(figure)
    except Exception as exc:
        (args.output_dir / "plot_error.txt").write_text(repr(exc) + "\n")
    save_montage(evaluation, evaluation_points, args.output_dir / "heldout_locator_and_crops.png", args.crop_size)
    print(json.dumps({"output": str(args.output_dir), "gate": summary["gate"],
                      "locator": locator_evaluation, "overall": overall, "near_port": near_metrics}, indent=2))


if __name__ == "__main__":
    main()
