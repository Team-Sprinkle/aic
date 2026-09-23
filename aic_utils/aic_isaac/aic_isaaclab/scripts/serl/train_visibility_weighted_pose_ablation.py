#!/usr/bin/env python3
"""Train and evaluate observation-only visibility-weighted multiview pose fusion."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
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


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


opening = load_module("opening_visibility", "train_opening_landmark_pose_probe.py")
pretrained = load_module("pretrained_visibility", "calibrate_pretrained_opening_landmarks.py")
temporal = load_module("temporal_visibility", "train_temporal_multiview_pose_ablation.py")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--spatial-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--visibility-updates", type=int, default=2500)
    parser.add_argument("--landmark-updates", type=int, default=2000)
    parser.add_argument("--finetune-landmark", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--residual-updates", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--history", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


class VisibilityHead(nn.Module):
    def __init__(self, input_dim=60):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, value):
        return self.net(value)


class ResidualPose(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 128), nn.GELU(),
                                 nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 3))

    def forward(self, value):
        return self.net(value)


def attach_visibility(rows, replay_path):
    transitions = torch.load(replay_path, map_location="cpu", weights_only=False)["transitions"]
    accepted, rejected = [], Counter()
    for row in rows:
        labels = (transitions[row["transition_index"]].get("metadata") or {}).get("offline_visibility_labels")
        if not labels or any(camera not in labels for camera in opening.CAMERAS):
            rejected["missing_offline_visibility_labels"] += 1
            continue
        value = dict(row)
        value["visibility"] = torch.tensor([
            [float(labels[camera]["plug_visible"]), float(labels[camera]["opening_clear"])]
            for camera in opening.CAMERAS
        ], dtype=torch.float32)
        value["rope_opening_fraction"] = [float(labels[camera]["rope_opening_fraction"])
                                            for camera in opening.CAMERAS]
        accepted.append(value)
    return accepted, dict(rejected)


def landmark_predict_features(model, rows, coarse_points, crop_size, device):
    images, truth, geometry = opening.native_arrays(rows, coarse_points, crop_size, jitter=False)
    coordinates, features = [], []
    model = model.to(device).eval()
    with torch.no_grad():
        for start in range(0, len(images), 128):
            image = images[start:start + 128].to(device).float() / 255.0
            value = (image - model.mean) / model.std
            taps = {}
            for index, layer in enumerate(model.features):
                value = layer(value)
                if index in (1, 3, 8):
                    taps[index] = value
            fused = model.lateral40(taps[1])
            fused = fused + F.interpolate(model.lateral20(taps[3]), size=fused.shape[-2:], mode="bilinear", align_corners=False)
            fused = fused + F.interpolate(model.lateral10(taps[8]), size=fused.shape[-2:], mode="bilinear", align_corners=False)
            logits = model.head(fused)
            coordinates.append(opening.decode(logits).cpu())
            probability = torch.softmax(logits.flatten(-2), -1)
            maximum = probability.max(-1).values
            entropy = -(probability * probability.clamp_min(1e-9).log()).sum(-1) / math.log(probability.shape[-1])
            representation = taps[8].mean((-2, -1))
            features.append(torch.cat((representation, maximum, entropy), dim=1).cpu())
    local = torch.cat(coordinates)
    feature = torch.cat(features)
    output = torch.empty(len(rows), 3, 6, 2)
    errors = []
    for prediction, target, (row, view, left, top, width, height) in zip(local, truth, geometry):
        pixels = prediction * (crop_size - 1) + torch.tensor([left, top])
        output[row, view] = pixels / torch.tensor([width - 1, height - 1])
        errors.append((pixels - (target * (crop_size - 1) + torch.tensor([left, top]))).norm(dim=1))
    errors = torch.stack(errors)
    report = {
        name: {"median_px": float(errors[:, index].median()), "p95_px": float(torch.quantile(errors[:, index], .95))}
        for index, name in enumerate(("plug", "entrance", *opening.CORNERS))
    }
    report["view_count"] = len(errors)
    return output, feature.reshape(len(rows), 3, -1), report


def classification(truth, probability):
    predicted = probability >= 0.5
    output = {}
    for index, name in enumerate(("plug_visible", "opening_clear")):
        y = truth[:, :, index].reshape(-1).numpy().astype(bool)
        p = predicted[:, :, index].reshape(-1).numpy().astype(bool)
        tp = int(np.sum(y & p)); fp = int(np.sum(~y & p)); fn = int(np.sum(y & ~p))
        output[name] = {
            "count": len(y), "positive_fraction": float(y.mean()),
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "accuracy": float((y == p).mean()),
        }
    return output


def train_visibility(fit, calibration, fit_feature, cal_feature, args, device):
    x = fit_feature.reshape(-1, fit_feature.shape[-1])
    y = torch.stack([row["visibility"] for row in fit]).reshape(-1, 2)
    xc = cal_feature.reshape(-1, cal_feature.shape[-1]).to(device)
    yc = torch.stack([row["visibility"] for row in calibration]).reshape(-1, 2).to(device)
    episode_counts = Counter(row["episode_id"] for row in fit)
    weights = torch.tensor([1.0 / episode_counts[row["episode_id"]] for row in fit for _ in range(3)])
    positive = y.sum(0); negative = len(y) - positive
    positive_weight = (negative / positive.clamp_min(1)).to(device)
    model = VisibilityHead(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    best, best_loss, best_step, history = None, float("inf"), 0, []
    for step in range(1, args.visibility_updates + 1):
        index = torch.multinomial(weights, args.batch_size, replacement=True, generator=generator)
        loss = F.binary_cross_entropy_with_logits(model(x[index].to(device)), y[index].to(device),
                                                  pos_weight=positive_weight)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step == 1 or step % 100 == 0:
            model.eval()
            with torch.no_grad():
                validation = F.binary_cross_entropy_with_logits(model(xc), yc, pos_weight=positive_weight)
            value = float(validation)
            history.append({"step": step, "fit_loss": float(loss), "calibration_loss": value})
            if value < best_loss:
                best_loss, best_step, best = value, step, copy.deepcopy(model.state_dict())
            elif step - best_step >= args.patience:
                break
            model.train()
    model.load_state_dict(best)
    return model.cpu(), {"best_step": best_step, "completed_step": step,
                         "best_calibration_loss": best_loss, "history": history}


def visibility_predict(model, feature, device):
    model = model.to(device).eval(); output = []
    flat = feature.reshape(-1, feature.shape[-1])
    with torch.no_grad():
        for start in range(0, len(flat), 256):
            output.append(torch.sigmoid(model(flat[start:start + 256].to(device))).cpu())
    return torch.cat(output).reshape(feature.shape[0], 3, 2)


def weighted_triangulate(per_view, row, point_offset, weights):
    matrix = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for normalized, camera, weight in zip(per_view, opening.camera_row(row), weights):
        u = float(normalized[point_offset]) * (camera["width"] - 1)
        v = float(normalized[point_offset + 1]) * (camera["height"] - 1)
        intrinsic = camera["K"]
        ray_camera = np.asarray(((u-intrinsic[0, 2])/intrinsic[0, 0],
                                 (v-intrinsic[1, 2])/intrinsic[1, 1], 1.0))
        ray_world = opening.tri.qrot(camera["quat"], ray_camera)
        ray_world /= max(np.linalg.norm(ray_world), 1e-12)
        projection = np.eye(3) - np.outer(ray_world, ray_world)
        weight = max(0.03, float(weight))
        matrix += weight * projection
        rhs += weight * projection @ camera["position"]
    return np.linalg.solve(matrix + 1e-9 * np.eye(3), rhs)


def relative_weighted(points, rows, probability):
    result = []
    for value, row, weights in zip(points, rows, probability):
        plug = weighted_triangulate(value, row, 0, weights[:, 0])
        opening_point = weighted_triangulate(value, row, 2, weights[:, 1])
        result.append((plug - opening_point) * 1000.0)
    return np.asarray(result)


def affine_prediction(fit_raw, fit, rows_raw):
    target = np.stack([row["translation_mm"].numpy() for row in fit])
    affine = opening.tri.affine_fit(fit_raw, target)
    return np.c_[rows_raw, np.ones(len(rows_raw))] @ affine, affine


def residual_features(coarse, landmarks, raw, visibility):
    return torch.cat((coarse.reshape(len(coarse), -1),
                      landmarks[:, :, [0, 2, 3, 4, 5]].reshape(len(coarse), -1),
                      torch.tensor(raw, dtype=torch.float32), visibility.reshape(len(coarse), -1)), dim=1)


def train_residual(fit, calibration, xfit, xcal, raw_fit, raw_cal, args, device, *, temporal_window=False):
    if temporal_window:
        xfit = temporal.causal_windows(fit, xfit, args.history)
        xcal = temporal.causal_windows(calibration, xcal, args.history)
    target_fit = torch.stack([row["translation_mm"] for row in fit]) - torch.tensor(raw_fit, dtype=torch.float32)
    target_cal = torch.stack([row["translation_mm"] for row in calibration]) - torch.tensor(raw_cal, dtype=torch.float32)
    xm, xs = xfit.mean(0), xfit.std(0).clamp_min(1e-5)
    ym, ys = target_fit.mean(0), target_fit.std(0).clamp_min(0.05)
    counts = Counter(row["episode_id"] for row in fit)
    weights = torch.tensor([1.0 / counts[row["episode_id"]] for row in fit])
    members, histories = [], []
    for member in range(args.ensemble_size):
        seed = args.seed + 100 + member
        torch.manual_seed(seed); random.seed(seed)
        model = ResidualPose(xfit.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(seed)
        best, best_score, best_step, history = None, float("inf"), 0, []
        for step in range(1, args.residual_updates + 1):
            index = torch.multinomial(weights, args.batch_size, replacement=True, generator=generator)
            output = model(((xfit[index] - xm) / xs).to(device))
            loss = F.smooth_l1_loss(output, ((target_fit[index] - ym) / ys).to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if step == 1 or step % 100 == 0:
                model.eval()
                with torch.no_grad():
                    residual = model(((xcal - xm) / xs).to(device)).cpu() * ys + ym
                    prediction = torch.tensor(raw_cal) + residual
                metric = temporal.score(calibration, prediction.numpy())
                score = metric["lateral_error_mm"]["median"] + metric["lateral_error_mm"]["p95"]
                history.append({"step": step, "fit_loss": float(loss), "selection_score": score})
                if score < best_score:
                    best_score, best_step, best = score, step, copy.deepcopy(model.state_dict())
                elif step - best_step >= args.patience:
                    break
                model.train()
        model.load_state_dict(best); members.append(model.cpu())
        histories.append({"member": member, "best_step": best_step, "completed_step": step,
                          "best_selection_score": best_score, "history": history})
    return members, {"input_mean": xm, "input_std": xs, "target_mean": ym, "target_std": ys,
                     "histories": histories, "temporal_window": temporal_window}


def residual_predict(models, stats, rows, feature, raw, args, device):
    if stats["temporal_window"]:
        feature = temporal.causal_windows(rows, feature, args.history)
    normalized = (feature - stats["input_mean"]) / stats["input_std"]
    values = []
    with torch.no_grad():
        for model in models:
            residual = model.to(device).eval()(normalized.to(device)).cpu()
            values.append(torch.tensor(raw) + residual * stats["target_std"] + stats["target_mean"])
    stack = torch.stack(values)
    return stack.mean(0).numpy(), stack.var(0, unbiased=False).numpy()


def report(rows, prediction, variance=None):
    if variance is None:
        variance = np.zeros_like(prediction)
    return temporal.model_report(rows, prediction, variance)


def save_overlay(rows, landmarks, visibility, output):
    chosen = np.linspace(0, len(rows) - 1, min(10, len(rows)), dtype=int)
    canvas = Image.new("RGB", (864, 256 * len(chosen)))
    for line, row_index in enumerate(chosen):
        for view_index, view in enumerate(rows[row_index]["highres"]):
            image = Image.open(view["path"]).convert("RGB")
            draw = ImageDraw.Draw(image)
            scale = np.asarray([view["width"] - 1, view["height"] - 1])
            points = landmarks[row_index, view_index].numpy() * scale
            for point_index, (x, y) in enumerate(points):
                color = "yellow" if point_index == 0 else "cyan"
                draw.ellipse((x-3, y-3, x+3, y+3), outline=color, width=2)
            draw.rectangle((4, 4, 220, 25), fill="black")
            draw.text((8, 8), f"pred vis plug={visibility[row_index,view_index,0]:.2f} opening={visibility[row_index,view_index,1]:.2f}", fill="white")
            canvas.paste(image.resize((288, 256)), (288 * view_index, 256 * line))
    canvas.save(output, quality=92)


def main():
    args = arguments(); args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    manifest = json.loads(args.dataset_manifest.read_text())
    split_ids = {key: set(value["episode_ids"]) for key, value in manifest["splits"].items()}
    allowed = set().union(*split_ids.values())
    rows, load_audit = opening.world.load([args.replay], allowed)
    rows, image_rejected = opening.attach(rows, args.replay)
    rows, visibility_rejected = attach_visibility(rows, args.replay)
    fit = [row for row in rows if row["episode_id"] in split_ids["fit"]]
    calibration = [row for row in rows if row["episode_id"] in split_ids["calibration"]]
    development = [row for row in rows if row["episode_id"] in split_ids["development"]]
    if not fit or not calibration or not development:
        raise RuntimeError(f"empty split: fit={len(fit)} calibration={len(calibration)} development={len(development)}")

    checkpoint = torch.load(args.spatial_checkpoint, map_location="cpu", weights_only=False)
    locator = opening.crop.Locator(); locator.load_state_dict(checkpoint["locator"])
    landmark = pretrained.MobileNetLandmarks(); landmark.load_state_dict(checkpoint["landmark"])
    coarse = {}
    for name, split in (("fit", fit), ("calibration", calibration), ("development", development)):
        coarse[name], _ = opening.crop.locator_predict(locator, opening.coarse_rows(split), device)
    landmark_training = None
    if args.finetune_landmark:
        # Reuse exactly the existing ImageNet MobileNetV3-small FPN architecture;
        # only its weights and the cable-varied fit/calibration data change.
        args.lr = args.learning_rate
        landmark, landmark_training = pretrained.train(
            landmark, fit, calibration, coarse["fit"], coarse["calibration"], args, device
        )
    landmark_points = {}
    representations = {}
    landmark_reports = {}
    for name, split in (("fit", fit), ("calibration", calibration), ("development", development)):
        landmark_points[name], representations[name], landmark_reports[name] = landmark_predict_features(
            landmark, split, coarse[name], args.crop_size, device)

    visibility_model, visibility_training = train_visibility(
        fit, calibration, representations["fit"], representations["calibration"], args, device)
    probabilities = {name: visibility_predict(visibility_model, representations[name], device)
                     for name in representations}
    visibility_metrics = {
        name: classification(torch.stack([row["visibility"] for row in split]), probabilities[name])
        for name, split in (("calibration", calibration), ("development", development))
    }

    selected = checkpoint["summary"]["selection"]["selected"]
    points = {
        name: opening.pair_points(coarse[name], landmark_points[name], selected["plug"], selected["port"])
        for name in coarse
    }
    ones = {name: torch.ones(len(split), 3, 2) for name, split in
            (("fit", fit), ("calibration", calibration), ("development", development))}
    oracle = {name: torch.stack([row["visibility"] for row in split]).clamp_min(0.03) for name, split in
              (("fit", fit), ("calibration", calibration), ("development", development))}

    raw = {}
    for kind, weight_map in (("equal", ones), ("predicted_visibility", probabilities), ("oracle_visibility", oracle)):
        raw[kind] = {
            "fit": relative_weighted(points["fit"], fit, weight_map["fit"]),
            "calibration": relative_weighted(points["calibration"], calibration, weight_map["calibration"]),
            "development": relative_weighted(points["development"], development, weight_map["development"]),
        }

    predictions = {}; affines = {}
    for kind in raw:
        predictions[kind] = {}
        for split_name in ("calibration", "development"):
            predictions[kind][split_name], affines[kind] = affine_prediction(
                raw[kind]["fit"], fit, raw[kind][split_name])

    fit_feature = residual_features(coarse["fit"], landmark_points["fit"],
                                    predictions["predicted_visibility"].get("fit", raw["predicted_visibility"]["fit"]),
                                    probabilities["fit"])
    # Residuals operate after the affine convention mapping.
    fit_affine_raw = np.c_[raw["predicted_visibility"]["fit"], np.ones(len(fit))] @ affines["predicted_visibility"]
    cal_affine_raw = predictions["predicted_visibility"]["calibration"]
    dev_affine_raw = predictions["predicted_visibility"]["development"]
    fit_feature = residual_features(coarse["fit"], landmark_points["fit"], fit_affine_raw, probabilities["fit"])
    cal_feature = residual_features(coarse["calibration"], landmark_points["calibration"], cal_affine_raw,
                                    probabilities["calibration"])
    dev_feature = residual_features(coarse["development"], landmark_points["development"], dev_affine_raw,
                                    probabilities["development"])
    current_models, current_stats = train_residual(fit, calibration, fit_feature, cal_feature,
                                                    fit_affine_raw, cal_affine_raw, args, device)
    temporal_models, temporal_stats = train_residual(fit, calibration, fit_feature, cal_feature,
                                                      fit_affine_raw, cal_affine_raw, args, device,
                                                      temporal_window=True)
    current_prediction, current_variance = residual_predict(current_models, current_stats, development,
                                                             dev_feature, dev_affine_raw, args, device)
    temporal_prediction, temporal_variance = residual_predict(temporal_models, temporal_stats, development,
                                                               dev_feature, dev_affine_raw, args, device)
    current_cal_prediction, current_cal_variance = residual_predict(
        current_models, current_stats, calibration, cal_feature, cal_affine_raw, args, device
    )
    temporal_cal_prediction, temporal_cal_variance = residual_predict(
        temporal_models, temporal_stats, calibration, cal_feature, cal_affine_raw, args, device
    )

    models = {
        "equal_view_triangulation": report(development, predictions["equal"]["development"]),
        "predicted_visibility_weighted": report(development, dev_affine_raw),
        "oracle_visibility_upper_bound": report(development, predictions["oracle_visibility"]["development"]),
        "visibility_weighted_current_residual": report(development, current_prediction, current_variance),
        "visibility_weighted_causal_temporal": report(development, temporal_prediction, temporal_variance),
    }
    calibration_models = {
        "predicted_visibility_weighted": report(calibration, cal_affine_raw),
        "visibility_weighted_current_residual": report(calibration, current_cal_prediction, current_cal_variance),
        "visibility_weighted_causal_temporal": report(calibration, temporal_cal_prediction, temporal_cal_variance),
    }
    selected_name = min(
        ("predicted_visibility_weighted", "visibility_weighted_current_residual", "visibility_weighted_causal_temporal"),
        key=lambda name: calibration_models[name]["near_port"]["lateral_error_mm"]["median"]
                         + calibration_models[name]["near_port"]["lateral_error_mm"]["p95"],
    )
    gate = models[selected_name]["gate"]
    gate["passed"] = all(gate.values())

    save_overlay(development, landmark_points["development"], probabilities["development"],
                 args.output_dir / "development_predicted_visibility.jpg")
    saved = {
        "locator": locator.state_dict(),
        "landmark": landmark.state_dict(),
        "visibility_head": visibility_model.state_dict(),
        "visibility_input_dim": representations["fit"].shape[-1],
        "selected_landmarks": selected,
        "affines": affines,
        "current": {**current_stats, "state_dicts": [model.state_dict() for model in current_models]},
        "temporal": {**temporal_stats, "state_dicts": [model.state_dict() for model in temporal_models]},
    }
    torch.save(saved, args.output_dir / "visibility_ablation_checkpoint.pt")
    report_payload = {
        "schema_version": 1,
        "status": "development_complete",
        "reserved_final_opened": False,
        "split": {
            name: {"episode_count": len({row["episode_id"] for row in split}),
                   "sequence_count": len({row["sequence_id"] for row in split}), "rows": len(split)}
            for name, split in (("fit", fit), ("calibration", calibration), ("development", development))
        },
        "audit": {"load": load_audit, "image_rejected": image_rejected,
                  "visibility_rejected": visibility_rejected},
        "visibility": {"training": visibility_training, "metrics": visibility_metrics,
                       "labels_used_only_offline": True},
        "landmarks": {"finetuned": bool(args.finetune_landmark), "training": landmark_training,
                      "splits": landmark_reports},
        "models": models,
        "calibration_models": calibration_models,
        "selection": {"rule": "calibration-only training and early stopping; report best deployable development model",
                      "selected_on_calibration": selected_name},
        "gate": gate,
        "policy_training_allowed": bool(gate["passed"]),
        "limitations": [
            "The eight cable templates were sampled from one earlier trajectory and are not a broad cable-shape distribution.",
            "Physics settling rejected many transitions and the held-out split contains only cable templates 6 and 7.",
            "Oracle visibility is an analysis upper bound; only RGB-predicted visibility is deployable.",
        ],
        "sources": {"replay": opening.world.file_id(args.replay),
                    "dataset_manifest": opening.world.file_id(args.dataset_manifest),
                    "spatial_checkpoint": opening.world.file_id(args.spatial_checkpoint)},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report_payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output_dir), "selected": selected_name,
                      "gate": gate, "development": models[selected_name]["near_port"],
                      "visibility": visibility_metrics["development"]}, indent=2))


if __name__ == "__main__":
    main()
