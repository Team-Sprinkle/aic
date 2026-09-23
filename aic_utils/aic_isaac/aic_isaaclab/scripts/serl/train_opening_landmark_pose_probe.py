#!/usr/bin/env python3
"""Train explicit opening landmarks with causal temporal smoothing and triangulation."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


world = load_module("world_probe_landmarks", HERE / "train_world_feature_pose_probe.py")
crop = load_module("crop_probe_landmarks", HERE / "train_highres_crop_pose_probe.py")
tri = load_module("tri_probe_landmarks", HERE / "train_triangulated_pose_probe.py")
CAMERAS = ("center_camera", "left_camera", "right_camera")
CORNERS = ("top_left", "top_right", "bottom_right", "bottom_left")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-replay", type=Path, required=True)
    p.add_argument("--evaluation-replay", type=Path, required=True)
    p.add_argument("--scene-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--locator-updates", type=int, default=5000)
    p.add_argument("--landmark-updates", type=int, default=5000)
    p.add_argument("--updates", type=int, default=4000)
    p.add_argument("--patience", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--ensemble-size", type=int, default=3)
    p.add_argument("--crop-size", type=int, default=160)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def attach(rows, replay_path):
    transitions = torch.load(replay_path, map_location="cpu", weights_only=False)["transitions"]
    sequence_by_transition = {}
    sequence_index = -1
    previous_episode = None
    previous_terminal = True
    for transition_index, transition in enumerate(transitions):
        metadata = transition.get("metadata") or {}
        episode = (metadata.get("causal_episode") or {}).get("episode_id")
        if previous_terminal or episode != previous_episode:
            sequence_index += 1
        sequence_by_transition[transition_index] = f"reset-{sequence_index:06d}:{episode}"
        previous_episode = episode
        previous_terminal = bool(metadata.get("terminated") or metadata.get("truncated"))
    accepted, rejected = [], Counter()
    for row in rows:
        item = transitions[row["transition_index"]]
        metadata = item.get("metadata") or {}
        cameras = ((metadata.get("highres_observation") or {}).get("cameras") or {})
        parsed = []
        for camera in CAMERAS:
            view = cameras.get(camera) or {}; labels = view.get("locator_supervision_xy") or {}
            corners = view.get("opening_corner_supervision_xy") or {}
            if (not view.get("path") or labels.get("plug") is None or labels.get("entrance") is None
                    or any(corners.get(name) is None for name in CORNERS)
                    or view.get("camera_position_world") is None
                    or view.get("camera_orientation_wxyz_ros") is None):
                break
            parsed.append({"path": view["path"], "width": view["width"], "height": view["height"],
                           "plug": labels["plug"], "entrance": labels["entrance"],
                           "corners": [corners[name] for name in CORNERS],
                           "intrinsic_matrix": view["intrinsic_matrix"],
                           "camera_position_world": view["camera_position_world"],
                           "camera_orientation_wxyz_ros": view["camera_orientation_wxyz_ros"]})
        if len(parsed) != 3:
            rejected["missing_image_landmark_or_calibration"] += 1
            continue
        row = dict(row)
        row["highres"] = parsed
        # ``episode_id`` names a reset configuration and can recur when a
        # collector cycles through its manifest.  Temporal state must reset at
        # every simulator episode rather than flow into the next occurrence of
        # the same configuration.
        global_episode_index = metadata.get("global_episode_index")
        if global_episode_index is None:
            global_episode_index = (metadata.get("causal_episode") or {}).get("global_episode_index")
        row["global_episode_index"] = global_episode_index
        row["sequence_id"] = (f"global-{global_episode_index}:{row['episode_id']}"
                              if global_episode_index is not None
                              else sequence_by_transition[row["transition_index"]])
        accepted.append(row)
    return accepted, dict(rejected)


def coarse_rows(rows):
    out = []
    for row in rows:
        row = dict(row); views = []
        for view in row["highres"]:
            item = dict(view); item["target"] = item["entrance"]; views.append(item)
        row["highres"] = views; out.append(row)
    return out


class LandmarkHeatmaps(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 96, 3, 1, 1), nn.GELU(),
            nn.Conv2d(96, 96, 3, 1, 1), nn.GELU(),
            nn.Conv2d(96, 6, 1),
        )

    def forward(self, image):
        return self.net(image)


def native_arrays(rows, coarse_points, crop_size, *, jitter=False, seed=0):
    images, coordinates, geometry = [], [], []
    rng = random.Random(seed)
    for row_index, (row, predicted_views) in enumerate(zip(rows, coarse_points)):
        for view_index, (view, prediction) in enumerate(zip(row["highres"], predicted_views)):
            if jitter:
                truth_center = np.mean(np.asarray([view["plug"], view["entrance"]]), axis=0)
                center = truth_center + np.asarray([rng.uniform(-20, 20), rng.uniform(-20, 20)])
            else:
                scale = torch.tensor([view["width"] - 1, view["height"] - 1] * 2)
                center = (prediction * scale).reshape(2, 2).mean(0).numpy()
            left = int(round(float(center[0]) - crop_size / 2)); top = int(round(float(center[1]) - crop_size / 2))
            image = Image.open(view["path"]).convert("RGB").crop((left, top, left + crop_size, top + crop_size))
            images.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).to(torch.uint8))
            points = np.asarray([view["plug"], view["entrance"], *view["corners"]], np.float32)
            local = (points - np.asarray([left, top], np.float32)) / (crop_size - 1)
            coordinates.append(torch.from_numpy(local))
            geometry.append((row_index, view_index, left, top, view["width"], view["height"]))
    return torch.stack(images), torch.stack(coordinates), geometry


def heatmap_loss(logits, coordinate, sigma=1.5):
    height, width = logits.shape[-2:]
    y, x = torch.meshgrid(torch.arange(height, device=logits.device),
                          torch.arange(width, device=logits.device), indexing="ij")
    target_x = coordinate[:, :, 0, None, None] * (width - 1)
    target_y = coordinate[:, :, 1, None, None] * (height - 1)
    target = torch.exp(-((x - target_x).square() + (y - target_y).square()) / (2 * sigma**2))
    target = target / target.sum((-2, -1), keepdim=True).clamp_min(1e-8)
    return -(target * F.log_softmax(logits.flatten(-2), -1).reshape_as(logits)).sum((-2, -1)).mean()


def decode(logits):
    batch, points, height, width = logits.shape
    probability = torch.softmax(logits.flatten(-2), -1)
    y, x = torch.meshgrid(torch.linspace(0, 1, height, device=logits.device),
                          torch.linspace(0, 1, width, device=logits.device), indexing="ij")
    return torch.stack(((probability * x.flatten()).sum(-1),
                        (probability * y.flatten()).sum(-1)), -1)


def train_landmarks(fit, calibration, fit_coarse, cal_coarse, a, device):
    x, y, _ = native_arrays(fit, fit_coarse, a.crop_size, jitter=True, seed=a.seed)
    xc, yc, _ = native_arrays(calibration, cal_coarse, a.crop_size, jitter=False)
    model = LandmarkHeatmaps().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(a.seed); torch.manual_seed(a.seed)
    best, best_loss, best_step, history = None, float("inf"), 0, []
    for step in range(1, a.landmark_updates + 1):
        index = torch.randint(len(x), (a.batch_size,), generator=generator)
        loss = heatmap_loss(model(x[index].to(device).float() / 255), y[index].to(device))
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step == 1 or step % 100 == 0:
            with torch.no_grad():
                values = [heatmap_loss(model(xc[s:s+128].to(device).float()/255), yc[s:s+128].to(device))
                          for s in range(0, len(xc), 128)]
                value = float(torch.stack(values).mean())
            history.append({"step": step, "fit_loss": float(loss), "calibration_loss": value})
            if value < best_loss:
                best_loss, best_step, best = value, step, copy.deepcopy(model.state_dict())
            elif step - best_step >= a.patience:
                break
    model.load_state_dict(best)
    return model.cpu(), {"best_step": best_step, "completed_step": step,
                         "best_calibration_loss": best_loss, "history": history}


def landmark_predict(model, rows, coarse_points, crop_size, device):
    x, truth, geometry = native_arrays(rows, coarse_points, crop_size, jitter=False)
    model = model.to(device).eval(); values = []
    with torch.no_grad():
        for start in range(0, len(x), 128):
            values.append(decode(model(x[start:start+128].to(device).float()/255)).cpu())
    local = torch.cat(values); output = torch.empty(len(rows), 3, 6, 2); errors = []
    for prediction, target, (row, view, left, top, width, height) in zip(local, truth, geometry):
        pixels = prediction * (crop_size - 1) + torch.tensor([left, top])
        output[row, view] = pixels / torch.tensor([width - 1, height - 1])
        errors.append((pixels - (target * (crop_size - 1) + torch.tensor([left, top]))).norm(dim=1))
    errors = torch.stack(errors)
    names = ("plug", "entrance", *CORNERS)
    report = {name: {"median_px": float(errors[:, i].median()),
                     "p95_px": float(torch.quantile(errors[:, i], .95))} for i, name in enumerate(names)}
    report.update({"view_count": len(errors), "episode_count": len(set(r["episode_id"] for r in rows))})
    return output, report


def smooth_port(points, rows, alpha):
    result = points.clone(); previous = {}
    for index, row in enumerate(rows):
        episode = row.get("sequence_id", row["episode_id"])
        value = points[index, :, 2:4].clone()
        if episode in previous:
            value = alpha * value + (1 - alpha) * previous[episode]
        previous[episode] = value; result[index, :, 2:4] = value
    return result


def camera_row(row):
    return [{"K": np.asarray(v["intrinsic_matrix"], np.float64),
             "position": np.asarray(v["camera_position_world"], np.float64),
             "quat": np.asarray(v["camera_orientation_wxyz_ros"], np.float64),
             "width": v["width"], "height": v["height"]} for v in row["highres"]]


def relative_world(points, rows):
    values = []
    for point, row in zip(points, rows):
        calibration = camera_row(row)
        plug = tri.triangulate_one(point, calibration, 0)
        opening = tri.triangulate_one(point, calibration, 2)
        values.append((plug - opening) * 1000)
    return np.asarray(values)


def pair_points(coarse_points, landmarks, plug_source, port_source):
    result = torch.empty(len(landmarks), 3, 4)
    result[:, :, :2] = coarse_points[:, :, :2] if plug_source == "coarse" else landmarks[:, :, 0]
    if port_source == "coarse": result[:, :, 2:] = coarse_points[:, :, 2:]
    elif port_source == "entrance": result[:, :, 2:] = landmarks[:, :, 1]
    else: result[:, :, 2:] = landmarks[:, :, 2:6].mean(2)
    return result


def save_montage(rows, predictions, output):
    chosen = np.linspace(0, len(rows) - 1, min(12, len(rows)), dtype=int)
    strips = []
    for index in chosen:
        views = []
        for view_index, view in enumerate(rows[index]["highres"]):
            image = Image.open(view["path"]).convert("RGB"); draw = ImageDraw.Draw(image)
            scale = np.asarray([view["width"] - 1, view["height"] - 1])
            predicted = predictions[index, view_index].numpy() * scale
            truth = np.asarray([view["plug"], view["entrance"], *view["corners"]])
            for point in truth:
                x, y = point; draw.ellipse((x-3, y-3, x+3, y+3), outline="lime", width=2)
            for point in predicted:
                x, y = point; draw.ellipse((x-3, y-3, x+3, y+3), outline="red", width=2)
            draw.line([*truth[2:].tolist(), truth[2].tolist()], fill="cyan", width=2)
            draw.line([*predicted[2:].tolist(), predicted[2].tolist()], fill="orange", width=2)
            views.append(image.resize((288, 256)))
        strip = Image.new("RGB", (864, 256))
        for view_index, image in enumerate(views): strip.paste(image, (288*view_index, 0))
        strips.append(strip)
    sheet = Image.new("RGB", (864, 256*len(strips)))
    for index, strip in enumerate(strips): sheet.paste(strip, (0, 256*index))
    sheet.save(output)


def main():
    a = arguments(); device = torch.device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(a.scene_manifest.read_text()); calibration_indices = {3, 7, 12, 16}
    fit_ids = [x["episode_id"] for i, x in enumerate(manifest["train"]) if i not in calibration_indices]
    cal_ids = [x["episode_id"] for i, x in enumerate(manifest["train"]) if i in calibration_indices]
    eval_ids = [x["episode_id"] for x in manifest["development"]]
    train, train_audit = world.load([a.train_replay], set(fit_ids + cal_ids)); evaluation, eval_audit = world.load([a.evaluation_replay], set(eval_ids))
    train, train_reject = attach(train, a.train_replay); evaluation, eval_reject = attach(evaluation, a.evaluation_replay)
    fit = [r for r in train if r["episode_id"] in fit_ids]; calibration = [r for r in train if r["episode_id"] in cal_ids]
    if not fit or not calibration or not evaluation: raise RuntimeError("empty fit/calibration/evaluation split")
    fit_c, cal_c, eval_c = coarse_rows(fit), coarse_rows(calibration), coarse_rows(evaluation)
    locator, locator_training = crop.train_locator(fit_c, cal_c, a, device)
    fit_coarse, locator_fit = crop.locator_predict(locator, fit_c, device)
    cal_coarse, locator_cal = crop.locator_predict(locator, cal_c, device)
    eval_coarse, locator_eval = crop.locator_predict(locator, eval_c, device)
    landmark, landmark_training = train_landmarks(fit, calibration, fit_coarse, cal_coarse, a, device)
    fit_land, landmark_fit = landmark_predict(landmark, fit, fit_coarse, a.crop_size, device)
    cal_land, landmark_cal = landmark_predict(landmark, calibration, cal_coarse, a.crop_size, device)
    eval_land, landmark_eval = landmark_predict(landmark, evaluation, eval_coarse, a.crop_size, device)
    save_montage(evaluation, eval_land, a.output_dir / "heldout_opening_landmarks.png")

    models, mean, std, histories = world.train_ensemble(fit, calibration, "feature", a, device)
    cpred, _, _, _ = world.predict(models, calibration, "feature", mean, std, device)
    cy = torch.stack([torch.cat((r["translation_mm"], r["rotation_deg"])) for r in calibration])
    residual = ((cpred - cy)**2).mean(0).numpy().clip(1e-6)
    base, variance, probability, phase = world.predict(models, evaluation, "feature", mean, std, device)

    target_fit = np.stack([r["translation_mm"].numpy() for r in fit]); candidates = []
    for plug_source in ("coarse", "landmark"):
        for port_source in ("coarse", "entrance", "corners"):
            for alpha in (1.0, 0.5, 0.25, 0.1):
                fp = smooth_port(pair_points(fit_coarse, fit_land, plug_source, port_source), fit, alpha)
                cp = smooth_port(pair_points(cal_coarse, cal_land, plug_source, port_source), calibration, alpha)
                affine = tri.affine_fit(relative_world(fp, fit), target_fit)
                translation = np.c_[relative_world(cp, calibration), np.ones(len(calibration))] @ affine
                prediction = cpred.clone(); prediction[:, :3] = torch.tensor(translation, dtype=torch.float32)
                near = [i for i, r in enumerate(calibration) if r["signed_depth_m"] >= -.003]
                metric = world.metrics([calibration[i] for i in near], prediction[near],
                                       torch.zeros(len(near), 2), torch.full((len(near), 3), 1/3),
                                       torch.zeros(len(near), 6), residual)[0]
                score = metric["lateral_error_mm"]["median"] + metric["lateral_error_mm"]["p95"]
                candidates.append({"plug": plug_source, "port": port_source, "alpha": alpha,
                                   "score": score, "metrics": metric, "affine": affine})
    selected = min(candidates, key=lambda x: x["score"])
    ep = smooth_port(pair_points(eval_coarse, eval_land, selected["plug"], selected["port"]), evaluation, selected["alpha"])
    translated = np.c_[relative_world(ep, evaluation), np.ones(len(evaluation))] @ selected["affine"]
    prediction = base.clone(); prediction[:, :3] = torch.tensor(translated, dtype=torch.float32)
    overall = world.metrics(evaluation, prediction, probability, phase, variance, residual)[0]
    near = [i for i, r in enumerate(evaluation) if r["signed_depth_m"] >= -.003]
    near_metrics = world.metrics([evaluation[i] for i in near], prediction[near], probability[near], phase[near], variance[near], residual)[0]

    true_points = torch.stack([torch.stack([torch.tensor([*v["plug"], *v["entrance"]]) /
                              torch.tensor([v["width"]-1, v["height"]-1]*2) for v in r["highres"]]) for r in evaluation])
    true_fit = torch.stack([torch.stack([torch.tensor([*v["plug"], *v["entrance"]]) /
                            torch.tensor([v["width"]-1, v["height"]-1]*2) for v in r["highres"]]) for r in fit])
    oracle_affine = tri.affine_fit(relative_world(true_fit, fit), target_fit)
    oracle_t = np.c_[relative_world(true_points, evaluation), np.ones(len(evaluation))] @ oracle_affine
    oracle_pred = base.clone(); oracle_pred[:, :3] = torch.tensor(oracle_t, dtype=torch.float32)
    oracle = world.metrics([evaluation[i] for i in near], oracle_pred[near], probability[near], phase[near], variance[near], residual)[0]
    passed = bool(near_metrics["lateral_error_mm"]["median"] <= .25 and near_metrics["lateral_error_mm"]["p95"] <= .5 and (near_metrics["lateral_correction_sign_accuracy"] or 0) >= .9)
    summary = {
        "schema_version": 1, "status": "perception_gate_passed" if passed else "perception_gate_failed",
        "split": {"unit": "complete reset configuration / episode_id", "fit_episode_ids": fit_ids,
                  "calibration_episode_ids": cal_ids, "evaluation_episode_ids": eval_ids,
                  "fit_rows": len(fit), "calibration_rows": len(calibration), "evaluation_rows": len(evaluation)},
        "audit": {"train": train_audit, "evaluation": eval_audit, "train_rejected": train_reject, "evaluation_rejected": eval_reject},
        "locator": {"training": locator_training, "fit": locator_fit, "calibration": locator_cal, "evaluation": locator_eval},
        "landmarks": {"training": landmark_training, "fit": landmark_fit, "calibration": landmark_cal, "evaluation": landmark_eval},
        "selection": {"rule": "minimum calibration near-port lateral median+p95", "selected": {k: selected[k] for k in ("plug", "port", "alpha", "score")},
                      "candidates": [{k: c[k] for k in ("plug", "port", "alpha", "score", "metrics")} for c in candidates]},
        "heldout": {"overall": overall, "near_port": near_metrics}, "oracle_near_port": oracle,
        "gate": {"required": "near-port lateral median <=0.25 mm, p95 <=0.5 mm, sign >=0.9", "passed": passed},
        "training": {"opening_dimensions_m": [0.014, 0.0089495], "crop_size": a.crop_size,
                     "landmark_channels": ["plug", "entrance", *CORNERS], "feature_auxiliary": histories},
        "sources": {"train_replay": world.file_id(a.train_replay), "evaluation_replay": world.file_id(a.evaluation_replay),
                    "scene_manifest": world.file_id(a.scene_manifest)},
    }
    torch.save({"locator": locator.state_dict(), "landmark": landmark.state_dict(),
                "selection": summary["selection"]["selected"], "translation_affine": selected["affine"],
                "feature_members": [m.state_dict() for m in models], "target_mean": mean, "target_std": std},
               a.output_dir / "opening_landmark_pose_probe.pt")
    (a.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(a.output_dir), "gate": summary["gate"],
                      "selected": summary["selection"]["selected"], "near_port": near_metrics,
                      "oracle_near_port": oracle}, indent=2))


if __name__ == "__main__":
    main()
