#!/usr/bin/env python3
"""Train an RGB keypoint locator and triangulate plug-to-opening translation."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(out)
    return out


world = module("world_probe", HERE / "train_world_feature_pose_probe.py")
crop = module("crop_probe", HERE / "train_highres_crop_pose_probe.py")


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-replay", type=Path, required=True)
    p.add_argument("--evaluation-replay", type=Path, required=True)
    p.add_argument("--scene-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--locator-updates", type=int, default=5000)
    p.add_argument("--refiner-updates", type=int, default=4000)
    p.add_argument("--crop-size", type=int, default=160)
    p.add_argument("--updates", type=int, default=4000)
    p.add_argument("--patience", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--ensemble-size", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def true_points(rows):
    result = []
    for row in rows:
        views = []
        for item in row["highres"]:
            scale = torch.tensor([item["width"] - 1, item["height"] - 1] * 2)
            views.append(torch.tensor([*item["plug"], *item["target"]]) / scale)
        result.append(torch.stack(views))
    return torch.stack(result)


class KeypointRefiner(nn.Module):
    """Shared native-crop heatmap refiner with differentiable subpixel output."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2, 2), nn.GELU(),
            nn.Conv2d(24, 48, 3, 2, 1), nn.GELU(),
            nn.Conv2d(48, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.GELU(),
            nn.Conv2d(64, 2, 1),
        )

    def forward(self, image):
        heatmap = self.features(image)
        batch, points, height, width = heatmap.shape
        probability = torch.softmax(heatmap.reshape(batch, points, -1) * 4.0, dim=-1)
        y, x = torch.meshgrid(
            torch.linspace(0, 1, height, device=image.device),
            torch.linspace(0, 1, width, device=image.device), indexing="ij")
        x = x.reshape(-1); y = y.reshape(-1)
        return torch.stack(((probability * x).sum(-1), (probability * y).sum(-1)), dim=-1).reshape(batch, 4)


def refinement_arrays(rows, coarse, crop_size):
    images, labels, geometry = [], [], []
    truth = true_points(rows)
    for row_index, (row, per_view) in enumerate(zip(rows, coarse)):
        for view_index, (item, points) in enumerate(zip(row["highres"], per_view)):
            scale = torch.tensor([item["width"] - 1, item["height"] - 1] * 2)
            pixel = points * scale
            center = pixel.reshape(2, 2).mean(0)
            left = int(round(float(center[0]) - crop_size / 2))
            top = int(round(float(center[1]) - crop_size / 2))
            image = Image.open(item["path"]).convert("RGB").crop(
                (left, top, left + crop_size, top + crop_size))
            images.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).to(torch.uint8))
            true_pixel = truth[row_index, view_index] * scale
            local = true_pixel.reshape(2, 2) - torch.tensor([left, top])
            labels.append((local / max(crop_size - 1, 1)).reshape(-1))
            geometry.append((row_index, view_index, left, top, item["width"], item["height"]))
    return torch.stack(images), torch.stack(labels), geometry


def train_refiner(fit, calibration, fit_coarse, cal_coarse, a, device):
    x, y, _ = refinement_arrays(fit, fit_coarse, a.crop_size)
    xc, yc, _ = refinement_arrays(calibration, cal_coarse, a.crop_size)
    model = KeypointRefiner().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(a.seed + 100)
    best, best_loss, best_step, history = None, float("inf"), 0, []
    torch.manual_seed(a.seed + 100)
    for step in range(1, a.refiner_updates + 1):
        index = torch.randint(len(x), (a.batch_size,), generator=generator)
        prediction = model(x[index].to(device).float() / 255.0)
        loss = F.smooth_l1_loss(prediction, y[index].to(device), beta=0.01)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step == 1 or step % 100 == 0:
            with torch.no_grad():
                values = [model(xc[start:start + 128].to(device).float() / 255.0).cpu()
                          for start in range(0, len(xc), 128)]
                value = float(F.smooth_l1_loss(torch.cat(values), yc, beta=0.01))
            history.append({"step": step, "fit_loss": float(loss), "calibration_loss": value})
            if value < best_loss:
                best_loss, best_step, best = value, step, copy.deepcopy(model.state_dict())
            elif step - best_step >= a.patience:
                break
    model.load_state_dict(best)
    return model.cpu(), {"best_step": best_step, "completed_step": step,
                         "best_calibration_loss": best_loss, "history": history}


def refine_predict(model, rows, coarse, crop_size, device):
    x, y, geometry = refinement_arrays(rows, coarse, crop_size)
    model = model.to(device).eval(); values = []
    with torch.no_grad():
        for start in range(0, len(x), 128):
            values.append(model(x[start:start + 128].to(device).float() / 255.0).cpu())
    local = torch.cat(values); output = torch.empty_like(coarse); errors = []
    for prediction, truth, (row, view, left, top, width, height) in zip(local, y, geometry):
        offset = torch.tensor([left, top] * 2, dtype=torch.float32)
        full = prediction * (crop_size - 1) + offset
        scale = torch.tensor([width - 1, height - 1] * 2, dtype=torch.float32)
        output[row, view] = full / scale
        errors.append(((prediction - truth) * (crop_size - 1)).reshape(2, 2).norm(dim=1))
    errors = torch.stack(errors)
    report = {
        "view_count": len(errors),
        "episode_count": len(set(row["episode_id"] for row in rows)),
        "plug_error_px": {"median": float(errors[:, 0].median()),
                          "p95": float(torch.quantile(errors[:, 0], .95))},
        "target_error_px": {"median": float(errors[:, 1].median()),
                            "p95": float(torch.quantile(errors[:, 1], .95))},
    }
    return output, report


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array((w1*w2-x1*x2-y1*y2-z1*z2,
                     w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,
                     w1*z2+x1*y2-y1*x2+z1*w2))


def qrot(q, value):
    q = np.asarray(q, np.float64); q /= np.linalg.norm(q)
    return qmul(qmul(q, np.r_[0.0, value]), np.array((q[0], -q[1], -q[2], -q[3])))[1:]


def calibration_from(rows):
    for row in rows:
        if all(item.get("camera_position_world") is not None and
               item.get("camera_orientation_wxyz_ros") is not None for item in row["highres"]):
            return [{"K": np.asarray(item["intrinsic_matrix"], np.float64),
                     "position": np.asarray(item["camera_position_world"], np.float64),
                     "quat": np.asarray(item["camera_orientation_wxyz_ros"], np.float64),
                     "width": item["width"], "height": item["height"]} for item in row["highres"]]
    raise RuntimeError("No rendered camera extrinsics in replay")


def triangulate_one(per_view, calibration, point_offset):
    matrix = np.zeros((3, 3), np.float64); rhs = np.zeros(3, np.float64)
    for normalized, camera in zip(per_view, calibration):
        u = float(normalized[point_offset]) * (camera["width"] - 1)
        v = float(normalized[point_offset + 1]) * (camera["height"] - 1)
        k = camera["K"]
        ray_camera = np.array(((u-k[0, 2])/k[0, 0], (v-k[1, 2])/k[1, 1], 1.0))
        ray_world = qrot(camera["quat"], ray_camera)
        ray_world /= np.linalg.norm(ray_world)
        projection = np.eye(3) - np.outer(ray_world, ray_world)
        matrix += projection; rhs += projection @ camera["position"]
    return np.linalg.solve(matrix + 1e-9*np.eye(3), rhs)


def relative_world(points, calibration):
    values = []
    for row in points:
        plug = triangulate_one(row, calibration, 0)
        opening = triangulate_one(row, calibration, 2)
        values.append((plug - opening) * 1000.0)
    return np.asarray(values)


def affine_fit(x, y):
    design = np.c_[x, np.ones(len(x))]
    ridge = 1e-5 * np.eye(design.shape[1]); ridge[-1, -1] = 0.0
    return np.linalg.solve(design.T @ design + ridge, design.T @ y)


def main():
    a = args(); a.output_dir.mkdir(parents=True, exist_ok=True); device = torch.device(a.device)
    manifest = json.loads(a.scene_manifest.read_text()); calibration_indices = {3, 5, 8, 14}
    fit_ids = [x["episode_id"] for i, x in enumerate(manifest["train"]) if i not in calibration_indices]
    cal_ids = [x["episode_id"] for i, x in enumerate(manifest["train"]) if i in calibration_indices]
    eval_ids = [x["episode_id"] for x in manifest["development"]]
    train, train_audit = world.load([a.train_replay], set(fit_ids + cal_ids))
    evaluation, eval_audit = world.load([a.evaluation_replay], set(eval_ids))
    train, reject_train = crop.attach_highres(train, a.train_replay)
    evaluation, reject_eval = crop.attach_highres(evaluation, a.evaluation_replay)
    fit = [r for r in train if r["episode_id"] in fit_ids]
    calibration = [r for r in train if r["episode_id"] in cal_ids]
    if not fit or not calibration or not evaluation:
        raise RuntimeError("empty split")

    locator, locator_training = crop.train_locator(fit, calibration, a, device)
    fit_coarse, locator_fit = crop.locator_predict(locator, fit, device)
    cal_coarse, locator_cal = crop.locator_predict(locator, calibration, device)
    eval_coarse, locator_eval = crop.locator_predict(locator, evaluation, device)
    refiner, refiner_training = train_refiner(
        fit, calibration, fit_coarse, cal_coarse, a, device)
    fit_pred, refiner_fit = refine_predict(refiner, fit, fit_coarse, a.crop_size, device)
    cal_pred, refiner_cal = refine_predict(refiner, calibration, cal_coarse, a.crop_size, device)
    eval_pred, refiner_eval = refine_predict(refiner, evaluation, eval_coarse, a.crop_size, device)
    fit_true, cal_true, eval_true = true_points(fit), true_points(calibration), true_points(evaluation)
    camera_calibration = calibration_from(evaluation)

    # The affine transform captures the fixed target-frame convention and the
    # entrance-to-seated offset. It is learned only from training episodes.
    target_fit = np.stack([r["translation_mm"].numpy() for r in fit])
    pred_affine = affine_fit(relative_world(fit_pred, camera_calibration), target_fit)
    coarse_affine = affine_fit(relative_world(fit_coarse, camera_calibration), target_fit)
    oracle_affine = affine_fit(relative_world(fit_true, camera_calibration), target_fit)
    translated_pred = np.c_[relative_world(eval_pred, camera_calibration), np.ones(len(evaluation))] @ pred_affine
    translated_coarse = np.c_[relative_world(eval_coarse, camera_calibration), np.ones(len(evaluation))] @ coarse_affine
    translated_oracle = np.c_[relative_world(eval_true, camera_calibration), np.ones(len(evaluation))] @ oracle_affine
    calibration_pred = np.c_[relative_world(cal_pred, camera_calibration), np.ones(len(calibration))] @ pred_affine
    calibration_coarse = np.c_[relative_world(cal_coarse, camera_calibration), np.ones(len(calibration))] @ coarse_affine

    # Reuse a small feature probe only for rotation/contact/phase, then replace
    # its translation with the geometry-derived estimate.
    models, mean, std, histories = world.train_ensemble(fit, calibration, "feature", a, device)
    cpred, _, _, _ = world.predict(models, calibration, "feature", mean, std, device)
    cy = torch.stack([torch.cat((r["translation_mm"], r["rotation_deg"])) for r in calibration])
    residual = ((cpred - cy) ** 2).mean(0).numpy().clip(1e-6)
    base, variance, probability, phase = world.predict(models, evaluation, "feature", mean, std, device)

    def evaluate_rows(rows, translation, feature_pred, feature_prob, feature_phase, feature_var):
        prediction = feature_pred.clone(); prediction[:, :3] = torch.tensor(translation, dtype=torch.float32)
        overall, _ = world.metrics(rows, prediction, feature_prob, feature_phase, feature_var, residual)
        near_idx = [i for i, r in enumerate(rows) if r["signed_depth_m"] >= -0.003]
        near, _ = world.metrics([rows[i] for i in near_idx], prediction[near_idx],
                                feature_prob[near_idx], feature_phase[near_idx], feature_var[near_idx], residual)
        return {"overall": overall, "near_port": near}

    calibration_refined = evaluate_rows(calibration, calibration_pred, cpred, torch.sigmoid(torch.zeros(len(calibration), 2)),
                                        torch.softmax(torch.zeros(len(calibration), 3), -1), torch.zeros_like(cpred))
    calibration_coarse_metrics = evaluate_rows(calibration, calibration_coarse, cpred, torch.sigmoid(torch.zeros(len(calibration), 2)),
                                               torch.softmax(torch.zeros(len(calibration), 3), -1), torch.zeros_like(cpred))
    refined_score = (calibration_refined["near_port"]["lateral_error_mm"]["median"]
                     + calibration_refined["near_port"]["lateral_error_mm"]["p95"])
    coarse_score = (calibration_coarse_metrics["near_port"]["lateral_error_mm"]["median"]
                    + calibration_coarse_metrics["near_port"]["lateral_error_mm"]["p95"])
    selected = "refined" if refined_score < coarse_score else "coarse"
    refined = evaluate_rows(evaluation, translated_pred, base, probability, phase, variance)
    coarse_result = evaluate_rows(evaluation, translated_coarse, base, probability, phase, variance)
    deployable = refined if selected == "refined" else coarse_result
    oracle = evaluate_rows(evaluation, translated_oracle, base, probability, phase, variance)
    near = deployable["near_port"]
    passed = bool(near["lateral_error_mm"]["median"] <= .25 and
                  near["lateral_error_mm"]["p95"] <= .5 and
                  (near["lateral_correction_sign_accuracy"] or 0) >= .9)
    result = {
        "schema_version": 1,
        "representation": "coarse full-view RGB locator plus native-crop heatmap refinement and multiview triangulation",
        "deployable_geometry": "camera calibration only; simulator plug/port geometry used for training labels only",
        "split": {"unit": "complete reset configuration / episode_id", "fit_episode_ids": fit_ids,
                  "calibration_episode_ids": cal_ids, "evaluation_episode_ids": eval_ids,
                  "fit_rows": len(fit), "calibration_rows": len(calibration), "evaluation_rows": len(evaluation)},
        "audit": {"train": train_audit, "evaluation": eval_audit,
                  "highres_train_rejected": reject_train, "highres_evaluation_rejected": reject_eval},
        "locator": {"training": locator_training, "fit": locator_fit,
                    "calibration": locator_cal, "evaluation": locator_eval},
        "refiner": {"training": refiner_training, "fit": refiner_fit,
                    "calibration": refiner_cal, "evaluation": refiner_eval,
                    "native_crop_pixels": a.crop_size},
        "coordinate_selection": {"rule": "minimum calibration near-port lateral median+p95",
                                 "selected": selected, "coarse_score": coarse_score,
                                 "refined_score": refined_score,
                                 "calibration_coarse": calibration_coarse_metrics,
                                 "calibration_refined": calibration_refined},
        "deployable_coarse_keypoints": coarse_result,
        "deployable_refined_keypoints": refined,
        "deployable_predicted_keypoints": deployable,
        "oracle_keypoint_ceiling": oracle,
        "gate": {"required": "near-port lateral median <=0.25 mm, p95 <=0.5 mm, sign >=0.9",
                 "passed": passed},
        "camera_calibration": camera_calibration,
        "translation_affine": pred_affine.tolist(),
        "oracle_translation_affine": oracle_affine.tolist(),
        "feature_auxiliary_training": histories,
        "sources": {"train_replay": world.file_id(a.train_replay),
                    "evaluation_replay": world.file_id(a.evaluation_replay),
                    "scene_manifest": world.file_id(a.scene_manifest)},
    }
    # Convert ndarray values held in calibration before JSON serialization.
    result["camera_calibration"] = [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in row.items()}
                                    for row in camera_calibration]
    torch.save({"locator": locator.state_dict(), "refiner": refiner.state_dict(),
                "translation_affine": pred_affine, "coarse_translation_affine": coarse_affine,
                "feature_members": [m.state_dict() for m in models], "target_mean": mean,
                "target_std": std}, a.output_dir / "triangulated_pose_probe.pt")
    (a.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(a.output_dir), "gate": result["gate"],
                      "locator": locator_eval, "deployable_near": near,
                      "oracle_near": oracle["near_port"]}, indent=2))


if __name__ == "__main__":
    main()
