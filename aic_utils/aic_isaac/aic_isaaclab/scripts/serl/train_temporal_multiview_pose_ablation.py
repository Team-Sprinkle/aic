#!/usr/bin/env python3
"""Calibration-only causal temporal and learned multiview pose ablations.

The pretrained locator and landmark network stay frozen.  Isaac geometry is
used only as a supervised target.  Every temporal window resets at a detected
simulator episode boundary and contains the current and earlier observations.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(value)
    return value


opening = load_module("opening_temporal_ablation", "train_opening_landmark_pose_probe.py")
pretrained = load_module("pretrained_temporal_ablation", "calibrate_pretrained_opening_landmarks.py")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--crop-size", type=int, default=160)
    p.add_argument("--updates", type=int, default=3000)
    p.add_argument("--patience", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--history", type=int, default=6)
    p.add_argument("--ensemble-size", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260921)
    return p.parse_args()


class ResidualPose(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 128), nn.GELU(),
                                 nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 3))

    def forward(self, value):
        return self.net(value)


def predicted_features(rows, coarse, landmarks, selected, affine):
    points = opening.pair_points(coarse, landmarks, selected["plug"], selected["port"])
    raw = np.c_[opening.relative_world(points, rows), np.ones(len(rows))] @ affine
    # All inputs are observation-derived: three coarse plug locations, four
    # opening corners in each view, and the geometric triangulation result.
    feature = torch.cat((coarse[:, :, :2].reshape(len(rows), -1),
                         landmarks[:, :, 2:6].reshape(len(rows), -1),
                         torch.tensor(raw, dtype=torch.float32)), dim=1)
    return feature, torch.tensor(raw, dtype=torch.float32)


def causal_windows(rows, feature, history):
    by_sequence = defaultdict(list)
    windows = []
    for index, row in enumerate(rows):
        past = by_sequence[row["sequence_id"]]
        indices = (past + [index])[-history:]
        valid = [0.0] * (history - len(indices)) + [1.0] * len(indices)
        indices = [indices[0]] * (history - len(indices)) + indices
        windows.append(torch.cat((feature[indices].reshape(-1), torch.tensor(valid))))
        past.append(index)
    return torch.stack(windows)


def score(rows, prediction):
    lateral, axial, translation, sign = [], [], [], []
    truth = np.stack([r["translation_mm"].numpy() for r in rows])
    error = prediction - truth
    for row, predicted, actual, delta in zip(rows, prediction, truth, error):
        axis = np.asarray(row["axis_local"], float)
        axis /= max(np.linalg.norm(axis), 1e-12)
        axial.append(abs(float(np.dot(delta, axis))))
        lateral.append(float(np.linalg.norm(delta - axis * np.dot(delta, axis))))
        translation.append(float(np.linalg.norm(delta)))
        predicted_lateral = predicted - axis * np.dot(predicted, axis)
        actual_lateral = actual - axis * np.dot(actual, axis)
        if np.linalg.norm(actual_lateral) > .25:
            sign.append(float(np.dot(predicted_lateral, actual_lateral) > 0))

    def q(values):
        values = np.asarray(values)
        return {"mean": float(values.mean()), "median": float(np.median(values)),
                "p95": float(np.quantile(values, .95))}

    return {"count": len(rows), "configuration_count": len({r["episode_id"] for r in rows}),
            "sequence_count": len({r["sequence_id"] for r in rows}),
            "translation_error_mm": q(translation), "axial_error_mm": q(axial),
            "lateral_error_mm": q(lateral),
            "lateral_correction_sign_accuracy": float(np.mean(sign)) if sign else None}


def slices(rows, prediction, key):
    result = {}
    for value in sorted({r[key] for r in rows}):
        indices = [i for i, row in enumerate(rows) if row[key] == value]
        result[value] = score([rows[i] for i in indices], prediction[indices])
    return result


def train_members(name, xfit, xcal, fit, calibration, raw_fit, raw_cal, args, device):
    target_fit = torch.stack([r["translation_mm"] for r in fit]) - raw_fit
    target_cal = torch.stack([r["translation_mm"] for r in calibration]) - raw_cal
    xm, xs = xfit.mean(0), xfit.std(0).clamp_min(1e-5)
    ym, ys = target_fit.mean(0), target_fit.std(0).clamp_min(.05)
    xfit_n = (xfit - xm) / xs
    xcal_n = ((xcal - xm) / xs).to(device)
    counts = Counter(r["episode_id"] for r in fit)
    weights = torch.tensor([(2.0 if r["signed_depth_m"] >= -.003 else 1.0) / counts[r["episode_id"]]
                            for r in fit])
    members, histories = [], []
    for member in range(args.ensemble_size):
        seed = args.seed + member
        torch.manual_seed(seed); random.seed(seed)
        model = ResidualPose(xfit.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(seed)
        best, best_score, best_step, history = None, float("inf"), 0, []
        for step in range(1, args.updates + 1):
            indices = torch.multinomial(weights, args.batch_size, replacement=True, generator=generator)
            output = model(xfit_n[indices].to(device))
            loss = F.smooth_l1_loss(output, ((target_fit[indices] - ym) / ys).to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if step == 1 or step % 100 == 0:
                model.eval()
                with torch.no_grad():
                    residual = model(xcal_n).cpu() * ys + ym
                    predicted = (raw_cal + residual).numpy()
                near = [i for i, row in enumerate(calibration) if row["signed_depth_m"] >= -.003]
                metric = score([calibration[i] for i in near], predicted[near])
                value = metric["lateral_error_mm"]["median"] + metric["lateral_error_mm"]["p95"]
                history.append({"step": step, "fit_loss": float(loss), "selection_score": value})
                if value < best_score:
                    best_score, best_step = value, step
                    best = copy.deepcopy(model.state_dict())
                elif step - best_step >= args.patience:
                    break
                model.train()
        model.load_state_dict(best); members.append(model.cpu())
        histories.append({"member": member, "seed": seed, "best_step": best_step,
                          "completed_step": step, "best_selection_score": best_score,
                          "history": history})
    predictions = []
    for model in members:
        model = model.to(device).eval()
        with torch.no_grad(): predictions.append(raw_cal + model(xcal_n).cpu() * ys + ym)
    stack = torch.stack(predictions)
    return stack.mean(0).numpy(), stack.var(0, unbiased=False).numpy(), {
        "name": name, "input_dim": xfit.shape[1],
        "parameters_per_member": sum(p.numel() for p in members[0].parameters()),
        "members": histories, "input_mean": xm, "input_std": xs,
        "target_mean": ym, "target_std": ys, "state_dicts": [m.state_dict() for m in members]}


def model_report(rows, prediction, variance):
    near = [i for i, row in enumerate(rows) if row["signed_depth_m"] >= -.003]
    near_metric = score([rows[i] for i in near], prediction[near])
    return {"overall": score(rows, prediction), "near_port": near_metric,
            "by_phase": slices(rows, prediction, "phase"),
            "by_motion_size": slices(rows, prediction, "motion_bin"),
            "predictive_variance_mm2": {"mean": np.mean(variance, axis=0).tolist()},
            "gate": {"median_lte_0p25mm": near_metric["lateral_error_mm"]["median"] <= .25,
                     "p95_lte_0p5mm": near_metric["lateral_error_mm"]["p95"] <= .5,
                     "sign_gte_0p9": (near_metric["lateral_correction_sign_accuracy"] or 0) >= .9}}


def public_training(training):
    return {key: value for key, value in training.items()
            if key not in {"state_dicts", "input_mean", "input_std", "target_mean", "target_std"}}


def main():
    args = arguments(); args.output_dir.mkdir(parents=True, exist_ok=True); device = torch.device(args.device)
    manifest = json.loads(args.manifest.read_text()); calibration_indices = {3, 7, 12, 16}
    fit_ids = [x["episode_id"] for i, x in enumerate(manifest["train"]) if i not in calibration_indices]
    cal_ids = [x["episode_id"] for i, x in enumerate(manifest["train"]) if i in calibration_indices]
    rows, audit = opening.world.load([args.replay], set(fit_ids + cal_ids)); rows, rejected = opening.attach(rows, args.replay)
    fit = [row for row in rows if row["episode_id"] in fit_ids]
    calibration = [row for row in rows if row["episode_id"] in cal_ids]
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    selected = saved["summary"]["selection"]["selected"]
    locator = opening.crop.Locator(); locator.load_state_dict(saved["locator"])
    landmark = pretrained.MobileNetLandmarks(); landmark.load_state_dict(saved["landmark"])
    fit_coarse, locator_fit = opening.crop.locator_predict(locator, opening.coarse_rows(fit), device)
    cal_coarse, locator_cal = opening.crop.locator_predict(locator, opening.coarse_rows(calibration), device)
    fit_land, landmark_fit = opening.landmark_predict(landmark, fit, fit_coarse, args.crop_size, device)
    cal_land, landmark_cal = opening.landmark_predict(landmark, calibration, cal_coarse, args.crop_size, device)
    initial_fit_points = opening.pair_points(fit_coarse, fit_land, selected["plug"], selected["port"])
    affine = opening.tri.affine_fit(opening.relative_world(initial_fit_points, fit),
                                    np.stack([r["translation_mm"].numpy() for r in fit]))
    fit_feature, raw_fit = predicted_features(fit, fit_coarse, fit_land, selected, affine)
    cal_feature, raw_cal = predicted_features(calibration, cal_coarse, cal_land, selected, affine)
    one_fit = torch.cat((fit_feature, torch.ones(len(fit), 1)), dim=1)
    one_cal = torch.cat((cal_feature, torch.ones(len(calibration), 1)), dim=1)
    temporal_fit = causal_windows(fit, fit_feature, args.history)
    temporal_cal = causal_windows(calibration, cal_feature, args.history)
    multi_pred, multi_var, multi_training = train_members("learned_multiview_current_frame", one_fit, one_cal,
        fit, calibration, raw_fit, raw_cal, args, device)
    temporal_pred, temporal_var, temporal_training = train_members("causal_temporal_multiview", temporal_fit,
        temporal_cal, fit, calibration, raw_fit, raw_cal, args, device)
    zero_var = np.zeros((len(calibration), 3))
    report = {"schema_version": 1, "status": "calibration_only", "development_set_opened": False,
        "scope": "translation-only residual ablation; orientation/contact heads are unchanged and excluded",
        "causality": {"window_frames": args.history, "current_and_past_only": True,
            "reset_key": "derived reset boundary because recorded global_episode_index is null",
            "boundary_rule": "new sequence on episode_id change or after terminated/truncated transition"},
        "split": {"unit": "complete reset configuration / episode_id", "fit_episode_ids": fit_ids,
            "calibration_episode_ids": cal_ids, "fit_rows": len(fit), "calibration_rows": len(calibration),
            "fit_sequences": len({r['sequence_id'] for r in fit}),
            "calibration_sequences": len({r['sequence_id'] for r in calibration})},
        "audit": audit, "rejected": rejected, "frozen_perception": {"selection": selected,
            "locator_fit": locator_fit, "locator_calibration": locator_cal,
            "landmark_fit": landmark_fit, "landmark_calibration": landmark_cal},
        "models": {"frozen_spatial_baseline": model_report(calibration, raw_cal.numpy(), zero_var),
            "learned_multiview_current_frame": model_report(calibration, multi_pred, multi_var),
            "causal_temporal_multiview": model_report(calibration, temporal_pred, temporal_var)},
        "selection_rule": "calibration near-port lateral median+p95; no development scenes used",
        "limitations": ["No independent cable-shape reset variable exists in this replay.",
            "Instance segmentation was not supplied to either model.",
            "Calibration configurations are held out, but share the same fixed scene and cable initialization family."],
        "training": {"updates": args.updates, "patience": args.patience, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "ensemble_size": args.ensemble_size,
            "multiview": public_training(multi_training),
            "temporal": public_training(temporal_training)}}
    for value in report["models"].values(): value["gate"]["passed"] = all(value["gate"].values())
    torch.save({"translation_affine": affine, "selected": selected,
                "multiview": multi_training, "temporal": temporal_training}, args.output_dir / "ablation_checkpoint.pt")
    (args.output_dir / "calibration_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output_dir), "split": report["split"],
                      "models": {k: v["near_port"] for k, v in report["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
