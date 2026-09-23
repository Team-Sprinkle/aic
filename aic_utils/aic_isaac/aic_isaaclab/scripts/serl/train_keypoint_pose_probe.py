#!/usr/bin/env python3
"""Test explicit multiview keypoints as a relative-pose representation.

This is deliberately an offline diagnostic. Simulator labels create the oracle
ceiling and train the RGB locator; only RGB-predicted keypoints are eligible for
the deployable result.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


world = load_module("world_probe", HERE / "train_world_feature_pose_probe.py")
crop = load_module("crop_probe", HERE / "train_highres_crop_pose_probe.py")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-replay", type=Path, required=True)
    p.add_argument("--evaluation-replay", type=Path, required=True)
    p.add_argument("--scene-manifest", type=Path, required=True)
    p.add_argument("--locator-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--updates", type=int, default=5000)
    p.add_argument("--patience", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ensemble-size", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def true_points(rows):
    values = []
    for row in rows:
        per_view = []
        for item in row["highres"]:
            scale = torch.tensor([item["width"] - 1, item["height"] - 1] * 2)
            raw = torch.tensor([*item["plug"], *item["target"]], dtype=torch.float32)
            per_view.append(raw / scale)
        values.append(torch.stack(per_view))
    return torch.stack(values)


def explicit_feature(row, points):
    # Absolute image locations preserve perspective/depth information. Relative
    # image displacement exposes the visual-servo correction directly.
    delta = points[:, :2] - points[:, 2:4]
    return torch.cat((row["state"], row["feature"], points.reshape(-1), delta.reshape(-1))).float()


def fit_and_score(name, fit, calibration, evaluation, fit_points, cal_points, eval_points, args, device):
    for rows, points in ((fit, fit_points), (calibration, cal_points), (evaluation, eval_points)):
        for row, point in zip(rows, points):
            row[name] = explicit_feature(row, point)
    models, mean, std, histories = world.train_ensemble(fit, calibration, name, args, device)
    cpred, _, _, _ = world.predict(models, calibration, name, mean, std, device)
    cy = torch.stack([torch.cat((r["translation_mm"], r["rotation_deg"])) for r in calibration])
    residual = ((cpred - cy) ** 2).mean(0).numpy().clip(1e-6)
    pred, variance, probability, phase = world.predict(models, evaluation, name, mean, std, device)
    overall, raw = world.metrics(evaluation, pred, probability, phase, variance, residual)
    near_idx = [i for i, row in enumerate(evaluation) if row["signed_depth_m"] >= -0.003]
    near = world.metrics(
        [evaluation[i] for i in near_idx], pred[near_idx], probability[near_idx],
        phase[near_idx], variance[near_idx], residual,
    )[0]
    torch.save(
        {"members": [m.state_dict() for m in models], "target_mean": mean,
         "target_std": std, "input_dim": fit[0][name].numel(), "histories": histories},
        args.output_dir / f"{name}.pt",
    )
    return {"overall": overall, "near_port": near, "training": histories,
            "calibration_residual_variance": residual.tolist()}, raw


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    manifest = json.loads(args.scene_manifest.read_text())
    calibration_indices = {3, 5, 8, 14}
    fit_ids = [x["episode_id"] for i, x in enumerate(manifest["train"]) if i not in calibration_indices]
    cal_ids = [x["episode_id"] for i, x in enumerate(manifest["train"]) if i in calibration_indices]
    eval_ids = [x["episode_id"] for x in manifest["development"]]
    train, train_audit = world.load([args.train_replay], set(fit_ids + cal_ids))
    evaluation, eval_audit = world.load([args.evaluation_replay], set(eval_ids))
    train, highres_train_rejected = crop.attach_highres(train, args.train_replay)
    evaluation, highres_eval_rejected = crop.attach_highres(evaluation, args.evaluation_replay)
    fit = [r for r in train if r["episode_id"] in fit_ids]
    calibration = [r for r in train if r["episode_id"] in cal_ids]
    if not fit or not calibration or not evaluation:
        raise RuntimeError("empty split")

    checkpoint = torch.load(args.locator_checkpoint, map_location="cpu", weights_only=False)
    locator = crop.Locator()
    locator.load_state_dict(checkpoint["locator"])
    fit_pred, fit_locator = crop.locator_predict(locator, fit, device)
    cal_pred, cal_locator = crop.locator_predict(locator, calibration, device)
    eval_pred, eval_locator = crop.locator_predict(locator, evaluation, device)
    fit_true, cal_true, eval_true = true_points(fit), true_points(calibration), true_points(evaluation)

    predicted, _ = fit_and_score(
        "predicted_keypoint_pose", fit, calibration, evaluation,
        fit_pred, cal_pred, eval_pred, args, device,
    )
    oracle, _ = fit_and_score(
        "oracle_keypoint_pose", fit, calibration, evaluation,
        fit_true, cal_true, eval_true, args, device,
    )
    near = predicted["near_port"]
    gate = bool(
        near["lateral_error_mm"]["median"] <= 0.25
        and near["lateral_error_mm"]["p95"] <= 0.5
        and (near["lateral_correction_sign_accuracy"] or 0.0) >= 0.9
    )
    result = {
        "schema_version": 1,
        "representation": "state + frozen full-image feature + explicit three-view plug/target coordinates and offsets",
        "deployable_input": "RGB-predicted keypoints; no simulator labels or geometry",
        "split": {"unit": "complete reset configuration / episode_id", "fit_episode_ids": fit_ids,
                  "calibration_episode_ids": cal_ids, "evaluation_episode_ids": eval_ids,
                  "fit_rows": len(fit), "calibration_rows": len(calibration),
                  "evaluation_rows": len(evaluation)},
        "audit": {"train": train_audit, "evaluation": eval_audit,
                  "highres_train_rejected": highres_train_rejected,
                  "highres_evaluation_rejected": highres_eval_rejected},
        "locator": {"fit": fit_locator, "calibration": cal_locator, "evaluation": eval_locator},
        "predicted_keypoints": predicted,
        "oracle_keypoint_ceiling": oracle,
        "gate": {"required": "near-port lateral median <=0.25 mm, p95 <=0.5 mm, sign >=0.9",
                 "passed": gate},
        "sources": {"train_replay": world.file_id(args.train_replay),
                    "evaluation_replay": world.file_id(args.evaluation_replay),
                    "scene_manifest": world.file_id(args.scene_manifest),
                    "locator_checkpoint": world.file_id(args.locator_checkpoint)},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output_dir), "gate": result["gate"],
                      "predicted_near": predicted["near_port"],
                      "oracle_near": oracle["near_port"]}, indent=2))


if __name__ == "__main__":
    main()
