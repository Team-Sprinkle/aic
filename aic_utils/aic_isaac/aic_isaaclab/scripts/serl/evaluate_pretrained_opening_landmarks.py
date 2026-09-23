#!/usr/bin/env python3
"""One-shot evaluation of the calibration-selected pretrained opening landmarks."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
def module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE/file); value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None; spec.loader.exec_module(value); return value
opening = module("opening_pretrained_eval", "train_opening_landmark_pose_probe.py")
pretrained = module("pretrained_opening_eval", "calibrate_pretrained_opening_landmarks.py")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-replay", type=Path, required=True); p.add_argument("--evaluation-replay", type=Path, required=True)
    p.add_argument("--train-manifest", type=Path, required=True); p.add_argument("--evaluation-manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--auxiliary-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--crop-size", type=int, default=160)
    p.add_argument("--device", default="cuda"); a = p.parse_args(); device = torch.device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True)
    tm = json.loads(a.train_manifest.read_text()); em = json.loads(a.evaluation_manifest.read_text()); calibration_indices = {3, 7, 12, 16}
    fit_ids = [x["episode_id"] for i, x in enumerate(tm["train"]) if i not in calibration_indices]
    cal_ids = [x["episode_id"] for i, x in enumerate(tm["train"]) if i in calibration_indices]
    eval_ids = [x["episode_id"] for x in em["development"]]
    train, train_audit = opening.world.load([a.train_replay], set(fit_ids+cal_ids)); evaluation, eval_audit = opening.world.load([a.evaluation_replay], set(eval_ids))
    train, train_reject = opening.attach(train, a.train_replay); evaluation, eval_reject = opening.attach(evaluation, a.evaluation_replay)
    fit = [r for r in train if r["episode_id"] in fit_ids]; calibration = [r for r in train if r["episode_id"] in cal_ids]
    if not fit or not calibration or not evaluation: raise RuntimeError(f"empty split: {len(fit)}/{len(calibration)}/{len(evaluation)}")
    saved = torch.load(a.checkpoint, map_location="cpu", weights_only=False); selected = saved["summary"]["selection"]["selected"]
    locator = opening.crop.Locator(); locator.load_state_dict(saved["locator"])
    landmark = pretrained.MobileNetLandmarks(); landmark.load_state_dict(saved["landmark"])
    fit_c, cal_c, eval_c = opening.coarse_rows(fit), opening.coarse_rows(calibration), opening.coarse_rows(evaluation)
    fit_coarse, locator_fit = opening.crop.locator_predict(locator, fit_c, device)
    cal_coarse, locator_cal = opening.crop.locator_predict(locator, cal_c, device)
    eval_coarse, locator_eval = opening.crop.locator_predict(locator, eval_c, device)
    fit_land, landmark_fit = opening.landmark_predict(landmark, fit, fit_coarse, a.crop_size, device)
    cal_land, landmark_cal = opening.landmark_predict(landmark, calibration, cal_coarse, a.crop_size, device)
    eval_land, landmark_eval = opening.landmark_predict(landmark, evaluation, eval_coarse, a.crop_size, device)
    opening.save_montage(evaluation, eval_land, a.output_dir/"heldout_pretrained_landmarks.png")
    fit_points = opening.pair_points(fit_coarse, fit_land, selected["plug"], selected["port"])
    eval_points = opening.pair_points(eval_coarse, eval_land, selected["plug"], selected["port"])
    affine = opening.tri.affine_fit(opening.relative_world(fit_points, fit), np.stack([r["translation_mm"].numpy() for r in fit]))
    translation = np.c_[opening.relative_world(eval_points, evaluation), np.ones(len(evaluation))] @ affine

    aux = torch.load(a.auxiliary_checkpoint, map_location="cpu", weights_only=False)
    models = [opening.world.Probe(384) for _ in aux["feature_members"]]
    for model, state in zip(models, aux["feature_members"]): model.load_state_dict(state)
    base, variance, probability, phase = opening.world.predict(models, evaluation, "feature", aux["target_mean"], aux["target_std"], device)
    cpred, _, _, _ = opening.world.predict(models, calibration, "feature", aux["target_mean"], aux["target_std"], device)
    cy = torch.stack([torch.cat((r["translation_mm"], r["rotation_deg"])) for r in calibration])
    residual = ((cpred-cy)**2).mean(0).numpy().clip(1e-6)
    prediction = base.clone(); prediction[:, :3] = torch.tensor(translation, dtype=torch.float32)
    overall = opening.world.metrics(evaluation, prediction, probability, phase, variance, residual)[0]
    near = [i for i, r in enumerate(evaluation) if r["signed_depth_m"] >= -.003]
    near_metrics = opening.world.metrics([evaluation[i] for i in near], prediction[near], probability[near], phase[near], variance[near], residual)[0]
    passed = bool(near_metrics["lateral_error_mm"]["median"] <= .25 and near_metrics["lateral_error_mm"]["p95"] <= .5 and (near_metrics["lateral_correction_sign_accuracy"] or 0) >= .9)
    summary = {"schema_version": 1, "status": "perception_gate_passed" if passed else "perception_gate_failed",
        "evaluation_policy": "one-shot fresh development evaluation after calibration-only model selection",
        "selection": selected, "model": saved["summary"]["model"],
        "split": {"unit": "complete reset configuration / episode_id", "fit_episode_ids": fit_ids,
                  "calibration_episode_ids": cal_ids, "evaluation_episode_ids": eval_ids,
                  "fit_rows": len(fit), "calibration_rows": len(calibration), "evaluation_rows": len(evaluation)},
        "audit": {"train": train_audit, "evaluation": eval_audit, "train_rejected": train_reject, "evaluation_rejected": eval_reject},
        "locator": {"fit": locator_fit, "calibration": locator_cal, "evaluation": locator_eval},
        "landmarks": {"fit": landmark_fit, "calibration": landmark_cal, "evaluation": landmark_eval},
        "heldout": {"overall": overall, "near_port": near_metrics,
                    "by_phase": opening.world.sliced(evaluation, prediction, probability, phase, variance, residual, "phase"),
                    "by_motion_size": opening.world.sliced(evaluation, prediction, probability, phase, variance, residual, "motion_bin")},
        "gate": {"required": "near-port lateral median <=0.25 mm, p95 <=0.5 mm, sign >=0.9", "passed": passed},
        "decision": "policy comparison permitted" if passed else "stop before policy and RL; reserved final remains sealed",
        "sources": {"train_replay": opening.world.file_id(a.train_replay), "evaluation_replay": opening.world.file_id(a.evaluation_replay),
                    "train_manifest": opening.world.file_id(a.train_manifest), "evaluation_manifest": opening.world.file_id(a.evaluation_manifest),
                    "checkpoint": opening.world.file_id(a.checkpoint), "auxiliary_checkpoint": opening.world.file_id(a.auxiliary_checkpoint)}}
    (a.output_dir/"metrics.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"output": str(a.output_dir), "gate": summary["gate"], "near_port": near_metrics}, indent=2))

if __name__ == "__main__": main()
