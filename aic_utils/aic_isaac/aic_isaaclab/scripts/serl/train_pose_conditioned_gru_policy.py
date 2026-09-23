#!/usr/bin/env python3
"""Train matched temporal BC heads with and without frozen predicted pose.

The deployable inputs are RGB-derived landmark features, robot state, measured
state change, force, previous executed action, predicted plug-to-opening
translation, ensemble spread, and an observation-derived phase.  Simulator
geometry is used by the frozen pose model's original supervision and by offline
reporting only; it is never included in the actor input.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
CAMERAS = ("center_camera", "left_camera", "right_camera")


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


visibility = load_module("pose_gru_visibility", "train_visibility_weighted_pose_ablation.py")
opening = visibility.opening
pretrained = visibility.pretrained


class PoseConditionedGRUPolicy(nn.Module):
    """Small causal controller that predicts four TCP-frame 6D deltas."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, action_dim: int = 24):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        encoded = self.input(sequence)
        output, _ = self.gru(encoded)
        return self.head(output[:, -1])


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--pose-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--history", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--cache", type=Path)
    return parser.parse_args()


def frozen_perception(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    manifest = json.loads(args.dataset_manifest.read_text())
    split_ids = {name: set(value["episode_ids"]) for name, value in manifest["splits"].items()}
    allowed = set().union(*split_ids.values())
    rows, load_audit = opening.world.load([args.replay], allowed)
    rows, image_rejected = opening.attach(rows, args.replay)
    rows, visibility_rejected = visibility.attach_visibility(rows, args.replay)
    splits = {name: [row for row in rows if row["episode_id"] in ids] for name, ids in split_ids.items()}
    checkpoint = torch.load(args.pose_checkpoint, map_location="cpu", weights_only=False)
    locator = opening.crop.Locator(); locator.load_state_dict(checkpoint["locator"])
    landmark = pretrained.MobileNetLandmarks(); landmark.load_state_dict(checkpoint["landmark"])
    vis_model = visibility.VisibilityHead(checkpoint["visibility_input_dim"])
    vis_model.load_state_dict(checkpoint["visibility_head"])
    output: dict[str, object] = {
        "schema_version": 1,
        "load_audit": load_audit,
        "image_rejected": image_rejected,
        "visibility_rejected": visibility_rejected,
        "splits": {},
    }
    for name, split in splits.items():
        coarse, _ = opening.crop.locator_predict(locator, opening.coarse_rows(split), device)
        landmarks, visual, _ = visibility.landmark_predict_features(landmark, split, coarse, 160, device)
        probability = visibility.visibility_predict(vis_model, visual, device)
        selected = checkpoint["selected_landmarks"]
        points = opening.pair_points(coarse, landmarks, selected["plug"], selected["port"])
        raw = visibility.relative_weighted(points, split, probability)
        affine = checkpoint["affines"]["predicted_visibility"]
        affine_raw = np.c_[raw, np.ones(len(raw))] @ affine
        residual_feature = visibility.residual_features(coarse, landmarks, affine_raw, probability)
        saved = checkpoint["current"]
        models = []
        for state in saved["state_dicts"]:
            model = visibility.ResidualPose(int(saved["input_mean"].numel()))
            model.load_state_dict(state)
            models.append(model)
        prediction, variance = visibility.residual_predict(
            models, saved, split, residual_feature, affine_raw, SimpleNamespace(history=6), device
        )
        output["splits"][name] = {
            "episode_ids": [row["episode_id"] for row in split],
            "transition_indices": [int(row["transition_index"]) for row in split],
            "visual": visual.reshape(len(split), -1).float(),
            "visibility": probability.reshape(len(split), -1).float(),
            "pose_mm": torch.tensor(prediction, dtype=torch.float32),
            "pose_variance_mm2": torch.tensor(variance, dtype=torch.float32),
        }
    return output


def phase_from_observation(pose_mm: torch.Tensor, force_xyz: torch.Tensor) -> torch.Tensor:
    distance = torch.linalg.norm(pose_mm)
    force = torch.linalg.norm(force_xyz)
    phase = torch.zeros(4)
    if float(force) >= 5.0:
        phase[3] = 1.0  # contact or blocked
    elif float(distance) > 3.0:
        phase[0] = 1.0  # approach
    elif float(distance) > 0.75:
        phase[1] = 1.0  # alignment
    else:
        phase[2] = 1.0  # insertion corridor
    return phase


def build_examples(cache: dict[str, object], replay: Path) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
    payload = torch.load(replay, map_location="cpu", weights_only=False)
    transitions = payload["transitions"]
    result: dict[str, dict[str, object]] = {}
    feature_dims: dict[str, int] = {}
    for split_name, saved in cache["splits"].items():
        rows = []
        previous_by_episode: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for local_index, (episode_id, transition_index) in enumerate(
            zip(saved["episode_ids"], saved["transition_indices"])
        ):
            transition = transitions[int(transition_index)]
            state = transition["obs"]["state"].reshape(-1).float()
            action = transition["action"].reshape(-1).float()
            if state.numel() != 42 or action.numel() != 24:
                raise ValueError(f"Unexpected state/action width at transition {transition_index}")
            previous = previous_by_episode.get(episode_id)
            state_delta = torch.zeros_like(state) if previous is None else state - previous[0]
            previous_action = torch.zeros_like(action) if previous is None else previous[1]
            force = torch.tensor(
                (transition.get("metadata") or {}).get("causal_force_xyz_n") or [0.0, 0.0, 0.0],
                dtype=torch.float32,
            )
            pose = saved["pose_mm"][local_index].float()
            uncertainty = saved["pose_variance_mm2"][local_index].clamp_min(0).sqrt().float()
            phase = phase_from_observation(pose, force)
            base = torch.cat((saved["visual"][local_index], state, state_delta, force, previous_action,
                              saved["visibility"][local_index]))
            conditioning = torch.cat((pose, uncertainty, phase))
            rows.append({
                "episode_id": episode_id,
                "transition_index": int(transition_index),
                "base": base,
                "conditioning": conditioning,
                "action": action,
                "force_n": float(torch.linalg.norm(force)),
            })
            previous_by_episode[episode_id] = (state, action)
        result[split_name] = {"rows": rows}
        feature_dims = {"base": int(rows[0]["base"].numel()), "conditioning": int(rows[0]["conditioning"].numel())}
    return result, feature_dims


def fit_normalization(examples: dict[str, dict[str, object]]) -> dict[str, torch.Tensor]:
    fit = examples["fit"]["rows"]
    base = torch.stack([row["base"] for row in fit])
    condition = torch.stack([row["conditioning"] for row in fit])
    action = torch.stack([row["action"] for row in fit])
    return {
        "base_mean": base.mean(0), "base_std": base.std(0).clamp_min(1e-5),
        "conditioning_mean": condition.mean(0), "conditioning_std": condition.std(0).clamp_min(1e-5),
        "action_mean": action.mean(0), "action_std": action.std(0).clamp_min(1e-6),
    }


def causal_windows(rows: list[dict[str, object]], normalization: dict[str, torch.Tensor], history: int,
                   conditioned: bool) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    normalized = []
    for row in rows:
        base = (row["base"] - normalization["base_mean"]) / normalization["base_std"]
        if conditioned:
            condition = (row["conditioning"] - normalization["conditioning_mean"]) / normalization["conditioning_std"]
        else:
            condition = torch.zeros_like(row["conditioning"])
        normalized.append(torch.cat((base, condition)))
    windows, targets, episode_ids = [], [], []
    for index, row in enumerate(rows):
        sequence = []
        start = max(0, index - history + 1)
        for prior in range(start, index + 1):
            if rows[prior]["episode_id"] == row["episode_id"]:
                sequence.append(normalized[prior])
        padding = [torch.zeros_like(normalized[index]) for _ in range(history - len(sequence))]
        windows.append(torch.stack(padding + sequence))
        targets.append((row["action"] - normalization["action_mean"]) / normalization["action_std"])
        episode_ids.append(str(row["episode_id"]))
    return torch.stack(windows), torch.stack(targets), episode_ids


def physical_metrics(pred_norm: torch.Tensor, target_norm: torch.Tensor,
                     normalization: dict[str, torch.Tensor]) -> dict[str, float]:
    pred = pred_norm.cpu() * normalization["action_std"] + normalization["action_mean"]
    target = target_norm.cpu() * normalization["action_std"] + normalization["action_mean"]
    pred = pred.reshape(-1, 4, 6); target = target.reshape(-1, 4, 6)
    translation = torch.abs(pred[..., :3] - target[..., :3]) * 1000.0
    rotation = torch.abs(pred[..., 3:] - target[..., 3:]) * (180.0 / math.pi)
    return {
        "translation_mae_mm": float(translation.mean()),
        "translation_p95_mm": float(torch.quantile(translation.reshape(-1), 0.95)),
        "rotation_mae_deg": float(rotation.mean()),
        "rotation_p95_deg": float(torch.quantile(rotation.reshape(-1), 0.95)),
    }


def train_one(name: str, examples: dict[str, dict[str, object]], normalization: dict[str, torch.Tensor],
              args: argparse.Namespace, device: torch.device, conditioned: bool) -> tuple[nn.Module, dict[str, object]]:
    tensors = {}
    for split, value in examples.items():
        tensors[split] = causal_windows(value["rows"], normalization, args.history, conditioned)
    xfit, yfit, fit_episodes = tensors["fit"]
    xcal, ycal, _ = tensors["calibration"]
    model = PoseConditionedGRUPolicy(xfit.shape[-1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    counts = Counter(fit_episodes)
    weights = torch.tensor([1.0 / counts[episode] for episode in fit_episodes], device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    best_state = copy.deepcopy(model.state_dict()); best_loss = float("inf"); best_update = 0; history = []
    xfit=xfit.to(device); yfit=yfit.to(device); xcal=xcal.to(device); ycal=ycal.to(device)
    for update in range(1, args.updates + 1):
        model.train()
        indices = torch.multinomial(weights, args.batch_size, replacement=True, generator=generator)
        prediction = model(xfit[indices])
        loss = F.smooth_l1_loss(prediction, yfit[indices])
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)); optimizer.step()
        if update == 1 or update % args.log_every == 0 or update == args.updates:
            model.eval()
            with torch.no_grad():
                calibration_prediction = model(xcal)
                calibration_loss = float(F.smooth_l1_loss(calibration_prediction, ycal))
            row = {"update": update, "fit_loss": float(loss), "calibration_loss": calibration_loss,
                   "grad_norm": grad_norm, **physical_metrics(calibration_prediction, ycal, normalization)}
            history.append(row); print(json.dumps({"model": name, **row}, sort_keys=True), flush=True)
            if calibration_loss < best_loss:
                best_loss=calibration_loss; best_update=update; best_state=copy.deepcopy(model.state_dict())
            elif update - best_update >= args.patience:
                break
    model.load_state_dict(best_state); model.eval()
    split_metrics = {}
    with torch.no_grad():
        for split, (x, y, episode_ids) in tensors.items():
            prediction = model(x.to(device)).cpu()
            split_metrics[split] = {
                "rows": len(x), "episodes": len(set(episode_ids)),
                "normalized_smooth_l1": float(F.smooth_l1_loss(prediction, y)),
                **physical_metrics(prediction, y, normalization),
            }
    report = {"conditioned": conditioned, "parameter_count": sum(p.numel() for p in model.parameters()),
              "best_update": best_update, "completed_updates": update, "best_calibration_loss": best_loss,
              "history": history, "split_metrics": split_metrics}
    return model.cpu(), report


def main() -> None:
    args = arguments()
    if args.history < 1 or args.updates < 1 or args.batch_size < 1:
        raise ValueError("history, updates, and batch-size must be positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cache_path = args.cache or (args.output_dir / "frozen_perception_cache.pt")
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        cache = frozen_perception(args, device)
        cache_path.parent.mkdir(parents=True, exist_ok=True); torch.save(cache, cache_path)
    examples, dimensions = build_examples(cache, args.replay)
    normalization = fit_normalization(examples)
    models={}; reports={}
    for index, (name, conditioned) in enumerate((("action_only", False), ("pose_conditioned", True))):
        torch.manual_seed(args.seed)  # identical initialization and optimizer sampling
        models[name], reports[name] = train_one(name, examples, normalization, args, device, conditioned)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema_version": 1,
        "architecture": "shared frozen RGB landmark features -> 256D GRU -> 256D MLP -> 24D TCP-frame delta chunk",
        "action_representation": "four_by_six_tcp_body_frame_delta",
        "simulator_geometry_actor_input": False,
        "pose_checkpoint": str(args.pose_checkpoint),
        "replay": str(args.replay),
        "dataset_manifest": str(args.dataset_manifest),
        "history": args.history, "hidden_dim": args.hidden_dim,
        "dimensions": dimensions,
        "normalization": normalization,
        "models": {name: model.state_dict() for name, model in models.items()},
        "reports": reports,
        "perception_audit": {key: cache[key] for key in ("load_audit", "image_rejected", "visibility_rejected")},
        "split_episode_ids": {
            split: sorted(set(row["episode_id"] for row in value["rows"])) for split, value in examples.items()
        },
        "reserved_final_opened": False,
        "rl_started": False,
    }
    torch.save(bundle, args.output_dir / "pose_gru_policy_checkpoint.pt")
    summary = {key: value for key, value in bundle.items() if key not in ("normalization", "models")}
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"reports": reports, "dimensions": dimensions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
