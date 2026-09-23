#!/usr/bin/env python3
"""Train a bounded pose-driven residual on top of the frozen action-only GRU."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pose_gru = load_module("explicit_correction_pose_gru", "train_pose_conditioned_gru_policy.py")


class ExplicitPoseCorrection(nn.Module):
    """Odd-symmetric pose map with context controlling magnitude only."""

    def __init__(self, context_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, hidden_dim), nn.GELU())
        self.context_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gate = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 24), nn.Sigmoid())
        self.pose_map = nn.Linear(3, 24, bias=False)
        nn.init.zeros_(self.pose_map.weight)
        nn.init.zeros_(self.gate[1].weight)
        nn.init.zeros_(self.gate[1].bias)
        scale = torch.tensor([0.0005, 0.0005, 0.0005, 0.002, 0.002, 0.002] * 4)
        self.register_buffer("output_scale", scale)

    def forward(self, pose_mm: torch.Tensor, context_sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.context(context_sequence)
        history, _ = self.context_gru(encoded)
        gain = 0.25 + 1.5 * self.gate(history[:, -1])
        # Pose is scaled in physical coordinates and the no-bias map makes
        # correction(-pose) == -correction(pose) for a fixed context.
        pose_value = pose_mm[:, -1] / 2.0
        residual = torch.tanh(self.pose_map(pose_value)) * self.output_scale * gain
        return residual, gain


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--pose-checkpoint", type=Path, required=True)
    parser.add_argument("--nominal-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=800)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--selection-metric", choices=("translation_mae", "normalized_action_loss"),
                        default="translation_mae")
    return parser.parse_args()


def correction_windows(rows, normalization, history: int):
    contexts, poses, targets, episodes = [], [], [], []
    normalized_context = []
    for row in rows:
        base = (row["base"] - normalization["base_mean"]) / normalization["base_std"]
        condition = row["conditioning"]
        # state delta 42, explicit force 3, previous action 24, visibility 6,
        # plus uncertainty 3 and phase 4. Current pose has its own mandatory path.
        context = torch.cat((base[222:297],
                             (condition[3:] - normalization["conditioning_mean"][3:]) /
                             normalization["conditioning_std"][3:]))
        normalized_context.append(context)
    for index, row in enumerate(rows):
        start = max(0, index - history + 1)
        context_sequence, pose_sequence = [], []
        for prior in range(start, index + 1):
            if rows[prior]["episode_id"] == row["episode_id"]:
                context_sequence.append(normalized_context[prior])
                pose_sequence.append(rows[prior]["conditioning"][:3])
        contexts.append(torch.stack([torch.zeros_like(normalized_context[index])] *
                                    (history-len(context_sequence)) + context_sequence))
        poses.append(torch.stack([torch.zeros(3)] * (history-len(pose_sequence)) + pose_sequence))
        targets.append((row["action"]-normalization["action_mean"])/normalization["action_std"])
        episodes.append(row["episode_id"])
    return torch.stack(poses), torch.stack(contexts), torch.stack(targets), episodes


def physical_metrics(pred_norm, target_norm, normalization):
    return pose_gru.physical_metrics(pred_norm, target_norm, normalization)


def dependence_metrics(model, pose, context, nominal_phys, target_phys, normalization):
    with torch.no_grad():
        residual, _ = model(pose, context)
        zero_residual, _ = model(torch.zeros_like(pose), context)
        order = torch.randperm(len(pose), generator=torch.Generator().manual_seed(20260923))
        shuffled_residual, _ = model(pose[order], context)
    scale = normalization["action_std"]
    mean = normalization["action_mean"]
    true_norm = (nominal_phys + residual - mean) / scale
    zero_norm = (nominal_phys + zero_residual - mean) / scale
    shuffled_norm = (nominal_phys + shuffled_residual - mean) / scale
    target_norm = (target_phys - mean) / scale
    vector = residual.reshape(-1, 4, 6)[..., :3].norm(dim=-1) * 1000.0
    target_residual = (target_phys-nominal_phys).reshape(-1,4,6)[...,:3]
    predicted_residual = residual.reshape(-1,4,6)[...,:3]
    mask = target_residual.norm(dim=-1) >= 3e-5
    sign = ((target_residual*predicted_residual).sum(-1) > 0)[mask]
    odd_a, _ = model(pose, context); odd_b, _ = model(-pose, context)
    odd_error = (odd_a+odd_b).abs().max()
    return {
        "true_pose": physical_metrics(true_norm, target_norm, normalization),
        "zero_pose": physical_metrics(zero_norm, target_norm, normalization),
        "shuffled_pose": physical_metrics(shuffled_norm, target_norm, normalization),
        "correction_vector_p50_mm": float(torch.quantile(vector, 0.5)),
        "correction_vector_p95_mm": float(torch.quantile(vector, 0.95)),
        "correction_target_sign_accuracy": float(sign.float().mean()) if sign.numel() else None,
        "odd_symmetry_max_abs": float(odd_error),
    }


def main() -> None:
    args = arguments()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache or args.output_dir / "frozen_perception_cache.pt"
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        cache = pose_gru.frozen_perception(SimpleNamespace(
            replay=args.replay, dataset_manifest=args.dataset_manifest, pose_checkpoint=args.pose_checkpoint
        ), device)
        torch.save(cache, cache_path)
    examples, dimensions = pose_gru.build_examples(cache, args.replay)
    replay = torch.load(args.replay, map_location="cpu", weights_only=False)
    # Prefer the stored unblended teacher chunk even though these collections
    # executed a 100% guide blend.
    by_index = {index: transition for index, transition in enumerate(replay["transitions"])}
    for split in examples.values():
        for row in split["rows"]:
            transition = by_index[row["transition_index"]]
            if transition.get("guide_action") is not None:
                row["action"] = transition["guide_action"].reshape(-1).float()

    nominal_bundle = torch.load(args.nominal_checkpoint, map_location="cpu", weights_only=False)
    normalization = {key: value.float() for key, value in nominal_bundle["normalization"].items()}
    history = int(nominal_bundle["history"])
    nominal = pose_gru.PoseConditionedGRUPolicy(
        nominal_bundle["dimensions"]["base"] + nominal_bundle["dimensions"]["conditioning"],
        int(nominal_bundle["hidden_dim"]),
    ).to(device)
    nominal.load_state_dict(nominal_bundle["models"]["action_only"])
    nominal.eval().requires_grad_(False)

    tensors = {}
    for split, value in examples.items():
        nominal_x, _, _ = pose_gru.causal_windows(value["rows"], normalization, history, False)
        pose, context, target, episode_ids = correction_windows(value["rows"], normalization, history)
        with torch.no_grad():
            nominal_norm = nominal(nominal_x.to(device)).cpu()
        nominal_phys = nominal_norm*normalization["action_std"]+normalization["action_mean"]
        target_phys = target*normalization["action_std"]+normalization["action_mean"]
        tensors[split] = (pose, context, target, nominal_phys, target_phys, episode_ids)

    context_dim = tensors["fit"][1].shape[-1]
    model = ExplicitPoseCorrection(context_dim).to(device)
    pose_fit, context_fit, target_fit, nominal_fit, target_phys_fit, fit_episodes = tensors["fit"]
    # Fit the mandatory pose path once by least squares before nonlinear
    # refinement. This uses predicted pose and teacher-minus-nominal commands
    # from fit episodes only; calibration remains unseen.
    with torch.no_grad():
        design = pose_fit[:, -1] / 2.0
        residual_target_physical = target_phys_fit - nominal_fit
        scale = model.output_scale.cpu().clamp_min(1e-8)
        inverse_target = torch.atanh((residual_target_physical / scale).clamp(-0.95, 0.95))
        solution = torch.linalg.lstsq(design, inverse_target).solution
        model.pose_map.weight.copy_(solution.T.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    counts = Counter(fit_episodes)
    weights = torch.tensor([1/counts[x] for x in fit_episodes], device=device)
    pose_fit=pose_fit.to(device); context_fit=context_fit.to(device); target_fit=target_fit.to(device)
    nominal_fit=nominal_fit.to(device); target_phys_fit=target_phys_fit.to(device)
    action_mean=normalization["action_mean"].to(device); action_std=normalization["action_std"].to(device)
    generator=torch.Generator(device=device).manual_seed(args.seed)
    best_state=copy.deepcopy(model.state_dict()); best_loss=float("inf"); best_update=0; log=[]
    for update in range(1,args.updates+1):
        model.train(); index=torch.multinomial(weights,args.batch_size,replacement=True,generator=generator)
        residual,_=model(pose_fit[index],context_fit[index])
        prediction=(nominal_fit[index]+residual-action_mean)/action_std
        action_loss=F.smooth_l1_loss(prediction,target_fit[index])
        residual_target=(target_phys_fit[index]-nominal_fit[index])/action_std
        residual_loss=F.smooth_l1_loss(residual/action_std,residual_target)
        pred_translation=residual.reshape(-1,4,6)[...,:3]
        target_translation=(target_phys_fit[index]-nominal_fit[index]).reshape(-1,4,6)[...,:3]
        valid=target_translation.norm(dim=-1)>=3e-5
        direction_loss=(1-F.cosine_similarity(pred_translation[valid],target_translation[valid],dim=-1)).mean() if valid.any() else action_loss*0
        loss=action_loss+0.5*residual_loss+0.1*direction_loss
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
        if update==1 or update%args.log_every==0 or update==args.updates:
            model.eval(); p,c,t,n,tp,_=tensors["calibration"]
            with torch.no_grad():
                r,_=model(p.to(device),c.to(device)); pred=(n.to(device)+r-action_mean)/action_std
                calibration_loss=float(F.smooth_l1_loss(pred,t.to(device)))
                calibration_translation_mae_mm = physical_metrics(
                    pred.cpu(), t, normalization)["translation_mae_mm"]
            selection_value = (calibration_translation_mae_mm
                               if args.selection_metric == "translation_mae" else calibration_loss)
            row={"update":update,"fit_loss":float(loss),"action_loss":float(action_loss),
                 "residual_loss":float(residual_loss),"direction_loss":float(direction_loss),
                 "calibration_loss":calibration_loss,
                 "calibration_translation_mae_mm":calibration_translation_mae_mm,
                 "selection_value":selection_value}
            log.append(row); print(json.dumps(row,sort_keys=True),flush=True)
            if selection_value<best_loss:
                best_loss=selection_value; best_update=update; best_state=copy.deepcopy(model.state_dict())
            elif update-best_update>=args.patience: break
    model.load_state_dict(best_state); model.eval().cpu()

    reports={}
    for split,(pose,context,target,nominal_phys,target_phys,episode_ids) in tensors.items():
        reports[split]={"rows":len(pose),"episodes":len(set(episode_ids)),
                        **dependence_metrics(model,pose,context,nominal_phys,target_phys,normalization)}
    calibration=reports["calibration"]
    true_mae=calibration["true_pose"]["translation_mae_mm"]
    zero_mae=calibration["zero_pose"]["translation_mae_mm"]
    gates={
        "true_pose_improves_translation_mae_by_5pct": true_mae <= 0.95*zero_mae,
        "correction_vector_p50_at_least_0p03mm": calibration["correction_vector_p50_mm"] >= 0.03,
        "correction_target_sign_accuracy_at_least_0p65":
            (calibration["correction_target_sign_accuracy"] or 0.0) >= 0.65,
    }
    gates["offline_gate_passed"] = all(gates.values())
    bundle={
        "schema_version":1,
        "architecture":"frozen action-only GRU plus mandatory odd-symmetric predicted-pose residual with temporal magnitude gate",
        "nominal_checkpoint":str(args.nominal_checkpoint),"pose_checkpoint":str(args.pose_checkpoint),
        "replay":str(args.replay),"dataset_manifest":str(args.dataset_manifest),
        "history":history,"context_dim":context_dim,"correction_hidden_dim":96,
        "normalization":normalization,"correction_state_dict":model.state_dict(),
        "reports":reports,"gates":gates,"best_update":best_update,"completed_updates":update,
        "selection_metric":args.selection_metric,"best_selection_value":best_loss,
        "training_log":log,"dimensions":dimensions,"reserved_final_opened":False,"rl_started":False,
    }
    torch.save(bundle,args.output_dir/"explicit_pose_correction_checkpoint.pt")
    summary={key:value for key,value in bundle.items() if key not in ("normalization","correction_state_dict")}
    (args.output_dir/"metrics.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"reports":reports,"gates":gates,"best_update":best_update},indent=2))


if __name__ == "__main__":
    main()
