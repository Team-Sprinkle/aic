#!/usr/bin/env python3
"""Distill deterministic RPDP BC into SAC-compatible trajectory policies."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


geo = load("rpdp_serl_geometry", "rpdp_geometry.py")
bc = load("rpdp_serl_bc", "train_rpdp_diffusion.py")
policy_module = load("rpdp_serl_policy_module", "rpdp_serl_policy.py")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=8000)
    parser.add_argument("--patience", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--covariance-rank", type=int, default=4)
    parser.add_argument("--mixture-lateral-std-units", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def pose9_to_pose6(value: torch.Tensor) -> torch.Tensor:
    p, q = geo.unpack_pose9(value)
    return torch.cat((p, geo.quat_to_rotvec(q)), dim=-1)


def pose6_to_action(
    future_pose6: torch.Tensor,
    current_pose9: torch.Tensor,
    adapter: object,
) -> torch.Tensor:
    current_p, current_q = geo.unpack_pose9(current_pose9)
    future_p = future_pose6[..., :3]
    future_q = geo.rotvec_to_quat(future_pose6[..., 3:6])
    return adapter.waypoint_chunk(current_p, current_q, future_p, future_q)


def tensors(rows: list[dict], condition_indices: list[int]):
    condition = torch.stack([row["condition"][condition_indices] for row in rows]).float()
    target = pose9_to_pose6(torch.stack([row["future_pose9"] for row in rows]).float())
    current = torch.stack([row["current_pose9"] for row in rows]).float()
    action = torch.stack([row.get("target_action", row["executed_action"]) for row in rows]).float()
    mask = torch.stack([row["future_mask"] for row in rows]).float()
    return condition, target, current, action, mask


def copy_bc_backbone(policy, bundle, device):
    config = bundle["model"]
    source = bc.TrajectoryDiffusion(
        config["condition_dim"],
        config["horizon"],
        config["width"],
        config["layers"],
        fusion=config.get("fusion", False),
        visual_dim=config.get("visual_dim", 0),
    ).to(device)
    source.load_state_dict(bundle["model_state_dict"])
    with torch.no_grad():
        if config.get("fusion", False):
            policy.visual_condition.load_state_dict(source.visual_condition.state_dict())
            policy.pose_condition.load_state_dict(source.pose_condition.state_dict())
            policy.visual_gate.load_state_dict(source.visual_gate.state_dict())
        else:
            policy.condition.load_state_dict(source.condition.state_dict())
        policy.blocks.load_state_dict(source.blocks.state_dict())
        policy.output_norm.load_state_dict(source.output[0].state_dict())
        zero = torch.zeros((1, config["horizon"], 9), device=device)
        t = torch.zeros((1,), dtype=torch.long, device=device)
        absorbed = source.input(zero) + source.position + source.time(bc.timestep_embedding(t, 64))[:, None]
        policy.query.copy_(absorbed)


def quantiles(value: torch.Tensor):
    flat = value.detach().float().reshape(-1).cpu()
    return {
        "mean": float(flat.mean()),
        "median": float(flat.median()),
        "p95": float(torch.quantile(flat, 0.95)),
        "max": float(flat.max()),
    }


@torch.no_grad()
def evaluate(policy, data, normalization, adapter, batch_size, device):
    condition, target, current, action, mask = data
    predictions = []
    selected_components = []
    latencies = []
    policy.eval()
    for start in range(0, len(condition), batch_size):
        c = ((condition[start : start + batch_size].to(device) - normalization["condition_mean"])
             / normalization["condition_std"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        normalized, selected = policy.mode(c)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - t0) * 1000.0 / max(1, len(c)))
        pred = normalized.reshape(-1, 4, 6) * normalization["target_std"] + normalization["target_mean"]
        predictions.append(pred.cpu())
        selected_components.append(selected.cpu())
    prediction = torch.cat(predictions)
    selected = torch.cat(selected_components)
    translation = torch.linalg.norm((prediction[..., :3] - target[..., :3]) * 1000.0, dim=-1)
    rotation = torch.linalg.norm(prediction[..., 3:] - target[..., 3:], dim=-1) * 180.0 / math.pi
    predicted_action = pose6_to_action(prediction, current, adapter)
    action_translation = torch.linalg.norm((predicted_action[..., :3] - action[..., :3]) * 1000.0, dim=-1)
    action_rotation = torch.linalg.norm(predicted_action[..., 3:] - action[..., 3:], dim=-1) * 180.0 / math.pi
    valid = mask > 0.5
    logits, means, _, _ = policy.parameters_for_distribution(
        ((condition[: min(len(condition), batch_size)].to(device) - normalization["condition_mean"])
         / normalization["condition_std"])
    )
    return {
        "rows": len(condition),
        "translation_error_mm": quantiles(translation[valid]),
        "orientation_error_deg": quantiles(rotation[valid]),
        "adapter_translation_error_mm": quantiles(action_translation[valid]),
        "adapter_orientation_error_deg": quantiles(action_rotation[valid]),
        "selected_component_counts": torch.bincount(selected, minlength=policy.config.components).tolist(),
        "component_separation_normalized": float(policy_module.component_separation(means).cpu()),
        "inference_ms_per_row": quantiles(torch.tensor(latencies)),
    }


def main():
    args = arguments()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=False)
    bundle = torch.load(args.bc_checkpoint, map_location="cpu", weights_only=False)
    if bundle.get("policy_mode") != "direct" or bundle.get("trajectory_semantics") != "microstep_chunk":
        raise ValueError("Expected the selected direct microstep-chunk BC checkpoint")
    indices = list(bundle["condition_indices"])
    split_data = {name: tensors(rows, indices) for name, rows in dataset["splits"].items()}
    fit_condition, fit_target, _, _, fit_mask = split_data["fit"]
    condition_mean = fit_condition.mean(0)
    condition_std = fit_condition.std(0).clamp_min(1.0e-6)
    target_flat = fit_target[fit_mask > 0.5]
    target_mean = target_flat.mean(0).reshape(1, 1, 6)
    target_std = target_flat.std(0).clamp_min(1.0e-6).reshape(1, 1, 6)
    normalization = {
        "condition_mean": condition_mean.to(device),
        "condition_std": condition_std.to(device),
        "target_mean": target_mean.to(device),
        "target_std": target_std.to(device),
    }
    config = policy_module.MixturePolicyConfig(
        condition_dim=len(indices),
        horizon=4,
        width=int(bundle["model"]["width"]),
        layers=int(bundle["model"]["layers"]),
        components=1,
        covariance_rank=int(args.covariance_rank),
        fusion=bool(bundle["model"].get("fusion", False)),
        visual_dim=int(bundle["model"].get("visual_dim", 0)),
    )
    policy = policy_module.PortTrajectoryMixturePolicy(config).to(device)
    copy_bc_backbone(policy, bundle, device)
    # Supervised distillation learns the complete minimal pose trajectory.  The
    # variance heads remain fixed until online SAC so BC fitting cannot collapse
    # their variance toward zero.
    for module in (policy.logit_head, policy.log_std_head, policy.factor_head):
        module.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=1.0e-5,
    )
    normalized_fit_condition = (fit_condition - condition_mean) / condition_std
    normalized_fit_target = (fit_target - target_mean.cpu()) / target_std.cpu()
    best = None
    best_update = 0
    stale = 0
    history = []
    completed_updates = 0
    for update in range(1, args.updates + 1):
        completed_updates = update
        policy.train()
        index = torch.randint(0, len(fit_condition), (args.batch_size,))
        condition_batch = normalized_fit_condition[index].to(device)
        target_batch = normalized_fit_target[index].reshape(args.batch_size, -1).to(device)
        mask_batch = fit_mask[index].to(device).unsqueeze(-1).expand(-1, -1, 6).reshape(args.batch_size, -1)
        prediction, _ = policy.mode(condition_batch)
        loss = F.smooth_l1_loss(prediction * mask_batch, target_batch * mask_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()
        if update == 1 or update % 100 == 0 or update == args.updates:
            report = evaluate(policy, split_data["calibration"], normalization, geo.ConnectorTCPAdapter(
                bundle["tcp_to_connector_p"], bundle["tcp_to_connector_q"]), args.batch_size, device)
            score = (
                report["adapter_translation_error_mm"]["median"]
                + 0.25 * report["adapter_translation_error_mm"]["p95"]
                + report["adapter_orientation_error_deg"]["median"]
            )
            history.append({"update": update, "loss": float(loss.detach().cpu()), "selection_score": score})
            if best is None or score < best:
                best = score
                best_update = update
                stale = 0
                torch.save(policy.state_dict(), args.output_dir / "single_best_state.pt")
            else:
                stale += 100
            if stale >= args.patience:
                break
    policy.load_state_dict(torch.load(args.output_dir / "single_best_state.pt", map_location=device, weights_only=True))
    for module in (policy.logit_head, policy.log_std_head, policy.factor_head):
        module.requires_grad_(True)
    policy.eval()
    mixture = policy_module.expand_single_to_mixture(
        policy,
        components=args.components,
        lateral_std_units=args.mixture_lateral_std_units,
    ).to(device).eval()
    adapter = geo.ConnectorTCPAdapter(bundle["tcp_to_connector_p"], bundle["tcp_to_connector_q"])
    reports = {
        "single": {name: evaluate(policy, data, normalization, adapter, args.batch_size, device)
                   for name, data in split_data.items()},
        "mixture_initialization": {name: evaluate(mixture, data, normalization, adapter, args.batch_size, device)
                                   for name, data in split_data.items()},
    }
    common = {
        "schema_version": 1,
        "source_bc_checkpoint": str(args.bc_checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "condition_indices": indices,
        "normalization": {key: value.detach().cpu() for key, value in normalization.items()},
        "tcp_to_connector_p": bundle["tcp_to_connector_p"],
        "tcp_to_connector_q": bundle["tcp_to_connector_q"],
        "trajectory_semantics": "four_port_frame_pose6_waypoints",
        "policy_action_dim": 24,
        "executed_action_dim": 24,
        "reserved_final_opened": False,
        "rl_started": False,
    }
    torch.save({**common, "config": dict(policy.config.__dict__), "model_state_dict": policy.state_dict()},
               args.output_dir / "single_gaussian_bc.pt")
    torch.save({**common, "config": dict(mixture.config.__dict__), "model_state_dict": mixture.state_dict()},
               args.output_dir / "mixture_bc.pt")
    metrics = {
        "best_update": best_update,
        "completed_updates": completed_updates,
        "history": history,
        "reports": reports,
        "single_summary": policy_module.policy_summary(policy),
        "mixture_summary": policy_module.policy_summary(mixture),
        "command": " ".join(sys.argv),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
