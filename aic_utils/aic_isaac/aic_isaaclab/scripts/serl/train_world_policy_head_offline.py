#!/usr/bin/env python3
"""Fit the deployable world-policy head to 24D guide chunks from Isaac replay."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from world_policy_actor import IsaacWorldPolicyActor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-policy-checkpoint", type=Path, required=True)
    parser.add_argument("--world-policy-source", type=Path, required=True)
    parser.add_argument("--replay", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--rotation-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--near-port-oversample",
        type=float,
        default=1.0,
        help="Sampling multiplier for rows at signed depth >= -3 mm; 1 disables phase weighting.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=800,
        help="Stop after this many updates without a validation improvement; <=0 disables early stopping.",
    )
    return parser.parse_args()


def load_examples(paths: list[Path]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    features: list[torch.Tensor] = []
    guides: list[torch.Tensor] = []
    near_port: list[bool] = []
    sources: list[dict[str, object]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        transitions = payload.get("transitions") if isinstance(payload, dict) else None
        if not isinstance(transitions, list):
            raise ValueError(f"Replay has no transition list: {path}")
        accepted = 0
        for transition in transitions:
            feature = (transition.get("obs") or {}).get("world_feature")
            guide = transition.get("guide_action")
            if not isinstance(feature, torch.Tensor) or not isinstance(guide, torch.Tensor):
                continue
            feature = feature.reshape(-1).float()
            guide = guide.reshape(-1).float()
            if feature.numel() != 384 or guide.numel() != 24:
                continue
            if not bool(torch.isfinite(feature).all()) or not bool(torch.isfinite(guide).all()):
                continue
            features.append(feature)
            guides.append(guide)
            geometry = (transition.get("metadata") or {}).get("post_step_insertion_geometry") or {}
            depth = geometry.get("signed_depth_m_mean")
            near_port.append(depth is not None and float(depth) >= -0.003)
            accepted += 1
        sources.append({"path": str(path), "transitions": len(transitions), "accepted": accepted})
    if not features:
        raise ValueError("No replay rows contain a finite 384D world feature and 24D guide chunk")
    return torch.stack(features), torch.stack(guides), torch.tensor(near_port, dtype=torch.bool), sources


def guide_loss(prediction: torch.Tensor, target: torch.Tensor, rotation_weight: float) -> tuple[torch.Tensor, float, float]:
    pred = prediction.reshape(-1, 4, 6)
    truth = target.reshape(-1, 4, 6)
    translation = F.l1_loss(pred[:, :, :3], truth[:, :, :3])
    rotation = F.l1_loss(pred[:, :, 3:], truth[:, :, 3:])
    return translation + float(rotation_weight) * rotation, float(translation.detach().cpu()), float(rotation.detach().cpu())


def main() -> None:
    args = parse_args()
    if args.updates <= 0 or args.batch_size <= 0:
        raise ValueError("--updates and --batch-size must be positive")
    if args.near_port_oversample <= 0.0:
        raise ValueError("--near-port-oversample must be positive")
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in [0, 1)")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    features, guides, near_port, sources = load_examples(args.replay)
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(features.shape[0], generator=generator)
    validation_count = min(features.shape[0] - 1, max(1, round(features.shape[0] * args.validation_fraction)))
    validation_indices = order[:validation_count]
    train_indices = order[validation_count:]
    train_features = features[train_indices].to(device)
    train_guides = guides[train_indices].to(device)
    train_weights = torch.where(
        near_port[train_indices].to(device),
        torch.full((train_indices.numel(),), max(float(args.near_port_oversample), 0.0), device=device),
        torch.ones((train_indices.numel(),), device=device),
    )
    validation_features = features[validation_indices].to(device)
    validation_guides = guides[validation_indices].to(device)

    actor = IsaacWorldPolicyActor(
        checkpoint=args.world_policy_checkpoint,
        source=args.world_policy_source,
        device=device,
    ).to(device)
    actor.eval()
    optimizer = torch.optim.Adam(actor.head.parameters(), lr=args.learning_rate)
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_update = 0
    best_head_state = copy.deepcopy(actor.head.state_dict())
    stopped_early = False
    for update in range(1, args.updates + 1):
        sample = torch.multinomial(train_weights, args.batch_size, replacement=True)
        prediction = actor.mean_action({"world_feature": train_features[sample]})
        loss, translation_loss, rotation_loss = guide_loss(
            prediction,
            train_guides[sample],
            args.rotation_loss_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(actor.head.parameters(), 10.0).detach().cpu())
        optimizer.step()
        if update == 1 or update % args.log_every == 0 or update == args.updates:
            with torch.no_grad():
                validation_prediction = actor.mean_action({"world_feature": validation_features})
                validation_loss, validation_translation, validation_rotation = guide_loss(
                    validation_prediction,
                    validation_guides,
                    args.rotation_loss_weight,
                )
            row = {
                "update": update,
                "train_loss": float(loss.detach().cpu()),
                "train_translation_l1": translation_loss,
                "train_rotation_l1": rotation_loss,
                "validation_loss": float(validation_loss.detach().cpu()),
                "validation_translation_l1": validation_translation,
                "validation_rotation_l1": validation_rotation,
                "head_grad_norm": grad_norm,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if float(validation_loss) < best_validation_loss:
                best_validation_loss = float(validation_loss)
                best_update = update
                best_head_state = copy.deepcopy(actor.head.state_dict())
            elif args.early_stop_patience > 0 and update - best_update >= args.early_stop_patience:
                stopped_early = True
                print(
                    f"Early stopping at update {update}; best validation loss was "
                    f"{best_validation_loss:.8g} at update {best_update}",
                    flush=True,
                )
                break

    actor.head.load_state_dict(best_head_state)

    summary = {
        "base_world_policy_checkpoint": str(args.world_policy_checkpoint),
        "world_policy_source": str(args.world_policy_source),
        "sources": sources,
        "example_count": int(features.shape[0]),
        "train_count": int(train_features.shape[0]),
        "validation_count": int(validation_features.shape[0]),
        "requested_updates": args.updates,
        "completed_updates": update,
        "updates": update,
        "best_update": best_update,
        "best_validation_loss": best_validation_loss,
        "stopped_early": stopped_early,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "rotation_loss_weight": args.rotation_loss_weight,
        "near_port_oversample": args.near_port_oversample,
        "near_port_example_count": int(near_port.sum()),
        "seed": args.seed,
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": actor.state_dict(), "world_policy_head_offline": summary}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote trained world-policy actor: {args.output}", flush=True)


if __name__ == "__main__":
    main()
