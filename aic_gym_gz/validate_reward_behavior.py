"""Validation-only reward sanity checks for the dense RL reward."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .validation_common import pearson_correlation, policy_specs, rollout_policy


def _policy_summary(rollout: dict[str, Any]) -> dict[str, Any]:
    records = rollout["records"]
    rewards = [float(record["rl_step_reward"]) for record in records]
    distances = [float(record["distance_to_target"]) for record in records]
    action_norms = [float(record["action_l2_norm"]) for record in records]
    contacts = [bool(record["off_limit_contact"]) for record in records]
    final_eval = rollout.get("final_evaluation") or {}
    cumulative_reward = float(sum(rewards))
    reward_nonzero_fraction = (
        float(sum(1 for reward in rewards if abs(reward) > 1e-9)) / len(rewards) if rewards else 0.0
    )
    return {
        "policy": rollout["policy"],
        "description": rollout["description"],
        "num_steps": rollout["num_steps"],
        "terminated": bool(rollout["terminated"]),
        "truncated": bool(rollout["truncated"]),
        "cumulative_rl_reward": cumulative_reward,
        "mean_rl_step_reward": float(np.mean(rewards)) if rewards else 0.0,
        "reward_nonzero_fraction": reward_nonzero_fraction,
        "distance_delta": (distances[-1] - distances[0]) if len(distances) >= 2 else 0.0,
        "reward_vs_distance_correlation": pearson_correlation(rewards, distances),
        "reward_vs_action_magnitude_correlation": pearson_correlation(rewards, action_norms),
        "max_force_l2_norm": max((float(record["wrench_force_l2_norm"]) for record in records), default=0.0),
        "contact_steps": sum(1 for contact in contacts if contact),
        "final_gym_final_score": final_eval.get("gym_final_score"),
        "final_official_eval_score": final_eval.get("official_eval_score"),
    }


def _write_csv(*, output_dir: Path, rollout: dict[str, Any]) -> None:
    path = output_dir / f"{rollout['policy']}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step_idx",
                "rl_step_reward",
                "cumulative_rl_reward",
                "distance_to_target",
                "distance_to_entrance",
                "action_l2_norm",
                "tcp_velocity_l2_norm",
                "wrench_force_l2_norm",
                "wrench_timestamp",
                "off_limit_contact",
                "reward_terms_json",
                "reward_metrics_json",
            ]
        )
        cumulative = 0.0
        for record in rollout["records"]:
            cumulative += float(record["rl_step_reward"])
            writer.writerow(
                [
                    record["step_idx"],
                    record["rl_step_reward"],
                    cumulative,
                    record["distance_to_target"],
                    record["distance_to_entrance"],
                    record["action_l2_norm"],
                    record["tcp_velocity_l2_norm"],
                    record["wrench_force_l2_norm"],
                    record["wrench_timestamp"],
                    int(record["off_limit_contact"]),
                    json.dumps(record["reward_terms"], sort_keys=True),
                    json.dumps(record["reward_metrics"], sort_keys=True),
                ]
            )


def run_validation(
    *,
    policies: list[str],
    seed: int,
    max_steps: int,
    ticks_per_step: int,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    rollouts = [
        rollout_policy(
            policy_name=policy,
            seed=seed + index,
            max_steps=max_steps,
            ticks_per_step=ticks_per_step,
        )
        for index, policy in enumerate(policies)
    ]
    summaries = [_policy_summary(rollout) for rollout in rollouts]
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for rollout in rollouts:
            _write_csv(output_dir=output_dir, rollout=rollout)
        (output_dir / "reward_behavior_summary.json").write_text(
            json.dumps({"rollouts": rollouts, "summaries": summaries}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {"rollouts": rollouts, "summaries": summaries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[
            "random",
            "heuristic",
            "toward_target",
            "aggressive_toward_target",
            "away_from_target",
            "oscillate_in_place",
            "collide_intentionally",
        ],
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--ticks-per-step", type=int, default=8)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    available = policy_specs(seed=args.seed)
    unknown = sorted(set(args.policies) - set(available))
    if unknown:
        raise SystemExit(f"Unknown policies: {unknown}. Available: {sorted(available)}")

    output_dir = Path(args.output_dir) if args.output_dir else None
    payload = run_validation(
        policies=args.policies,
        seed=args.seed,
        max_steps=args.max_steps,
        ticks_per_step=args.ticks_per_step,
        output_dir=output_dir,
    )
    for summary in payload["summaries"]:
        print(
            f"policy={summary['policy']} steps={summary['num_steps']} "
            f"cumulative_rl_reward={summary['cumulative_rl_reward']:.3f} "
            f"distance_delta={summary['distance_delta']:.4f} "
            f"nonzero_fraction={summary['reward_nonzero_fraction']:.3f} "
            f"contact_steps={summary['contact_steps']} "
            f"reward_vs_distance_corr={summary['reward_vs_distance_correlation']} "
            f"reward_vs_action_corr={summary['reward_vs_action_magnitude_correlation']} "
            f"gym_final_score={summary['final_gym_final_score']}"
        )
    print(json.dumps(payload["summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
