"""Compare dense RL reward accumulation against the local final episode score."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .validation_common import pearson_correlation, policy_specs, rollout_policy


def _episode_summary(rollout: dict[str, Any]) -> dict[str, Any]:
    final_report = rollout.get("final_evaluation") or {}
    total_rl_reward = float(sum(record["rl_step_reward"] for record in rollout["records"]))
    return {
        "policy": rollout["policy"],
        "seed": rollout["seed"],
        "trial_id": rollout["trial_id"],
        "num_steps": rollout["num_steps"],
        "terminated": bool(rollout["terminated"]),
        "truncated": bool(rollout["truncated"]),
        "total_rl_reward": total_rl_reward,
        "gym_final_score": final_report.get("gym_final_score"),
        "official_eval_score": final_report.get("official_eval_score"),
    }


def _teacher_rollout_from_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "total_rl_reward" not in payload or "gym_final_score" not in payload:
        raise ValueError(
            f"{path} must contain at least total_rl_reward and gym_final_score for teacher trajectory import."
        )
    return {
        "policy": f"teacher:{path.stem}",
        "seed": payload.get("seed"),
        "trial_id": payload.get("trial_id"),
        "num_steps": payload.get("num_steps"),
        "terminated": payload.get("terminated"),
        "truncated": payload.get("truncated"),
        "total_rl_reward": float(payload["total_rl_reward"]),
        "gym_final_score": float(payload["gym_final_score"]),
        "official_eval_score": payload.get("official_eval_score"),
    }


def run_comparison(
    *,
    policies: list[str],
    episodes: int,
    seed: int,
    max_steps: int,
    ticks_per_step: int,
    teacher_episode_json: list[Path] | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    available = policy_specs(seed=seed)
    unknown = sorted(set(policies) - set(available))
    if unknown:
        raise KeyError(f"Unknown policies: {unknown}. Available: {sorted(available)}")

    episodes_data: list[dict[str, Any]] = []
    for policy in policies:
        for episode_idx in range(episodes):
            rollout = rollout_policy(
                policy_name=policy,
                seed=seed + episode_idx,
                max_steps=max_steps,
                ticks_per_step=ticks_per_step,
            )
            episodes_data.append(_episode_summary(rollout))
    for path in teacher_episode_json or []:
        episodes_data.append(_teacher_rollout_from_json(path))

    rl_rewards = [
        float(item["total_rl_reward"])
        for item in episodes_data
        if item.get("gym_final_score") is not None
    ]
    gym_final_scores = [
        float(item["gym_final_score"])
        for item in episodes_data
        if item.get("gym_final_score") is not None
    ]
    per_policy: dict[str, dict[str, Any]] = {}
    for policy in sorted({str(item["policy"]) for item in episodes_data}):
        rows = [item for item in episodes_data if item["policy"] == policy and item.get("gym_final_score") is not None]
        per_policy[policy] = {
            "num_episodes": len(rows),
            "mean_total_rl_reward": (
                sum(float(item["total_rl_reward"]) for item in rows) / len(rows) if rows else None
            ),
            "mean_gym_final_score": (
                sum(float(item["gym_final_score"]) for item in rows) / len(rows) if rows else None
            ),
            "reward_final_score_correlation": pearson_correlation(
                [float(item["total_rl_reward"]) for item in rows],
                [float(item["gym_final_score"]) for item in rows],
            ),
        }
    payload = {
        "episodes": episodes_data,
        "overall": {
            "num_episodes": len(gym_final_scores),
            "reward_final_score_correlation": pearson_correlation(rl_rewards, gym_final_scores),
        },
        "per_policy": per_policy,
        "notes": [
            "This checks positive alignment between total rl_step_reward and gym_final_score. It does not try to force equality.",
            "official_eval_score remains None unless an external official toolkit result is injected.",
        ],
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        csv_path = output_path.with_suffix(".csv")
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "policy",
                    "seed",
                    "trial_id",
                    "num_steps",
                    "terminated",
                    "truncated",
                    "total_rl_reward",
                    "gym_final_score",
                    "official_eval_score",
                ]
            )
            for item in episodes_data:
                writer.writerow(
                    [
                        item["policy"],
                        item["seed"],
                        item["trial_id"],
                        item["num_steps"],
                        item["terminated"],
                        item["truncated"],
                        item["total_rl_reward"],
                        item["gym_final_score"],
                        item["official_eval_score"],
                    ]
                )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", default=["random", "heuristic"])
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--ticks-per-step", type=int, default=8)
    parser.add_argument("--teacher-episode-json", nargs="*", default=[])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    payload = run_comparison(
        policies=args.policies,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        ticks_per_step=args.ticks_per_step,
        teacher_episode_json=[Path(path) for path in args.teacher_episode_json],
        output_path=Path(args.output) if args.output else None,
    )
    print(
        f"overall_reward_final_score_correlation={payload['overall']['reward_final_score_correlation']} "
        f"episodes={payload['overall']['num_episodes']}"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
