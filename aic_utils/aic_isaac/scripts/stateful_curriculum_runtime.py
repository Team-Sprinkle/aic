#!/usr/bin/env python3
"""Episode accounting and curriculum decisions, usable without Isaac or Torch."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


class EpisodeTracker:
    """Count completed vector-env episodes using terminal signals, not geometry samples."""

    def __init__(self, num_envs: int):
        self.indices = [0] * num_envs
        self.lengths = [0] * num_envs
        self.completed = 0
        self.succeeded = 0

    def observe(self, *, step, terminated, truncated, success, metadata, reasons):
        size = len(self.indices)
        if any(len(values) != size for values in (terminated, truncated, success, metadata, reasons)):
            raise ValueError("Episode accounting arrays must match num_envs")
        outcomes = []
        for env_id in range(size):
            self.lengths[env_id] += 1
            if not (terminated[env_id] or truncated[env_id]):
                continue
            # A simultaneous failure term takes precedence over success. A time
            # limit alone may coincide with successful final seating.
            failures = set(reasons[env_id]) - {"target_success", "time_out"}
            succeeded = None if success[env_id] is None else bool(success[env_id] and not failures)
            outcomes.append({
                "env_index": env_id,
                "episode_index": self.indices[env_id],
                "config_episode_id": metadata[env_id].get("episode_id"),
                "step": step,
                "length": self.lengths[env_id],
                "terminated": bool(terminated[env_id]),
                "truncated": bool(truncated[env_id]),
                "success": succeeded,
                "termination_reasons": list(reasons[env_id]),
            })
            self.completed += 1
            self.succeeded += int(succeeded is True)
            self.indices[env_id] += 1
            self.lengths[env_id] = 0
        return outcomes


def progression_settings(config):
    progression = config.get("progression") or {}
    episodes = int(config.get("episodes", 2000))
    interval = int(progression.get("eval_every_episodes", 10))
    if episodes <= 0 or interval <= 0:
        raise ValueError("episodes and eval_every_episodes must be positive")
    levels = int(progression.get("level_count", math.ceil(episodes / interval)))
    rate = float(progression.get("promotion_success_rate", 0.8))
    demotion = int(progression.get("demotion_on_consecutive_failures", 3))
    if progression.get("policy") != "promotion_gated":
        raise ValueError("progression.policy must be promotion_gated")
    if levels <= 0 or not 0 < rate <= 1 or demotion <= 0:
        raise ValueError("Invalid level_count, promotion_success_rate, or demotion threshold")
    return episodes, interval, levels, rate, demotion


def summarize_run(run_dir: Path, *, min_episodes=1, evaluation=False, expected_ids=None):
    """Reject legacy/sampled or incomplete evidence instead of guessing success."""
    config = json.loads((run_dir / "train_config.json").read_text())
    args = config.get("args") or {}
    contract_keys = (
        "terminate_on_target_success", "target_reward_consistency_body",
        "target_success_axial_threshold", "target_success_lateral_threshold",
        "target_success_orientation_threshold", "target_success_consistency_axial_threshold",
        "target_success_consistency_lateral_threshold", "target_reward_orientation_error_mode",
        "target_reward_orientation_axis_local", "episode_target_orientation_source",
        "replace_sfp_body_sdf_collision_with_shrunk_sdf_boxes", "replace_nic_cage_p0_with_aligned_cubes",
    )
    contract = {key: args.get(key) for key in contract_keys}
    if evaluation:
        if args.get("terminate_on_target_success") is not True or args.get("target_reward_consistency_body") != "sfp_module_link":
            raise ValueError("Curriculum evaluation requires target-success termination and SFP module consistency")
        for key in contract_keys[2:7]:
            value = float(args.get(key) or 0)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Missing or disabled success threshold: {key}")
    result = config.get("result") or {}
    if result.get("stop_reason") not in {"max_completed_episodes", "max_steps", "max_wall_time", "target_updates"}:
        raise ValueError(f"Run did not finish normally: {result.get('stop_reason')}")
    if evaluation and result.get("updates_done") != 0:
        raise ValueError("Evaluation performed gradient updates")
    outcomes = []
    seen = set()
    count = 0
    last_step = 0
    maxima = {"guide_blend_max": 0.0, "guard_max": 0.0, "executed_minus_actor_max": 0.0}
    keys = {
        "target_action_guide_collect_blend_effective": "guide_blend_max",
        "insertion_action_guard_applied_fraction": "guard_max",
        "executed_minus_actor_l1_mean": "executed_minus_actor_max",
    }
    with (run_dir / "metrics.jsonl").open() as source:
        for line_no, line in enumerate(source, 1):
            try:
                row = json.loads(line)
                if int(row["step"]) <= last_step:
                    raise ValueError("Steps must increase")
                last_step = int(row["step"])
                current = row["episode_outcomes"]
                for outcome in current:
                    identity = (outcome["env_index"], outcome["episode_index"])
                    if identity in seen or not isinstance(outcome["success"], bool):
                        raise ValueError("Duplicate episode or unknown terminal success")
                    if not (outcome["terminated"] or outcome["truncated"]):
                        raise ValueError("Outcome is not a completed episode")
                    seen.add(identity)
                    outcomes.append(outcome)
                if row["completed_episodes_total"] != len(outcomes):
                    raise ValueError("Episode counter disagrees with terminal records")
                for key, destination in keys.items():
                    value = row.get(key)
                    if value is None and key == "target_action_guide_collect_blend_effective":
                        value = 0.0
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError(f"Nonfinite {key}")
                    maxima[destination] = max(maxima[destination], abs(value))
                count += 1
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid accounting in metrics.jsonl line {line_no}: {exc}") from exc
    if count == 0:
        raise ValueError("Empty metrics; no episode evidence")
    if result.get("episodes_completed") != len(outcomes) or result.get("steps_completed") != last_step:
        raise ValueError("Final result disagrees with metrics")
    if evaluation and any(value > 1e-9 for value in maxima.values()):
        raise ValueError("Evaluation executed guide/guard/overridden actions")
    if evaluation and float((config.get("args") or {}).get("actor_exploration_noise_std", 0.0)) != 0.0:
        raise ValueError("Evaluation enabled exploration noise")
    completed_ids = {outcome["config_episode_id"] for outcome in outcomes}
    missing_ids = sorted(set(expected_ids or ()) - completed_ids)
    successes = sum(outcome["success"] is True for outcome in outcomes)
    return {
        "schema_version": 1,
        "rows": count,
        "last_step": last_step,
        "completed_episodes": len(outcomes),
        "successful_episodes": successes,
        "success_rate": successes / len(outcomes) if outcomes else None,
        "evaluation_complete": len(outcomes) >= min_episodes and not missing_ids,
        "missing_episode_config_ids": missing_ids,
        "episode_outcomes": outcomes,
        "evaluation_contract": contract,
        **maxima,
    }


def promotion_decision(summary, *, level, level_count, failures, success_rate, demote_after):
    if not summary.get("evaluation_complete") or not summary.get("completed_episodes"):
        raise ValueError("Cannot promote or demote from incomplete evaluation")
    if not 0 <= level < level_count or failures < 0 or not 0 < success_rate <= 1 or demote_after <= 0:
        raise ValueError("Invalid progression state")
    successes = summary["successful_episodes"]
    completed = summary["completed_episodes"]
    if not 0 <= successes <= completed:
        raise ValueError("Invalid success denominator")
    if successes / completed >= success_rate:
        return level + 1, 0, "promoted"
    failures += 1
    if failures >= demote_after:
        return max(0, level - 1), 0, "demoted" if level else "held_at_level_zero"
    return level, failures, "held"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    settings = commands.add_parser("settings")
    settings.add_argument("config", type=Path)
    summary = commands.add_parser("summarize")
    summary.add_argument("run_dir", type=Path)
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--min-episodes", type=int, default=1)
    summary.add_argument("--evaluation", action="store_true")
    summary.add_argument("--episode-config-dir", type=Path)
    decision = commands.add_parser("decide")
    decision.add_argument("summary", type=Path)
    decision.add_argument("--level", type=int, required=True)
    decision.add_argument("--level-count", type=int, required=True)
    decision.add_argument("--failures", type=int, required=True)
    decision.add_argument("--success-rate", type=float, required=True)
    decision.add_argument("--demote-after", type=int, required=True)
    args = parser.parse_args()
    if args.operation == "settings":
        print(*progression_settings(yaml.safe_load(args.config.read_text())))
    elif args.operation == "summarize":
        expected = None
        if args.episode_config_dir:
            expected = [yaml.safe_load(p.read_text())["episode_id"]
                        for p in sorted(args.episode_config_dir.glob("episode_*.yaml"))]
            if not expected:
                raise ValueError("No expected evaluation episodes")
        report = summarize_run(args.run_dir, min_episodes=args.min_episodes,
                               evaluation=args.evaluation, expected_ids=expected)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if not report["evaluation_complete"]:
            raise ValueError("Insufficient completed episodes or missing evaluation configurations")
        print(report["completed_episodes"])
    else:
        print(*promotion_decision(json.loads(args.summary.read_text()), level=args.level,
                                  level_count=args.level_count, failures=args.failures,
                                  success_rate=args.success_rate, demote_after=args.demote_after))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
