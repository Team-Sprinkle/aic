"""Validation-only observation and F/T temporal behavior checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .env import make_default_env


def _phase_action(phase: str) -> np.ndarray:
    action = np.zeros(6, dtype=np.float32)
    if phase == "descend":
        action[2] = -0.25
    elif phase == "backoff":
        action[2] = 0.25
    return action


def run_temporal_validation(
    *,
    ticks_per_step_values: list[int],
    descend_ticks: int,
    backoff_ticks: int,
    seed: int,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    experiments: list[dict[str, Any]] = []
    for ticks_per_step in ticks_per_step_values:
        env = make_default_env(ticks_per_step=ticks_per_step, enable_randomization=False)
        env.task.max_episode_steps = max(
            env.task.max_episode_steps,
            int(np.ceil(descend_ticks / ticks_per_step)) + int(np.ceil(backoff_ticks / ticks_per_step)) + 8,
        )
        try:
            observation, info = env.reset(seed=seed)
            del info
            records: list[dict[str, Any]] = []
            for phase, total_ticks in (("descend", descend_ticks), ("backoff", backoff_ticks)):
                action = _phase_action(phase)
                num_steps = max(1, int(np.ceil(total_ticks / ticks_per_step)))
                for _ in range(num_steps):
                    observation, reward, terminated, truncated, step_info = env.step(action)
                    records.append(
                        {
                            "phase": phase,
                            "sim_tick": int(observation["sim_tick"]),
                            "sim_time": float(observation["sim_time"]),
                            "wrench_timestamp": float(observation["wrench_timestamp"][0]),
                            "wrench_force_l2_norm": float(np.linalg.norm(observation["wrench"][:3])),
                            "tcp_velocity_l2_norm": float(np.linalg.norm(observation["tcp_velocity"])),
                            "distance_to_target": float(step_info["distance_to_target"]),
                            "off_limit_contact": bool(observation["off_limit_contact"][0] > 0.5),
                            "rl_step_reward": float(reward),
                            "reward_terms": dict(step_info["reward_terms"]),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                    )
                    if terminated or truncated:
                        break
                if records and (records[-1]["terminated"] or records[-1]["truncated"]):
                    break
            timestamps = [float(record["wrench_timestamp"]) for record in records]
            contacts = [bool(record["off_limit_contact"]) for record in records]
            first_contact_tick = next(
                (int(record["sim_tick"]) for record in records if record["off_limit_contact"]),
                None,
            )
            first_backoff_clear_tick = next(
                (
                    int(record["sim_tick"])
                    for record in records
                    if first_contact_tick is not None
                    and int(record["sim_tick"]) > first_contact_tick
                    and not record["off_limit_contact"]
                ),
                None,
            )
            experiment = {
                "ticks_per_step": ticks_per_step,
                "num_records": len(records),
                "first_contact_tick": first_contact_tick,
                "first_backoff_clear_tick": first_backoff_clear_tick,
                "wrench_timestamp_monotonic": all(
                    later >= earlier for earlier, later in zip(timestamps, timestamps[1:])
                ),
                "max_wrench_force_l2_norm": max(
                    (float(record["wrench_force_l2_norm"]) for record in records),
                    default=0.0,
                ),
                "max_tcp_velocity_l2_norm": max(
                    (float(record["tcp_velocity_l2_norm"]) for record in records),
                    default=0.0,
                ),
                "contact_steps": sum(1 for contact in contacts if contact),
                "observable_at_policy_level": [
                    "wrench",
                    "wrench_timestamp",
                    "off_limit_contact",
                    "tcp_velocity",
                    "reward_terms",
                ],
                "transient_summary_fields_present": False,
                "records": records,
            }
            experiments.append(experiment)
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                csv_path = output_dir / f"observation_temporal_tps_{ticks_per_step}.csv"
                with csv_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        [
                            "phase",
                            "sim_tick",
                            "sim_time",
                            "wrench_timestamp",
                            "wrench_force_l2_norm",
                            "tcp_velocity_l2_norm",
                            "distance_to_target",
                            "off_limit_contact",
                            "rl_step_reward",
                            "terminated",
                            "truncated",
                            "reward_terms_json",
                        ]
                    )
                    for record in records:
                        writer.writerow(
                            [
                                record["phase"],
                                record["sim_tick"],
                                record["sim_time"],
                                record["wrench_timestamp"],
                                record["wrench_force_l2_norm"],
                                record["tcp_velocity_l2_norm"],
                                record["distance_to_target"],
                                int(record["off_limit_contact"]),
                                record["rl_step_reward"],
                                int(record["terminated"]),
                                int(record["truncated"]),
                                json.dumps(record["reward_terms"], sort_keys=True),
                            ]
                        )
        finally:
            env.close()
    payload = {
        "experiments": experiments,
        "summary": {
            "ticks_per_step_values": ticks_per_step_values,
            "policy_level_observation_is_final_sample_only": True,
            "policy_level_max_or_windowed_ft_summary_present": False,
        },
    }
    if output_dir is not None:
        (output_dir / "observation_temporal_summary.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks-per-step", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--descend-ticks", type=int, default=800)
    parser.add_argument("--backoff-ticks", type=int, default=240)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    payload = run_temporal_validation(
        ticks_per_step_values=args.ticks_per_step,
        descend_ticks=args.descend_ticks,
        backoff_ticks=args.backoff_ticks,
        seed=args.seed,
        output_dir=output_dir,
    )
    for experiment in payload["experiments"]:
        print(
            f"ticks_per_step={experiment['ticks_per_step']} "
            f"first_contact_tick={experiment['first_contact_tick']} "
            f"first_backoff_clear_tick={experiment['first_backoff_clear_tick']} "
            f"contact_steps={experiment['contact_steps']} "
            f"max_wrench_force={experiment['max_wrench_force_l2_norm']:.4f} "
            f"timestamp_monotonic={experiment['wrench_timestamp_monotonic']} "
            f"transient_summary_fields_present={experiment['transient_summary_fields_present']}"
        )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
