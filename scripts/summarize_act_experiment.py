#!/usr/bin/env python3
"""Summarize saved ACT training and completed official runtime attempts."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def summarize(root):
    training, evaluation = [], []
    for path in sorted(root.glob("act_*/training_config.json")):
        config = json.loads(path.read_text())
        directory = path.parent
        completion_path = directory / "completion.json"
        completion = json.loads(completion_path.read_text()) if completion_path.exists() else None
        metrics = read_jsonl(directory / "metrics.jsonl")
        validation = read_jsonl(directory / "validation.jsonl")
        best = min(validation, key=lambda row: row["first_action_translation_error_m"]) if validation else None
        training.append({"run": directory.name, "complete": completion is not None,
                         "actual_updates": completion["actual_updates"] if completion else None,
                         "last_logged_step": metrics[-1]["step"] if metrics else None,
                         "train_episode_count": len(config.get("sampled_training_episodes", config["train_episodes"])),
                         "normalization_episode_count": len(config.get("normalization_training_episodes", config["train_episodes"])),
                         "validation_episode_count": len(config["validation_episodes"]),
                         "best_heldout_frame_metric": best, "args": config["args"],
                         "training_config": str(path)})
    # Only the per-checkpoint summary is authoritative. It identifies the selected
    # attempt; do not count both it and its attempt-directory copy.
    for path in sorted(root.glob("*/eval_*/*/eval_summary.json")):
        summary = json.loads(path.read_text())
        score_path = Path(summary.get("scoring_yaml", ""))
        trials = []
        if score_path.is_file():
            scores = yaml.safe_load(score_path.read_text()) or {}
            for name, value in scores.items():
                if not isinstance(value, dict) or "tier_3" not in value:
                    continue
                tiers = {key: float(value.get(key, {}).get("score", 0)) for key in ("tier_1", "tier_2", "tier_3")}
                trials.append({"trial": name, "total_score": sum(tiers.values()), **tiers,
                               "inserted": abs(tiers["tier_3"] - 75.) < 1e-6,
                               "message": value["tier_3"].get("message"),
                               "categories": value.get("tier_2", {}).get("categories", {})})
        kind = summary.get("evaluation_kind")
        if kind is None:
            module = summary.get("policy_module", "")
            kind = ("privileged_expert_diagnostic" if module.endswith(".CheatCode") else
                    "recorded_trajectory_diagnostic" if module.endswith(".RunRecordedACTCommands") else "policy_rollout")
        evaluation.append({"run": path.parents[2].name, "evaluation": path.parents[1].name,
                           "checkpoint": path.parent.name, "evaluation_kind": kind,
                           "complete": bool(summary.get("evaluation_complete")),
                           "insertions": sum(row["inserted"] for row in trials),
                           "trials": trials, "mean_score": sum(row["total_score"] for row in trials) / len(trials) if trials else None,
                           "runtime_settings": summary.get("runtime_settings"), "summary": str(path)})
    return {"updated_utc": datetime.now(timezone.utc).isoformat(), "training": training, "evaluation": evaluation,
            "note": "Pilot scenes may recur across candidates. Never treat this aggregate as a fresh independent reliability sample. Incomplete attempts and privileged/replay diagnostics are not learned-policy successes."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = summarize(args.root)
    (args.root / "results_latest.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# ACT experiment artifact summary", "", report["note"], "", "## Training", "",
             "| Run | Finished updates / latest logged | Train / validation episodes | Best held-out translation error |",
             "| --- | --- | --- | --- |"]
    for row in report["training"]:
        best = row["best_heldout_frame_metric"]
        error = f"{best['first_action_translation_error_m'] * 1000:.3f} mm at {best['step']}" if best else "—"
        steps = str(row["actual_updates"]) if row["complete"] else f"running / {row['last_logged_step']}"
        lines.append(f"| {row['run']} | {steps} | {row['train_episode_count']} / {row['validation_episode_count']} | {error} |")
    lines += ["", "## Official evaluation", "", "| Run / checkpoint | Evaluation | Kind | Complete | Insertions / trials | Mean total |", "| --- | --- | --- | --- | --- | --- |"]
    for row in report["evaluation"]:
        score = f"{row['mean_score']:.2f}" if row["mean_score"] is not None else "—"
        lines.append(f"| {row['run']} / {row['checkpoint']} | {row['evaluation']} | {row['evaluation_kind']} | {row['complete']} | {row['insertions']} / {len(row['trials'])} | {score} |")
    (args.root / "results_latest.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"training_runs": len(report["training"]), "evaluation_groups": len(report["evaluation"]),
                      "report": str(args.root / "results_latest.md")}))


if __name__ == "__main__":
    main()
