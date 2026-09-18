#!/usr/bin/env python3
"""Validate a frozen ACT bundle and report every predeclared final trial."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import yaml


RUNTIME_KEYS = (
    "policy_module", "policy_device", "command_mode", "command_frame",
    "control_clock", "control_hz", "n_action_steps", "max_runtime_sec",
    "translation_deadband", "rotation_deadband", "image_channel_order",
    "translation_limit_mode", "max_translation_delta", "max_rotation_delta",
    "delta_pose_reference", "temporal_ensemble_coeff", "start_delay_sec",
    "diagnostic_ground_truth", "connect_scoring_world_frames",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(bundle):
    bundle = Path(bundle).resolve()
    frozen = json.loads((bundle / "frozen_selection.json").read_text())
    for relative, expected in frozen["file_sha256"].items():
        if digest(bundle / relative) != expected:
            raise ValueError(f"Frozen file changed: {relative}")
    trials = frozen["trials"]
    expected = frozen["expected_trials"]
    if (len(trials) != expected or len({x["trial"] for x in trials}) != expected
            or len({x["scene_task_sha256"] for x in trials}) != expected):
        raise ValueError("Final trial IDs and scenes must be unique and match the declared count")
    selected = frozen["settings_inherited_from_development"]
    export = bundle / "act_selected_cuda0.pt"
    metadata = json.loads(export.with_suffix(".json").read_text())
    checkpoint = Path(metadata["checkpoint_dir"]).resolve()
    rows = []
    for trial in trials:
        row = {"trial": trial["trial"], "scene_task_sha256": trial["scene_task_sha256"],
               "summary": trial["expected_summary"], "complete": False, "inserted": False}
        path = Path(row["summary"])
        if not path.is_file():
            row["status"] = "missing"
            rows.append(row)
            continue
        summary = json.loads(path.read_text())
        runtime = summary["runtime_settings"]
        if (summary.get("evaluation_kind") != "policy_rollout"
                or summary.get("evaluation_purpose") != "final_reliability"
                or runtime.get("diagnostic_ground_truth") is not False):
            raise ValueError(f"Not an unprivileged final policy evaluation: {path}")
        for key in RUNTIME_KEYS:
            if runtime.get(key) != selected.get(key):
                raise ValueError(f"Final runtime changed {key}: {path}")
        if (summary["runtime_source_sha256"] != frozen["runtime_source_sha256"]
                or Path(summary["checkpoint"]).resolve() != checkpoint
                or Path(runtime["act_torchscript"]).resolve() != export
                or runtime["container"] != trial.get("container", selected["container"])
                or digest(runtime["engine_config_host"]) != trial["config_sha256"]):
            raise ValueError(f"Final provenance disagrees with frozen selection: {path}")
        if not summary.get("evaluation_complete") or not summary.get("policy_ready") or summary.get("engine_returncode") != 0:
            row["status"] = "incomplete"
            rows.append(row)
            continue
        scores = yaml.safe_load(Path(summary["scoring_yaml"]).read_text())
        named = {k: v for k, v in scores.items() if isinstance(v, dict) and "tier_3" in v}
        if set(named) != {trial["trial"]}:
            raise ValueError(f"Unexpected or duplicate official trial mapping: {path}")
        score = named[trial["trial"]]
        tiers = {key: float(score[key]["score"]) for key in ("tier_1", "tier_2", "tier_3")}
        if not all(math.isfinite(value) for value in tiers.values()):
            raise ValueError(f"Nonfinite official score: {path}")
        row.update(status="scored", complete=True, inserted=abs(tiers["tier_3"] - 75.) < 1e-6,
                   total_score=sum(tiers.values()), **tiers, message=score["tier_3"].get("message"),
                   categories=score["tier_2"].get("categories", {}), container=runtime["container"],
                   scoring_yaml=summary["scoring_yaml"])
        rows.append(row)
    complete = [row for row in rows if row["complete"]]
    successes = sum(row["inserted"] for row in complete)
    return {"generated_utc": datetime.now(timezone.utc).isoformat(), "bundle": str(bundle),
            "expected_trials": expected, "completed_trials": len(complete), "insertions": successes,
            "insertion_fraction_of_declared_trials": successes / expected,
            "target_insertions": frozen["target_insertions"],
            "target_met": len(complete) == expected and successes >= frozen["target_insertions"],
            "mean_total_score_completed": sum(row["total_score"] for row in complete) / len(complete) if complete else None,
            "trials": rows,
            "scope": "Only the frozen setting/reset distribution. Missing/incomplete trials do not count as successful. No population-level reliability guarantee."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    result = summarize(args.bundle)
    (args.bundle / "final_results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Frozen ACT final evaluation", "",
             f"Completed {result['completed_trials']}/{result['expected_trials']} trials; "
             f"insertions {result['insertions']}/{result['expected_trials']}; target met: {result['target_met']}.", "",
             result["scope"], "", "| Trial | Status | Total | Tier 3 | Inserted |",
             "| --- | --- | ---: | ---: | --- |"]
    for row in result["trials"]:
        total = f"{row['total_score']:.2f}" if row["complete"] else "—"
        tier3 = f"{row['tier_3']:.2f}" if row["complete"] else "—"
        lines.append(f"| {row['trial']} | {row['status']} | {total} | {tier3} | {row['inserted']} |")
    (args.bundle / "final_results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "trials"}, indent=2))


if __name__ == "__main__":
    main()
