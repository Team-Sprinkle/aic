#!/usr/bin/env python3
"""Summarize completed matched SC LLB transport BC runs without rollout claims."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    records = []
    for root in a.runs:
        manifest = json.loads((root / "manifest.json").read_text())
        summary = json.loads((root / "summary.json").read_text())
        history = json.loads((root / "history.json").read_text())
        best = next(row for row in history if row["update"] == summary["best_update"])
        metrics = {
            family: {
                "first_target_mm_p95": best["validation"][family]["first_target_mm"]["p95"],
                "fourth_target_mm_p95": best["validation"][family]["last_target_mm"]["p95"],
                "orientation_deg_p95": best["validation"][family]["orientation_deg"]["p95"],
                "sample_count": best["validation"][family]["first_target_mm"]["count"],
            }
            for family in ("sfp", "sc")
        }
        records.append({
            "run": str(root.resolve()),
            "initialization": manifest["initialization"],
            "sc_source_sha256": next(value for key, value in manifest["source_sha256"].items()
                                     if key.endswith("/transport_chunks.jsonl")),
            "completed_updates": summary["completed_updates"],
            "best_update": summary["best_update"],
            "stop_reason": summary["stop_reason"],
            "elapsed_minutes": summary["elapsed_minutes"],
            "selection_score": summary["best_selection_score"],
            "heldout": metrics,
            "autonomous_rollouts_completed": summary["autonomous_rollouts_completed"],
            "checkpoint": str((root / "best.pt").resolve()),
        })
    report = {"schema": "aic_sc_llb_transport_bc_comparison/v1", "runs": records,
              "comparison_scope": "Offline held-out recorded-action regression only",
              "reserved_final_scenes_opened": False, "rl_started": False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"runs": len(records), "output": str(a.output)}))


if __name__ == "__main__":
    main()
