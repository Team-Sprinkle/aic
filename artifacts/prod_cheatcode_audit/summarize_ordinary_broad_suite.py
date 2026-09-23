#!/usr/bin/env python3
"""Join the generated manifest, official scores, bag metrics, and engine warnings."""

import collections
import json
from pathlib import Path
import re
import sys

import yaml


def outcome(message):
    if "Cable insertion successful" in message:
        return "full"
    if "Partial insertion" in message:
        return "partial"
    if "No insertion" in message:
        return "none"
    return "unscored"


def physics_errors_by_trial(path):
    counts = collections.Counter()
    active = None
    marker = re.compile(r"Trial (\d+)/(\d+): (trial_[A-Za-z0-9_]+)")
    with path.open(errors="replace") as stream:
        for line in stream:
            found = marker.search(line)
            if found:
                active = found.group(3)
            if active and "Internal error: a physics entity ptr" in line:
                counts[active] += 1
    return counts


def main(audit, dest):
    manifest = json.loads((audit / "manifest.json").read_text())
    scores_path = audit / "results/scoring.yaml"
    scores = yaml.safe_load(scores_path.read_text()) if scores_path.exists() else {}
    warnings = physics_errors_by_trial(audit / "engine.log")
    records = []
    for entry in manifest["trials"]:
        trial = entry["trial"]
        score = scores.get(trial) or {}
        tier3 = score.get("tier_3") or {}
        message = str(tier3.get("message") or "")
        index = int(trial.split("_")[1])
        analysis_path = audit / f"trial_{index:02d}_analysis.json"
        analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else {}
        bags = sorted((audit / "results").glob(f"bag_{trial}_*"))
        record = {
            **entry,
            "outcome": outcome(message),
            "tier3_score": tier3.get("score"),
            "tier3_message": message,
            "total_score": (score.get("tier_1") or {}).get("score", 0)
                + (score.get("tier_2") or {}).get("score", 0)
                + (tier3.get("score") or 0) if score else None,
            "bag": str(bags[0]) if bags else None,
            "physics_entity_errors_during_trial": warnings[trial],
            "force_max_n": analysis.get("force_max_n"),
            "terminal_lateral_mm": (analysis.get("final") or {}).get("lateral_mm"),
            "terminal_axial_mm": (analysis.get("final") or {}).get("axial_mm"),
            "terminal_orientation_deg": (analysis.get("final") or {}).get("orientation_deg"),
            "terminal_tcp_command_error_mm": (analysis.get("final") or {}).get("tcp_command_error_mm"),
        }
        records.append(record)
    summary = {
        "schema": "aic_ordinary_broad_suite/v1",
        "classification": manifest["classification"],
        "trial_count": len(records),
        "officially_scored": sum(record["outcome"] != "unscored" for record in records),
        "outcomes": dict(collections.Counter(record["outcome"] for record in records)),
        "task_outcomes": {family: dict(collections.Counter(record["outcome"] for record in records if record["family"] == family))
                          for family in ("sfp_to_nic", "sc_to_sc")},
        "records": records,
        "caveat": "Physics-entity errors invalidate collider-specific conclusions until reproduced in a fresh scene. Tier-3 messages are insertion labels, not cable-snag labels.",
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=2) + "\n")
    print(dest, summary["outcomes"], summary["task_outcomes"])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
