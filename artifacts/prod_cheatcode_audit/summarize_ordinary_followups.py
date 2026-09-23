#!/usr/bin/env python3
"""Join broad and isolated CheatCode replays without laundering invalid trials."""

import collections
from functools import lru_cache
import json
from pathlib import Path
import sys

import yaml

from summarize_ordinary_broad_suite import outcome, physics_errors_by_trial


@lru_cache(maxsize=None)
def warnings_for_log(path):
    return physics_errors_by_trial(path)


def read_attempt(root, trial):
    if not root.exists():
        return None
    score_file = root / "results/scoring.yaml"
    scores = yaml.safe_load(score_file.read_text()) or {} if score_file.exists() else {}
    score = scores.get(trial) or {}
    tier1 = score.get("tier_1") or {}
    tier3 = score.get("tier_3") or {}
    log = root / "engine.log"
    warnings = warnings_for_log(log).get(trial, 0) if log.exists() else None
    bags = sorted((root / "results").glob(f"bag_{trial}_*"))
    message = str(tier3.get("message") or "")
    return {
        "root": str(root), "tier1_message": tier1.get("message"),
        "tier3_message": message, "tier3_score": tier3.get("score"),
        "outcome": outcome(message), "physics_entity_errors": warnings,
        "bag": str(bags[0]) if bags else None,
        "engine_exit": (root / "engine.exit").read_text().strip()
            if (root / "engine.exit").exists() else None,
    }


def main(broad, fresh_roots, dest):
    manifest = json.loads((broad / "manifest.json").read_text())
    records = []
    for meta in manifest["trials"]:
        trial = meta["trial"]
        attempts = []
        original = read_attempt(broad, trial)
        if original:
            attempts.append({"kind": "broad_batch", **original})
        for root in fresh_roots:
            attempt = read_attempt(root / trial, trial)
            if attempt:
                attempts.append({"kind": "isolated", **attempt})
        for attempt in attempts:
            attempt["valid"] = bool(
                attempt["outcome"] != "unscored"
                and attempt["tier1_message"] == "Model validation succeeded."
                and attempt["physics_entity_errors"] == 0
                and attempt["bag"]
            )
        selected = next((a for a in reversed(attempts) if a["valid"]), None)
        records.append({**meta, "selected": selected, "attempts": attempts})
    summary = {
        "schema": "aic_ordinary_followups/v1",
        "suite_manifest": str(broad / "manifest.json"),
        "source_sha256": manifest["source_sha256"],
        "selection_rule": "Latest officially scored attempt with successful model validation, a bag, and zero Gazebo physics-entity errors. Invalid attempts retained below.",
        "trial_count": len(records),
        "valid_trial_count": sum(r["selected"] is not None for r in records),
        "selected_outcomes": dict(collections.Counter(
            r["selected"]["outcome"] if r["selected"] else "missing_valid"
            for r in records)),
        "records": records,
        "caveat": "A tier-3 insertion result is not a cable-snag diagnosis. This stratified sample is not an estimate of deployment frequency.",
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=2) + "\n")
    print(dest, summary["valid_trial_count"], summary["selected_outcomes"])


if __name__ == "__main__":
    main(Path(sys.argv[1]), [Path(path) for path in sys.argv[2:-1]], Path(sys.argv[-1]))
