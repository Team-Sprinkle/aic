#!/usr/bin/env python3
"""Merge scored native SC TF manifests without duplicate trial IDs.

Repeated physical reset configurations are retained and grouped by the
downstream split audit; they are not automatically independent scenes.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    by_trial = {}
    for source in args.inputs:
        for row in json.loads(source.read_text())["rows"]:
            if row["trial"] in by_trial:
                raise ValueError(f"Duplicate trial {row['trial']}")
            by_trial[row["trial"]] = row
    scenes = [row["scene_sha256"] for row in by_trial.values()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "sc_native_batch_edges/v1",
                                       "source_manifests": [str(x) for x in args.inputs],
                                       "distinct_reset_configurations": len(set(scenes)),
                                       "rows": [by_trial[key] for key in sorted(by_trial)]}, indent=2) + "\n")
    print(json.dumps({"trial_count": len(by_trial), "distinct_reset_configurations": len(set(scenes)), "full_tier3": sum(
        row["official_tier3"] == 75 for row in by_trial.values())}))


if __name__ == "__main__":
    main()
