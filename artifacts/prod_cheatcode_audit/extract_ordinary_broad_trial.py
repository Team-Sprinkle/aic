#!/usr/bin/env python3
"""Copy one scene into a fresh one-trial Gazebo replay configuration."""

import json
from pathlib import Path
import sys

import yaml


def main(source, trial, out):
    config = yaml.safe_load((source / "eval_config.yaml").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    if trial not in config["trials"]:
        raise KeyError(trial)
    config["trials"] = {trial: config["trials"][trial]}
    entry = next(row for row in manifest["trials"] if row["trial"] == trial)
    out.mkdir(parents=True, exist_ok=False)
    (out / "eval_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (out / "manifest.json").write_text(json.dumps({
        "classification": "fresh single-scene diagnostic replay; not an independent randomized scene",
        "source_suite": str(source), "source_trial": trial,
        "trial_count": 1, "trials": [entry],
    }, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
