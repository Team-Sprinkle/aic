#!/usr/bin/env python3
"""Build a bounded native-image SC development batch from scored trial seeds."""

import argparse
import json
from pathlib import Path

import yaml


DEFAULT_TRIALS = ("trial_000501", "trial_000502", "trial_000504", "trial_000505",
                  "trial_000506", "trial_000510", "trial_000512", "trial_000514",
                  "trial_000516")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", nargs="+", default=DEFAULT_TRIALS)
    args = parser.parse_args()
    trials = tuple(args.trials)
    sources = {}
    base = None
    for port in ("batch_port0", "batch_port1"):
        path = args.source / port / "engine_config.yaml"
        config = yaml.safe_load(path.read_text())
        if base is None:
            base = {k: v for k, v in config.items() if k != "trials"}
        for key, value in config["trials"].items():
            if key in trials:
                sources[key] = (value, str(path))
    missing = set(trials) - set(sources)
    if missing:
        raise ValueError(f"Missing trials: {sorted(missing)}")
    base["trials"] = {key: sources[key][0] for key in trials}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "engine_config.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
    (args.output / "collection_config.json").write_text(json.dumps({
        "kind": "native_sc_pose_ablation_development", "training_only": True,
        "source_trials": {key: sources[key][1] for key in trials},
        "policy": "privileged corrective teacher; never autonomous evidence",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
