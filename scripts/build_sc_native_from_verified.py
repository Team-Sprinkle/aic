#!/usr/bin/env python3
"""Recollect native RGB for selected scored SC development resets.

The source episode index is retained because old collections reused trial IDs.
This script never reads or selects the four reserved final IK configurations.
"""

import argparse
import json
from pathlib import Path

import yaml


DEFAULT_EPISODES = (251, 252, 253, 254, 255, 256, 257, 258, 259, 287)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified-manifest", type=Path,
                        default=Path("outputs/trajectory_datasets/expert_verified/manifest.json"))
    parser.add_argument("--episode-indices", nargs="+", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = {row["episode_index"]: row for row in json.loads(args.verified_manifest.read_text())["episodes"]}
    selected = [entries[index] for index in args.episode_indices]
    if len(set(args.episode_indices)) != len(args.episode_indices):
        raise ValueError("Duplicate source episode")
    config = None
    lineage = {}
    for row in selected:
        if row["task"]["task_family"] != "sc_to_sc" or row["official_tier3"] != 75:
            raise ValueError(f"Source episode {row['episode_index']} is not verified SC insertion")
        source = yaml.safe_load(Path(row["trial_yaml"]).read_text())
        if len(source["trials"]) != 1:
            raise ValueError("Expected one trial in source YAML")
        if config is None:
            config = {key: value for key, value in source.items() if key != "trials"}
            config["trials"] = {}
        elif {key: value for key, value in source.items() if key != "trials"} != {
                key: value for key, value in config.items() if key != "trials"}:
            raise ValueError(f"Top-level engine config mismatch in source episode {row['episode_index']}")
        new_trial = f"trial_100{row['episode_index']}"
        config["trials"][new_trial] = next(iter(source["trials"].values()))
        lineage[new_trial] = {"source_episode_index": row["episode_index"],
                              "source_trial_yaml": row["trial_yaml"],
                              "source_scene_sha256": row["scene_sha256"],
                              "source_split": row["split"]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "engine_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (args.output / "collection_config.json").write_text(json.dumps({
        "kind": "native_sc_verified_scene_recollection", "training_only": True,
        "source_episode_indices": lineage,
        "policy": "privileged corrective teacher; no autonomous evidence",
    }, indent=2) + "\n")
    print(json.dumps({"trials": len(lineage), "train": sum(v["source_split"] == "train" for v in lineage.values()),
                      "validation": sum(v["source_split"] == "validation" for v in lineage.values())}))


if __name__ == "__main__": main()
