#!/usr/bin/env python3
"""Join native SC episodes to scored bags by wall clock and extract TF labels.

Run inside the official rootless ROS container. The tight wall-time join is
checked before using TF; it cannot silently pair a reset with another trial.
"""

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import yaml

from inspect_sc_port_tf import extract_edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load((args.batch / "engine_config.yaml").read_text())
    collection_meta = json.loads((args.batch / "collection_config.json").read_text())
    source_indices = collection_meta.get("source_episode_indices", {})
    if (not source_indices and collection_meta.get('kind') == 'fresh_frozen_SC_pose_development_check'):
        source_indices = {trial: {'source_split': 'new_development_validation'}
                          for trial in collection_meta.get('source', {})}
    trial_names = list(config["trials"])
    score_file = args.batch / "eval_collection/collection_config/attempt_0001/scoring.yaml"
    scores = yaml.safe_load(score_file.read_text())
    attempt = score_file.parent
    bags = {}
    for bag in attempt.glob("bag_trial_*"):
        match = re.fullmatch(r"bag_(trial_\d+)_(\d{8})_(\d{6})_\d+", bag.name)
        if not match:
            continue
        seconds = dt.datetime.strptime(match.group(2) + match.group(3), "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc).timestamp()
        bags[match.group(1)] = (bag, seconds)
    episodes = sorted((args.batch / "episodes").glob("episode_*"),
                      key=lambda p: int(p.name.rsplit("_", 1)[-1]))
    if len(episodes) != len(trial_names) or set(bags) != set(trial_names):
        raise ValueError(f"Incomplete batch: {len(episodes)} episodes, {len(bags)} bags, {len(trial_names)} trials")
    rows = []
    for episode, trial in zip(episodes, trial_names):
        episode_start = int(episode.name.rsplit("_", 1)[-1]) / 1e9
        bag, bag_start = bags[trial]
        offset_s = episode_start - bag_start
        if not 0 <= offset_s <= 15:
            raise ValueError(f"Unsafe episode/bag join {episode} ↔ {bag}: {offset_s:.3f} s")
        meta = json.loads((episode / "episode.json").read_text())
        task = config["trials"][trial]["tasks"]["task_1"]
        if meta["task"]["target_module_name"] != task["target_module_name"]:
            raise ValueError(f"Task mismatch in {trial}")
        edges = extract_edges(bag, first_seconds=3, include_camera_chain=True)
        if len(edges) < 15:
            raise ValueError(f"Missing SC/camera TF in {bag}: {len(edges)} edges")
        rows.append({"trial": trial, "episode": str(episode), "bag": str(bag),
                     "bag_to_episode_wall_s": offset_s,
                     "official_tier3": scores[trial]["tier_3"]["score"],
                     "target_module_name": task["target_module_name"],
                     "source_episode_index": source_indices.get(trial, {}).get("source_episode_index"),
                     "source_split": source_indices.get(trial, {}).get("source_split"),
                     "scene_sha256": hashlib.sha256(json.dumps({
                         "scene": config["trials"][trial]["scene"],
                         "robot": config.get("robot")}, sort_keys=True).encode()).hexdigest(),
                     "edges": edges})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "sc_native_batch_edges/v1", "rows": rows}, indent=2) + "\n")
    print(json.dumps({"episodes": len(rows), "tier3_full": sum(r["official_tier3"] == 75 for r in rows)}))


if __name__ == "__main__":
    main()
