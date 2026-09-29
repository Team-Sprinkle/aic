#!/usr/bin/env python3
"""Freeze metadata for the shared SFP/SC continuation without opening final scenes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    manifest_path = repo / "outputs/trajectory_datasets/expert_verified/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    episodes = manifest["episodes"]
    counts = Counter()
    scene_splits: dict[str, set[str]] = defaultdict(set)
    for episode in episodes:
        task = episode["task"]
        family = task["task_family"]
        counts[(family, episode["split"], episode["nic_count"], task["target_port_index"])] += 1
        scene_splits[episode["scene_sha256"]].add(episode["split"])
        if (
            not episode["bc_eligible"]
            or not episode["verified_success_and_lineage"]
            or episode.get("official_insertion_success") is False
        ):
            raise ValueError(f"Non-success or ineligible episode in canonical set: {episode['episode_index']}")
    overlaps = sorted(key for key, splits in scene_splits.items() if len(splits) > 1)
    if overlaps:
        raise ValueError(f"Scene hashes span train and validation: {overlaps[:5]}")
    final_root = repo / "outputs/experiments/2026-09-20_isaac_world_rl/configs/varied_final/episodes"
    final_files = sorted(final_root.glob("episode_*.yaml"))
    if len(final_files) != 4:
        raise ValueError(f"Expected four sealed final configs, found {len(final_files)}")
    checkpoints = [
        repo / "outputs/experiments/2026-09-22_rpdp_dppo/bc_repair_direct/training/aic_rpdp/checkpoint.pt",
        repo / "outputs/experiments/2026-09-22_serl_recovery/online_selected/checkpoint_latest.pt",
    ]
    result = {
        "schema": "aic_shared_sc_inventory/v1",
        "source_manifest": str(manifest_path.relative_to(repo)),
        "source_manifest_sha256": digest(manifest_path),
        "canonical_episodes": len(episodes),
        "training_episodes": sum(e["split"] == "train" for e in episodes),
        "validation_episodes": sum(e["split"] == "validation" for e in episodes),
        "scene_hashes": len(scene_splits),
        "cross_split_scene_hashes": len(overlaps),
        "task_encoding": manifest["task_encoding"],
        "by_task_split_cards_target_port": [
            {"task": task, "split": split, "nic_cards": cards, "target_port": port, "episodes": count}
            for (task, split, cards, port), count in sorted(counts.items())
        ],
        "sealed_final": {
            "source": str(final_root.relative_to(repo)),
            "evaluation_opened": False,
            "configs": [
                {"path": str(path.relative_to(repo)), "sha256": digest(path)}
                for path in final_files
            ],
        },
        "historical_checkpoints_not_shared_sc_actor": [
            {"path": str(path.relative_to(repo)), "sha256": digest(path)}
            for path in checkpoints
        ],
        "important_limitations": [
            "Only 21 canonical SC successes; one to three NIC cards and restricted target-port coverage.",
            "Old selected RPDP/SERL policy is Isaac SFP near-port, not shared-task/full-episode SC.",
            "Hashing sealed YAML bytes does not evaluate or tune on their scenes.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
