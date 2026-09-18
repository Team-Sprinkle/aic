#!/usr/bin/env python3
"""Materialize the score/label-verified expert collection and an ACT cache manifest.

Historical mixed data are preserved by an atomic directory rename. Only complete
CheatCode sources whose every accepted episode passed the audit are hard-linked.
Aligned collections are selected one episode at a time. Questionable agent
labels remain outside the canonical training collection, with explicit reasons.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "aic_utils/lerobot_robot_aic"))
from lerobot_robot_aic.task_encoding import encode_task_vector, task_encoding_schema


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def link_tree(source, destination):
    """Share immutable bytes without aliasing mutable directory structures."""
    shutil.copytree(source, destination, copy_function=os.link)


def task_and_scene(path):
    config = yaml.safe_load(path.read_text())
    trial = next(iter(config["trials"].values()))
    task = next(iter(trial["tasks"].values()))
    if task["plug_type"] == "sfp":
        card = int(task["target_module_name"].removeprefix("nic_card_mount_"))
        port = int(task["port_name"].removeprefix("sfp_port_"))
        fields = dict(task_family="sfp_to_nic", target_card_index=card,
                      target_port_index=port, target_card_valid=1)
    elif task["plug_type"] == "sc":
        port = int(task["target_module_name"].removeprefix("sc_port_"))
        fields = dict(task_family="sc_to_sc", target_card_index=-1,
                      target_port_index=port, target_card_valid=0)
    else:
        raise ValueError(f"Unsupported task: {task}")
    vector = encode_task_vector(**fields)
    board = trial["scene"]["task_board"]
    counts = {kind + "_count": sum(bool(board.get(f"{kind}_rail_{i}", {}).get("entity_present"))
                                   for i in range(number)) for kind, number in [("nic", 5), ("sc", 2)]}
    target = {k: v for k, v in task.items() if k != "time_limit"}
    scene_hash = hashlib.sha256(json.dumps({"scene": trial["scene"], "task": target},
                                           sort_keys=True).encode()).hexdigest()
    return fields, vector, counts, scene_hash


def selected_shards(cache, cache_root, start, stop, destination_start):
    shards = cache.get("image_shards", [{"root": str(cache_root.resolve()), "from_index": 0,
                                       "to_index": cache["frames"],
                                       "image_channel_order": cache["image_channel_order"]}])
    result = []
    for shard in shards:
        lo, hi = max(start, shard["from_index"]), min(stop, shard["to_index"])
        if lo < hi:
            result.append({"root": shard["root"],
                           "source_from_index": shard.get("source_from_index", 0) + lo - shard["from_index"],
                           "from_index": destination_start + lo - start,
                           "to_index": destination_start + hi - start,
                           "image_channel_order": shard["image_channel_order"]})
    if sum(x["to_index"] - x["from_index"] for x in result) != stop - start:
        raise ValueError("Incomplete source image coverage")
    return result


def grouped_split(records, fraction=.1, seed=918):
    """Stratify by task and clutter while keeping identical scenes together."""
    groups = defaultdict(list)
    for record in records:
        groups[record["scene_sha256"]].append(record)
    strata = defaultdict(list)
    for key, members in groups.items():
        first = members[0]
        stratum = (first["task"]["task_family"], first["nic_count"], first["sc_count"])
        strata[stratum].append(key)
    rng = np.random.default_rng(seed)
    validation = set()
    for stratum, keys in sorted(strata.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        n = min(len(keys) - 1, max(1, round(len(keys) * fraction))) if len(keys) > 1 else 0
        validation.update(keys[:n])
    for record in records:
        record["split"] = "validation" if record["scene_sha256"] in validation else "train"
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-experiment", type=Path, required=True)
    parser.add_argument("--output-experiment", type=Path, required=True)
    parser.add_argument("--smoke-cache", type=Path, required=True)
    args = parser.parse_args()
    original = REPO / "outputs/trajectory_datasets/clean"
    archive = original.with_name("clean_including_no_insert_trajs")
    destination = original.with_name("expert_verified")
    pending = original.with_name("successful_pending_label_repair")
    if destination.exists() or pending.exists():
        raise FileExistsError("Canonical output already exists; never merge a partial rebuild silently")
    audit_path = args.old_experiment / "verification/episodes.json"
    audit = json.loads(audit_path.read_text())
    sources = [(args.old_experiment / "cache_cheat130", False), (args.smoke_cache, False),
               (args.old_experiment / "cache_cheat130_corrective24", True),
               (args.old_experiment / "cache_cheat130_aligned87", True)]
    # Preflight before renaming the source or publishing any canonical output.
    for cache_root, recent_only in sources:
        cache = json.loads((cache_root / "cache.json").read_text())
        for key in ["states", "actions", "timestamps"]:
            if not (cache_root / (key + ".npy")).is_file():
                raise FileNotFoundError(cache_root / (key + ".npy"))
        for episode in cache["episodes"]:
            if recent_only and not episode.get("collection"):
                continue
            trial_path = Path(episode["trial_yaml"])
            if not trial_path.exists() and archive.exists():
                trial_path = Path(str(trial_path).replace(str(original),str(archive)))
            task_and_scene(trial_path)
            if episode.get("episode_dir") and not Path(episode["episode_dir"]).is_dir():
                raise FileNotFoundError(episode["episode_dir"])
    if original.exists() and archive.exists():
        raise ValueError("Both original and archive exist; resolve ambiguous source first")
    if original.exists():
        original.rename(archive)
    if not archive.is_dir():
        raise FileNotFoundError(archive)
    destination.mkdir(); pending.mkdir()
    write_json(destination / "BUILD_IN_PROGRESS.json", {"status": "building; do not train until manifest.json exists"})
    relocation = {"old_root": str(original), "new_root": str(archive),
                  "operation": "directory rename; original bytes preserved",
                  "historical_manifests": "Preserved unchanged; apply this prefix substitution when reading historical paths"}
    write_json(args.output_experiment / "source_relocation.json", relocation)
    write_json(archive / "SOURCE_RELOCATION.json", relocation)

    def relocated(value):
        text = str(value)
        return Path(text.replace(str(original), str(archive)).replace(
            "outputs/trajectory_datasets/clean/", "outputs/trajectory_datasets/clean_including_no_insert_trajs/"))

    def relocate_tree(value):
        if isinstance(value, dict):
            return {k: relocate_tree(v) for k,v in value.items()}
        if isinstance(value,list):
            return [relocate_tree(v) for v in value]
        if isinstance(value,str) and "outputs/trajectory_datasets/clean/" in value:
            return str(relocated(value))
        return value

    source_groups = defaultdict(list)
    for episode in audit:
        source_groups[episode["source"]].append(episode)
    for source, episodes in source_groups.items():
        if "/cheatcode/" not in source:
            continue
        if not all(e["verified_success_and_lineage"] for e in episodes):
            raise ValueError(f"Cannot hardlink a source containing unverified episodes: {source}")
        src = archive / source / "accepted_dataset"
        target = destination / source / "accepted_dataset"
        target.mkdir(parents=True)
        for child in ["data", "meta", "videos"]:
            link_tree(src / child, target / child)
        write_json(target.parent / "provenance.json", {"archived_source": str(src),
                   "original_episode_indices": [e["episode_index"] for e in episodes],
                   "official_success_and_raw_lineage": True,
                   "image_channel_order": "bgr", "timing": "legacy wall-clock recording"})
    excluded, awaiting = [], []
    for episode in audit:
        e = relocate_tree(copy.deepcopy(episode))
        e["dataset"] = str(relocated(e["dataset"]))
        if not e["verified_success_and_lineage"]:
            e["exclusion_reason"] = "scored_noninsertion" if e["official_scores"] and not e["official_insertion_success"] else "unresolved_score_or_raw_lineage"
            excluded.append(e)
        elif "/agent/" in e["source"]:
            e["bc_eligible"] = False
            e["exclusion_reason"] = "Joint-space motion / expired Cartesian-command labels; exact observation-to-issued-command alignment is not established"
            awaiting.append(e)
    write_json(pending / "manifest.json", {"status": "successful_outcomes_pending_action_label_repair",
               "episodes": awaiting, "source_relocation": relocation,
               "note": "Data are preserved at archived_source paths. No episode is silently relabeled from observed movement."})
    write_json(destination / "excluded_episodes.json", excluded)

    arrays = {key: [] for key in ("states", "actions", "timestamps", "episodes", "task_vectors")}
    records, shards, duplicates = [], [], []
    seen_identity, seen_data, offset = set(), {}, 0
    for source_root, recent_only in sources:
        cache = json.loads((source_root / "cache.json").read_text())
        source_arrays = {k: np.load(source_root / (k + ".npy"), mmap_mode="r") for k in ("states", "actions", "timestamps")}
        for episode in cache["episodes"]:
            recent = bool(episode.get("collection"))
            if recent_only and not recent:
                continue
            if recent:
                if (abs(float(episode.get("official_tier3", -1)) - 75.) > 1e-6
                        or not 0 <= float(episode.get("target_position_roundtrip_error_m", float("inf"))) <= 1e-6
                        or not 0 <= float(episode.get("target_rotation_roundtrip_error_rad", float("inf"))) <= 1e-6):
                    raise ValueError("Aligned cache lacks verified score/label roundtrip evidence")
            elif not (episode.get("verified_success_and_lineage")
                      and episode.get("official_insertion_success")
                      and episode.get("action_state_matches_raw")):
                raise ValueError("Legacy cache lacks official success and matching raw arrays")
            identity = (episode.get("collection", episode.get("source")), episode["trial_id"])
            if identity in seen_identity:
                raise ValueError(f"Duplicate source input: {identity}")
            seen_identity.add(identity)
            trial_path = Path(episode["trial_yaml"]) if recent else relocated(episode["trial_yaml"])
            fields, vector, counts, scene_hash = task_and_scene(trial_path)
            start, stop = episode["cache_from_index"], episode["cache_to_index"]
            n = stop - start
            state = np.array(source_arrays["states"][start:stop], dtype=np.float32)
            action = np.array(source_arrays["actions"][start:stop], dtype=np.float32)
            timestamps = np.array(source_arrays["timestamps"][start:stop], dtype=np.float32)
            if state.shape != (n, 32) or action.shape != (n, 6) or not all(np.isfinite(x).all() for x in [state, action, timestamps]):
                raise ValueError(f"Invalid source arrays: {identity}")
            if n < 2 or not np.all(np.diff(timestamps) > 0):
                raise ValueError(f"Invalid source frame ordering: {identity}")
            digest = hashlib.sha256(scene_hash.encode() + state.tobytes() + action.tobytes() + timestamps.tobytes()).hexdigest()
            if digest in seen_data:
                duplicates.append({"source": identity, "duplicate_of": seen_data[digest], "content_sha256": digest})
                continue
            seen_data[digest] = identity
            index = len(records)
            record = relocate_tree(copy.deepcopy(episode))
            record.update(episode_index=index, source_episode_index=episode["episode_index"],
                          source_cache=str(source_root.resolve()), task=fields, task_vector=vector.tolist(),
                          scene_sha256=scene_hash, content_sha256=digest, **counts,
                          frames=n, cache_from_index=offset, cache_to_index=offset+n,
                          verified_success_and_lineage=True, bc_eligible=True,
                          source_image_channel_order="rgb" if recent else "bgr")
            record["trial_yaml"] = str(trial_path.resolve())
            if recent:
                source_dir = Path(episode["episode_dir"])
                dest_dir = destination / fields["task_family"] / "aligned" / Path(episode["collection"]).name / source_dir.name
                link_tree(source_dir, dest_dir)
                record["canonical_episode_dir"] = str(dest_dir.resolve())
                record["label_status"] = "Aligned observed state and privileged teacher target; numerical action/target roundtrip verified"
            else:
                record["dataset"] = str((destination / episode["source"] / "accepted_dataset").resolve())
                record["archived_dataset"] = str(relocated(episode["dataset"]).resolve())
                record["label_status"] = "Original recorded CheatCode Cartesian command; historical wall-clock sampling limitation retained"
            arrays["states"].append(state); arrays["actions"].append(action); arrays["timestamps"].append(timestamps)
            arrays["episodes"].append(np.full(n, index, np.int64))
            arrays["task_vectors"].append(np.tile(vector, (n, 1)))
            shards.extend(selected_shards(cache, source_root, start, stop, offset))
            records.append(record); offset += n
    grouped_split(records)
    splits = defaultdict(lambda: {"train": [], "validation": []})
    for record in records:
        splits[str(record["nic_count"])][record["split"]].append(record["episode_index"])
    cache_root = args.output_experiment / "cache_all_verified"
    cache_root.mkdir(exist_ok=False)
    for key, values in arrays.items():
        np.save(cache_root / (key + ".npy"), np.concatenate(values))
    metadata = dict(source_dataset=str(destination.resolve()), frames=offset, fps=20,
                    camera_keys=cache["camera_keys"], image_shape_hwc=cache["image_shape_hwc"],
                    state_dim=32, action_dim=6, image_channel_order="rgb", image_shards=shards,
                    action_semantics="Full TCP-relative command, converted to absolute target by trainer",
                    task_encoding=task_encoding_schema(), splits_by_nic_count=dict(splits), episodes=records,
                    split_policy="~90/10 stratified task/NIC/SC count; identical scene/task configurations grouped",
                    seed=918, cross_collection_deduplication="Exact scene+state+action+timestamp hash; source identity checked",
                    duplicates=duplicates)
    write_json(cache_root / "cache.json", metadata)
    write_json(destination / "manifest.json", {"schema_version": 1, "episodes": records,
               "source_relocation": relocation, "task_encoding": task_encoding_schema(),
               "training_cache": str(cache_root.resolve()), "excluded_count": len(excluded),
               "pending_label_repair_count": len(awaiting), "duplicates": duplicates})
    distribution = Counter((e["task"]["task_family"],e["nic_count"],e["sc_count"],e["split"]) for e in records)
    summary = dict(eligible_episodes=len(records), frames=offset, historical_excluded=len(excluded),
                   pending_label_repair=len(awaiting), duplicates=len(duplicates),
                   train_episodes=sum(e["split"]=="train" for e in records),
                   validation_episodes=sum(e["split"]=="validation" for e in records),
                   distribution=[dict(task_family=f,nic_count=n,sc_count=s,split=t,episodes=v)
                                 for (f,n,s,t),v in sorted(distribution.items())])
    write_json(args.output_experiment / "dataset_summary.json", summary)
    (destination / "BUILD_IN_PROGRESS.json").unlink()
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()
