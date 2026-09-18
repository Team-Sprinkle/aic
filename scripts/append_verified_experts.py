#!/usr/bin/env python3
"""Append newly verified aligned experts to an immutable ACT cache version.

The existing cache and split assignments are never edited. A dry run performs
score, label, scene, source, and file preflight without writing any files. An
actual append stages new images/data and canonical episodes before atomically
replacing the canonical manifest; earlier manifest versions remain available.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
from PIL import Image
import yaml

from curate_verified_experts import grouped_split, selected_shards, task_and_scene, write_json
from merge_corrective_act_cache import inspect_collection
from lerobot_robot_aic.task_encoding import task_encoding_schema


def link_or_copy(source, destination):
    """Preserve bytes across filesystems; other link failures are not concealed."""
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, destination)
    return str(destination)


def episode_identity(record):
    return (str(Path(record.get("collection", record.get("source", ""))).resolve()), record["trial_id"])


def content_digest(scene_hash, states, actions, timestamps):
    return hashlib.sha256(scene_hash.encode() + states.tobytes() + actions.tobytes() + timestamps.tobytes()).hexdigest()


def forbidden_scene_hashes(path):
    if path is None:
        return set()
    result = set()
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key.endswith("scene_sha256") and isinstance(child, str):
                    result.add(child)
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, str) and len(child) == 64:
                    result.add(child)
                else:
                    visit(child)
    visit(json.loads(path.read_text()))
    if not result:
        raise ValueError("Forbidden-scene manifest contains no scene hashes")
    return result


def assign_new_splits(existing, added, seed=918):
    assigned = {}
    for record in existing:
        key, split = record["scene_sha256"], record["split"]
        if split not in {"train", "validation"} or (key in assigned and assigned[key] != split):
            raise ValueError("Existing cache contains inconsistent scene splits")
        assigned[key] = split
    fresh = [record for record in added if record["scene_sha256"] not in assigned]
    # Keep newly represented target identities in the held-out set when each
    # has at least two scenes. The existing task/count stratifier alone can put
    # every SC port-1 episode in training when port 0 shares its card counts.
    by_target = {}
    for record in fresh:
        task = record["task"]
        identity = (task["task_family"], task.get("target_card_index", -1),
                    task.get("target_port_index", -1), task.get("target_card_valid", 0))
        by_target.setdefault(identity, []).append(record)
    for identity in sorted(by_target):
        grouped_split(by_target[identity], fraction=.1, seed=seed)
    for record in added:
        if record["scene_sha256"] in assigned:
            record["split"] = assigned[record["scene_sha256"]]


def preflight(base_cache, collections, canonical_root, output_dir, forbidden_manifest=None, minimum_score=80.):
    if not np.isfinite(minimum_score) or minimum_score < 80:
        raise ValueError("Canonical expert acceptance requires total score >=80")
    if output_dir.exists():
        raise FileExistsError(f"New cache output must not already exist: {output_dir}")
    if (canonical_root / "BUILD_IN_PROGRESS.json").exists():
        raise ValueError("Initial canonical dataset build has not completed")
    old_bytes = (canonical_root / "manifest.json").read_bytes()
    manifest = json.loads(old_bytes)
    base = json.loads((base_cache / "cache.json").read_text())
    if Path(manifest["training_cache"]).resolve() != base_cache.resolve():
        raise ValueError("Base cache is not the canonical manifest's current version")
    signature = lambda rows: [(r["episode_index"], r["scene_sha256"], r["content_sha256"], r["split"]) for r in rows]
    if signature(manifest["episodes"]) != signature(base["episodes"]):
        raise ValueError("Canonical and base-cache episode identities/splits disagree")
    if base.get("task_encoding") != task_encoding_schema() or base.get("image_channel_order") != "rgb":
        raise ValueError("Base cache requires canonical task metadata and RGB image semantics")
    if len({str(path.resolve()) for path in collections}) != len(collections):
        raise ValueError("Collection repeated in append request")
    arrays = {name: np.load(base_cache / (name + ".npy"), mmap_mode="r")
              for name in ("states", "actions", "timestamps", "episodes", "task_vectors")}
    for name, width in [("states", 32), ("actions", 6), ("task_vectors", 10)]:
        if arrays[name].shape != (base["frames"], width):
            raise ValueError(f"Unexpected base {name} shape")
    if any(len(value) != base["frames"] for value in arrays.values()):
        raise ValueError("Base cache arrays have inconsistent frame counts")
    seen_sources = {episode_identity(record) for record in base["episodes"]}
    seen_hashes = {record["content_sha256"] for record in base["episodes"]}
    forbidden = forbidden_scene_hashes(forbidden_manifest)
    accepted, rejected, destinations = [], [], set()
    next_episode = max(record["episode_index"] for record in base["episodes"]) + 1
    offset = base["frames"]
    for collection in collections:
        good, bad = inspect_collection(collection, minimum_score)
        for record in bad:
            rejected.append({**record, "rejection_reason": "not_full_insertion_or_total_below_threshold"})
        engine = yaml.safe_load((collection / "engine_config.yaml").read_text())
        for record, rows, selected, grid, states, actions in good:
            score = yaml.safe_load(Path(record["official_score_path"]).read_text())[record["trial_id"]]
            tier_scores = np.asarray([score[key]["score"] for key in ("tier_1", "tier_2", "tier_3")], dtype=float)
            if not np.isfinite(tier_scores).all():
                raise ValueError("Official scores must be finite")
            contacts = score.get("tier_2", {}).get("categories", {}).get("contacts")
            if contacts is None:
                raise ValueError("Official score lacks prohibited-contact evidence")
            if float(contacts["score"]) != 0 or contacts.get("message") != "No contact detected.":
                rejected.append({**record, "rejection_reason": "prohibited_contact_or_unconfirmed_contact_status"})
                continue
            force = score.get("tier_2", {}).get("categories", {}).get("insertion force")
            if force is None or not np.isfinite(float(force["score"])):
                raise ValueError("Official score lacks finite insertion-force evidence")
            if float(force["score"]) != 0:
                rejected.append({**record, "rejection_reason": "insertion_force_penalty"})
                continue
            trial_path = Path(record["trial_yaml"])
            trial_file = yaml.safe_load(trial_path.read_text())
            if (len(trial_file["trials"]) != 1 or next(iter(trial_file["trials"].values()))
                    != engine["trials"][record["trial_id"]]):
                raise ValueError(f"Trial snapshot disagrees with collection engine config: {trial_path}")
            task, vector, counts, scene_hash = task_and_scene(trial_path)
            if scene_hash in forbidden:
                raise ValueError(f"Collection overlaps a forbidden evaluation scene: {record['trial_id']}")
            identity = episode_identity(record)
            if identity in seen_sources:
                raise ValueError(f"Duplicate collection/trial source: {identity}")
            if states.shape != (len(grid), 32) or actions.shape != (len(grid), 6):
                raise ValueError("New aligned arrays have incorrect shapes")
            digest = content_digest(scene_hash, states, actions, grid)
            if digest in seen_hashes:
                raise ValueError(f"Duplicate trajectory content: {identity}")
            seen_sources.add(identity); seen_hashes.add(digest)
            source_dir = Path(record["episode_dir"])
            relative = Path(task["task_family"]) / "aligned" / collection.name / source_dir.name
            destination = canonical_root / relative
            if destination.exists() or str(relative) in destinations:
                raise FileExistsError(f"Canonical episode destination already exists: {destination}")
            destinations.add(str(relative))
            # Image verification is read-only and includes every recorded frame,
            # even frames omitted by the causal 20Hz resampling.
            for row in rows:
                for image_name in row["images"].values():
                    image_path = (source_dir / image_name).resolve()
                    if not image_path.is_relative_to(source_dir.resolve()):
                        raise ValueError("Recorded image path escapes its episode directory")
                    with Image.open(image_path) as image:
                        if list(image.size) != list(reversed(base["image_shape_hwc"][:2])):
                            raise ValueError(f"Unexpected source image dimensions: {image_path}")
                        image.verify()
            record.update(episode_index=next_episode, cache_from_index=offset,
                          cache_to_index=offset + len(grid), frames=len(grid), task=task,
                          task_vector=vector.tolist(), scene_sha256=scene_hash, content_sha256=digest,
                          **counts, verified_success_and_lineage=True, bc_eligible=True,
                          source_kind="score_verified_privileged_corrective_labels",
                          source_image_channel_order="rgb", canonical_episode_dir=str(destination.resolve()),
                          label_status="Aligned recorded observation and privileged teacher target; numerical target roundtrip verified",
                          prohibited_contact_verified_absent=True,
                          force_penalty_verified_absent=True)
            accepted.append((record, rows, selected, grid, states, actions, relative))
            next_episode += 1; offset += len(grid)
    assign_new_splits(base["episodes"], [entry[0] for entry in accepted])
    return dict(base=base, manifest=manifest, manifest_bytes=old_bytes, arrays=arrays,
                accepted=accepted, rejected=rejected, output_frames=offset)


def summary(plan):
    records = [entry[0] for entry in plan["accepted"]]
    combined = plan["base"]["episodes"] + records
    counts = Counter((r["task"]["task_family"], r["nic_count"], r["sc_count"], r["split"]) for r in combined)
    return {"base_episodes": len(plan["base"]["episodes"]), "accepted_new_episodes": len(records),
            "rejected_new_episodes": len(plan["rejected"]), "new_train_episodes": sum(r["split"] == "train" for r in records),
            "new_validation_episodes": sum(r["split"] == "validation" for r in records),
            "total_train_episodes": sum(r["split"] == "train" for r in combined),
            "total_validation_episodes": sum(r["split"] == "validation" for r in combined),
            "total_frames": plan["output_frames"], "new_episodes": records, "rejected": plan["rejected"],
            "distribution": [dict(task_family=f, nic_count=n, sc_count=s, split=t, episodes=c)
                             for (f, n, s, t), c in sorted(counts.items())],
            "split_policy": "Existing assignments retained. New scene groups use approximately 10% holdouts within task family, target card/port/validity, NIC-count, and SC-count strata; strata with at least two scenes get at least one holdout, while singleton strata stay in training. Small strata can therefore exceed a 10% validation share."}


def materialize(plan, base_cache, canonical_root, output_dir):
    if not plan["accepted"]:
        raise ValueError("No new episodes passed every expert acceptance gate")
    token = uuid.uuid4().hex
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.with_name(f".{output_dir.name}.building_{token}")
    canonical_stage = canonical_root / f".append_building_{token}"
    moved, published_cache, committed = [], False, False
    try:
        stage.mkdir(); canonical_stage.mkdir()
        image_root = stage / "aligned_images"; image_root.mkdir()
        base = plan["base"]
        added_frames = plan["output_frames"] - base["frames"]
        images = {key: np.lib.format.open_memmap(image_root / (key + ".npy"), mode="w+", dtype=np.uint8,
                    shape=(added_frames, *base["image_shape_hwc"])) for key in base["camera_keys"]}
        pieces = {name: [value] for name, value in plan["arrays"].items()}
        records, offset = [], 0
        for record, rows, selected, grid, states, actions, relative in plan["accepted"]:
            source = Path(record["episode_dir"])
            target = canonical_stage / relative
            shutil.copytree(source, target, copy_function=link_or_copy)
            provenance = target / "provenance"
            provenance.mkdir()
            for original, name in [(record["official_score_path"], "scoring.yaml"),
                                   (record["trial_yaml"], "trial.yaml"),
                                   (str(Path(record["collection"]) / "engine_config.yaml"), "engine_config.yaml"),
                                   (str(Path(record["collection"]) / "eval_collection/collection_config/eval_summary.json"), "eval_summary.json")]:
                shutil.copy2(original, provenance / name)
            write_json(provenance / "acceptance.json", record)
            for key, output in images.items():
                camera = key.removeprefix("observation.images.").removesuffix("_camera")
                previous_index, pixels = None, None
                for position, index in enumerate(selected):
                    if index != previous_index:
                        with Image.open(source / rows[index]["images"][camera]) as image:
                            pixels = np.asarray(image.convert("RGB"))
                        if tuple(pixels.shape) != tuple(base["image_shape_hwc"]):
                            raise ValueError("Decoded image shape differs from accepted preflight")
                        previous_index = index
                    output[offset + position] = pixels
            for name, values in [("states", states), ("actions", actions), ("timestamps", grid),
                                 ("episodes", np.full(len(grid), record["episode_index"], np.int64)),
                                 ("task_vectors", np.tile(np.asarray(record["task_vector"], np.float32), (len(grid), 1)))]:
                pieces[name].append(values)
            records.append(copy.deepcopy(record)); offset += len(grid)
        for image in images.values():
            image.flush()
        images.clear()
        for name, values in pieces.items():
            path = stage / (name + ".npy")
            combined = np.concatenate(values)
            np.save(path, combined)
            restored = np.load(path, mmap_mode="r")
            if restored.dtype != combined.dtype or not np.array_equal(restored, combined):
                raise ValueError(f"Saved {name} array failed its exact roundtrip check")
        metadata = copy.deepcopy(base)
        metadata["episodes"].extend(records)
        splits = copy.deepcopy(base["splits_by_nic_count"])
        for record in records:
            splits.setdefault(str(record["nic_count"]), {"train": [], "validation": []})[record["split"]].append(record["episode_index"])
        metadata.update(frames=plan["output_frames"], base_cache=str(base_cache.resolve()),
                        source_dataset=str(canonical_root.resolve()), splits_by_nic_count=splits,
                        image_shards=selected_shards(base, base_cache, 0, base["frames"], 0) + [{
                            "root": str(output_dir / "aligned_images"), "from_index": base["frames"],
                            "to_index": plan["output_frames"], "image_channel_order": "rgb"}],
                        append_acceptance="Tier 3 exactly 75; total >=80; no prohibited contact or insertion-force penalty; complete matching scene, timestamp, image, and teacher-label lineage",
                        split_policy=summary(plan)["split_policy"])
        write_json(stage / "cache.json", metadata)
        write_json(stage / "base_cache_manifest.json", base)
        write_json(stage / "append_audit.json", summary(plan))
        current = copy.deepcopy(plan["manifest"])
        current["episodes"].extend(records)
        current["training_cache"] = str(output_dir.resolve())
        current.setdefault("append_history", []).append({"cache": str(output_dir.resolve()),
            "base_cache": str(base_cache.resolve()), "accepted_new_episodes": len(records),
            "rejected_new_episodes": len(plan["rejected"]), "audit": str(output_dir / "append_audit.json")})
        # The lock and manifest-content compare prevent an append based on a
        # stale version from replacing another completed append.
        with (canonical_root / ".append.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (canonical_root / "manifest.json").read_bytes() != plan["manifest_bytes"]:
                raise ValueError("Canonical manifest changed after preflight; append must be replanned")
            if output_dir.exists():
                raise FileExistsError(output_dir)
            backup_dir = canonical_root / "manifest_versions"; backup_dir.mkdir(exist_ok=True)
            backup = backup_dir / f"manifest_{time.time_ns()}_{hashlib.sha256(plan['manifest_bytes']).hexdigest()[:12]}.json"
            backup.write_bytes(plan["manifest_bytes"])
            stage.rename(output_dir); published_cache = True
            for record, _, _, _, _, _, relative in plan["accepted"]:
                destination = canonical_root / relative
                if destination.exists():
                    raise FileExistsError(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                (canonical_stage / relative).rename(destination)
                moved.append(destination)
            temporary = canonical_root / f".manifest_{token}.json"
            write_json(temporary, current)
            temporary.replace(canonical_root / "manifest.json")
            committed = True
    finally:
        if not committed:
            for destination in reversed(moved):
                shutil.rmtree(destination)
            if published_cache:
                shutil.rmtree(output_dir)
        if stage.exists():
            shutil.rmtree(stage)
        if canonical_stage.exists():
            shutil.rmtree(canonical_stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--collections", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--forbidden-scene-manifest", type=Path)
    parser.add_argument("--minimum-total-score", type=float, default=80.)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.base_cache = args.base_cache.resolve(); args.output_dir = args.output_dir.resolve()
    args.canonical_root = args.canonical_root.resolve()
    args.collections = [path.resolve() for path in args.collections]
    plan = preflight(args.base_cache, args.collections, args.canonical_root, args.output_dir,
                     args.forbidden_scene_manifest, args.minimum_total_score)
    if not args.dry_run:
        materialize(plan, args.base_cache, args.canonical_root, args.output_dir)
    print(json.dumps({"dry_run": args.dry_run, **summary(plan)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
