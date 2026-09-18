"""Append acceptance, immutable splits, and publication checks on temporary data."""
import copy
import errno
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import pytest
import yaml


def appender():
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("append_verified_experts_tested", scripts / "append_verified_experts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_base(tmp_path, module):
    canonical, cache = tmp_path / "expert_verified", tmp_path / "initial_cache"
    canonical.mkdir(); cache.mkdir()
    keys = [f"observation.images.{name}_camera" for name in ("center", "left", "right")]
    states = np.zeros((4, 32), np.float32); states[:, 6] = 1
    arrays = {"states": states, "actions": np.zeros((4, 6), np.float32),
              "timestamps": np.array([0, .05, 0, .05], np.float32), "episodes": np.array([0, 0, 1, 1], np.int64),
              "task_vectors": np.tile([1, 0, 0, 1, 1, 0, 0, 0, 0, 1], (4, 1)).astype(np.float32)}
    for name, values in arrays.items():
        np.save(cache / (name + ".npy"), values)
    for key in keys:
        np.save(cache / (key + ".npy"), np.zeros((4, 4, 4, 3), np.uint8))
    records = [{"episode_index": index, "source": "historical", "trial_id": f"old_{index}",
                "scene_sha256": f"old_scene_{index}", "content_sha256": f"old_content_{index}",
                "split": split, "task": {"task_family": "sfp_to_nic"}, "nic_count": 1, "sc_count": 0,
                "cache_from_index": index * 2, "cache_to_index": index * 2 + 2, "frames": 2}
               for index, split in enumerate(["train", "validation"])]
    metadata = {"frames": 4, "image_shape_hwc": [4, 4, 3], "camera_keys": keys, "image_channel_order": "rgb",
                "task_encoding": module.task_encoding_schema(), "episodes": records,
                "splits_by_nic_count": {"1": {"train": [0], "validation": [1]}},
                "source_dataset": str(canonical)}
    module.write_json(cache / "cache.json", metadata)
    module.write_json(canonical / "manifest.json", {"schema_version": 1, "training_cache": str(cache),
                      "episodes": records, "task_encoding": module.task_encoding_schema()})
    return canonical, cache, arrays


def make_collection(root, count=3):
    trials, scores = {}, {}
    (root / "trials").mkdir(parents=True)
    for number in range(1, count + 1):
        name = f"trial_{number:06d}"
        episode = root / "episodes" / f"episode_{number:04d}_{100 + number}"
        episode.mkdir(parents=True)
        state = np.zeros(32); state[0] = number * .001; state[6] = 1
        rows = []
        for frame, elapsed in enumerate([0., .05, .1]):
            images = {}
            for camera in ["center", "left", "right"]:
                image_name = f"{camera}_{frame:06d}.jpg"
                Image.fromarray(np.full((4, 4, 3), number * 20 + frame, np.uint8)).save(episode / image_name)
                images[camera] = image_name
            target = state[:7].tolist()
            rows.append({"state": state.tolist(), "action": [0.] * 6, "teacher_target_pose": target,
                         "executed_target_pose": target, "elapsed_sim_time": elapsed,
                         "images": images, "perturbation_xy_rotvec": [0.] * 5})
        (episode / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        (episode / "episode.json").write_text(json.dumps({"frames": 3, "image_channel_order": "rgb",
            "kind": "privileged_aligned_demonstration", "perturbation_config": {"scale": 0}}))
        trial = {"scene": {"task_board": {"sc_rail_0": {"entity_present": True},
                                          "nic_rail_0": {"entity_present": False}, "pose": {"x": number * .01}}},
                 "tasks": {"task_1": {"plug_type": "sc", "target_module_name": "sc_port_0", "port_name": "sc_port"}}}
        trials[name] = trial
        (root / "trials" / (name + ".yaml")).write_text(yaml.safe_dump({"trials": {name: trial}}))
        scores[name] = {"tier_1": {"score": 1}, "tier_2": {"score": 10,
            "categories": {"contacts": {"score": 0, "message": "No contact detected."},
                           "insertion force": {"score": 0, "message": "No excessive force detected"}}},
                        "tier_3": {"score": 75}}
    (root / "engine_config.yaml").write_text(yaml.safe_dump({"trials": trials}))
    (root / "score.yaml").write_text(yaml.safe_dump(scores))
    summary = root / "eval_collection/collection_config/eval_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"evaluation_complete": True, "evaluation_kind": "privileged_corrective_data_collection",
        "scoring_yaml": str(root / "score.yaml"), "runtime_settings": {"corrective_perturbation_scale": 0}}))
    return root


def test_preflight_has_no_mutation_and_append_preserves_original_arrays_and_splits(tmp_path):
    module = appender()
    canonical, base, old_arrays = make_base(tmp_path, module)
    collection = make_collection(tmp_path / "sc_new")
    destination = tmp_path / "new_version"
    original_manifest = (canonical / "manifest.json").read_bytes()
    original_cache = (base / "cache.json").read_bytes()
    plan = module.preflight(base, [collection], canonical, destination)
    assert not destination.exists()
    assert list(canonical.iterdir()) == [canonical / "manifest.json"]
    assert module.summary(plan)["new_validation_episodes"] == 1
    module.materialize(plan, base, canonical, destination)
    result = json.loads((destination / "cache.json").read_text())
    manifest = json.loads((canonical / "manifest.json").read_text())
    assert (base / "cache.json").read_bytes() == original_cache
    for name, original in old_arrays.items():
        np.testing.assert_array_equal(np.load(base / (name + ".npy")), original)
        np.testing.assert_array_equal(np.load(destination / (name + ".npy"))[:4], original)
    assert result["episodes"][:2] == json.loads(original_cache)["episodes"]
    assert result["splits_by_nic_count"]["1"] == {"train": [0], "validation": [1]}
    assert len(result["splits_by_nic_count"]["0"]["train"]) == 2
    assert len(result["splits_by_nic_count"]["0"]["validation"]) == 1
    assert manifest["training_cache"] == str(destination)
    assert next((canonical / "manifest_versions").iterdir()).read_bytes() == original_manifest
    vector = np.load(destination / "task_vectors.npy")[4]
    np.testing.assert_array_equal(vector, [0, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    for record in result["episodes"][2:]:
        saved = Path(record["canonical_episode_dir"])
        assert saved.is_relative_to(canonical / "sc_to_sc/aligned")
        assert (saved / "provenance/scoring.yaml").is_file()
        assert (saved / "provenance/engine_config.yaml").is_file()
    from act_cache import load_cached_images
    images = load_cached_images(destination, result)
    assert images[result["camera_keys"][0]][[0, 4, 12]].shape == (3, 4, 4, 3)


def test_known_scene_keeps_its_split_and_fresh_singleton_has_no_fake_holdout():
    module = appender()
    old = [{"scene_sha256": "known", "split": "validation"}]
    added = [{"scene_sha256": scene, "task": {"task_family": "sc_to_sc"}, "nic_count": 0, "sc_count": 1}
             for scene in ["known", "new", "new"]]
    module.assign_new_splits(old, added)
    assert [row["split"] for row in added] == ["validation", "train", "train"]


def test_new_target_identities_each_receive_a_holdout_without_moving_old_scenes():
    module = appender()
    old = [{"scene_sha256": "old_port1", "split": "train"}]
    added = [{"scene_sha256": f"port{port}_scene{scene}",
              "task": {"task_family": "sc_to_sc", "target_port_index": port,
                       "target_card_index": -1, "target_card_valid": 0},
              "nic_count": 1, "sc_count": 1}
             for port in (0, 1) for scene in (0, 1)]
    added.append({**copy.deepcopy(added[-1]), "scene_sha256": "old_port1"})
    module.assign_new_splits(old, added)
    for port in (0, 1):
        fresh = [r for r in added if r["task"]["target_port_index"] == port
                 and r["scene_sha256"] != "old_port1"]
        assert sorted(r["split"] for r in fresh) == ["train", "validation"]
    assert added[-1]["split"] == "train"


def test_forbidden_scene_and_duplicate_content_fail_before_publication(tmp_path):
    module = appender(); canonical, base, _ = make_base(tmp_path, module)
    first = make_collection(tmp_path / "sc_first", count=1)
    scene = module.task_and_scene(first / "trials/trial_000001.yaml")[-1]
    forbidden = tmp_path / "evaluation.json"; forbidden.write_text(json.dumps({"configs": [{"scene_sha256": scene}]}))
    with pytest.raises(ValueError, match="forbidden"):
        module.preflight(base, [first], canonical, tmp_path / "out", forbidden)
    second = make_collection(tmp_path / "sc_second", count=1)
    with pytest.raises(ValueError, match="Duplicate trajectory content"):
        module.preflight(base, [first, second], canonical, tmp_path / "out")
    with pytest.raises(ValueError, match="Collection repeated"):
        module.preflight(base, [first, first], canonical, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_contacts_and_noninsertion_are_excluded_even_with_other_high_scores(tmp_path):
    module = appender(); canonical, base, _ = make_base(tmp_path, module)
    collection = make_collection(tmp_path / "sc_contacts", count=3)
    path = collection / "score.yaml"; scores = yaml.safe_load(path.read_text())
    scores["trial_000001"]["tier_2"]["categories"]["contacts"] = {"score": -24, "message": "Contacts detected."}
    scores["trial_000002"]["tier_3"]["score"] = 74.9
    path.write_text(yaml.safe_dump(scores))
    plan = module.preflight(base, [collection], canonical, tmp_path / "out")
    assert len(plan["accepted"]) == 1 and len(plan["rejected"]) == 2
    assert plan["accepted"][0][0]["trial_id"] == "trial_000003"
    assert module.summary(plan)["new_validation_episodes"] == 0


def test_nonfinite_scores_cannot_silently_pass_threshold_comparisons(tmp_path):
    module = appender(); canonical, base, _ = make_base(tmp_path, module)
    collection = make_collection(tmp_path / "sc_nonfinite", count=1)
    path = collection / "score.yaml"; scores = yaml.safe_load(path.read_text())
    scores["trial_000001"]["tier_1"]["score"] = float("nan")
    path.write_text(yaml.safe_dump(scores))
    with pytest.raises(ValueError, match="scores must be finite"):
        module.preflight(base, [collection], canonical, tmp_path / "out")
    with pytest.raises(ValueError, match="acceptance requires"):
        module.preflight(base, [collection], canonical, tmp_path / "out", minimum_score=float("nan"))
    assert not (tmp_path / "out").exists()


def test_insertion_force_penalty_excludes_an_otherwise_successful_episode(tmp_path):
    module = appender(); canonical, base, _ = make_base(tmp_path, module)
    collection = make_collection(tmp_path / "force_penalty", count=2)
    path = collection / "score.yaml"; scores = yaml.safe_load(path.read_text())
    scores["trial_000001"]["tier_2"]["categories"]["insertion force"]["score"] = -1.
    path.write_text(yaml.safe_dump(scores))
    plan = module.preflight(base, [collection], canonical, tmp_path / "out")
    assert [r[0]["trial_id"] for r in plan["accepted"]] == ["trial_000002"]
    assert plan["accepted"][0][0]["force_penalty_verified_absent"] is True
    assert plan["rejected"][0]["rejection_reason"] == "insertion_force_penalty"


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_nonfinite_force_evidence_fails_before_publication(tmp_path, missing):
    module = appender(); canonical, base, _ = make_base(tmp_path, module)
    collection = make_collection(tmp_path / "unknown_force", count=1)
    path = collection / "score.yaml"; scores = yaml.safe_load(path.read_text())
    categories = scores["trial_000001"]["tier_2"]["categories"]
    if missing:
        del categories["insertion force"]
    else:
        categories["insertion force"]["score"] = float("nan")
    path.write_text(yaml.safe_dump(scores))
    with pytest.raises(ValueError, match="finite insertion-force evidence"):
        module.preflight(base, [collection], canonical, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_stale_manifest_rolls_back_staged_output_without_touching_original_cache(tmp_path):
    module = appender(); canonical, base, _ = make_base(tmp_path, module)
    collection = make_collection(tmp_path / "sc_new", count=1)
    destination = tmp_path / "out"
    plan = module.preflight(base, [collection], canonical, destination)
    manifest = json.loads((canonical / "manifest.json").read_text()); manifest["note"] = "changed concurrently"
    module.write_json(canonical / "manifest.json", manifest)
    with pytest.raises(ValueError, match="changed after preflight"):
        module.materialize(plan, base, canonical, destination)
    assert not destination.exists()
    assert not list(canonical.glob(".append_building_*"))
    assert not (canonical / "sc_to_sc").exists()
    assert json.loads((canonical / "manifest.json").read_text())["note"] == "changed concurrently"
    assert (base / "states.npy").is_file()


def test_cross_filesystem_link_falls_back_only_for_exdev(tmp_path, monkeypatch):
    module = appender(); source = tmp_path / "source"; source.write_bytes(b"recorded source bytes")
    def fail_cross_device(*args):
        raise OSError(errno.EXDEV, "different filesystems")
    monkeypatch.setattr(module.os, "link", fail_cross_device)
    module.link_or_copy(source, tmp_path / "copied")
    assert (tmp_path / "copied").read_bytes() == source.read_bytes()
    def fail_permissions(*args):
        raise OSError(errno.EACCES, "denied")
    monkeypatch.setattr(module.os, "link", fail_permissions)
    with pytest.raises(OSError) as error:
        module.link_or_copy(source, tmp_path / "must_not_exist")
    assert error.value.errno == errno.EACCES
