"""Protect scene-group holdouts and image offsets in the canonical expert set."""
import copy
import importlib.util
from pathlib import Path

import pytest


def curator():
    path = Path(__file__).resolve().parents[3] / "scripts/curate_verified_experts.py"
    spec = importlib.util.spec_from_file_location("curate_verified_experts_tested", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(scene, nic_count=1, sc_count=0, task_family="sfp_to_nic"):
    return {"scene_sha256": scene, "nic_count": nic_count, "sc_count": sc_count,
            "task": {"task_family": task_family}}


def test_repeated_scene_cannot_leak_across_training_and_validation():
    split = curator().grouped_split
    records = [record(f"scene_{index}") for index in range(20)]
    records += [copy.deepcopy(records[0]), copy.deepcopy(records[0]), copy.deepcopy(records[12])]
    first = split(copy.deepcopy(records), fraction=.2, seed=18)
    second = split(copy.deepcopy(list(reversed(records))), fraction=.2, seed=18)
    assignments = {}
    for item in first:
        assignments.setdefault(item["scene_sha256"], set()).add(item["split"])
    assert all(len(values) == 1 for values in assignments.values())
    assert {item["split"] for item in first} == {"train", "validation"}
    assert {item["scene_sha256"]: item["split"] for item in first} == {
        item["scene_sha256"]: item["split"] for item in second}


def test_sparse_strata_keep_training_and_split_when_two_distinct_scenes_exist():
    records = [record("single_sfp", nic_count=5), record("single_sfp", nic_count=5),
               record("sc_first", nic_count=2, sc_count=2, task_family="sc_to_sc"),
               record("sc_second", nic_count=2, sc_count=2, task_family="sc_to_sc")]
    result = curator().grouped_split(records, fraction=.1)
    assert all(item["split"] == "train" for item in result if item["nic_count"] == 5)
    assert {item["split"] for item in result if item["nic_count"] == 2} == {"train", "validation"}


def test_selected_image_shards_preserve_nested_source_offsets_and_detect_gaps(tmp_path):
    select = curator().selected_shards
    cache = {"frames": 30, "image_channel_order": "rgb", "image_shards": [
        {"root": "/first", "from_index": 0, "to_index": 10, "source_from_index": 100,
         "image_channel_order": "bgr"},
        {"root": "/second", "from_index": 10, "to_index": 20, "source_from_index": 50,
         "image_channel_order": "rgb"},
        {"root": "/third", "from_index": 20, "to_index": 30, "source_from_index": 7,
         "image_channel_order": "rgb"}]}
    result = select(cache, tmp_path, 3, 17, 200)
    assert result == [
        {"root": "/first", "source_from_index": 103, "from_index": 200, "to_index": 207,
         "image_channel_order": "bgr"},
        {"root": "/second", "source_from_index": 50, "from_index": 207, "to_index": 214,
         "image_channel_order": "rgb"}]
    del cache["image_shards"][1]
    with pytest.raises(ValueError, match="Incomplete"):
        select(cache, tmp_path, 3, 17, 200)
