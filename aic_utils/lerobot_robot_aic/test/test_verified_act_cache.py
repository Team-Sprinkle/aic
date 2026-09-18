"""Protect channel conventions and score/label gates used by ACT training."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml


def load_script(name):
    path = Path(__file__).resolve().parents[3] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shards_preserve_frame_order_and_convert_only_needed_channels(tmp_path):
    loader = load_script("act_cache")
    key = "observation.images.center_camera"
    a, b = tmp_path / "old", tmp_path / "new"
    a.mkdir(); b.mkdir()
    np.save(a / (key + ".npy"), np.array([[[[30, 20, 10]]], [[[60, 50, 40]]]], np.uint8))
    np.save(b / (key + ".npy"), np.array([[[[70, 80, 90]]]], np.uint8))
    cache = {"frames": 3, "image_shape_hwc": [1, 1, 3], "camera_keys": [key], "image_channel_order": "bgr",
             "image_shards": [{"root": str(a), "from_index": 0, "to_index": 2, "image_channel_order": "bgr"},
                              {"root": str(b), "from_index": 2, "to_index": 3, "image_channel_order": "rgb"}]}
    images = loader.load_cached_images(tmp_path, cache)[key]
    assert images[[2, 0, 2, 1]][:, 0, 0].tolist() == [[90, 80, 70], [30, 20, 10], [90, 80, 70], [60, 50, 40]]
    with pytest.raises(IndexError):
        images[[3]]


def test_adjacent_episode_shards_share_mapping_without_merging_source_gaps(tmp_path):
    loader = load_script("act_cache")
    key = "observation.images.center_camera"
    source = np.arange(20 * 2 * 2 * 3, dtype=np.uint8).reshape(20, 2, 2, 3)
    np.save(tmp_path / (key + ".npy"), source)
    cache = {"frames": 9, "image_shape_hwc": [2, 2, 3], "camera_keys": [key], "image_channel_order": "rgb",
             "image_shards": [{"root": str(tmp_path), "from_index": start, "to_index": start + 3,
                               "source_from_index": offset, "image_channel_order": "bgr"}
                              for start, offset in [(0, 5), (3, 8), (6, 17)]]}
    images = loader.load_cached_images(tmp_path, cache)[key]
    assert images.source_shard_count == 3 and len(images.shards) == 2
    assert images.shards[0][3] is images.shards[1][3]
    np.testing.assert_array_equal(images[[8, 0, 5, 6, 3]], source[[19, 5, 10, 17, 8], ..., ::-1])
    np.testing.assert_array_equal(images[0], source[5, ..., ::-1])


def test_parallel_camera_collation_preserves_keys_pixels_and_repeated_frames():
    from concurrent.futures import ThreadPoolExecutor
    loader = load_script("act_cache")
    random = np.random.default_rng(74)
    images = {name: random.integers(0, 256, (11, 3, 4, 3), dtype=np.uint8)
              for name in ["center", "left", "right"]}
    indices = np.array([10, 4, 4, 0])
    serial = dict(loader.prepare_camera_arrays(images, indices))
    with ThreadPoolExecutor(max_workers=3) as pool:
        threaded = dict(loader.prepare_camera_arrays(images, indices, pool))
    assert list(serial) == list(threaded) == list(images)
    for name in images:
        np.testing.assert_array_equal(serial[name], threaded[name])
        np.testing.assert_array_equal(threaded[name], images[name][indices].transpose(0, 3, 1, 2))
        assert threaded[name].flags.c_contiguous and threaded[name].dtype == np.uint8


def collection_fixture(root):
    episode = root / "episodes/episode_0001_100"
    episode.mkdir(parents=True)
    (episode / "episode.json").write_text(json.dumps({"frames": 2, "image_channel_order": "rgb"}))
    state = np.zeros(32); state[6] = 1
    rows = [{"state": state.tolist(), "action": [0.] * 6, "teacher_target_pose": [0., 0., 0., 0., 0., 0., 1.],
             "elapsed_sim_time": t, "images": {k: k + ".jpg" for k in ("center", "left", "right")},
             "perturbation_xy_rotvec": [.001, 0., 0., 0., 0.]} for t in (0., .1)]
    for camera in ("center", "left", "right"):
        (episode / (camera + ".jpg")).write_bytes(b"existence checked here; decoder checks pixels during materialization")
    (episode / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (root / "engine_config.yaml").write_text(yaml.safe_dump({"trials": {"trial_1": {"scene": {"task_board": {"nic_rail_0": {"entity_present": True}}}}}}))
    score = root / "score.yaml"
    score.write_text(yaml.safe_dump({"trial_1": {"tier_1": {"score": 1}, "tier_2": {"score": 4}, "tier_3": {"score": 75}}}))
    summary = root / "eval_collection/collection_config/eval_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"evaluation_complete": True, "evaluation_kind": "privileged_corrective_data_collection", "scoring_yaml": str(score)}))
    return episode, score, rows


def test_incremental_cache_reuses_images_and_rejects_duplicate_collections(tmp_path):
    merge = load_script("merge_corrective_act_cache")
    old, new = tmp_path / "collection_old", tmp_path / "collection_new"
    shard = {"root": str(tmp_path / "original_images"), "from_index": 0, "to_index": 7,
             "image_channel_order": "rgb"}
    base = {"frames": 7, "episodes": [{"collection": str(old)}], "image_shards": [shard]}
    inherited = merge.inherited_image_shards(base, tmp_path / "derived_cache", [new])
    assert inherited == [shard]
    inherited[0]["to_index"] = 8
    assert base["image_shards"][0]["to_index"] == 7
    with pytest.raises(ValueError, match="already present"):
        merge.inherited_image_shards(base, tmp_path, [old / "."])
    with pytest.raises(ValueError, match="repeated"):
        merge.inherited_image_shards(base, tmp_path, [new, new / "."])


def test_corrective_score_gate_and_causal_timestamp_resampling(tmp_path):
    merge = load_script("merge_corrective_act_cache")
    _, score, _ = collection_fixture(tmp_path)
    accepted, rejected = merge.inspect_collection(tmp_path, 80.)
    assert len(accepted) == 1 and not rejected
    assert accepted[0][2].tolist() == [0, 0, 1]
    value = yaml.safe_load(score.read_text()); value["trial_1"]["tier_3"]["score"] = 38
    score.write_text(yaml.safe_dump(value))
    accepted, rejected = merge.inspect_collection(tmp_path, 0.)
    assert not accepted and len(rejected) == 1


def test_corrective_labels_must_reconstruct_teacher_target(tmp_path):
    merge = load_script("merge_corrective_act_cache")
    episode, _, rows = collection_fixture(tmp_path)
    rows[0]["teacher_target_pose"][0] = .02
    (episode / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="labels do not reconstruct"):
        merge.inspect_collection(tmp_path, 80.)


def test_unperturbed_collection_requires_explicit_nominal_metadata_and_runtime(tmp_path):
    merge = load_script("merge_corrective_act_cache")
    episode, _, rows = collection_fixture(tmp_path)
    for row in rows:
        row["perturbation_xy_rotvec"] = [0.] * 5
        row["executed_target_pose"] = row["teacher_target_pose"]
    (episode / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    metadata = json.loads((episode / "episode.json").read_text())
    metadata.update(kind="privileged_aligned_demonstration", perturbation_config={"scale": 0})
    (episode / "episode.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="No recorded perturbations"):
        merge.inspect_collection(tmp_path, 80.)
    path = tmp_path / "eval_collection/collection_config/eval_summary.json"
    summary = json.loads(path.read_text()); summary["runtime_settings"] = {"corrective_perturbation_scale": 0}
    path.write_text(json.dumps(summary))
    accepted, rejected = merge.inspect_collection(tmp_path, 80.)
    assert len(accepted) == 1 and not rejected
    assert accepted[0][0]["demonstration_kind"] == "nominal_aligned_expert"


def test_student_correction_requires_lineage_bounded_execution_and_matching_labels(tmp_path):
    import hashlib
    merge = load_script("merge_corrective_act_cache")
    episode, _, rows = collection_fixture(tmp_path)
    actor = tmp_path / "student.pt"; actor.write_bytes(b"student checkpoint identity")
    metadata = json.loads((episode / "episode.json").read_text())
    metadata.update(kind="privileged_corrective_demonstration", student_recorded_frames=2,
                    student_correction={"torchscript": str(actor), "sha256": hashlib.sha256(actor.read_bytes()).hexdigest(),
                                        "probability_per_cycle": 1.})
    (episode / "episode.json").write_text(json.dumps(metadata))
    for row in rows:
        row.update(student_active=True, student_absolute_action=[.001, 0, 0, 0, 0, 0],
                   perturbation_xyz_rotvec=[.001, 0, 0, 0, 0, 0],
                   executed_target_pose=[.001, 0, 0, 0, 0, 0, 1.])
    def save_rows():
        (episode / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    save_rows()
    summary_path = tmp_path / "eval_collection/collection_config/eval_summary.json"
    summary = json.loads(summary_path.read_text()); summary["runtime_settings"] = {"corrective_student_probability": 1.}
    summary_path.write_text(json.dumps(summary))
    accepted, rejected = merge.inspect_collection(tmp_path, 80.)
    assert len(accepted) == 1 and not rejected
    assert accepted[0][0]["demonstration_kind"] == "student_corrective_expert"
    rows[0]["executed_target_pose"][0] = .01; save_rows()
    with pytest.raises(ValueError, match="does not match executed"):
        merge.inspect_collection(tmp_path, 80.)
    rows[0]["executed_target_pose"][0] = .04; rows[0]["perturbation_xyz_rotvec"][0] = .04; save_rows()
    with pytest.raises(ValueError, match="Invalid bounded"):
        merge.inspect_collection(tmp_path, 80.)
    rows[0]["executed_target_pose"][0] = .001; rows[0]["perturbation_xyz_rotvec"][0] = .001; save_rows()
    actor.write_bytes(b"changed student")
    with pytest.raises(ValueError, match="hash mismatch"):
        merge.inspect_collection(tmp_path, 80.)


def test_unused_student_is_nominal_only_when_execution_exactly_matches_expert(tmp_path):
    import hashlib
    merge = load_script("merge_corrective_act_cache")
    episode, _, rows = collection_fixture(tmp_path)
    actor = tmp_path / "student.pt"; actor.write_bytes(b"student")
    metadata = json.loads((episode / "episode.json").read_text())
    metadata.update(kind="privileged_corrective_demonstration", student_recorded_frames=0, perturbation_config={"scale": 0},
                    student_correction={"torchscript": str(actor), "sha256": hashlib.sha256(actor.read_bytes()).hexdigest(),
                                        "probability_per_cycle": .5})
    (episode / "episode.json").write_text(json.dumps(metadata))
    for row in rows:
        row.update(student_active=False, student_absolute_action=None,
                   perturbation_xy_rotvec=[0.] * 5, perturbation_xyz_rotvec=[0.] * 6,
                   executed_target_pose=list(row["teacher_target_pose"]))
    def save():
        (episode / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    save()
    path = tmp_path / "eval_collection/collection_config/eval_summary.json"
    summary = json.loads(path.read_text());summary['runtime_settings'] = {'corrective_student_probability': .5, 'corrective_perturbation_scale': 0}
    path.write_text(json.dumps(summary))
    accepted, rejected = merge.inspect_collection(tmp_path, 80.)
    assert not rejected and accepted[0][0]['demonstration_kind'] == 'nominal_expert_no_student_intervention'
    rows[0]['executed_target_pose'][0] = .001;save()
    with pytest.raises(ValueError, match='Nominal collection changed'):
        merge.inspect_collection(tmp_path, 80.)
