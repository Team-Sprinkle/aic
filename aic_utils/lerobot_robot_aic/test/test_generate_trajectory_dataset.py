from __future__ import annotations

import importlib.util
import json
import math
import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pytest
import yaml

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generate_trajectory_dataset.py"
)
spec = importlib.util.spec_from_file_location("generate_trajectory_dataset", SCRIPT)
gtd = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(gtd)

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

from lerobot_robot_aic.dataset_schema import summarize_dataset_schema  # noqa: E402
from lerobot_robot_aic.policy_recorder import should_save_episode  # noqa: E402
from lerobot_robot_aic.task_metadata import (  # noqa: E402
    join_task_vector_to_frames,
    load_episode_task_vectors,
    parse_task_metadata_csv,
    task_vector_from_fields,
    validate_task_vector,
)
from action_msgs.msg import GoalStatus  # noqa: E402

TRAIN_SCRIPT = PACKAGE_DIR / "scripts" / "train_act_policy.py"
train_spec = importlib.util.spec_from_file_location("train_act_policy", TRAIN_SCRIPT)
train_act_policy = importlib.util.module_from_spec(train_spec)
assert train_spec.loader is not None
train_spec.loader.exec_module(train_act_policy)


def base_request(tmp_path: Path, task_family: str = "sfp_to_nic") -> dict:
    request = {
        "root_dir": str(tmp_path / "outputs"),
        "task_family": task_family,
        "suffix": "unit",
        "generation": {
            "target_accepted_trajectories": 2,
            "max_attempts": 3,
            "policy": "cheatcode",
            "seed": 7,
            "append_if_exists": True,
        },
        "acceptance": {"success_only": True, "min_score": 90.0},
        "scene": {},
    }
    if task_family == "sfp_to_nic":
        request["scene"]["nic_cards"] = {"count": 1}
    else:
        request["scene"]["sc_ports"] = {"count": 2}
    return request


def test_yaml_parsing(tmp_path: Path) -> None:
    path = tmp_path / "request.yaml"
    path.write_text(yaml.safe_dump(base_request(tmp_path)), encoding="utf-8")
    loaded = gtd.load_request(path)
    gtd.validate_request(loaded)
    assert loaded["task_family"] == "sfp_to_nic"


def test_policy_recorder_drops_failed_episodes_by_default() -> None:
    assert should_save_episode(GoalStatus.STATUS_SUCCEEDED, save_failed_episodes=False)
    assert not should_save_episode(GoalStatus.STATUS_ABORTED, save_failed_episodes=False)
    assert not should_save_episode(GoalStatus.STATUS_CANCELED, save_failed_episodes=False)
    assert should_save_episode(GoalStatus.STATUS_ABORTED, save_failed_episodes=True)


def test_output_directory_derivation(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    out = gtd.derive_output_dir(request)
    assert out == (
        tmp_path
        / "outputs"
        / "sfp_to_nic"
        / "cheatcode"
        / "nic_cards_1"
        / "n2__unit"
    )
    request["scene"]["nic_cards"]["count"] = [1, 2]
    assert "nic_cards_mixed" in str(gtd.derive_output_dir(request))
    request["output_dir"] = str(tmp_path / "custom_output")
    assert gtd.derive_output_dir(request) == tmp_path / "custom_output"


def test_derived_dataset_repo_id_is_hf_validation_safe(tmp_path: Path) -> None:
    output_dir = (
        tmp_path
        / "outputs"
        / "sfp_to_nic"
        / "cheatcode"
        / "nic_cards_1"
        / "n10__hybrid_nominal_sfp_to_nic_cheatcode_with_extra_suffix"
    )
    repo_id = gtd.derived_dataset_repo_id(output_dir)
    assert repo_id.startswith("local/")
    assert len(repo_id) <= 96


def test_sample_value_scalar_list_and_minmax() -> None:
    rng = gtd.random.Random(1)
    assert gtd.sample_value(4, None, rng) == 4
    assert gtd.sample_value(["a"], None, rng) == "a"
    assert gtd.sample_value({"min": 2.0, "max": 2.0}, None, rng) == 2.0
    val = gtd.sample_value({"min": 1.0, "max": 3.0}, None, rng)
    assert 1.0 <= val <= 3.0


def test_degree_to_radian_conversion(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["scene"]["board"] = {"yaw_deg": {"min": 180.0, "max": 180.0}}
    trial = next(iter(gtd.generate_trials(request, 1).values()))
    assert trial["scene"]["task_board"]["pose"]["yaw"] == pytest.approx(math.pi, abs=1e-5)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("nic_cards", "yaw_deg", {"min": -11.0, "max": 0.0}),
        ("fixture_mounts", "yaw_deg", {"min": -61.0, "max": 0.0}),
        ("nic_cards", "translation", {"min": -0.03, "max": 0.0}),
        ("sc_ports", "translation", {"min": -0.07, "max": 0.0}),
    ],
)
def test_validation_rejects_out_of_range(
    tmp_path: Path, section: str, field: str, value: dict
) -> None:
    request = base_request(tmp_path)
    request["scene"].setdefault(section, {})
    request["scene"][section][field] = value
    with pytest.raises(ValueError):
        gtd.validate_override_limits(request)


def test_exact_nic_count_behavior(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["scene"]["nic_cards"] = {"count": 3}
    trial = next(iter(gtd.generate_trials(request, 1).values()))
    board = trial["scene"]["task_board"]
    assert sum(1 for rail in gtd.NIC_RAILS if board[rail]["entity_present"]) == 3


def test_explicit_target_card_is_present_with_single_nic(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["scene"]["nic_cards"] = {
        "count": 1,
        "rails": gtd.NIC_RAILS,
        "target_card": 1,
        "target_port": "sfp_port_1",
    }
    trial = next(iter(gtd.generate_trials(request, 1).values()))
    board = trial["scene"]["task_board"]
    assert board["nic_rail_1"]["entity_present"]
    assert sum(1 for rail in gtd.NIC_RAILS if board[rail]["entity_present"]) == 1
    assert trial["tasks"]["task_1"]["target_module_name"] == "nic_card_mount_1"
    assert trial["tasks"]["task_1"]["port_name"] == "sfp_port_1"


def test_explicit_target_card_is_present_with_mixed_nic_counts(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["generation"]["seed"] = 11
    request["scene"]["nic_cards"] = {
        "count": [1, 2, 3],
        "rails": gtd.NIC_RAILS,
        "target_card": 1,
    }
    trials = gtd.generate_trials(request, 20)
    seen_counts = set()
    for trial in trials.values():
        board = trial["scene"]["task_board"]
        present_count = sum(1 for rail in gtd.NIC_RAILS if board[rail]["entity_present"])
        seen_counts.add(present_count)
        assert board["nic_rail_1"]["entity_present"]
        assert trial["tasks"]["task_1"]["target_module_name"] == "nic_card_mount_1"
    assert seen_counts <= {1, 2, 3}
    assert len(seen_counts) > 1


def test_default_sfp_target_port_randomizes_both_ports(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["generation"]["seed"] = 5
    request["scene"]["nic_cards"] = {"count": 1, "target_card": 0}
    trials = gtd.generate_trials(request, 40)
    ports = {trial["tasks"]["task_1"]["port_name"] for trial in trials.values()}
    assert ports == {"sfp_port_0", "sfp_port_1"}


def test_exact_sc_count_behavior(tmp_path: Path) -> None:
    request = base_request(tmp_path, task_family="sc_to_sc")
    request["scene"]["sc_ports"] = {"count": 1}
    trial = next(iter(gtd.generate_trials(request, 1).values()))
    board = trial["scene"]["task_board"]
    assert sum(1 for rail in gtd.SC_RAILS if board[rail]["entity_present"]) == 1


def test_sc_target_port_index_is_derived_from_target_module(tmp_path: Path) -> None:
    request = base_request(tmp_path, task_family="sc_to_sc")
    request["scene"]["sc_ports"] = {"count": 1, "target_port": 1}
    trial = next(iter(gtd.generate_trials(request, 1).values()))
    assert trial["tasks"]["task_1"]["target_module_name"] == "sc_port_1"
    metadata = gtd._task_metadata_from_trial("sc_to_sc", trial)
    assert metadata["task"]["target_port_index"] == 1
    assert metadata["task"]["target_card_index"] == -1


def test_inconsistent_cable_type_override_is_rejected(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["scene"]["cable"] = {"cable_type": "sfp_sc_cable_reversed"}
    with pytest.raises(ValueError, match="requires scene.cable.cable_type"):
        gtd.generate_trials(request, 1)


def test_recording_outputs_complete_accepts_failed_trial_rows(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    (output_dir / "raw_dataset" / "meta").mkdir(parents=True)
    (output_dir / "raw_dataset" / "meta" / "info.json").write_text("{}", encoding="utf-8")
    (output_dir / "scores").mkdir()
    (output_dir / "scores" / "score_summary.csv").write_text(
        "\n".join(
            [
                "run_index,trial_id,status,total_score,scoring_yaml",
                "1,trial_000001,OK,95,path/to/scoring.yaml",
                "2,trial_000002,FAILED,0,path/to/scoring.yaml",
                "3,trial_000003,OK,94,path/to/scoring.yaml",
            ]
        ),
        encoding="utf-8",
    )

    complete, reason = gtd.recording_outputs_complete(output_dir, expected_trials=3)

    assert complete
    assert "3/3" in reason


def test_recording_outputs_complete_rejects_incomplete_score_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    (output_dir / "raw_dataset" / "meta").mkdir(parents=True)
    (output_dir / "raw_dataset" / "meta" / "info.json").write_text("{}", encoding="utf-8")
    (output_dir / "scores").mkdir()
    (output_dir / "scores" / "score_summary.csv").write_text(
        "\n".join(
            [
                "run_index,trial_id,status,total_score,scoring_yaml",
                "1,trial_000001,OK,95,path/to/scoring.yaml",
            ]
        ),
        encoding="utf-8",
    )

    complete, reason = gtd.recording_outputs_complete(output_dir, expected_trials=3)

    assert not complete
    assert "1/3" in reason


def test_dry_run_creates_expected_files(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["suffix"] = "smoke_test"
    request_path = tmp_path / "request.yaml"
    request_path.write_text(yaml.safe_dump(request), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--request-yaml",
            str(request_path),
            "--dry-run",
            "--skip-recording",
            "--target-accepted-override",
            "2",
            "--max-attempts-override",
            "3",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    out = (
        tmp_path
        / "outputs"
        / "sfp_to_nic"
        / "cheatcode"
        / "nic_cards_1"
        / "n2__smoke_test"
    )
    assert (out / "request.yaml").exists()
    assert (out / "engine_config.yaml").exists()
    assert (out / "trials" / "trial_000001.yaml").exists()
    assert (out / "generation_summary.json").exists()
    assert (out / "manifests" / "attempts.csv").exists()
    assert (out / "manifests" / "accepted.csv").exists()
    assert (out / "manifests" / "episode_task_metadata.jsonl").exists()
    assert not (out / "raw_dataset").exists()
    assert not (out / "accepted_dataset").exists()
    summary = json.loads((out / "generation_summary.json").read_text(encoding="utf-8"))
    assert summary["action_mode"] == "cartesian"
    assert "manifests" in summary
    manifest_rows = parse_task_metadata_csv(out / "manifests" / "attempts.csv")
    assert manifest_rows
    assert manifest_rows[0]["task_family"] == "sfp_to_nic"
    assert len(manifest_rows[0]["task_vector"]) == 10


def test_manifest_generation_joins_scores_and_selection_report(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["scene"]["nic_cards"] = {"count": 1, "target_card": 3, "target_port": "sfp_port_1"}
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _, trials = gtd.write_engine_configs(request, output_dir, 1)
    (output_dir / "scores").mkdir(exist_ok=True)
    (output_dir / "scores" / "score_summary.csv").write_text(
        "run_index,trial_id,status,total_score,scoring_yaml\n"
        "1,trial_000001,OK,97.5,scores/trial_000001.yaml\n",
        encoding="utf-8",
    )
    (output_dir / "accepted_dataset").mkdir()
    (output_dir / "accepted_dataset" / "selection_report.csv").write_text(
        "dataset_root,score_csv,trial_id,run_index,status,total_score,mapped_episode_index,selected,reason\n"
        "raw,scores,trial_000001,1,OK,97.5,0,True,selected\n",
        encoding="utf-8",
    )

    paths = gtd.write_task_manifests(output_dir, "sfp_to_nic", trials)

    rows = parse_task_metadata_csv(Path(paths["accepted_csv"]))
    assert rows[0]["accepted_episode_index"] == 0
    assert rows[0]["source_episode_index"] == 0
    assert rows[0]["target_card_index"] == 3
    assert rows[0]["target_port_index"] == 1
    assert rows[0]["task_vector"] == [1, 0, 0, 1, 0, 0, 0, 1, 0, 1]
    vectors = load_episode_task_vectors(Path(paths["episode_task_metadata_jsonl"]))
    assert vectors[0] == rows[0]["task_vector"]


def test_task_vector_examples_and_validation() -> None:
    assert task_vector_from_fields("sfp_to_nic", 1, 3) == [1, 0, 0, 1, 0, 0, 0, 1, 0, 1]
    assert task_vector_from_fields("sc_to_sc", 0, -1) == [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError):
        validate_task_vector([1, 0, 0], task_family="sfp_to_nic")
    with pytest.raises(ValueError):
        task_vector_from_fields("sc_to_sc", 0, 2)


def test_join_task_vector_to_frames_broadcasts_by_episode() -> None:
    frames = [{"episode_index": 0, "x": 1}, {"episode_index": 0, "x": 2}, {"episode_index": 1, "x": 3}]
    vectors = {
        0: task_vector_from_fields("sfp_to_nic", 1, 3),
        1: task_vector_from_fields("sc_to_sc", 0, -1),
    }
    joined = join_task_vector_to_frames(frames, vectors)
    assert joined[0]["task_vector"] == vectors[0]
    assert joined[1]["task_vector"] == vectors[0]
    assert joined[2]["task_vector"] == vectors[1]


def write_fake_info(dataset_root: Path, action_names: list[str]) -> None:
    meta = dataset_root / "meta"
    meta.mkdir(parents=True)
    info = {
        "fps": 20,
        "robot_type": "ur5e_aic",
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [len(action_names)],
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [31],
                "names": ["tcp_pose.position.x"],
            },
        },
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")


def test_dataset_schema_detects_cartesian_action_mode(tmp_path: Path) -> None:
    dataset_root = tmp_path / "accepted_dataset"
    write_fake_info(
        dataset_root,
        [
            "delta_position.x",
            "delta_position.y",
            "delta_position.z",
            "delta_rotation.x",
            "delta_rotation.y",
            "delta_rotation.z",
        ],
    )
    summary = summarize_dataset_schema(dataset_root)
    assert summary.action_mode == "cartesian"
    assert summary.fps == 20
    assert summary.robot_type == "ur5e_aic"


def test_dataset_schema_detects_joint_action_mode(tmp_path: Path) -> None:
    dataset_root = tmp_path / "accepted_dataset"
    write_fake_info(
        dataset_root,
        [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
    )
    assert summarize_dataset_schema(dataset_root).action_mode == "joint"


def test_act_training_command_uses_local_root_and_act_policy(tmp_path: Path) -> None:
    dataset_root = tmp_path / "accepted_dataset"
    write_fake_info(
        dataset_root,
        [
            "delta_position.x",
            "delta_position.y",
            "delta_position.z",
            "delta_rotation.x",
            "delta_rotation.y",
            "delta_rotation.z",
        ],
    )
    args = argparse.Namespace(
        dataset_root=dataset_root,
        dataset_repo_id=None,
        output_dir=tmp_path / "train",
        job_name="act_smoke",
        steps=200,
        batch_size=4,
        device="cpu",
        num_workers=1,
        lr="1e-4",
        dataset_video_backend="pyav",
        chunk_size=16,
        n_action_steps=8,
        n_obs_steps=1,
        wandb=False,
        policy_repo_id=None,
        extra_arg=[],
    )
    cmd = train_act_policy.build_lerobot_train_cmd(args)
    assert "lerobot-train" == cmd[0]
    assert "--policy.type=act" in cmd
    assert f"--dataset.root={dataset_root.resolve()}" in cmd
    assert "--dataset.video_backend=pyav" in cmd
    assert "--optimizer.lr=1e-4" in cmd
    assert "--policy.optimizer_lr=1e-4" in cmd
    assert "--policy.optimizer_lr_backbone=1e-4" in cmd
    assert "--policy.chunk_size=16" in cmd
    assert "--policy.n_action_steps=8" in cmd
    assert "--policy.n_obs_steps=1" in cmd
    assert "--policy.push_to_hub=false" in cmd
    assert "--wandb.enable=false" in cmd
