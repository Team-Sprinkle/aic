from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml

from gazebo_rl.gym_env import GazeboRLEnv
from gazebo_rl.serl_train import run_training
from test_serl_train import _args


def _request(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "task_family": "sfp_to_nic",
                "generation": {"target_accepted_trajectories": 1, "seed": 1},
                "scene": {
                    "nic_cards": {"count": 1, "target_card": 0, "target_port": 0},
                    "start_near_gate": {
                        "axial_distance_m": 0.012,
                        "lateral_distance_m": 0.003,
                        "min_clearance_m": 0.010,
                    },
                    "end_effector_tip": {
                        "body_name": "sfp_tip_link",
                        "body_position_offset": [0.0, 0.0, 0.0],
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_gazebo_serl_dry_run_materializes_start_near_gate_yaml(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.user_config_yaml = _request(tmp_path / "request.yaml")
    args.episode_config_count = 1
    args.reward_preset = "cheatcode_insertion_v1"
    args._explicit_cli_flags = {"--reward-preset", "--user-config-yaml", "--episode-config-count"}

    summary = run_training(args)

    assert summary["status"] == "dry_run"
    assert summary["reward_preset"] == "cheatcode_insertion_v1"
    start = summary["first_episode_config"]["scene"]["start_near_gate"]
    assert start["reset_mode"] == "body_start_position_world"
    assert start["achieved_axial_distance_m"] == pytest.approx(0.012, abs=1e-6)
    assert start["achieved_lateral_distance_m"] == pytest.approx(0.003, abs=1e-6)
    axis = start["target_gate_axis_world"]
    ref = start["reference_tip_center_position_world"]
    gate = start["target_gate_position"]
    axial_delta = sum((ref[i] - gate[i]) * axis[i] for i in range(3))
    assert axial_delta < 0.0
    assert (tmp_path / "out" / "dry_run_summary.json").is_file()
    assert (tmp_path / "out" / "train_config.json").is_file()


def test_start_near_gate_requires_explicit_reward_only_override(tmp_path: Path) -> None:
    episode = {
        "scene": {
            "start_near_gate": {"reset_mode": "body_start_position_world"},
            "target": {},
        }
    }
    with pytest.raises(RuntimeError, match="allow-reward-only-curriculum"):
        GazeboRLEnv(
            workspace_dir=tmp_path,
            max_steps=1,
            per_trial_timeout_sec=1.0,
            episode_config=episode,
        )


def test_start_near_gate_reward_only_override_is_recorded(tmp_path: Path) -> None:
    episode = {
        "scene": {
            "start_near_gate": {"reset_mode": "body_start_position_world"},
            "target": {},
        }
    }
    env = GazeboRLEnv(
        workspace_dir=tmp_path,
        max_steps=1,
        per_trial_timeout_sec=1.0,
        episode_config=episode,
        allow_reward_only_curriculum=True,
    )
    try:
        assert env.curriculum_start_mode == "reward_only"
    finally:
        env.close()
