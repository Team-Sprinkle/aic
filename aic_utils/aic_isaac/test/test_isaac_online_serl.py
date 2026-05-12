from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_isaac_online_serl.py"
spec = importlib.util.spec_from_file_location("train_isaac_online_serl", SCRIPT)
isaac_online_serl = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = isaac_online_serl
spec.loader.exec_module(isaac_online_serl)


def test_isaac_serl_plan_inspects_adapter_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "adapter_serl.pt"
    torch.save(
        {
            "step": 200,
            "actor": {},
            "critic1": {},
            "critic2": {},
            "vision_offline_serl_config": {"actor_mode": "act_adapter", "freeze_act": True},
            "dataset_summary": {"action_horizon": 8},
            "warmstart_report": {"mode": "act_adapter"},
        },
        checkpoint,
    )
    args = argparse.Namespace(
        task="AIC-Task-v0",
        num_envs=2,
        seed=3,
        device="cpu",
        headless=True,
        rendering_mode="performance",
        kit_args=None,
        checkpoint=checkpoint,
        act_torchscript=tmp_path / "act_ts.pt",
        output_dir=tmp_path / "out",
        steps=10,
        updates=2,
        replay_capacity=100,
        batch_size=4,
        warmup_steps=2,
        max_wall_time_minutes=30.0,
        log_every=5,
        gamma=0.99,
        tau=0.005,
        adapter_lr=1e-4,
        critic_lr=1e-4,
        act_lr=1e-5,
        adapter_penalty_weight=1e-2,
        act_preservation_weight=1e-1,
        adapter_delta_clip=0.05,
        action_clip=0.05,
        isaac_action_scale=1.0,
        freeze_act=True,
        randomization_profile="none",
        insertion_distance_weight=0.2,
        insertion_close_weight=0.3,
        insertion_orientation_weight=0.0,
        insertion_reaching_weight=1.0,
        insertion_lateral_weight=-0.1,
        force_delta_penalty_weight=0.2,
        force_delta_threshold=3.0,
        force_delta_reference=20.0,
        target_reward_body="sfp_tip_link",
        target_reward_distance_std=0.02,
        target_reward_close_sigma=0.01,
        target_reward_reaching_threshold=0.01,
        target_reward_position_offset=None,
        target_reward_body_position_offset=None,
        state_source="lerobot_compatible",
        task_family="sfp_to_nic",
        target_port_index=0,
        target_card_index=0,
        target_card_valid=1,
        task_distribution_yaml=None,
        isaac_user_config_yaml=None,
        isaac_user_config_dir=None,
        isaac_user_config_filenames="",
        max_gpus=1,
        episode_config_dir=None,
        episode_config_count=None,
        gripper_joint_position=0.0035405,
        initial_arm_joint_pos="-0.37,-1.62,-1.75,-1.43,1.93,1.31",
        isaaclab="isaaclab",
        run_name="test_serl",
        debug_timing=True,
        disable_fabric=False,
        save_step_images=False,
        image_log_every=1,
        max_logged_image_steps=200,
        dry_run=True,
        extra_arg=[],
    )

    plan = isaac_online_serl.build_plan(args)
    assert plan["status"] == "implemented_short_run_capable"
    assert plan["checkpoint"]["has_actor"]
    assert plan["checkpoint"]["has_critics"]
    assert plan["checkpoint"]["vision_offline_serl_config"]["actor_mode"] == "act_adapter"
    assert plan["replay_buffer"]["capacity"] == 100

    cmd, env = isaac_online_serl.build_command(args)
    assert cmd[:2] == ["isaaclab", "-p"]
    assert "--checkpoint" in cmd
    assert str(checkpoint) in cmd
    assert "--act_torchscript" in cmd
    assert str(tmp_path / "act_ts.pt") in cmd
    assert "--updates" in cmd
    assert "2" in cmd
    assert "--rendering_mode" in cmd
    assert "performance" in cmd
    assert "--max_wall_time_minutes" in cmd
    assert "30.0" in cmd
    assert "--adapter_penalty_weight" in cmd
    assert "0.01" in cmd
    assert "--act_preservation_weight" in cmd
    assert "0.1" in cmd
    assert "--adapter_delta_clip" in cmd
    assert "0.05" in cmd
    assert "--force_delta_penalty_weight" in cmd
    assert "0.2" in cmd
    assert "--action_clip" in cmd
    assert "--isaac_action_scale" in cmd
    assert "1.0" in cmd
    assert "--state_source" in cmd
    assert "lerobot_compatible" in cmd
    assert "--task_family" in cmd
    assert "sfp_to_nic" in cmd
    assert env["AIC_ISAAC_DISABLE_CAMERAS"] == "0"
    assert env["AIC_ISAAC_RANDOMIZATION_PROFILE"] == "none"
    assert env["AIC_ISAAC_INSERTION_DISTANCE_WEIGHT"] == "0.2"
    assert env["AIC_ISAAC_INSERTION_LATERAL_WEIGHT"] == "-0.1"
    assert env["AIC_ISAAC_FORCE_DELTA_PENALTY_WEIGHT"] == "0.2"
    assert env["AIC_ISAAC_INITIAL_ARM_JOINT_POS"] == "-0.37,-1.62,-1.75,-1.43,1.93,1.31"


def test_isaac_serl_dry_run_does_not_require_checkpoint(tmp_path: Path) -> None:
    args = isaac_online_serl.parse_args(
        [
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--act-torchscript",
            str(tmp_path / "missing_ts.pt"),
            "--dry-run",
        ]
    )

    plan = isaac_online_serl.build_plan(args, inspect_required=False)
    assert plan["checkpoint"]["exists"] is False
    assert plan["checkpoint"]["inspect_skipped"] is True


def test_isaac_user_config_materializes_episode_yamls(tmp_path: Path) -> None:
    request = tmp_path / "request.yaml"
    request.write_text(
        """
root_dir: outputs/isaac
task_family: sfp_to_nic
generation:
  target_accepted_trajectories: 2
  max_attempts: 2
  policy: agent
  seed: 7
acceptance:
  success_only: false
  min_score: 0.0
  stop_near_gate:
    max_lateral_error_m: 0.004
scene:
  start_near_gate:
    axial_distance_m: 0.05
    lateral_distance_m: 0.01
  nic_cards:
    count: 1
    target_card: 0
    target_port: sfp_port_1
""",
        encoding="utf-8",
    )

    summary = isaac_online_serl.materialize_episode_configs(request, tmp_path / "episodes")
    episode_path = Path(summary["episodes_dir"]) / "episode_000001.yaml"
    assert episode_path.exists()
    text = episode_path.read_text(encoding="utf-8")
    assert "task_description:" in text
    assert "target_pose_world:" in text
    assert "stop_near_gate:" in text
    assert "achieved_axial_distance_m: 0.05" in text
    assert Path(summary["manifest_csv"]).exists()
    assert Path(summary["task_distribution_yaml"]).exists()


def test_isaac_user_config_wrapper_sets_episode_config_dir(tmp_path: Path) -> None:
    request = tmp_path / "request.yaml"
    request.write_text(
        """
task_family: sc_to_sc
generation:
  target_accepted_trajectories: 1
  max_attempts: 1
  policy: agent
  seed: 3
acceptance:
  success_only: false
  min_score: 0.0
scene:
  start_near_gate:
    distance: 0.08
  sc_ports:
    count: 2
    target_port: 1
""",
        encoding="utf-8",
    )
    args = isaac_online_serl.parse_args(
        [
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--act-torchscript",
            str(tmp_path / "missing_ts.pt"),
            "--isaac-user-config-yaml",
            str(request),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )

    summary = isaac_online_serl.prepare_user_episode_configs(args)
    assert summary is not None
    assert args.episode_config_dir == Path(summary["episodes_dir"])
    assert args.task_distribution_yaml == Path(summary["task_distribution_yaml"])
    cmd, env = isaac_online_serl.build_command(args)
    assert "--episode_config_dir" in cmd
    assert str(args.episode_config_dir) in cmd
    assert env["AIC_ISAAC_EPISODE_CONFIG_DIR"] == str(args.episode_config_dir)


def test_multi_minimal_yaml_materialization_shards_curriculum_order(tmp_path: Path) -> None:
    requests = tmp_path / "requests"
    requests.mkdir()
    (requests / "a.yaml").write_text(
        """
task_family: sfp_to_nic
generation:
  target_accepted_trajectories: 3
  seed: 1
scene:
  nic_cards:
    count: 1
    target_card: 0
    target_port: sfp_port_0
""",
        encoding="utf-8",
    )
    (requests / "b.yaml").write_text(
        """
task_family: sc_to_sc
generation:
  target_accepted_trajectories: 2
  seed: 2
scene:
  sc_ports:
    count: 2
    target_port: auto
""",
        encoding="utf-8",
    )

    summary = isaac_online_serl.materialize_many_episode_configs(
        input_dir=requests,
        output_dir=tmp_path / "episodes",
        filenames=["a.yaml", "b.yaml"],
        max_gpus=2,
    )
    assert summary["episode_count"] == 5
    assert [s["episode_count"] for s in summary["shards"]] == [3, 2]
    manifest = Path(summary["manifest_csv"]).read_text(encoding="utf-8").splitlines()
    assert "episode_000001" in manifest[1]
    assert "a.yaml" in manifest[1]
    assert ",0," in manifest[1]
    assert "episode_000002" in manifest[2]
    assert "a.yaml" in manifest[2]
    assert ",1," in manifest[2]
    assert "episode_000004" in manifest[4]
    assert "b.yaml" in manifest[4]
