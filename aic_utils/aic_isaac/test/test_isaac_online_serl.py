from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import yaml

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
        act_torchscript_device="auto",
        output_dir=tmp_path / "out",
        steps=10,
        updates=2,
        replay_capacity=100,
        batch_size=4,
        warmup_steps=2,
        actor_update_start_steps=80,
        actor_update_end_steps=400,
        max_wall_time_minutes=30.0,
        log_every=5,
        gamma=0.99,
        tau=0.005,
        adapter_lr=1e-4,
        critic_lr=1e-4,
        act_lr=1e-5,
        actor_q_weight=0.25,
        adapter_penalty_weight=1e-2,
        act_preservation_weight=1e-1,
        adapter_delta_clip=0.05,
        tcp_translation_action_clip=0.002,
        tcp_rotation_action_clip=0.003,
        target_action_guide_mode="cheatcode_transform",
        target_action_guide_lateral_switch_m=0.002,
        target_action_guide_axial_blend_lateral_m=0.006,
        target_action_guide_prefix_decay=True,
        action_clip=0.05,
        isaac_action_scale=1.0,
        freeze_act=True,
        randomization_profile="none",
        insertion_distance_weight=0.2,
        insertion_close_weight=0.3,
        insertion_orientation_weight=0.0,
        insertion_reaching_weight=1.0,
        insertion_lateral_weight=-0.1,
        insertion_lateral_gate_sigma=0.012,
        insertion_lateral_error_scale=0.006,
        insertion_corridor_weight=0.4,
        insertion_corridor_sigma=0.0025,
        insertion_bypass_penalty_scale=1.0,
        insertion_axis=0,
        force_delta_penalty_weight=0.3,
        force_delta_threshold=10.0,
        force_delta_reference=20.0,
        force_delta_saturation=30.0,
        force_delta_knee_penalty_fraction=0.1,
        isaac_force_observation_clip_n=35.0,
        act_normalized_state_clip=10.0,
        target_reward_body="sfp_tip_link",
        target_reward_distance_std=0.02,
        target_reward_close_sigma=0.01,
        target_reward_reaching_threshold=0.01,
        target_success_termination_threshold=0.0035,
        terminate_on_target_success=True,
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
        enable_contact_sensor=True,
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
    assert "--act_torchscript_device" in cmd
    assert "auto" in cmd
    assert "--updates" in cmd
    assert "2" in cmd
    assert "--actor_update_start_steps" in cmd
    assert "80" in cmd
    assert "--rendering_mode" in cmd
    assert "performance" in cmd
    assert "--max_wall_time_minutes" in cmd
    assert "30.0" in cmd
    assert "--adapter_penalty_weight" in cmd
    assert "0.01" in cmd
    assert "--act_preservation_weight" in cmd
    assert "0.1" in cmd
    assert "--actor_q_weight" in cmd
    assert "0.25" in cmd
    assert "--actor_update_end_steps" in cmd
    assert "400" in cmd
    assert "--adapter_delta_clip" in cmd
    assert "0.05" in cmd
    assert "--tcp_translation_action_clip" in cmd
    assert "0.002" in cmd
    assert "--tcp_rotation_action_clip" in cmd
    assert "0.003" in cmd
    assert "--target_action_guide_lateral_switch_m" in cmd
    assert "--target_action_guide_mode" in cmd
    mode_idx = cmd.index("--target_action_guide_mode")
    assert cmd[mode_idx + 1] == "cheatcode_transform"
    assert "--target_action_guide_axial_blend_lateral_m" in cmd
    assert "--target_action_guide_prefix_decay" in cmd
    assert "--force_delta_penalty_weight" in cmd
    assert "0.3" in cmd
    assert "--force_delta_saturation" in cmd
    assert "30.0" in cmd
    assert "--force_delta_knee_penalty_fraction" in cmd
    assert "--isaac_force_observation_clip_n" in cmd
    assert "--enable_contact_sensor" in cmd
    assert "--target_reward_lateral_gate_sigma" in cmd
    assert "--target_reward_insertion_corridor_weight" in cmd
    idx = cmd.index("--target_reward_insertion_corridor_weight")
    assert cmd[idx + 1] == "0.4"
    assert "--target_reward_insertion_corridor_sigma" in cmd
    assert "--target_reward_insertion_bypass_penalty_scale" in cmd
    assert "--target_reward_insertion_orientation_gate_std" in cmd
    assert "--target_reward_insertion_axis" in cmd
    assert "--terminate_on_target_success" in cmd
    assert "--target_success_termination_threshold" in cmd
    assert "0.0035" in cmd
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
    assert env["AIC_ISAAC_FORCE_DELTA_PENALTY_WEIGHT"] == "0.3"
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


def test_positive_lateral_weight_is_treated_as_penalty_magnitude(tmp_path: Path) -> None:
    args = isaac_online_serl.parse_args(
        [
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--act-torchscript",
            str(tmp_path / "missing_ts.pt"),
            "--insertion-lateral-weight",
            "0.08",
            "--dry-run",
        ]
    )

    plan = isaac_online_serl.build_plan(args, inspect_required=False)
    cmd, env = isaac_online_serl.build_command(args)

    assert plan["insertion_lateral_weight"] == -0.08
    assert env["AIC_ISAAC_INSERTION_LATERAL_WEIGHT"] == "-0.08"
    idx = cmd.index("--target_reward_lateral_weight")
    assert cmd[idx + 1] == "-0.08"


def test_reward_preset_keeps_explicit_cli_overrides(tmp_path: Path) -> None:
    args = isaac_online_serl.parse_args(
        [
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--act-torchscript",
            str(tmp_path / "missing_ts.pt"),
            "--reward-preset",
            "near_gate_corridor_v1",
            "--insertion-axial-progress-weight",
            "0.5",
            "--insertion-corridor-weight",
            "1.0",
            "--dry-run",
        ]
    )

    isaac_online_serl.apply_reward_preset(args)
    cmd, _ = isaac_online_serl.build_command(args)

    assert args.insertion_axial_progress_weight == 0.5
    assert args.insertion_corridor_weight == 1.0
    assert args.insertion_lateral_progress_weight == 0.25
    assert args.insertion_bypass_penalty_scale == 2.0
    assert args.insertion_corridor_orientation_gate_std == 0.10
    axial_idx = cmd.index("--target_reward_axial_progress_weight")
    corridor_idx = cmd.index("--target_reward_insertion_corridor_weight")
    orientation_gate_idx = cmd.index("--target_reward_insertion_orientation_gate_std")
    assert cmd[axial_idx + 1] == "0.5"
    assert cmd[corridor_idx + 1] == "1.0"
    assert cmd[orientation_gate_idx + 1] == "0.1"


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


def test_near_gate_axial_start_is_outside_entrance(tmp_path: Path) -> None:
    request = tmp_path / "request.yaml"
    request.write_text(
        """
task_family: sfp_to_nic
generation:
  target_accepted_trajectories: 1
  seed: 11
scene:
  start_near_gate:
    axial_distance_m: 0.006
    lateral_distance_m: 0.006
    min_clearance_m: 0.004
  nic_cards:
    count: 1
    target_card: 0
    target_port: sfp_port_0
""",
        encoding="utf-8",
    )

    summary = isaac_online_serl.materialize_episode_configs(request, tmp_path / "episodes")
    episode = yaml.safe_load((Path(summary["episodes_dir"]) / "episode_000001.yaml").read_text(encoding="utf-8"))
    start = episode["scene"]["start_near_gate"]
    gate = start["target_gate_position"]
    body = start["body_start_position_world"]
    axis = start["target_gate_axis_world"]
    signed_depth = sum((float(body[i]) - float(gate[i])) * float(axis[i]) for i in range(3))

    assert signed_depth < 0.0
    assert abs(abs(signed_depth) - 0.006) < 1e-4
    assert start["achieved_axial_distance_m"] == 0.006


def test_near_gate_can_reset_controlled_body_with_reference_offset(tmp_path: Path) -> None:
    request = tmp_path / "request.yaml"
    request.write_text(
        """
task_family: sfp_to_nic
generation:
  target_accepted_trajectories: 1
  seed: 11
scene:
  start_near_gate:
    axial_distance_m: 0.004
    lateral_distance_m: 0.0005
    min_clearance_m: 0.004
    reset_body_name: gripper_tcp
    reset_body_offset_from_reference_world: [-0.007149, 0.002556, 0.059066]
    reset_body_orientation_wxyz: [0.026548, 0.013188, 0.991236, 0.128732]
  nic_cards:
    count: 1
    target_card: 0
    target_port: sfp_port_0
""",
        encoding="utf-8",
    )

    summary = isaac_online_serl.materialize_episode_configs(request, tmp_path / "episodes")
    episode = yaml.safe_load((Path(summary["episodes_dir"]) / "episode_000001.yaml").read_text(encoding="utf-8"))
    start = episode["scene"]["start_near_gate"]
    reference = start["reference_reward_body_start_position_world"]
    body = start["body_start_position_world"]

    assert start["reset_body_name"] == "gripper_tcp"
    assert start["reference_reward_body_name"] == "sfp_tip_link"
    assert start["reset_body_offset_from_reference_world"] == [-0.007149, 0.002556, 0.059066]
    assert start["body_start_orientation_wxyz"] == [0.026548, 0.013188, 0.991236, 0.128732]
    assert start["tcp_start_orientation_world"] == [0.026548, 0.013188, 0.991236, 0.128732]
    assert body == [
        round(reference[0] - 0.007149, 6),
        round(reference[1] + 0.002556, 6),
        round(reference[2] + 0.059066, 6),
    ]
    assert start["reference_body_position"] == reference
    assert start["tcp_start_position_world"] == body


def test_near_gate_distances_are_to_semantic_tip_center_with_body_offset(tmp_path: Path) -> None:
    request = tmp_path / "request.yaml"
    request.write_text(
        """
task_family: sfp_to_nic
generation:
  target_accepted_trajectories: 1
  seed: 11
scene:
  end_effector_tip:
    body_name: sfp_module_link
    body_position_offset: [0.0, 0.02365, 0.0]
  start_near_gate:
    axial_distance_m: 0.006
    lateral_distance_m: 0.006
    min_clearance_m: 0.004
  nic_cards:
    count: 1
    target_card: 0
    target_port: sfp_port_0
""",
        encoding="utf-8",
    )

    summary = isaac_online_serl.materialize_episode_configs(request, tmp_path / "episodes")
    episode = yaml.safe_load((Path(summary["episodes_dir"]) / "episode_000001.yaml").read_text(encoding="utf-8"))
    target = episode["scene"]["target"]
    start = episode["scene"]["start_near_gate"]

    assert target["target_reward_body"] == "sfp_module_link"
    assert target["body_position_offset"] == [0.0, 0.02365, 0.0]
    assert start["reference_reward_body_name"] == "sfp_module_link"
    assert start["reference_reward_body_position_offset"] == [0.0, 0.02365, 0.0]
    assert start["reference_tip_center_position_world"] == start["reference_body_position"]
    assert start["body_start_position_world"] != start["reference_tip_center_position_world"]
    gate = start["target_gate_position"]
    tip = start["reference_tip_center_position_world"]
    axis = start["target_gate_axis_world"]
    signed_depth = sum((float(tip[i]) - float(gate[i])) * float(axis[i]) for i in range(3))
    assert signed_depth < 0.0
    assert abs(abs(signed_depth) - 0.006) < 1e-4
    assert start["achieved_lateral_distance_m"] == 0.006


def test_sc_near_gate_uses_sc_tip_link_by_default(tmp_path: Path) -> None:
    request = tmp_path / "request.yaml"
    request.write_text(
        """
task_family: sc_to_sc
generation:
  target_accepted_trajectories: 1
  seed: 7
scene:
  start_near_gate:
    axial_distance_m: 0.006
    lateral_distance_m: 0.006
    min_clearance_m: 0.004
  sc_ports:
    count: 1
    target_port: sc_port_0
""",
        encoding="utf-8",
    )

    summary = isaac_online_serl.materialize_episode_configs(request, tmp_path / "episodes")
    episode = yaml.safe_load((Path(summary["episodes_dir"]) / "episode_000001.yaml").read_text(encoding="utf-8"))
    target = episode["scene"]["target"]
    start = episode["scene"]["start_near_gate"]

    assert target["target_reward_body"] == "sc_tip_link"
    assert target["body_position_offset"] == [0.0, 0.0, 0.0]
    assert start["reference_reward_body_name"] == "sc_tip_link"
    assert start["achieved_axial_distance_m"] == 0.006
    assert start["achieved_lateral_distance_m"] == 0.006


def test_sfp_entrance_axis_offset_shifts_semantic_gate(tmp_path: Path) -> None:
    request = tmp_path / "request.yaml"
    request.write_text(
        """
task_family: sfp_to_nic
generation:
  target_accepted_trajectories: 1
  seed: 11
scene:
  target:
    entrance_axis_offset_m: -0.0009
  start_near_gate:
    axial_distance_m: 0.002
    lateral_distance_m: 0.0005
    min_clearance_m: 0.001
  nic_cards:
    count: 1
    target_card: 0
    target_port: sfp_port_0
""",
        encoding="utf-8",
    )

    summary = isaac_online_serl.materialize_episode_configs(request, tmp_path / "episodes")
    episode = yaml.safe_load((Path(summary["episodes_dir"]) / "episode_000001.yaml").read_text(encoding="utf-8"))
    target = episode["scene"]["target"]
    start = episode["scene"]["start_near_gate"]
    entrance = target["entrance_pose_world"]["position"]
    start_gate = start["target_gate_position"]
    axis = target["insertion_axis_world"]
    signed_depth_to_target = sum(
        (float(target["target_pose_world"]["position"][i]) - float(entrance[i])) * float(axis[i])
        for i in range(3)
    )

    assert target["entrance_axis_offset_m"] == -0.0009
    assert entrance == start_gate
    assert signed_depth_to_target > 0.002


def test_sfp_seated_depth_override_places_target_along_entrance_axis(tmp_path: Path) -> None:
    request = tmp_path / "request.yaml"
    request.write_text(
        """
task_family: sfp_to_nic
generation:
  target_accepted_trajectories: 1
  seed: 11
scene:
  target:
    entrance_axis_offset_m: -0.0009
    seated_depth_m: 0.008
  start_near_gate:
    axial_distance_m: 0.002
    lateral_distance_m: 0.0005
    min_clearance_m: 0.001
  nic_cards:
    count: 1
    target_card: 0
    target_port: sfp_port_0
""",
        encoding="utf-8",
    )

    summary = isaac_online_serl.materialize_episode_configs(request, tmp_path / "episodes")
    episode = yaml.safe_load((Path(summary["episodes_dir"]) / "episode_000001.yaml").read_text(encoding="utf-8"))
    target = episode["scene"]["target"]
    entrance = target["entrance_pose_world"]["position"]
    axis = target["insertion_axis_world"]
    signed_depth_to_target = sum(
        (float(target["target_pose_world"]["position"][i]) - float(entrance[i])) * float(axis[i])
        for i in range(3)
    )

    assert target["seated_depth_m"] == 0.008
    assert signed_depth_to_target == pytest.approx(0.008, abs=5e-6)


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
