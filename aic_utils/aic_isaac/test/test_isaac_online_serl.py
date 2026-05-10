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
        gripper_joint_position=0.0035405,
        initial_arm_joint_pos="-0.37,-1.62,-1.75,-1.43,1.93,1.31",
        isaaclab="isaaclab",
        run_name="test_serl",
        debug_timing=True,
        disable_fabric=False,
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
