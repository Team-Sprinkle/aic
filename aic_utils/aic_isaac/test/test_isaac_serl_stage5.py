from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_isaac_serl_stage5.py"
spec = importlib.util.spec_from_file_location("train_isaac_serl_stage5", SCRIPT)
stage5 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = stage5
spec.loader.exec_module(stage5)


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
        freeze_act=True,
        isaaclab="isaaclab",
        run_name="test_serl",
        dry_run=True,
        extra_arg=[],
    )

    plan = stage5.build_plan(args)
    assert plan["status"] == "implemented_short_run_capable"
    assert plan["checkpoint"]["has_actor"]
    assert plan["checkpoint"]["has_critics"]
    assert plan["checkpoint"]["vision_offline_serl_config"]["actor_mode"] == "act_adapter"
    assert plan["replay_buffer"]["capacity"] == 100

    cmd, env = stage5.build_command(args)
    assert cmd[:2] == ["isaaclab", "-p"]
    assert "--checkpoint" in cmd
    assert str(checkpoint) in cmd
    assert "--act_torchscript" in cmd
    assert str(tmp_path / "act_ts.pt") in cmd
    assert "--updates" in cmd
    assert "2" in cmd
    assert "--max_wall_time_minutes" in cmd
    assert "30.0" in cmd
    assert "--adapter_penalty_weight" in cmd
    assert "0.01" in cmd
    assert "--act_preservation_weight" in cmd
    assert "0.1" in cmd
    assert "--adapter_delta_clip" in cmd
    assert "0.05" in cmd
    assert "--action_clip" in cmd
    assert env["AIC_ISAAC_DISABLE_CAMERAS"] == "0"
