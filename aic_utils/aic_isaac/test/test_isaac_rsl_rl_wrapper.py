from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_isaac_rsl_rl.py"
spec = importlib.util.spec_from_file_location("train_isaac_rsl_rl", SCRIPT)
isaac_rsl_rl = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(isaac_rsl_rl)


def test_isaac_rsl_rl_command_sets_randomization_env(tmp_path: Path) -> None:
    args = argparse.Namespace(
        task="AIC-Task-v0",
        num_envs=4,
        max_iterations=1,
        seed=7,
        device=None,
        headless=True,
        randomization_profile="heavy",
        output_dir=tmp_path / "isaac",
        isaaclab="isaaclab",
        run_name="isaac_rsl_rl_test",
        resume=False,
        checkpoint=None,
        load_run=None,
        init_policy_checkpoint=None,
        insertion_distance_weight=0.05,
        insertion_lateral_weight=-0.01,
        dry_run=True,
        extra_arg=[],
    )
    cmd, env = isaac_rsl_rl.build_command(args)
    assert cmd[:2] == ["isaaclab", "-p"]
    assert "--task" in cmd
    assert "AIC-Task-v0" in cmd
    assert "--headless" in cmd
    assert "--enable_cameras" in cmd
    assert env["AIC_ISAAC_RANDOMIZATION_PROFILE"] == "heavy"
    assert env["AIC_ISAAC_INSERTION_DISTANCE_WEIGHT"] == "0.05"
    assert env["AIC_ISAAC_INSERTION_LATERAL_WEIGHT"] == "-0.01"
    assert env["AIC_ISAAC_DISABLE_CAMERAS"] == "0"
    assert env["AIC_ISAAC_OUTPUT_DIR"] == str(tmp_path / "isaac")


def test_isaac_rsl_rl_passes_offline_serl_init_checkpoint(tmp_path: Path) -> None:
    args = argparse.Namespace(
        task="AIC-Task-v0",
        num_envs=4,
        max_iterations=1,
        seed=7,
        device=None,
        headless=True,
        randomization_profile="light",
        output_dir=tmp_path / "isaac",
        isaaclab="isaaclab",
        run_name="isaac_rsl_rl_test",
        resume=False,
        checkpoint=None,
        load_run=None,
        init_policy_checkpoint=tmp_path / "offline_serl.pt",
        insertion_distance_weight=0.0,
        insertion_lateral_weight=0.0,
        dry_run=True,
        extra_arg=[],
    )
    cmd, _ = isaac_rsl_rl.build_command(args)
    assert "--init_policy_checkpoint" in cmd
    assert str(tmp_path / "offline_serl.pt") in cmd
