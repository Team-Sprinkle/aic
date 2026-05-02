from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_hydra_offline_config_resolves_without_training(tmp_path: Path) -> None:
    script = REPO_ROOT / "aic_utils" / "lerobot_robot_aic" / "scripts" / "hydra_train.py"
    env = dict(os.environ)
    env.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "train.dry_run=true",
            "hardware.dry_run_selection=true",
            "hardware.set_cuda_visible_devices=false",
            f"run.output_dir={tmp_path / 'hydra_run'}",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "hydra_run" / "resolved_config.yaml").is_file()
    assert (tmp_path / "hydra_run" / "task_encoding_schema.json").is_file()
    assert "train_vision_offline_serl.py" in result.stdout
    assert "frozen_act_base_plus_mlp_delta_adapter" in (
        tmp_path / "hydra_run" / "resolved_config.yaml"
    ).read_text()


def test_hydra_offline_config_can_wrap_torchrun(tmp_path: Path) -> None:
    script = REPO_ROOT / "aic_utils" / "lerobot_robot_aic" / "scripts" / "hydra_train.py"
    env = dict(os.environ)
    env.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "train.dry_run=true",
            "hardware.dry_run_selection=true",
            "hardware.set_cuda_visible_devices=false",
            "hardware.distributed.enabled=true",
            "hardware.distributed.nproc_per_node=2",
            f"run.output_dir={tmp_path / 'hydra_ddp_run'}",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "torch.distributed.run" in result.stdout
    assert "--nproc-per-node" in result.stdout


def test_hydra_act_config_can_wrap_lerobot_torchrun(tmp_path: Path) -> None:
    script = REPO_ROOT / "aic_utils" / "lerobot_robot_aic" / "scripts" / "hydra_train.py"
    env = dict(os.environ)
    env.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "train=act",
            "model=act",
            "train.task_conditioning=off",
            "train.dry_run=true",
            "hardware.dry_run_selection=true",
            "hardware.set_cuda_visible_devices=false",
            "hardware.distributed.enabled=true",
            "hardware.distributed.nproc_per_node=2",
            f"run.output_dir={tmp_path / 'hydra_act_ddp_run'}",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "torch.distributed.run" in result.stdout
    assert "lerobot.scripts.lerobot_train" in result.stdout
    assert "train_act_policy.py" not in result.stdout
