"""Tests for the PPO training script dependency and smoke gating behavior."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def test_training_worker_fails_fast_when_training_dependencies_are_missing() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "aic_utils" / "aic_gazebo_env" / "scripts" / "train_sb3_ppo.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "aic_utils" / "aic_gazebo_env")

    result = subprocess.run(
        [sys.executable, str(script), "--worker-train", "--json-only", "--smoke"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error_category"] == "missing_training_dependencies"
    assert "stable_baselines3" in payload["dependency_report"]["missing"]
