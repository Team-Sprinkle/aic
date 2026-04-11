"""Optional live training smoke tests for the SB3 PPO entry point."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


RUN_LIVE = os.environ.get("AIC_GAZEBO_ENV_RUN_LIVE_TRAINING") == "1"
HAS_TRAINING_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "gymnasium", "stable_baselines3")
)


@pytest.mark.skipif(not RUN_LIVE, reason="live training smoke tests are disabled")
@pytest.mark.skipif(not HAS_TRAINING_DEPS, reason="training dependencies are unavailable")
def test_live_training_smoke_run() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "aic_utils" / "aic_gazebo_env" / "scripts" / "train_sb3_ppo.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "aic_utils" / "aic_gazebo_env")

    command = [sys.executable, str(script), "--smoke", "--json-only"]
    if os.environ.get("AIC_GAZEBO_ENV_AUTO_BUILD") == "1":
        command.append("--auto-build")
    if os.environ.get("AIC_GAZEBO_ENV_AUTO_LAUNCH") == "1":
        command.append("--auto-launch")

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=3600.0,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload["training"]["ok"] is True
