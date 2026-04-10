"""Gated live e2e tests for the training-only Gazebo runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


LIVE_E2E_ENABLED = os.environ.get("AIC_GAZEBO_ENV_RUN_LIVE_E2E") == "1"


def _script_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_live_e2e.py"
    )


def _run_live_e2e(*extra_args: str) -> dict[str, object]:
    command = [
        sys.executable,
        str(_script_path()),
        "--json-only",
        *extra_args,
    ]
    if os.environ.get("AIC_GAZEBO_ENV_AUTO_BUILD") == "1":
        command.append("--auto-build")
    if os.environ.get("AIC_GAZEBO_ENV_AUTO_LAUNCH") == "1":
        command.append("--auto-launch")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0:
        pytest.fail(
            "Live e2e runner failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


@pytest.mark.skipif(
    not LIVE_E2E_ENABLED,
    reason="Set AIC_GAZEBO_ENV_RUN_LIVE_E2E=1 to enable live e2e tests.",
)
def test_live_e2e_smoke() -> None:
    payload = _run_live_e2e()
    assert payload["result"]["health"]["no_op_step_ok"] is True
    assert payload["result"]["smoke"]["ok"] is True


@pytest.mark.skipif(
    not LIVE_E2E_ENABLED,
    reason="Set AIC_GAZEBO_ENV_RUN_LIVE_E2E=1 to enable live e2e tests.",
)
def test_live_e2e_parity() -> None:
    payload = _run_live_e2e()
    assert payload["result"]["health"]["first_observation_ok"] is True
    assert payload["result"]["parity"]["ok"] is True
