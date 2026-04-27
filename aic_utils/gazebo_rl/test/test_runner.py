from pathlib import Path
import subprocess

import pytest

from gazebo_rl import runner as runner_module
from gazebo_rl.runner import GazeboRLRunner, GazeboRLRunnerConfig


def _runner(sim_distrobox=None) -> GazeboRLRunner:
    return GazeboRLRunner(
        GazeboRLRunnerConfig(
            workspace_dir=Path("/tmp/aic_ws"),
            sim_distrobox=sim_distrobox,
            ground_truth=False,
            gazebo_gui=False,
            launch_rviz=False,
        )
    )


def test_simulation_command_without_distrobox_uses_local_pixi_launch():
    cmd = _runner(sim_distrobox=None).simulation_command()
    assert cmd[:5] == ["pixi", "run", "ros2", "launch", "aic_bringup"]
    assert "aic_gz_bringup.launch.py" in cmd
    assert "ground_truth:=false" in cmd
    assert not any("distrobox" in token for token in cmd)


def test_simulation_command_with_distrobox_uses_exact_user_name():
    cmd = _runner(sim_distrobox="my_box").simulation_command()
    assert cmd[:5] == ["distrobox", "enter", "-r", "--no-tty", "my_box"]
    assert "--name" not in cmd
    assert "aic_eval" not in cmd
    assert cmd[5:] == [
        "--",
        "/entrypoint.sh",
        "ground_truth:=false",
        "start_aic_engine:=true",
        "gazebo_gui:=false",
        "launch_rviz:=false",
    ]


def test_distrobox_preflight_is_skipped_when_not_configured(monkeypatch):
    def fail_run(*args, **kwargs):
        raise AssertionError("distrobox list should not be called")

    monkeypatch.setattr(runner_module.subprocess, "run", fail_run)
    monkeypatch.setattr(runner_module.shutil, "which", lambda _: None)
    _runner(sim_distrobox=None)._validate_distrobox()


def test_distrobox_preflight_missing_binary_has_clear_error(monkeypatch):
    monkeypatch.setattr(runner_module.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError) as exc_info:
        _runner(sim_distrobox="my_box")._validate_distrobox()
    assert "distrobox is not installed or is not on PATH" in str(exc_info.value)


def test_distrobox_preflight_accepts_requested_container(monkeypatch):
    monkeypatch.setattr(runner_module.shutil, "which", lambda _: "/usr/bin/distrobox")
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="ID | NAME | STATUS | IMAGE\n12 | my_box | running | image\n",
            stderr="",
        ),
    )
    _runner(sim_distrobox="my_box")._validate_distrobox()


def test_distrobox_preflight_missing_container_has_clear_error(monkeypatch):
    monkeypatch.setattr(runner_module.shutil, "which", lambda _: "/usr/bin/distrobox")
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="ID | NAME | STATUS | IMAGE\n12 | other_box | running | image\n",
            stderr="",
        ),
    )
    with pytest.raises(RuntimeError) as exc_info:
        _runner(sim_distrobox="my_box")._validate_distrobox()
    message = str(exc_info.value)
    assert "Distrobox container 'my_box' was not found" in message
    assert "user-created container name" in message
    assert "--sim-distrobox <your-container-name>" in message
    assert "aic_eval" not in message
