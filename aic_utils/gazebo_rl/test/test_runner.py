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


def test_recorder_command_uses_existing_policy_recorder():
    runner = GazeboRLRunner(
        GazeboRLRunnerConfig(
            workspace_dir=Path("/tmp/aic_ws"),
            results_dir=Path("/tmp/aic_ws/outputs/gazebo_rl/results/iter_000"),
            record_lerobot=True,
            record_root=Path("/tmp/aic_ws/outputs/gazebo_rl/rollout_dataset"),
            record_repo_id="local/test_rollout",
            record_video=True,
            record_fps=15,
        )
    )
    cmd = runner.recorder_command()
    assert cmd[:3] == ["pixi", "run", "aic-policy-recorder"]
    assert "--dataset.repo_id=local/test_rollout" in cmd
    assert "--dataset.root=/tmp/aic_ws/outputs/gazebo_rl/rollout_dataset" in cmd
    assert "--dataset.fps=15" in cmd
    assert "--dataset.video" in cmd
    assert "--save_failed_episodes" in cmd
    assert "--action_mode=cartesian" in cmd


def test_start_does_not_precreate_lerobot_dataset_root(monkeypatch, tmp_path):
    class FakePopen:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def poll(self):
            return None

    record_root = tmp_path / "new_lerobot_dataset"
    runner = GazeboRLRunner(
        GazeboRLRunnerConfig(
            workspace_dir=tmp_path,
            results_dir=tmp_path / "results",
            record_lerobot=True,
            record_root=record_root,
        )
    )
    monkeypatch.setattr(runner, "_validate_distrobox", lambda: None)
    monkeypatch.setattr(runner, "_cleanup_stale_zenoh_router", lambda: None)
    monkeypatch.setattr(runner_module.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(runner_module.time, "sleep", lambda *_: None)

    try:
        runner.start()
        assert (tmp_path / "results").is_dir()
        assert not record_root.exists()
    finally:
        runner.processes.clear()


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


def test_start_runs_stale_zenoh_cleanup_before_launch(monkeypatch, tmp_path):
    events = []

    class FakePopen:
        def __init__(self, *args, **kwargs):
            events.append("popen")
            self.args = args
            self.kwargs = kwargs

        def poll(self):
            return None

    runner = GazeboRLRunner(
        GazeboRLRunnerConfig(
            workspace_dir=tmp_path,
            results_dir=tmp_path / "results",
        )
    )
    monkeypatch.setattr(runner, "_validate_distrobox", lambda: events.append("validate"))
    monkeypatch.setattr(runner, "_cleanup_stale_zenoh_router", lambda: events.append("cleanup"))
    monkeypatch.setattr(runner_module.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(runner_module.time, "sleep", lambda *_: None)

    try:
        runner.start()
        assert events[:3] == ["validate", "cleanup", "popen"]
    finally:
        runner.processes.clear()


def test_cleanup_stale_zenoh_kills_host_and_distrobox(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    runner = _runner(sim_distrobox="my_box")
    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    monkeypatch.setattr(runner_module.time, "sleep", lambda *_: None)

    runner._cleanup_stale_zenoh_router()

    assert ["pkill", "-f", "rmw_zenohd"] in calls
    assert ["pkill", "-f", "rmw_zenoh_cpp rmw_zenohd"] in calls
    assert any(call[:6] == ["distrobox", "enter", "-r", "--no-tty", "my_box", "--"] for call in calls)
