"""Tests for runtime executable and live preflight discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from aic_gazebo_env.discovery import resolve_transport_helper_executable
from aic_gazebo_env.live_runtime import LiveRuntimeManager
from aic_gazebo_env.transport_bridge import GazeboTransportBridge, GazeboTransportBridgeConfig


def _fake_repo_root(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / "aic_utils").mkdir()
    return repo_root


def test_helper_discovery_from_repo_relative_install_path(tmp_path: Path) -> None:
    repo_root = _fake_repo_root(tmp_path)
    helper_path = (
        repo_root
        / "install"
        / "aic_gazebo_transport_bridge"
        / "lib"
        / "aic_gazebo_transport_bridge"
        / "aic_gz_transport_bridge"
    )
    helper_path.parent.mkdir(parents=True)
    helper_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    helper_path.chmod(0o755)

    resolution = resolve_transport_helper_executable(repo_root=repo_root)

    assert resolution.resolved_path == str(helper_path.resolve())
    assert str(helper_path) in resolution.searched_locations


def test_transport_bridge_error_includes_searched_locations(tmp_path: Path) -> None:
    bridge = GazeboTransportBridge(
        GazeboTransportBridgeConfig(
            world_name="test_world",
            state_topic="/world/test_world/state",
            pose_topic="/world/test_world/pose/info",
            helper_executable=str(tmp_path / "missing_helper"),
            startup_timeout_s=0.1,
        )
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        bridge.start()

    message = str(exc_info.value)
    assert "Searched:" in message
    assert str(tmp_path / "missing_helper") in message


def test_live_preflight_classifies_missing_gz_and_helper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = _fake_repo_root(tmp_path)
    manager = LiveRuntimeManager(repo_root=repo_root)
    monkeypatch.setattr(
        "aic_gazebo_env.live_runtime.live_prerequisites",
        lambda _repo_root: {
            "repo_root": str(repo_root),
            "setup_script": "/tmp/ws_overlay/install/setup.bash",
            "setup_explanation": "found nearby overlay setup script",
            "searched_setup_locations": ["/tmp/ws_overlay/install/setup.bash"],
            "gz_resolved_path": None,
            "gz_status": "gz_not_found",
            "gz_setup_status": "workspace_setup_script_found_but_not_sourced",
            "searched_gz_locations": ["/usr/bin/gz"],
            "helper_resolved_path": None,
            "helper_status": "helper_not_found",
            "helper_setup_status": "workspace_setup_script_found_but_not_sourced",
            "searched_helper_locations": ["/tmp/ws_overlay/install/aic_gazebo_transport_bridge/lib/aic_gazebo_transport_bridge/aic_gz_transport_bridge"],
            "distrobox_path": "/usr/bin/distrobox",
            "docker_path": "/usr/bin/docker",
            "colcon_path": "/usr/bin/colcon",
            "ros_setup_exists": True,
        },
    )

    preflight = manager.preflight()

    assert preflight["gz_status"] == "gz_not_found"
    assert preflight["helper_status"] == "helper_not_found"
    assert (
        preflight["helper_setup_status"]
        == "workspace_setup_script_found_but_not_sourced"
    )
    assert preflight["setup_script"] == "/tmp/ws_overlay/install/setup.bash"
    assert "source /tmp/ws_overlay/install/setup.bash" in preflight["recommendation"]


def test_live_recommendation_includes_source_command(tmp_path: Path) -> None:
    repo_root = _fake_repo_root(tmp_path)
    manager = LiveRuntimeManager(repo_root=repo_root)
    recommendation = manager.source_recommendation("/tmp/ws_overlay/install/setup.bash")

    assert "source /tmp/ws_overlay/install/setup.bash" in recommendation
    assert "python3 aic_utils/aic_gazebo_env/scripts/run_live_e2e.py" in recommendation
