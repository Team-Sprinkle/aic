"""Tests for runtime executable and benchmark preflight discovery."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from aic_gazebo_env.discovery import ExecutableResolution, resolve_transport_helper_executable
from aic_gazebo_env.transport_bridge import GazeboTransportBridge, GazeboTransportBridgeConfig


def _fake_repo_root(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / "aic_utils").mkdir()
    return repo_root


def _load_benchmark_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "live_transport_benchmark.py"
    )
    spec = spec_from_file_location("test_live_transport_benchmark", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark script from {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_benchmark_preflight_classifies_missing_gz_and_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_benchmark_module()
    monkeypatch.setattr(module, "find_repo_root", lambda _: Path("/repo"))
    monkeypatch.setattr(
        module,
        "resolve_gz_executable",
        lambda *_args, **_kwargs: ExecutableResolution(
            requested_name="gz",
            resolved_path=None,
            searched_locations=("/usr/bin/gz",),
            discovered_setup_script="/tmp/ws_overlay/install/setup.bash",
            setup_explanation="found nearby overlay setup script",
            status="gz_not_found",
            setup_status="workspace_setup_script_found_but_not_sourced",
        ),
    )
    monkeypatch.setattr(
        module,
        "resolve_transport_helper_executable",
        lambda *_args, **_kwargs: ExecutableResolution(
            requested_name="aic_gz_transport_bridge",
            resolved_path=None,
            searched_locations=("/tmp/ws_overlay/install/aic_gazebo_transport_bridge/lib/aic_gazebo_transport_bridge/aic_gz_transport_bridge",),
            discovered_setup_script="/tmp/ws_overlay/install/setup.bash",
            setup_explanation="found nearby overlay setup script",
            status="helper_not_found",
            setup_status="workspace_setup_script_found_but_not_sourced",
        ),
    )

    preflight = module.build_preflight()

    assert preflight["gz_status"] == "gz_not_found"
    assert preflight["helper_status"] == "helper_not_found"
    assert (
        preflight["helper_setup_status"]
        == "workspace_setup_script_found_but_not_sourced"
    )
    assert preflight["discovered_setup_script"] == "/tmp/ws_overlay/install/setup.bash"


def test_benchmark_recommendation_includes_source_command(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_benchmark_module()
    monkeypatch.setattr(module, "find_repo_root", lambda _: Path("/repo"))

    recommendation = module.build_source_recommendation("/tmp/ws_overlay/install/setup.bash")

    assert "source /tmp/ws_overlay/install/setup.bash" in recommendation
    assert "python3 " in recommendation
    assert "live_transport_benchmark.py" in recommendation
