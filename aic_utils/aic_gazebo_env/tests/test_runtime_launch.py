"""Launch-smoke tests for the Gazebo subprocess runtime."""

from __future__ import annotations

from pathlib import Path
import sys
import textwrap

import pytest

from aic_gazebo_env import GazeboRuntime, GazeboRuntimeConfig


def _write_fake_gz_script(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import os
            import signal
            import sys
            import time

            mode = os.environ.get("AIC_FAKE_GZ_MODE", "run")

            def _exit_cleanly(signum, frame):
                del signum, frame
                sys.exit(0)

            signal.signal(signal.SIGTERM, _exit_cleanly)

            if len(sys.argv) < 3 or sys.argv[1] != "sim":
                print("unexpected argv", sys.argv, file=sys.stderr)
                sys.exit(3)

            if mode == "crash":
                print("startup crash", file=sys.stderr, flush=True)
                sys.exit(17)

            if mode == "ignore-term":
                signal.signal(signal.SIGTERM, signal.SIG_IGN)

            print("fake gz running", flush=True)
            while True:
                time.sleep(0.1)
            """
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_world(path: Path) -> None:
    path.write_text("<sdf version='1.9'></sdf>\n", encoding="utf-8")


def test_runtime_can_start_gazebo_successfully(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_gz = tmp_path / "fake_gz.py"
    world = tmp_path / "world.sdf"
    _write_fake_gz_script(fake_gz)
    _write_world(world)

    monkeypatch.setenv("AIC_FAKE_GZ_MODE", "run")
    runtime = GazeboRuntime(
        GazeboRuntimeConfig(
            world_path=str(world),
            headless=True,
            timeout=0.2,
            executable=str(fake_gz),
        )
    )

    runtime.start()

    assert runtime.process is not None
    assert runtime.process.poll() is None

    runtime.stop()


def test_runtime_can_stop_gazebo_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_gz = tmp_path / "fake_gz.py"
    world = tmp_path / "world.sdf"
    _write_fake_gz_script(fake_gz)
    _write_world(world)

    monkeypatch.setenv("AIC_FAKE_GZ_MODE", "run")
    runtime = GazeboRuntime(
        GazeboRuntimeConfig(
            world_path=str(world),
            timeout=0.2,
            executable=str(fake_gz),
        )
    )

    runtime.start()
    process = runtime.process

    runtime.stop()

    assert process is not None
    assert process.poll() == 0
    assert runtime.process is None


def test_invalid_world_path_fails_with_clear_error() -> None:
    runtime = GazeboRuntime(
        GazeboRuntimeConfig(
            world_path="/definitely/missing/world.sdf",
            timeout=0.2,
            executable=sys.executable,
        )
    )

    with pytest.raises(FileNotFoundError, match="Gazebo world file does not exist"):
        runtime.start()


def test_process_crash_is_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_gz = tmp_path / "fake_gz.py"
    world = tmp_path / "world.sdf"
    _write_fake_gz_script(fake_gz)
    _write_world(world)

    monkeypatch.setenv("AIC_FAKE_GZ_MODE", "crash")
    runtime = GazeboRuntime(
        GazeboRuntimeConfig(
            world_path=str(world),
            timeout=0.2,
            executable=str(fake_gz),
        )
    )

    with pytest.raises(RuntimeError, match="crashed during startup"):
        runtime.start()
