"""Tests for the persistent Gazebo topic transport path."""

from __future__ import annotations

from pathlib import Path
import tempfile
import textwrap

from aic_gazebo_env.gazebo_client import (
    GazeboCliClient,
    GazeboCliClientConfig,
    PersistentGazeboTopicReader,
)


def _write_streaming_topic_script(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import signal
            import sys
            import time

            def _exit_cleanly(signum, frame):
                del signum, frame
                sys.exit(0)

            signal.signal(signal.SIGTERM, _exit_cleanly)
            signal.signal(signal.SIGINT, _exit_cleanly)

            if sys.argv[1:3] != ["topic", "-e"]:
                print(f"unexpected argv: {sys.argv}", file=sys.stderr, flush=True)
                sys.exit(3)

            print(
                "world: \\"test_world\\"\\nstep_count: 0\\nentity {\\n  id: 1\\n  name: \\"robot\\"\\n}",
                flush=True,
            )
            time.sleep(0.15)
            print(
                "world: \\"test_world\\"\\nstep_count: 1\\nentity {\\n  id: 1\\n  name: \\"robot\\"\\n}",
                flush=True,
            )
            while True:
                time.sleep(0.1)
            """
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_persistent_topic_reader_returns_newer_samples() -> None:
    with tempfile.TemporaryDirectory(prefix="aic_gz_persistent_reader_") as tmp_dir:
        script = Path(tmp_dir) / "fake_gz_stream.py"
        _write_streaming_topic_script(script)

        reader = PersistentGazeboTopicReader(
            executable=str(script),
            topic="/world/test_world/state",
            quiet_period_s=0.02,
        )
        try:
            first_sample, first_generation = reader.get_sample(timeout=1.0)
            second_sample, second_generation = reader.get_sample(
                after_generation=first_generation,
                timeout=1.0,
            )
        finally:
            reader.stop()

    assert "step_count: 0" in first_sample
    assert "step_count: 1" in second_sample
    assert second_generation > first_generation


def test_cli_client_auto_transport_only_enables_persistent_mode_for_real_gz() -> None:
    fake_world = "/tmp/fake_world.sdf"

    auto_fake = GazeboCliClient(
        GazeboCliClientConfig(
            executable="/tmp/fake_gz.py",
            world_path=fake_world,
            timeout=0.2,
            observation_transport="auto",
        )
    )
    auto_real = GazeboCliClient(
        GazeboCliClientConfig(
            executable="gz",
            world_path=fake_world,
            timeout=0.2,
            observation_transport="auto",
        )
    )

    assert auto_fake._uses_persistent_observation_transport() is False
    assert auto_real._uses_persistent_observation_transport() is True


def test_persistent_state_read_falls_back_to_world_step_when_first_sample_times_out() -> None:
    client = GazeboCliClient(
        GazeboCliClientConfig(
            executable="gz",
            world_path="/tmp/fake_world.sdf",
            timeout=5.0,
            observation_transport="persistent",
        )
    )

    class Reader:
        def __init__(self) -> None:
            self.calls = 0

        def get_sample(
            self,
            *,
            after_generation: int | None = None,
            timeout: float,
        ) -> tuple[str, int]:
            del after_generation, timeout
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("no first sample yet")
            return "step_count: 1", 1

    reader = Reader()
    client._state_reader = reader  # type: ignore[assignment]
    client._read_state_sample_after_world_step = lambda topic: "step_count: 1"  # type: ignore[method-assign]

    payload, generation = client._read_state_sample(
        topic="/world/test_world/state",
        after_generation=None,
    )

    assert payload == "step_count: 1"
    assert generation is None
    assert reader.calls == 1


def test_persistent_pose_read_returns_none_when_no_sample_arrives_quickly() -> None:
    client = GazeboCliClient(
        GazeboCliClientConfig(
            executable="gz",
            world_path="/tmp/fake_world.sdf",
            timeout=5.0,
            observation_transport="persistent",
        )
    )

    class Reader:
        def get_sample(
            self,
            *,
            after_generation: int | None = None,
            timeout: float,
        ) -> tuple[str, int]:
            del after_generation, timeout
            raise TimeoutError("pose stream is quiet")

    client._pose_reader = Reader()  # type: ignore[assignment]

    assert client._read_pose_sample(topic="/world/test_world/pose/info") is None
