"""Tests for the persistent C++ transport client integration surface."""

from __future__ import annotations

from pathlib import Path
import textwrap

from aic_gazebo_env.gazebo_client import GazeboTransportClient
from aic_gazebo_env.runtime import GazeboRuntime, GazeboRuntimeConfig


class _RunningProcess:
    def __init__(self) -> None:
        self._returncode: int | None = None

    def poll(self) -> None:
        return self._returncode

    def terminate(self) -> None:
        self._returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self._returncode = 0
        return 0


def _write_fake_transport_bridge(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import sys

            state = {
                "generation": 1,
                "step_count": 0,
                "robot_position": [1.0, 2.0, 3.0],
                "joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }

            def state_text():
                joints = "\\n".join(
                    [
                        f'joint {{\\n  name: "{name}"\\n  position: {position}\\n}}'
                        for name, position in zip(
                            [
                                "shoulder_pan_joint",
                                "shoulder_lift_joint",
                                "elbow_joint",
                                "wrist_1_joint",
                                "wrist_2_joint",
                                "wrist_3_joint",
                            ],
                            state["joint_positions"],
                        )
                    ]
                )
                robot = state["robot_position"]
                return (
                    f'world: "test_world"\\n'
                    f"step_count: {state['step_count']}\\n"
                    f'entity {{\\n  id: 101\\n  name: "robot"\\n  pose {{\\n'
                    f"    position {{\\n      x: {robot[0]}\\n      y: {robot[1]}\\n      z: {robot[2]}\\n    }}\\n"
                    f"    orientation {{\\n      x: 0.0\\n      y: 0.0\\n      z: 0.0\\n      w: 1.0\\n    }}\\n"
                    f"  }}\\n}}\\n"
                    f'entity {{\\n  id: 202\\n  name: "task_board"\\n  pose {{\\n'
                    f"    position {{\\n      x: 4.0\\n      y: 5.0\\n      z: 6.0\\n    }}\\n"
                    f"    orientation {{\\n      x: 0.0\\n      y: 0.0\\n      z: 0.707\\n      w: 0.707\\n    }}\\n"
                    f"  }}\\n}}\\n"
                    + joints
                )

            def pose_text():
                robot = state["robot_position"]
                return (
                    f'pose {{\\n  name: "robot"\\n  id: 101\\n'
                    f"  position {{\\n    x: {robot[0]}\\n    y: {robot[1]}\\n    z: {robot[2]}\\n  }}\\n"
                    f"  orientation {{\\n    x: 0.0\\n    y: 0.0\\n    z: 0.0\\n    w: 1.0\\n  }}\\n}}\\n"
                    f'pose {{\\n  name: "task_board"\\n  id: 202\\n'
                    f"  position {{\\n    x: 4.0\\n    y: 5.0\\n    z: 6.0\\n  }}\\n"
                    f"  orientation {{\\n    x: 0.0\\n    y: 0.0\\n    z: 0.707\\n    w: 0.707\\n  }}\\n}}"
                )

            for line in sys.stdin:
                request = json.loads(line)
                response = {"id": request["id"], "ok": True}
                op = request["op"]
                if op == "ping":
                    pass
                elif op == "shutdown":
                    print(json.dumps(response), flush=True)
                    sys.exit(0)
                elif op == "get_observation":
                    after_generation = int(request.get("after_generation") or 0)
                    if after_generation >= state["generation"]:
                        state["generation"] += 1
                    response["state_generation"] = state["generation"]
                    response["pose_generation"] = state["generation"]
                    response["state_text"] = state_text()
                    response["pose_text"] = pose_text()
                elif op == "world_control":
                    if request.get("reset_all"):
                        state["generation"] += 1
                        state["step_count"] = 0
                        state["robot_position"] = [1.0, 2.0, 3.0]
                        state["joint_positions"] = [0.0] * 6
                    if "multi_step" in request:
                        state["generation"] += 1
                        state["step_count"] += int(request["multi_step"])
                    response["reply_text"] = "data: true\\n"
                elif op == "set_pose":
                    state["generation"] += 1
                    state["robot_position"] = [float(v) for v in request["position"]]
                    response["reply_text"] = "data: true\\n"
                elif op == "joint_target":
                    state["generation"] += 1
                    fields = {}
                    for field in request["request_text"].split(";"):
                        if "=" not in field:
                            continue
                        key, value = field.split("=", 1)
                        fields[key] = value
                    state["joint_positions"] = [
                        float(value) for value in fields["positions"].split(",")
                    ]
                    response["reply_text"] = "data: true\\n"
                else:
                    response = {"id": request["id"], "ok": False, "error": f"unsupported op {op}"}
                print(json.dumps(response), flush=True)
            """
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_world(path: Path) -> None:
    path.write_text(
        "<sdf version='1.9'><world name='test_world'></world></sdf>\n",
        encoding="utf-8",
    )


def test_runtime_auto_prefers_transport_helper_when_available(tmp_path: Path) -> None:
    helper = tmp_path / "fake_transport_bridge.py"
    world = tmp_path / "world.sdf"
    _write_fake_transport_bridge(helper)
    _write_world(world)

    runtime = GazeboRuntime(
        GazeboRuntimeConfig(
            world_path=str(world),
            world_name="test_world",
            executable="gz",
            timeout=0.2,
            transport_backend="auto",
            transport_helper_executable=str(helper),
        )
    )

    client = runtime._client()
    assert isinstance(client, GazeboTransportClient)

    try:
        runtime.process = _RunningProcess()
        observation, info = runtime.get_observation()
        assert observation["world_name"] == "test_world"
        assert observation["step_count"] == 0
        assert info["transport_backend"] == "transport"
    finally:
        runtime.stop()


def test_transport_client_step_reads_fresh_post_action_state(tmp_path: Path) -> None:
    helper = tmp_path / "fake_transport_bridge.py"
    world = tmp_path / "world.sdf"
    _write_fake_transport_bridge(helper)
    _write_world(world)

    runtime = GazeboRuntime(
        GazeboRuntimeConfig(
            world_path=str(world),
            world_name="test_world",
            executable="gz",
            timeout=0.2,
            transport_backend="transport",
            transport_helper_executable=str(helper),
        )
    )

    try:
        runtime.process = _RunningProcess()
        observation, reward, terminated, truncated, info = runtime.step(
            {
                "set_entity_position": {
                    "name": "robot",
                    "position": [7.0, 8.0, 9.0],
                },
                "multi_step": 3,
            }
        )

        assert observation["step_count"] == 3
        assert observation["entities_by_name"]["robot"]["position"] == [7.0, 8.0, 9.0]
        assert info["transport_backend"] == "transport"
        assert reward == -5.196152422706632
        assert terminated is False
        assert truncated is False
    finally:
        runtime.stop()
