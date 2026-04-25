"""Tests for the persistent C++ transport client integration surface."""

from __future__ import annotations

from pathlib import Path
import textwrap
import pytest

from aic_gazebo_env.gazebo_client import GazeboCliClient, GazeboCliClientConfig, GazeboTransportClient
from aic_gazebo_env.runtime import GazeboRuntime, GazeboRuntimeConfig
from aic_gazebo_env.transport_bridge import (
    GazeboTransportBridge,
    GazeboTransportBridgeConfig,
    GazeboTransportBridgeError,
)
from aic_gazebo_env.protocol import GetObservationRequest, ResetRequest


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
                "generation": 0,
                "pose_generation": 0,
                "state_callback_count": 0,
                "pose_callback_count": 0,
                "state_parse_failures": 0,
                "pose_parse_failures": 0,
                "step_count": 0,
                "robot_position": [1.0, 2.0, 3.0],
                "joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
            never_ready = "never_ready" in sys.argv[0]

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
                elif op == "status":
                    response.update(
                        {
                            "state_generation": state["generation"],
                            "pose_generation": state["pose_generation"],
                            "state_callback_count": state["state_callback_count"],
                            "state_parse_failures": state["state_parse_failures"],
                            "pose_callback_count": state["pose_callback_count"],
                            "pose_parse_failures": state["pose_parse_failures"],
                        }
                    )
                elif op == "wait_until_ready":
                    if never_ready:
                        response = {
                            "id": request["id"],
                            "ok": False,
                            "error": "timed out waiting for initial transport samples",
                            "state_generation": state["generation"],
                            "pose_generation": state["pose_generation"],
                            "state_callback_count": state["state_callback_count"],
                            "state_parse_failures": state["state_parse_failures"],
                            "pose_callback_count": state["pose_callback_count"],
                            "pose_parse_failures": state["pose_parse_failures"],
                        }
                    else:
                        state["generation"] = max(state["generation"], 1)
                        state["pose_generation"] = max(state["pose_generation"], 1)
                        state["state_callback_count"] = max(state["state_callback_count"], 1)
                        state["pose_callback_count"] = max(state["pose_callback_count"], 1)
                        response.update(
                            {
                                "state_generation": state["generation"],
                                "pose_generation": state["pose_generation"],
                                "state_callback_count": state["state_callback_count"],
                                "state_parse_failures": state["state_parse_failures"],
                                "pose_callback_count": state["pose_callback_count"],
                                "pose_parse_failures": state["pose_parse_failures"],
                            }
                        )
                elif op == "shutdown":
                    print(json.dumps(response), flush=True)
                    sys.exit(0)
                elif op == "get_observation":
                    after_generation = int(request.get("after_generation") or 0)
                    if after_generation >= state["generation"]:
                        state["generation"] += 1
                    state["pose_generation"] = max(state["pose_generation"], state["generation"])
                    state["state_callback_count"] = max(state["state_callback_count"], state["generation"])
                    state["pose_callback_count"] = max(state["pose_callback_count"], state["pose_generation"])
                    response["state_generation"] = state["generation"]
                    response["pose_generation"] = state["pose_generation"]
                    response["state_text"] = state_text()
                    response["pose_text"] = pose_text()
                elif op == "world_control":
                    if request.get("reset_all"):
                        state["generation"] += 1
                        state["pose_generation"] += 1
                        state["state_callback_count"] += 1
                        state["pose_callback_count"] += 1
                        state["step_count"] = 0
                        state["robot_position"] = [1.0, 2.0, 3.0]
                        state["joint_positions"] = [0.0] * 6
                    if "multi_step" in request:
                        state["generation"] += 1
                        state["pose_generation"] += 1
                        state["state_callback_count"] += 1
                        state["pose_callback_count"] += 1
                        state["step_count"] += int(request["multi_step"])
                    response["reply_text"] = "data: true\\n"
                elif op == "set_pose":
                    state["generation"] += 1
                    state["pose_generation"] += 1
                    state["state_callback_count"] += 1
                    state["pose_callback_count"] += 1
                    state["robot_position"] = [float(v) for v in request["position"]]
                    response["reply_text"] = "data: true\\n"
                elif op == "joint_target":
                    state["generation"] += 1
                    state["pose_generation"] += 1
                    state["state_callback_count"] += 1
                    state["pose_callback_count"] += 1
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


class _StubBridge:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, payload, *, timeout_s=None):
        del timeout_s
        self.calls.append(dict(payload))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        pass

    def health_flags(self):
        return {"helper_startup_ok": True, "helper_ready_ok": True}


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
        assert info["sim_step_count_raw"] == 3
        assert reward == -5.196152422706632
        assert terminated is False
        assert truncated is False
    finally:
        runtime.stop()


def test_transport_bridge_start_requires_sample_readiness(tmp_path: Path) -> None:
    helper = tmp_path / "fake_transport_bridge_never_ready.py"
    _write_fake_transport_bridge(helper)
    bridge = GazeboTransportBridge(
        GazeboTransportBridgeConfig(
            world_name="test_world",
            state_topic="/world/test_world/state",
            pose_topic="/world/test_world/pose/info",
            helper_executable=str(helper),
            startup_timeout_s=0.2,
        )
    )
    with pytest.raises(GazeboTransportBridgeError, match="failed readiness handshake"):
        bridge.start()


def test_transport_bridge_allows_world_control_before_sample_readiness(tmp_path: Path) -> None:
    helper = tmp_path / "fake_transport_bridge_never_ready.py"
    _write_fake_transport_bridge(helper)
    bridge = GazeboTransportBridge(
        GazeboTransportBridgeConfig(
            world_name="test_world",
            state_topic="/world/test_world/state",
            pose_topic="/world/test_world/pose/info",
            helper_executable=str(helper),
            startup_timeout_s=0.2,
        )
    )
    try:
        response = bridge.request(
            {
                "op": "world_control",
                "service": "/world/test_world/control",
                "multi_step": 1,
                "timeout_ms": 200,
            }
        )
        assert response["reply_text"] == "data: true\n"
        flags = bridge.health_flags()
        assert flags["helper_startup_ok"] is True
        assert flags["helper_ready_ok"] is False
    finally:
        bridge.close()


def test_transport_bridge_start_succeeds_when_helper_reports_ready(tmp_path: Path) -> None:
    helper = tmp_path / "fake_transport_bridge_ready.py"
    _write_fake_transport_bridge(helper)
    bridge = GazeboTransportBridge(
        GazeboTransportBridgeConfig(
            world_name="test_world",
            state_topic="/world/test_world/state",
            pose_topic="/world/test_world/pose/info",
            helper_executable=str(helper),
            startup_timeout_s=0.5,
        )
    )
    try:
        bridge.start()
        flags = bridge.health_flags()
        assert flags["helper_startup_ok"] is True
        assert flags["helper_ready_ok"] is True
        assert flags["helper_last_status"]["state_generation"] > 0
        assert flags["helper_last_status"]["state_callback_count"] > 0
    finally:
        bridge.close()


def test_transport_client_reset_advances_world_before_fresh_observation(tmp_path: Path) -> None:
    world = tmp_path / "world.sdf"
    _write_world(world)
    client = GazeboTransportClient(
        GazeboCliClientConfig(
            world_path=str(world),
            world_name="test_world",
            executable="gz",
            timeout=0.2,
            transport_backend="transport",
        )
    )
    client._bridge = _StubBridge(
        [
            {"state_generation": 3},
            {"requested": True, "result": True, "reply_text": "data: true\n"},
            {"requested": True, "result": True, "reply_text": "data: true\n"},
            {"pose_generation": 3},
                {
                    "state_generation": 4,
                    "pose_generation": 4,
                    "state_text": 'world: "test_world"\nstep_count: 0\nentity {\n  id: 101\n  name: "robot"\n  pose {\n    position {\n      x: 1.0\n      y: 2.0\n      z: 3.0\n    }\n    orientation {\n      x: 0.0\n      y: 0.0\n      z: 0.0\n      w: 1.0\n    }\n  }\n}\nentity {\n  id: 202\n  name: "task_board"\n  pose {\n    position {\n      x: 4.0\n      y: 5.0\n      z: 6.0\n    }\n    orientation {\n      x: 0.0\n      y: 0.0\n      z: 0.707\n      w: 0.707\n    }\n  }\n}\n',
                    "pose_text": 'pose {\n  name: "robot"\n  id: 101\n  position {\n    x: 1.0\n    y: 2.0\n    z: 3.0\n  }\n  orientation {\n    w: 1.0\n  }\n}\npose {\n  name: "task_board"\n  id: 202\n  position {\n    x: 4.0\n    y: 5.0\n    z: 6.0\n  }\n  orientation {\n    z: 0.707\n    w: 0.707\n  }\n}\n',
                },
                {
                    "state_generation": 4,
                    "pose_generation": 4,
                    "state_text": 'world: "test_world"\nstep_count: 0\nentity {\n  id: 101\n  name: "robot"\n  pose {\n    position {\n      x: 1.0\n      y: 2.0\n      z: 3.0\n    }\n    orientation {\n      x: 0.0\n      y: 0.0\n      z: 0.0\n      w: 1.0\n    }\n  }\n}\nentity {\n  id: 202\n  name: "task_board"\n  pose {\n    position {\n      x: 4.0\n      y: 5.0\n      z: 6.0\n    }\n    orientation {\n      x: 0.0\n      y: 0.0\n      z: 0.707\n      w: 0.707\n    }\n  }\n}\n',
                    "pose_text": 'pose {\n  name: "robot"\n  id: 101\n  position {\n    x: 1.0\n    y: 2.0\n    z: 3.0\n  }\n  orientation {\n    w: 1.0\n  }\n}\npose {\n  name: "task_board"\n  id: 202\n  position {\n    x: 4.0\n    y: 5.0\n    z: 6.0\n  }\n  orientation {\n    z: 0.707\n    w: 0.707\n  }\n}\n',
                },
            ]
        )
    response = client.reset(ResetRequest(seed=1, options={"mode": "test"}))
    assert response.observation["world_name"] == "test_world"
    assert response.info["reset_ok"] is True
    assert response.info["world_control_ok"] is True
    assert client._bridge.calls[1]["reset_all"] is True
    assert client._bridge.calls[2]["multi_step"] == client.config.reset_post_reset_ticks
    assert client._bridge.calls[4]["after_generation"] == 3
    assert client._bridge.calls[5]["pose_after_generation"] == 3


def test_transport_client_observation_fallback_only_on_transport_freshness_failure(tmp_path: Path) -> None:
    world = tmp_path / "world.sdf"
    _write_world(world)
    client = GazeboTransportClient(
        GazeboCliClientConfig(
            world_path=str(world),
            world_name="test_world",
            executable="gz",
            timeout=0.2,
            transport_backend="transport",
        )
    )
    client._bridge = _StubBridge(
        [GazeboTransportBridgeError("timed out waiting for state sample")]
    )
    client._cli_read_topic_sample = lambda topic: (
        'world: "test_world"\nstep_count: 1\nentity {\n  id: 101\n  name: "robot"\n  pose {\n    position {\n      x: 1.0\n      y: 2.0\n      z: 3.0\n    }\n    orientation {\n      x: 0.0\n      y: 0.0\n      z: 0.0\n      w: 1.0\n    }\n  }\n}\nentity {\n  id: 202\n  name: "task_board"\n  pose {\n    position {\n      x: 4.0\n      y: 5.0\n      z: 6.0\n    }\n    orientation {\n      x: 0.0\n      y: 0.0\n      z: 0.707\n      w: 0.707\n    }\n  }\n}\n'
    )
    client._cli_read_pose_sample = lambda topic: None
    response = client.get_observation(GetObservationRequest())
    assert response.info["transport_backend"] == "transport_cli_fallback"
    assert response.info["fallback_used"] is True


def test_transport_client_normalizes_parent_local_link_poses_into_world_space(tmp_path: Path) -> None:
    world = tmp_path / "world.sdf"
    _write_world(world)
    client = GazeboTransportClient(
        GazeboCliClientConfig(
            world_path=str(world),
            world_name="test_world",
            executable="gz",
            timeout=0.2,
            transport_backend="transport",
            source_entity_name="ati/tool_link",
            target_entity_name="sfp_port_0_link",
        )
    )
    observation = client._parse_live_observation(
        state_payload='world: "test_world"\nstep_count: 0\n',
        pose_payload=(
            'pose {\n'
            '  name: "task_board"\n'
            '  position {\n    x: 0.15\n    y: -0.2\n    z: 1.14\n  }\n'
            '  orientation {\n    w: 1.0\n  }\n'
            '}\n'
            'pose {\n'
            '  name: "nic_card_mount_0"\n'
            '  position {\n    x: -0.045418\n    y: -0.1745\n    z: 0.012\n  }\n'
            '  orientation {\n    w: 1.0\n  }\n'
            '}\n'
            'pose {\n'
            '  name: "nic_card_link"\n'
            '  position {\n    x: -0.002\n    y: -0.01785\n    z: 0.0899\n  }\n'
            '  orientation {\n    x: -0.70710678\n    w: 0.70710678\n  }\n'
            '}\n'
            'pose {\n'
            '  name: "sfp_port_0_link"\n'
            '  position {\n    x: 0.01095\n    y: -0.012865\n    z: 0.121476\n  }\n'
            '  orientation {\n    x: 0.99998\n    w: 0.006321\n  }\n'
            '}\n'
            'pose {\n'
            '  name: "sfp_port_0_link_entrance"\n'
            '  position {\n    x: 0.01095\n    y: -0.012286\n    z: 0.167272\n  }\n'
            '  orientation {\n    x: 0.99998\n    w: 0.006321\n  }\n'
            '}\n'
            'pose {\n'
            '  name: "cable_0"\n'
            '  position {\n    x: 0.1725\n    y: 0.0244\n    z: 1.5084\n  }\n'
            '  orientation {\n    w: 1.0\n  }\n'
            '}\n'
            'pose {\n'
            '  name: "lc_plug_link"\n'
            '  position {\n    x: -0.01386\n    y: -0.01737\n    z: -0.03445\n  }\n'
            '  orientation {\n    w: 1.0\n  }\n'
            '}\n'
        ),
        world_name="test_world",
    )
    entities = observation["entities_by_name"]
    assert entities["nic_card_mount_0"]["pose_frame_status"] == "parent_composed_world"
    assert entities["nic_card_link"]["pose_frame_status"] == "parent_composed_world"
    assert entities["sfp_port_0_link"]["pose_frame_status"] == "parent_composed_world"
    assert entities["sfp_port_0_link_entrance"]["pose_frame_status"] == "parent_composed_world"
    assert entities["lc_plug_link"]["pose_frame_status"] == "parent_composed_world"
    assert float(entities["lc_plug_link"]["position"][2]) > 1.4
    assert float(entities["sfp_port_0_link"]["position"][2]) > 1.2


def test_client_prefers_configured_world_name_without_service_scan(tmp_path: Path) -> None:
    world = tmp_path / "world.sdf"
    _write_world(world)
    client = GazeboTransportClient(
        GazeboCliClientConfig(
            world_path=str(world),
            world_name="test_world",
            executable="gz",
            timeout=0.2,
            transport_backend="transport",
        )
    )
    client._discover_running_world_name = lambda: (_ for _ in ()).throw(
        AssertionError("service scan should not run when world_name is configured")
    )
    assert client._world_name() == "test_world"


def test_cli_client_does_not_force_world_step_when_observation_timeout_fallback_disabled(
    tmp_path: Path,
) -> None:
    world = tmp_path / "world.sdf"
    _write_world(world)
    client = GazeboCliClient(
        GazeboCliClientConfig(
            world_path=str(world),
            world_name="test_world",
            executable="gz",
            timeout=0.2,
            transport_backend="cli",
            allow_world_step_on_observation_timeout=False,
            observation_transport="persistent",
        )
    )
    client._state_topic_reader = lambda topic: type(
        "_Reader",
        (),
        {
            "get_sample": staticmethod(
                lambda after_generation=None, timeout=None: (_ for _ in ()).throw(TimeoutError())
            )
        },
    )()
    client._read_state_sample_after_world_step = lambda topic: (_ for _ in ()).throw(
        AssertionError("world-step fallback should stay disabled")
    )
    with pytest.raises(TimeoutError):
        client.get_observation(GetObservationRequest())


def test_transport_client_reset_failure_exposes_reset_service_category(tmp_path: Path) -> None:
    world = tmp_path / "world.sdf"
    _write_world(world)
    client = GazeboTransportClient(
        GazeboCliClientConfig(
            world_path=str(world),
            world_name="test_world",
            executable="gz",
            timeout=0.2,
            transport_backend="transport",
        )
    )
    client._bridge = _StubBridge(
        [
            {"state_generation": 1},
            GazeboTransportBridgeError("world control request failed"),
        ]
    )
    with pytest.raises(RuntimeError, match="reset service failure"):
        client.reset(ResetRequest(seed=1, options={}))
