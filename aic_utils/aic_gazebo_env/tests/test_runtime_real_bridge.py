"""Focused tests for the real Gazebo runtime bridge path."""

from __future__ import annotations

from pathlib import Path

from aic_gazebo_env import GazeboEnv, GazeboRuntime, GazeboRuntimeConfig, GymnasiumGazeboEnv
from aic_gazebo_env.gazebo_client import GazeboCliClient, GazeboCliClientConfig
from aic_gazebo_env.protocol import GetObservationResponse


def _write_fake_gz_bridge_script(path: Path) -> None:
    script = "\n".join(
        [
            "#!/usr/bin/env python3",
            "import json",
            "from pathlib import Path",
            "import re",
            "import signal",
            "import sys",
            "import time",
            "",
            "state_file = Path(sys.argv[0]).with_suffix('.state')",
            "",
            "def _read_state():",
            "    return json.loads(state_file.read_text(encoding='utf-8'))",
            "",
            "def _joint_names():",
            "    return [",
            "        'shoulder_pan_joint',",
            "        'shoulder_lift_joint',",
            "        'elbow_joint',",
            "        'wrist_1_joint',",
            "        'wrist_2_joint',",
            "        'wrist_3_joint',",
            "    ]",
            "",
            "def _tcp_pose_from_joints(joints):",
            "    return {",
            "        'position': [",
            "            1.0 + joints['shoulder_pan_joint'] + 0.5 * joints['shoulder_lift_joint'],",
            "            2.0 + joints['elbow_joint'] + 0.5 * joints['wrist_1_joint'],",
            "            3.0 + joints['wrist_2_joint'] + 0.5 * joints['wrist_3_joint'],",
            "        ],",
            "        'orientation': [0.0, 0.0, 0.0, 1.0],",
            "    }",
            "",
            "def _base_state():",
            "    joints = {name: 0.0 for name in _joint_names()}",
            "    tcp_pose = _tcp_pose_from_joints(joints)",
            "    return {",
            "        'step_count': 0,",
            "        'joints': joints,",
            "        'entities': {",
            "            'robot': {",
            "                'id': 101,",
            "                'position': [1.0, 2.0, 3.0],",
            "                'orientation': [0.0, 0.0, 0.0, 1.0],",
            "            },",
            "            'task_board': {",
            "                'id': 202,",
            "                'position': [4.0, 5.0, 6.0],",
            "                'orientation': [0.0, 0.0, 0.707, 0.707],",
            "            },",
            "            'inspection_target': {",
            "                'id': 303,",
            "                'position': [10.0, 12.0, 14.0],",
            "                'orientation': [0.0, 0.0, 0.0, 1.0],",
            "            },",
            "            'gripper/tcp': {",
            "                'id': 404,",
            "                'position': list(tcp_pose['position']),",
            "                'orientation': list(tcp_pose['orientation']),",
            "            },",
            "        },",
            "    }",
            "",
            "def _refresh_tcp_entity(state):",
            "    tcp_pose = _tcp_pose_from_joints(state['joints'])",
            "    state['entities']['gripper/tcp']['position'] = list(tcp_pose['position'])",
            "    state['entities']['gripper/tcp']['orientation'] = list(tcp_pose['orientation'])",
            "",
            "def _write_state(state):",
            "    _refresh_tcp_entity(state)",
            "    state_file.write_text(json.dumps(state), encoding='utf-8')",
            "",
            "def _exit_cleanly(signum, frame):",
            "    del signum, frame",
            "    sys.exit(0)",
            "",
            "signal.signal(signal.SIGTERM, _exit_cleanly)",
            "",
            "if len(sys.argv) >= 3 and sys.argv[1] == 'sim':",
            "    print('fake gz sim running', flush=True)",
            "    while True:",
            "        time.sleep(0.1)",
            "",
            "if not state_file.exists():",
            "    _write_state(_base_state())",
            "",
            "if sys.argv[1:3] == ['topic', '-e']:",
            "    topic = sys.argv[-1]",
            "    state = _read_state()",
            "    print(",
            "        f'''world: \"test_world\"",
            "step_count: {state['step_count']}",
            "entity {{",
            "  id: {state['entities']['robot']['id']}",
            "  name: \"robot\"",
            "  pose {{",
            "    position {{",
            "      x: {state['entities']['robot']['position'][0]}",
            "      y: {state['entities']['robot']['position'][1]}",
            "      z: {state['entities']['robot']['position'][2]}",
            "    }}",
            "    orientation {{",
            "      x: {state['entities']['robot']['orientation'][0]}",
            "      y: {state['entities']['robot']['orientation'][1]}",
            "      z: {state['entities']['robot']['orientation'][2]}",
            "      w: {state['entities']['robot']['orientation'][3]}",
            "    }}",
            "  }}",
            "}}",
            "entity {{",
            "  id: {state['entities']['task_board']['id']}",
            "  name: \"task_board\"",
            "  pose {{",
            "    position {{",
            "      x: {state['entities']['task_board']['position'][0]}",
            "      y: {state['entities']['task_board']['position'][1]}",
            "      z: {state['entities']['task_board']['position'][2]}",
            "    }}",
            "    orientation {{",
            "      x: {state['entities']['task_board']['orientation'][0]}",
            "      y: {state['entities']['task_board']['orientation'][1]}",
            "      z: {state['entities']['task_board']['orientation'][2]}",
            "      w: {state['entities']['task_board']['orientation'][3]}",
            "    }}",
            "  }}",
            "}}",
            "entity {{",
            "  id: {state['entities']['inspection_target']['id']}",
            "  name: \"inspection_target\"",
            "  pose {{",
            "    position {{",
            "      x: {state['entities']['inspection_target']['position'][0]}",
            "      y: {state['entities']['inspection_target']['position'][1]}",
            "      z: {state['entities']['inspection_target']['position'][2]}",
            "    }}",
            "    orientation {{",
            "      x: {state['entities']['inspection_target']['orientation'][0]}",
            "      y: {state['entities']['inspection_target']['orientation'][1]}",
            "      z: {state['entities']['inspection_target']['orientation'][2]}",
            "      w: {state['entities']['inspection_target']['orientation'][3]}",
            "    }}",
            "  }}",
            "}}",
            "entity {{",
            "  id: {state['entities']['gripper/tcp']['id']}",
            "  name: \"gripper/tcp\"",
            "  pose {{",
            "    position {{",
            "      x: {state['entities']['gripper/tcp']['position'][0]}",
            "      y: {state['entities']['gripper/tcp']['position'][1]}",
            "      z: {state['entities']['gripper/tcp']['position'][2]}",
            "    }}",
            "    orientation {{",
            "      x: {state['entities']['gripper/tcp']['orientation'][0]}",
            "      y: {state['entities']['gripper/tcp']['orientation'][1]}",
            "      z: {state['entities']['gripper/tcp']['orientation'][2]}",
            "      w: {state['entities']['gripper/tcp']['orientation'][3]}",
            "    }}",
            "  }}",
            "}}",
            "joint {{",
            "  name: \"shoulder_pan_joint\"",
            "  position: {state['joints']['shoulder_pan_joint']}",
            "}}",
            "joint {{",
            "  name: \"shoulder_lift_joint\"",
            "  position: {state['joints']['shoulder_lift_joint']}",
            "}}",
            "joint {{",
            "  name: \"elbow_joint\"",
            "  position: {state['joints']['elbow_joint']}",
            "}}",
            "joint {{",
            "  name: \"wrist_1_joint\"",
            "  position: {state['joints']['wrist_1_joint']}",
            "}}",
            "joint {{",
            "  name: \"wrist_2_joint\"",
            "  position: {state['joints']['wrist_2_joint']}",
            "}}",
            "joint {{",
            "  name: \"wrist_3_joint\"",
            "  position: {state['joints']['wrist_3_joint']}",
            "}}",
            "topic: \"{topic}\"''',",
            "        flush=True,",
            "    )",
            "    sys.exit(0)",
            "",
            "if sys.argv[1:3] == ['service', '-s']:",
            "    service = sys.argv[3]",
            "    request_payload = sys.argv[sys.argv.index('--req') + 1]",
            "    state = _read_state()",
            "    if 'multi_step:' in request_payload:",
            "        increment = int(request_payload.split('multi_step:', 1)[1].strip())",
            "        state['step_count'] += increment",
            "    if 'reset:' in request_payload:",
            "        state = _base_state()",
            "    if service.endswith('/set_pose'):",
            "        name_match = re.search(r'name:\\s*\"([^\"]+)\"', request_payload)",
            "        position_match = re.search(",
            "            r'position:\\s*\\{x:\\s*([^,]+),\\s*y:\\s*([^,]+),\\s*z:\\s*([^}]+)\\}',",
            "            request_payload,",
            "        )",
            "        orientation_match = re.search(",
            "            r'orientation:\\s*\\{x:\\s*([^,]+),\\s*y:\\s*([^,]+),\\s*z:\\s*([^,]+),\\s*w:\\s*([^}]+)\\}',",
            "            request_payload,",
            "        )",
            "        if name_match and position_match:",
            "            state['entities'][name_match.group(1)]['position'] = [",
            "                float(position_match.group(1)),",
            "                float(position_match.group(2)),",
            "                float(position_match.group(3)),",
            "            ]",
            "        if name_match and orientation_match:",
            "            state['entities'][name_match.group(1)]['orientation'] = [",
            "                float(orientation_match.group(1)),",
            "                float(orientation_match.group(2)),",
            "                float(orientation_match.group(3)),",
            "                float(orientation_match.group(4)),",
            "            ]",
            "    if service.endswith('/joint_target'):",
            "        data_match = re.search(r'data:\\s*\"([^\"]+)\"', request_payload)",
            "        if data_match:",
            "            fields = {}",
            "            for field in data_match.group(1).split(';'):",
            "                if '=' not in field:",
            "                    continue",
            "                key, value = field.split('=', 1)",
            "                fields[key] = value",
            "            joint_names = [name for name in fields.get('joint_names', '').split(',') if name]",
            "            positions = [float(value) for value in fields.get('positions', '').split(',') if value]",
            "            for joint_name, position in zip(joint_names, positions):",
            "                state['joints'][joint_name] = position",
            "    _write_state(state)",
            "    print(",
            "        json.dumps(",
            "            {",
            "                'service': service,",
            "                'request': request_payload,",
            "                'success': True,",
            "            }",
            "        ),",
            "        flush=True,",
            "    )",
            "    sys.exit(0)",
            "",
            "print(f'unexpected argv: {sys.argv}', file=sys.stderr, flush=True)",
            "sys.exit(3)",
            "",
        ]
    )
    path.write_text(
        script,
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_world(path: Path) -> None:
    path.write_text(
        "<sdf version='1.9'><world name='test_world'></world></sdf>\n",
        encoding="utf-8",
    )


def test_runtime_can_get_real_observation_and_reset_via_client(tmp_path: Path) -> None:
    fake_gz = tmp_path / "fake_gz.py"
    world = tmp_path / "world.sdf"
    _write_fake_gz_bridge_script(fake_gz)
    _write_world(world)

    runtime = GazeboRuntime(
        GazeboRuntimeConfig(
            world_path=str(world),
            world_name="test_world",
            timeout=0.2,
            executable=str(fake_gz),
        )
    )

    runtime.start()
    try:
        observation, info = runtime.get_observation()
        assert observation["world_name"] == "test_world"
        assert observation["step_count"] == 0
        assert observation["entity_count"] == 4
        assert observation["entity_names"] == ["robot", "task_board", "inspection_target", "gripper/tcp"]
        assert observation["joint_count"] == 6
        assert observation["joint_names"] == [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        assert observation["joint_positions"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert observation["entities"] == [
            {
                "name": "robot",
                "id": 101,
                "position": [1.0, 2.0, 3.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "pose": {
                    "position": [1.0, 2.0, 3.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
            },
            {
                "name": "task_board",
                "id": 202,
                "position": [4.0, 5.0, 6.0],
                "orientation": [0.0, 0.0, 0.707, 0.707],
                "pose": {
                    "position": [4.0, 5.0, 6.0],
                    "orientation": [0.0, 0.0, 0.707, 0.707],
                },
            },
            {
                "name": "inspection_target",
                "id": 303,
                "position": [10.0, 12.0, 14.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "pose": {
                    "position": [10.0, 12.0, 14.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
            },
            {
                "name": "gripper/tcp",
                "id": 404,
                "position": [1.0, 2.0, 3.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "pose": {
                    "position": [1.0, 2.0, 3.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
            },
        ]
        assert observation["entities_by_name"]["robot"] == observation["entities"][0]
        assert observation["entities_by_name"]["task_board"] == observation["entities"][1]
        assert observation["entities_by_name"]["inspection_target"] == observation["entities"][2]
        assert observation["entities_by_name"]["gripper/tcp"] == observation["entities"][3]
        assert observation["task_geometry"] == {
            "tracked_entity_pair": {
                "source": "robot",
                "target": "task_board",
                "relative_position": [3.0, 3.0, 3.0],
                "distance": 5.196152422706632,
                "source_orientation": [0.0, 0.0, 0.0, 1.0],
                "target_orientation": [0.0, 0.0, 0.707, 0.707],
                "relative_orientation": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
                "orientation_error": 1.5707963267948966,
                "success_threshold": 1.0,
                "orientation_success_threshold": None,
                "distance_success": False,
                "orientation_success": True,
                "success": False,
            },
            "robot_to_task_board": {
                "source": "robot",
                "target": "task_board",
                "relative_position": [3.0, 3.0, 3.0],
                "distance": 5.196152422706632,
                "source_orientation": [0.0, 0.0, 0.0, 1.0],
                "target_orientation": [0.0, 0.0, 0.707, 0.707],
                "relative_orientation": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
                "orientation_error": 1.5707963267948966,
                "success_threshold": 1.0,
                "orientation_success_threshold": None,
                "distance_success": False,
                "orientation_success": True,
                "success": False,
            }
        }
        assert info["backend"] == "gazebo"
        assert info["observation_topic"] == "/world/test_world/state"
        assert 'topic: "/world/test_world/state"' in info["state_text"]

        reset_observation, reset_info = runtime.reset(seed=7, options={"mode": "smoke"})
        assert reset_observation == observation
        assert reset_info["backend"] == "gazebo"
        assert reset_info["runtime"] == "gazebo"
        assert reset_info["seed"] == 7
        assert reset_info["options"] == {"mode": "smoke"}
        assert "/world/test_world/control" in reset_info["reset_service"]

        step_observation, reward, terminated, truncated, step_info = runtime.step(
            {
                "set_entity_position": {
                    "name": "robot",
                    "position": [7.0, 8.0, 9.0],
                },
                "multi_step": 3,
            }
        )
        assert step_observation["world_name"] == "test_world"
        assert step_observation["step_count"] == 3
        assert step_observation["entity_count"] == 4
        assert step_observation["entity_names"] == ["robot", "task_board", "inspection_target", "gripper/tcp"]
        assert step_observation["joint_count"] == 6
        assert step_observation["joint_positions"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert step_observation["entities"] == [
            {
                "name": "robot",
                "id": 101,
                "position": [7.0, 8.0, 9.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "pose": {
                    "position": [7.0, 8.0, 9.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
            },
            {
                "name": "task_board",
                "id": 202,
                "position": [4.0, 5.0, 6.0],
                "orientation": [0.0, 0.0, 0.707, 0.707],
                "pose": {
                    "position": [4.0, 5.0, 6.0],
                    "orientation": [0.0, 0.0, 0.707, 0.707],
                },
            },
            {
                "name": "inspection_target",
                "id": 303,
                "position": [10.0, 12.0, 14.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "pose": {
                    "position": [10.0, 12.0, 14.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
            },
            {
                "name": "gripper/tcp",
                "id": 404,
                "position": [1.0, 2.0, 3.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "pose": {
                    "position": [1.0, 2.0, 3.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
            },
        ]
        assert step_observation["entities_by_name"]["robot"] == step_observation["entities"][0]
        assert step_observation["entities_by_name"]["robot"]["position"] == [7.0, 8.0, 9.0]
        assert step_observation["task_geometry"] == {
            "tracked_entity_pair": {
                "source": "robot",
                "target": "task_board",
                "relative_position": [-3.0, -3.0, -3.0],
                "distance": 5.196152422706632,
                "source_orientation": [0.0, 0.0, 0.0, 1.0],
                "target_orientation": [0.0, 0.0, 0.707, 0.707],
                "relative_orientation": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
                "orientation_error": 1.5707963267948966,
                "success_threshold": 1.0,
                "orientation_success_threshold": None,
                "distance_success": False,
                "orientation_success": True,
                "success": False,
            },
            "robot_to_task_board": {
                "source": "robot",
                "target": "task_board",
                "relative_position": [-3.0, -3.0, -3.0],
                "distance": 5.196152422706632,
                "source_orientation": [0.0, 0.0, 0.0, 1.0],
                "target_orientation": [0.0, 0.0, 0.707, 0.707],
                "relative_orientation": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
                "orientation_error": 1.5707963267948966,
                "success_threshold": 1.0,
                "orientation_success_threshold": None,
                "distance_success": False,
                "orientation_success": True,
                "success": False,
            }
        }
        assert reward == -5.196152422706632
        assert terminated is False
        assert truncated is False
        assert step_info["backend"] == "gazebo"
        assert step_info["runtime"] == "gazebo"
        assert step_info["applied_action"] == {
            "set_entity_position": {
                "name": "robot",
                "position": [7.0, 8.0, 9.0],
            },
            "multi_step": 3,
        }
        assert '"service": "/world/test_world/set_pose"' in step_info["pose_service"]
        assert step_info["multi_step"] == 3
        assert '"request": "multi_step: 3"' in step_info["step_service"]
        assert step_info["source_entity_name"] == "robot"
        assert step_info["target_entity_name"] == "task_board"
        assert step_info["success_distance_threshold"] == 1.0
        assert step_info["orientation_success_threshold"] is None
        assert step_info["success_bonus"] == 10.0
        assert step_info["terminated"] is False
        assert step_info["reward"] == -5.196152422706632

        updated_observation, updated_info = runtime.get_observation()
        assert updated_observation == step_observation
        assert updated_info["observation_topic"] == "/world/test_world/state"
        assert "step_count: 3" in updated_info["state_text"]
    finally:
        runtime.stop()


def test_runtime_real_step_terminates_when_distance_is_below_threshold(tmp_path: Path) -> None:
    fake_gz = tmp_path / "fake_gz.py"
    world = tmp_path / "world.sdf"
    _write_fake_gz_bridge_script(fake_gz)
    _write_world(world)

    runtime = GazeboRuntime(
        GazeboRuntimeConfig(
            world_path=str(world),
            world_name="test_world",
            timeout=0.2,
            executable=str(fake_gz),
            success_distance_threshold=1.0,
            success_bonus=10.0,
        )
    )

    runtime.start()
    try:
        far_observation, far_reward, far_terminated, far_truncated, far_info = runtime.step(
            {
                "set_entity_position": {
                    "name": "robot",
                    "position": [7.0, 8.0, 9.0],
                },
                "multi_step": 1,
            }
        )
        assert far_observation["task_geometry"]["tracked_entity_pair"]["distance"] == 5.196152422706632
        assert far_observation["task_geometry"]["tracked_entity_pair"]["orientation_error"] == 1.5707963267948966
        assert far_observation["task_geometry"]["tracked_entity_pair"]["success"] is False
        assert far_reward == -5.196152422706632
        assert far_terminated is False
        assert far_truncated is False
        assert far_info["terminated"] is False

        success_observation, success_reward, success_terminated, success_truncated, success_info = runtime.step(
            {
                "set_entity_position": {
                    "name": "robot",
                    "position": [4.0, 5.0, 6.0],
                },
                "multi_step": 1,
            }
        )
        assert success_observation["task_geometry"]["tracked_entity_pair"]["distance"] == 0.0
        assert success_observation["task_geometry"]["tracked_entity_pair"]["orientation_error"] == 1.5707963267948966
        assert success_observation["task_geometry"]["tracked_entity_pair"]["success"] is True
        assert success_reward == 10.0
        assert success_terminated is True
        assert success_truncated is False
        assert success_info["terminated"] is True
        assert success_info["reward"] == 10.0
    finally:
        runtime.stop()


def test_runtime_real_geometry_uses_overridden_entity_pair_and_threshold(tmp_path: Path) -> None:
    fake_gz = tmp_path / "fake_gz.py"
    world = tmp_path / "world.sdf"
    _write_fake_gz_bridge_script(fake_gz)
    _write_world(world)

    runtime = GazeboRuntime(
        GazeboRuntimeConfig(
            world_path=str(world),
            world_name="test_world",
            timeout=0.2,
            executable=str(fake_gz),
            source_entity_name="task_board",
            target_entity_name="inspection_target",
            success_distance_threshold=11.0,
            success_bonus=10.0,
        )
    )

    runtime.start()
    try:
        observation, info = runtime.get_observation()
        assert observation["task_geometry"] == {
            "tracked_entity_pair": {
                "source": "task_board",
                "target": "inspection_target",
                "relative_position": [6.0, 7.0, 8.0],
                "distance": 12.206555615733702,
                "source_orientation": [0.0, 0.0, 0.707, 0.707],
                "target_orientation": [0.0, 0.0, 0.0, 1.0],
                "relative_orientation": [0.0, 0.0, -0.7071067811865476, 0.7071067811865476],
                "orientation_error": 1.5707963267948966,
                "success_threshold": 11.0,
                "orientation_success_threshold": None,
                "distance_success": False,
                "orientation_success": True,
                "success": False,
            }
        }
        assert "robot_to_task_board" not in observation["task_geometry"]
        assert info["observation_topic"] == "/world/test_world/state"

        step_observation, reward, terminated, truncated, step_info = runtime.step(
            {
                "set_entity_position": {
                    "name": "inspection_target",
                    "position": [5.0, 6.0, 7.0],
                },
                "multi_step": 1,
            }
        )
        assert step_observation["task_geometry"] == {
            "tracked_entity_pair": {
                "source": "task_board",
                "target": "inspection_target",
                "relative_position": [1.0, 1.0, 1.0],
                "distance": 1.7320508075688772,
                "source_orientation": [0.0, 0.0, 0.707, 0.707],
                "target_orientation": [0.0, 0.0, 0.0, 1.0],
                "relative_orientation": [0.0, 0.0, -0.7071067811865476, 0.7071067811865476],
                "orientation_error": 1.5707963267948966,
                "success_threshold": 11.0,
                "orientation_success_threshold": None,
                "distance_success": True,
                "orientation_success": True,
                "success": True,
            }
        }
        assert reward == 8.267949192431123
        assert terminated is True
        assert truncated is False
        assert step_info["source_entity_name"] == "task_board"
        assert step_info["target_entity_name"] == "inspection_target"
        assert step_info["success_distance_threshold"] == 11.0
        assert step_info["reward"] == 8.267949192431123
        assert step_info["terminated"] is True
    finally:
        runtime.stop()


def test_cli_client_waits_for_joint_target_settle_before_returning_observation(tmp_path: Path) -> None:
    world = tmp_path / "world.sdf"
    _write_world(world)
    client = GazeboCliClient(
        GazeboCliClientConfig(
            executable="/bin/true",
            world_path=str(world),
            world_name="test_world",
            timeout=0.2,
            joint_settle_tolerance=0.01,
            joint_settle_timeout_s=0.5,
            joint_settle_poll_interval_s=0.0,
        )
    )

    observations = iter(
        [
            GetObservationResponse(
                observation={
                    "joint_names": [
                        "shoulder_pan_joint",
                        "shoulder_lift_joint",
                        "elbow_joint",
                        "wrist_1_joint",
                        "wrist_2_joint",
                        "wrist_3_joint",
                    ],
                    "joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                }
            ),
            GetObservationResponse(
                observation={
                    "joint_names": [
                        "shoulder_pan_joint",
                        "shoulder_lift_joint",
                        "elbow_joint",
                        "wrist_1_joint",
                        "wrist_2_joint",
                        "wrist_3_joint",
                    ],
                    "joint_positions": [0.999, -0.999, 0.499, 0.0, 0.0, 0.0],
                }
            ),
        ]
    )
    client.get_observation = lambda request: next(observations)

    response, settled = client._wait_for_joint_target_settle(
        [1.0, -1.0, 0.5, 0.0, 0.0, 0.0]
    )

    assert settled is True
    assert response.observation["joint_positions"] == [0.999, -0.999, 0.499, 0.0, 0.0, 0.0]


def test_cli_client_reports_unsettled_joint_target_after_timeout(tmp_path: Path) -> None:
    world = tmp_path / "world.sdf"
    _write_world(world)
    client = GazeboCliClient(
        GazeboCliClientConfig(
            executable="/bin/true",
            world_path=str(world),
            world_name="test_world",
            timeout=0.2,
            joint_settle_tolerance=1e-6,
            joint_settle_timeout_s=0.0,
            joint_settle_poll_interval_s=0.0,
        )
    )

    expected_response = GetObservationResponse(
        observation={
            "joint_names": [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            "joint_positions": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
        }
    )
    client.get_observation = lambda request: expected_response

    response, settled = client._wait_for_joint_target_settle(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )

    assert settled is False
    assert response is expected_response


def test_runtime_real_orientation_threshold_can_gate_success(tmp_path: Path) -> None:
    fake_gz = tmp_path / "fake_gz.py"
    world = tmp_path / "world.sdf"
    _write_fake_gz_bridge_script(fake_gz)
    _write_world(world)

    runtime = GazeboRuntime(
        GazeboRuntimeConfig(
            world_path=str(world),
            world_name="test_world",
            timeout=0.2,
            executable=str(fake_gz),
            source_entity_name="robot",
            target_entity_name="task_board",
            success_distance_threshold=1.0,
            orientation_success_threshold=0.2,
            success_bonus=10.0,
        )
    )

    runtime.start()
    try:
        step_observation, reward, terminated, truncated, step_info = runtime.step(
            {
                "set_entity_position": {
                    "name": "robot",
                    "position": [4.0, 5.0, 6.0],
                },
                "multi_step": 1,
            }
        )
        assert step_observation["task_geometry"]["tracked_entity_pair"]["distance"] == 0.0
        assert step_observation["task_geometry"]["tracked_entity_pair"]["orientation_error"] == 1.5707963267948966
        assert step_observation["task_geometry"]["tracked_entity_pair"]["orientation_success_threshold"] == 0.2
        assert step_observation["task_geometry"]["tracked_entity_pair"]["distance_success"] is True
        assert step_observation["task_geometry"]["tracked_entity_pair"]["orientation_success"] is False
        assert step_observation["task_geometry"]["tracked_entity_pair"]["success"] is False
        assert reward == 0.0
        assert terminated is False
        assert truncated is False
        assert step_info["orientation_success_threshold"] == 0.2
        assert step_info["terminated"] is False
        assert step_info["truncated"] is False
        assert step_info["reward"] == 0.0
    finally:
        runtime.stop()


def test_runtime_real_step_budget_controls_truncation(tmp_path: Path) -> None:
    fake_gz = tmp_path / "fake_gz.py"
    world = tmp_path / "world.sdf"
    _write_fake_gz_bridge_script(fake_gz)
    _write_world(world)

    runtime = GazeboRuntime(
        GazeboRuntimeConfig(
            world_path=str(world),
            world_name="test_world",
            timeout=0.2,
            executable=str(fake_gz),
            max_episode_steps=3,
        )
    )

    runtime.start()
    try:
        below_budget_observation, below_budget_reward, below_budget_terminated, below_budget_truncated, below_budget_info = runtime.step(
            {
                "set_entity_position": {
                    "name": "robot",
                    "position": [7.0, 8.0, 9.0],
                },
                "multi_step": 1,
            }
        )
        assert below_budget_observation["step_count"] == 1
        assert below_budget_observation["task_geometry"]["tracked_entity_pair"]["success"] is False
        assert below_budget_reward == -5.196152422706632
        assert below_budget_terminated is False
        assert below_budget_truncated is False
        assert below_budget_info["max_episode_steps"] == 3
        assert below_budget_info["truncated"] is False

        reset_observation, _ = runtime.reset(seed=1, options={"mode": "budget-reset"})
        assert reset_observation["step_count"] == 0

        success_observation, success_reward, success_terminated, success_truncated, success_info = runtime.step(
            {
                "set_entity_position": {
                    "name": "robot",
                    "position": [4.0, 5.0, 6.0],
                },
                "multi_step": 1,
            }
        )
        assert success_observation["step_count"] == 1
        assert success_observation["task_geometry"]["tracked_entity_pair"]["success"] is True
        assert success_reward == 10.0
        assert success_terminated is True
        assert success_truncated is False
        assert success_info["truncated"] is False

        runtime.reset(seed=2, options={"mode": "budget-truncate"})
        truncated_observation, truncated_reward, truncated_terminated, truncated_truncated, truncated_info = runtime.step(
            {
                "set_entity_position": {
                    "name": "robot",
                    "position": [7.0, 8.0, 9.0],
                },
                "multi_step": 3,
            }
        )
        assert truncated_observation["step_count"] == 3
        assert truncated_observation["task_geometry"]["tracked_entity_pair"]["success"] is False
        assert truncated_reward == -5.196152422706632
        assert truncated_terminated is False
        assert truncated_truncated is True
        assert truncated_info["max_episode_steps"] == 3
        assert truncated_info["truncated"] is True
    finally:
        runtime.stop()


def test_runtime_real_pose_action_updates_orientation_and_can_drive_success(tmp_path: Path) -> None:
    fake_gz = tmp_path / "fake_gz.py"
    world = tmp_path / "world.sdf"
    _write_fake_gz_bridge_script(fake_gz)
    _write_world(world)

    runtime = GazeboRuntime(
        GazeboRuntimeConfig(
            world_path=str(world),
            world_name="test_world",
            timeout=0.2,
            executable=str(fake_gz),
            source_entity_name="robot",
            target_entity_name="task_board",
            success_distance_threshold=1.0,
            orientation_success_threshold=0.2,
            success_bonus=10.0,
        )
    )

    runtime.start()
    try:
        position_only_observation, position_only_reward, position_only_terminated, position_only_truncated, position_only_info = runtime.step(
            {
                "set_entity_position": {
                    "name": "robot",
                    "position": [4.0, 5.0, 6.0],
                },
                "multi_step": 1,
            }
        )
        assert position_only_observation["entities_by_name"]["robot"]["orientation"] == [0.0, 0.0, 0.0, 1.0]
        assert position_only_observation["task_geometry"]["tracked_entity_pair"]["orientation_error"] == 1.5707963267948966
        assert position_only_observation["task_geometry"]["tracked_entity_pair"]["orientation_success"] is False
        assert position_only_observation["task_geometry"]["tracked_entity_pair"]["success"] is False
        assert position_only_reward == 0.0
        assert position_only_terminated is False
        assert position_only_truncated is False
        assert '"request": "name: \\"robot\\" position: {x: 4.0, y: 5.0, z: 6.0} orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}"' in position_only_info["pose_service"]

        runtime.reset(seed=3, options={"mode": "pose-success"})
        pose_observation, pose_reward, pose_terminated, pose_truncated, pose_info = runtime.step(
            {
                "set_entity_pose": {
                    "name": "robot",
                    "position": [4.0, 5.0, 6.0],
                    "orientation": [0.0, 0.0, 0.707, 0.707],
                },
                "multi_step": 1,
            }
        )
        assert pose_observation["entities_by_name"]["robot"]["orientation"] == [0.0, 0.0, 0.707, 0.707]
        assert pose_observation["entities_by_name"]["robot"]["pose"]["orientation"] == [0.0, 0.0, 0.707, 0.707]
        assert pose_observation["task_geometry"]["tracked_entity_pair"]["source_orientation"] == [0.0, 0.0, 0.707, 0.707]
        assert pose_observation["task_geometry"]["tracked_entity_pair"]["target_orientation"] == [0.0, 0.0, 0.707, 0.707]
        assert pose_observation["task_geometry"]["tracked_entity_pair"]["relative_orientation"] == [0.0, 0.0, 0.0, 1.0]
        assert pose_observation["task_geometry"]["tracked_entity_pair"]["orientation_error"] == 0.0
        assert pose_observation["task_geometry"]["tracked_entity_pair"]["orientation_success"] is True
        assert pose_observation["task_geometry"]["tracked_entity_pair"]["success"] is True
        assert pose_reward == 10.0
        assert pose_terminated is True
        assert pose_truncated is False
        assert pose_info["applied_action"] == {
            "set_entity_pose": {
                "name": "robot",
                "position": [4.0, 5.0, 6.0],
                "orientation": [0.0, 0.0, 0.707, 0.707],
            },
            "multi_step": 1,
        }
        assert '"request": "name: \\"robot\\" position: {x: 4.0, y: 5.0, z: 6.0} orientation: {x: 0.0, y: 0.0, z: 0.707, w: 0.707}"' in pose_info["pose_service"]
    finally:
        runtime.stop()


def test_runtime_real_delta_source_pose_accumulates_from_current_pose(tmp_path: Path) -> None:
    fake_gz = tmp_path / "fake_gz.py"
    world = tmp_path / "world.sdf"
    _write_fake_gz_bridge_script(fake_gz)
    _write_world(world)

    runtime = GazeboRuntime(
        GazeboRuntimeConfig(
            world_path=str(world),
            world_name="test_world",
            timeout=0.2,
            executable=str(fake_gz),
            source_entity_name="robot",
            target_entity_name="task_board",
        )
    )

    runtime.start()
    try:
        runtime.step(
            {
                "set_entity_pose": {
                    "name": "robot",
                    "position": [4.0, 5.0, 6.0],
                    "orientation": [0.0, 0.0, 0.707, 0.707],
                },
                "multi_step": 1,
            }
        )

        first_delta_observation, first_delta_reward, first_delta_terminated, first_delta_truncated, first_delta_info = runtime.step(
            {
                "delta_source_pose": {
                    "position_delta": [1.0, 0.0, 0.0],
                },
                "multi_step": 1,
            }
        )
        assert first_delta_observation["entities_by_name"]["robot"]["position"] == [5.0, 5.0, 6.0]
        assert first_delta_observation["entities_by_name"]["robot"]["orientation"] == [
            0.0,
            0.0,
            0.7071067811865476,
            0.7071067811865476,
        ]
        assert first_delta_observation["task_geometry"]["tracked_entity_pair"]["relative_position"] == [-1.0, 0.0, 0.0]
        assert first_delta_observation["task_geometry"]["tracked_entity_pair"]["distance"] == 1.0
        assert first_delta_reward == 9.0
        assert first_delta_terminated is True
        assert first_delta_truncated is False
        assert first_delta_info["applied_action"] == {
            "delta_source_pose": {
                "position_delta": [1.0, 0.0, 0.0],
            },
            "multi_step": 1,
        }
        assert '"request": "name: \\"robot\\" position: {x: 5.0, y: 5.0, z: 6.0} orientation: {x: 0.0, y: 0.0, z: 0.7071067811865476, w: 0.7071067811865476}"' in first_delta_info["pose_service"]

        second_delta_observation, second_delta_reward, second_delta_terminated, second_delta_truncated, _ = runtime.step(
            {
                "delta_source_pose": {
                    "position_delta": [0.0, -1.0, 0.0],
                    "orientation_delta": [0.0, 0.0, -0.7071067811865476, 0.7071067811865476],
                },
                "multi_step": 1,
            }
        )
        assert second_delta_observation["entities_by_name"]["robot"]["position"] == [5.0, 4.0, 6.0]
        assert second_delta_observation["entities_by_name"]["robot"]["orientation"] == [0.0, 0.0, 0.0, 1.0]
        assert second_delta_observation["task_geometry"]["tracked_entity_pair"]["relative_position"] == [-1.0, 1.0, 0.0]
        assert second_delta_observation["task_geometry"]["tracked_entity_pair"]["distance"] == 1.4142135623730951
        assert second_delta_observation["task_geometry"]["tracked_entity_pair"]["orientation_error"] == 1.5707963267948966
        assert second_delta_reward == -1.4142135623730951
        assert second_delta_terminated is False
        assert second_delta_truncated is False
    finally:
        runtime.stop()


def test_runtime_real_get_observation_uses_same_config_as_step(tmp_path: Path) -> None:
    fake_gz = tmp_path / "fake_gz.py"
    world = tmp_path / "world.sdf"
    _write_fake_gz_bridge_script(fake_gz)
    _write_world(world)

    runtime = GazeboRuntime(
        GazeboRuntimeConfig(
            world_path=str(world),
            world_name="test_world",
            timeout=0.2,
            executable=str(fake_gz),
            source_entity_name="robot",
            target_entity_name="task_board",
            success_distance_threshold=1.0,
            orientation_success_threshold=0.2,
            success_bonus=10.0,
        )
    )

    runtime.start()
    try:
        step_observation, step_reward, step_terminated, step_truncated, step_info = runtime.step(
            {
                "set_entity_pose": {
                    "name": "robot",
                    "position": [5.0, 5.0, 6.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
                "multi_step": 1,
            }
        )
        assert step_observation["task_geometry"]["tracked_entity_pair"]["orientation_success_threshold"] == 0.2
        assert step_observation["task_geometry"]["tracked_entity_pair"]["orientation_error"] == 1.5707963267948966
        assert step_observation["task_geometry"]["tracked_entity_pair"]["distance_success"] is True
        assert step_observation["task_geometry"]["tracked_entity_pair"]["orientation_success"] is False
        assert step_observation["task_geometry"]["tracked_entity_pair"]["success"] is False
        assert step_reward == -1.0
        assert step_terminated is False
        assert step_truncated is False
        assert step_info["orientation_success_threshold"] == 0.2

        updated_observation, updated_info = runtime.get_observation()
        assert updated_observation == step_observation
        assert updated_observation["task_geometry"]["tracked_entity_pair"]["orientation_success_threshold"] == 0.2
        assert updated_observation["task_geometry"]["tracked_entity_pair"]["orientation_error"] == 1.5707963267948966
        assert updated_observation["task_geometry"]["tracked_entity_pair"]["distance_success"] is True
        assert updated_observation["task_geometry"]["tracked_entity_pair"]["orientation_success"] is False
        assert updated_observation["task_geometry"]["tracked_entity_pair"]["success"] is False
        assert updated_info["observation_topic"] == "/world/test_world/state"
    finally:
        runtime.stop()


def test_runtime_real_policy_action_translates_to_delta_source_pose(tmp_path: Path) -> None:
    fake_gz = tmp_path / "fake_gz.py"
    world = tmp_path / "world.sdf"
    _write_fake_gz_bridge_script(fake_gz)
    _write_world(world)

    runtime = GazeboRuntime(
        GazeboRuntimeConfig(
            world_path=str(world),
            world_name="test_world",
            timeout=0.2,
            executable=str(fake_gz),
            source_entity_name="robot",
            target_entity_name="task_board",
        )
    )

    runtime.start()
    try:
        first_observation, first_reward, first_terminated, first_truncated, first_info = runtime.step(
            {
                "policy_action": {
                    "position_delta": [1.0, 0.0, 0.0],
                },
                "multi_step": 1,
            }
        )
        assert first_observation["entities_by_name"]["robot"]["position"] == [2.0, 2.0, 3.0]
        assert first_observation["task_geometry"]["tracked_entity_pair"]["relative_position"] == [2.0, 3.0, 3.0]
        assert first_reward == -4.69041575982343
        assert first_terminated is False
        assert first_truncated is False
        assert first_info["applied_action"] == {
            "delta_source_pose": {
                "position_delta": [1.0, 0.0, 0.0],
            },
            "multi_step": 1,
        }

        second_observation, second_reward, second_terminated, second_truncated, second_info = runtime.step(
            {
                "policy_action": {
                    "position_delta": [0.0, 1.0, 0.0],
                    "orientation_delta": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
                },
                "multi_step": 1,
            }
        )
        assert second_observation["entities_by_name"]["robot"]["position"] == [2.0, 3.0, 3.0]
        assert second_observation["entities_by_name"]["robot"]["orientation"] == [
            0.0,
            0.0,
            0.7071067811865476,
            0.7071067811865476,
        ]
        assert second_observation["task_geometry"]["tracked_entity_pair"]["relative_position"] == [2.0, 2.0, 3.0]
        assert second_observation["task_geometry"]["tracked_entity_pair"]["orientation_error"] == 0.0
        assert second_reward == -4.123105625617661
        assert second_terminated is False
        assert second_truncated is False
        assert second_info["applied_action"] == {
            "delta_source_pose": {
                "position_delta": [0.0, 1.0, 0.0],
                "orientation_delta": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
            },
            "multi_step": 1,
        }
    finally:
        runtime.stop()


def test_env_real_policy_action_reaches_bridge_and_accumulates(tmp_path: Path) -> None:
    fake_gz = tmp_path / "fake_gz.py"
    world = tmp_path / "world.sdf"
    _write_fake_gz_bridge_script(fake_gz)
    _write_world(world)

    env = GazeboEnv(
        runtime=GazeboRuntime(
            GazeboRuntimeConfig(
                world_path=str(world),
                world_name="test_world",
                timeout=0.2,
                executable=str(fake_gz),
                source_entity_name="robot",
                target_entity_name="task_board",
            )
        )
    )

    try:
        env.reset(seed=12, options={"mode": "env-policy"})
        first_observation, first_reward, first_terminated, first_truncated, first_info = env.step(
            {
                "policy_action": {
                    "position_delta": [1.0, 0.0, 0.0],
                },
                "multi_step": 1,
            }
        )
        assert first_observation["entities_by_name"]["robot"]["position"] == [2.0, 2.0, 3.0]
        assert first_observation["task_geometry"]["tracked_entity_pair"]["relative_position"] == [2.0, 3.0, 3.0]
        assert first_reward == -4.69041575982343
        assert first_terminated is False
        assert first_truncated is False
        assert first_info["applied_action"] == {
            "delta_source_pose": {
                "position_delta": [1.0, 0.0, 0.0],
            },
            "multi_step": 1,
        }

        second_observation, second_reward, second_terminated, second_truncated, second_info = env.step(
            {
                "policy_action": {
                    "position_delta": [0.0, 1.0, 0.0],
                    "orientation_delta": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
                },
                "multi_step": 1,
            }
        )
        assert second_observation["entities_by_name"]["robot"]["position"] == [2.0, 3.0, 3.0]
        assert second_observation["entities_by_name"]["robot"]["orientation"] == [
            0.0,
            0.0,
            0.7071067811865476,
            0.7071067811865476,
        ]
        assert second_observation["task_geometry"]["tracked_entity_pair"]["relative_position"] == [2.0, 2.0, 3.0]
        assert second_observation["task_geometry"]["tracked_entity_pair"]["orientation_error"] == 0.0
        assert second_reward == -4.123105625617661
        assert second_terminated is False
        assert second_truncated is False
        assert second_info["applied_action"] == {
            "delta_source_pose": {
                "position_delta": [0.0, 1.0, 0.0],
                "orientation_delta": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
            },
            "multi_step": 1,
        }
    finally:
        env.close()


def test_env_real_clean_action_normalizes_to_policy_action_and_accumulates(tmp_path: Path) -> None:
    fake_gz = tmp_path / "fake_gz.py"
    world = tmp_path / "world.sdf"
    _write_fake_gz_bridge_script(fake_gz)
    _write_world(world)

    env = GazeboEnv(
        runtime=GazeboRuntime(
            GazeboRuntimeConfig(
                world_path=str(world),
                world_name="test_world",
                timeout=0.2,
                executable=str(fake_gz),
                source_entity_name="robot",
                target_entity_name="task_board",
            )
        )
    )

    try:
        env.reset(seed=34, options={"mode": "env-clean"})
        first_observation, first_reward, first_terminated, first_truncated, first_info = env.step(
            {
                "position_delta": [1.0, 0.0, 0.0],
                "multi_step": 1,
            }
        )
        assert first_observation["entities_by_name"]["robot"]["position"] == [2.0, 2.0, 3.0]
        assert first_observation["task_geometry"]["tracked_entity_pair"]["relative_position"] == [2.0, 3.0, 3.0]
        assert first_reward == -4.69041575982343
        assert first_terminated is False
        assert first_truncated is False
        assert first_info["applied_action"] == {
            "delta_source_pose": {
                "position_delta": [1.0, 0.0, 0.0],
            },
            "multi_step": 1,
        }

        second_observation, second_reward, second_terminated, second_truncated, second_info = env.step(
            {
                "position_delta": [0.0, 1.0, 0.0],
                "orientation_delta": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
                "multi_step": 1,
            }
        )
        assert second_observation["entities_by_name"]["robot"]["position"] == [2.0, 3.0, 3.0]
        assert second_observation["entities_by_name"]["robot"]["orientation"] == [
            0.0,
            0.0,
            0.7071067811865476,
            0.7071067811865476,
        ]
        assert second_observation["task_geometry"]["tracked_entity_pair"]["relative_position"] == [2.0, 2.0, 3.0]
        assert second_observation["task_geometry"]["tracked_entity_pair"]["orientation_error"] == 0.0
        assert second_reward == -4.123105625617661
        assert second_terminated is False
        assert second_truncated is False
        assert second_info["applied_action"] == {
            "delta_source_pose": {
                "position_delta": [0.0, 1.0, 0.0],
                "orientation_delta": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
            },
            "multi_step": 1,
        }
    finally:
        env.close()


def test_env_real_ee_delta_action_normalizes_and_accumulates(tmp_path: Path) -> None:
    fake_gz = tmp_path / "fake_gz.py"
    world = tmp_path / "world.sdf"
    _write_fake_gz_bridge_script(fake_gz)
    _write_world(world)

    env = GazeboEnv(
        runtime=GazeboRuntime(
            GazeboRuntimeConfig(
                world_path=str(world),
                world_name="test_world",
                timeout=0.2,
                executable=str(fake_gz),
                source_entity_name="robot",
                target_entity_name="task_board",
            )
        )
    )

    try:
        env.reset(seed=35, options={"mode": "env-ee"})
        first_observation, first_reward, first_terminated, first_truncated, first_info = env.step(
            {
                "ee_delta_action": {
                    "position_delta": [1.0, 0.0, 0.0],
                },
                "multi_step": 1,
            }
        )
        assert first_observation["entities_by_name"]["robot"]["position"] == [2.0, 2.0, 3.0]
        assert first_observation["task_geometry"]["tracked_entity_pair"]["relative_position"] == [2.0, 3.0, 3.0]
        assert first_reward == -4.69041575982343
        assert first_terminated is False
        assert first_truncated is False
        assert first_info["applied_action"] == {
            "delta_source_pose": {
                "position_delta": [1.0, 0.0, 0.0],
            },
            "multi_step": 1,
        }

        second_observation, second_reward, second_terminated, second_truncated, second_info = env.step(
            {
                "ee_delta_action": {
                    "position_delta": [0.0, 1.0, 0.0],
                    "orientation_delta": [0.0, 0.0, 2.0, 2.0],
                },
                "multi_step": 1,
            }
        )
        assert second_observation["entities_by_name"]["robot"]["position"] == [2.0, 3.0, 3.0]
        assert second_observation["entities_by_name"]["robot"]["orientation"] == [
            0.0,
            0.0,
            0.7071067811865476,
            0.7071067811865476,
        ]
        assert second_observation["task_geometry"]["tracked_entity_pair"]["orientation_error"] == 0.0
        assert second_reward == -4.123105625617661
        assert second_terminated is False
        assert second_truncated is False
        assert second_info["applied_action"] == {
            "delta_source_pose": {
                "position_delta": [0.0, 1.0, 0.0],
                "orientation_delta": [0.0, 0.0, 0.7071067811865475, 0.7071067811865475],
            },
            "multi_step": 1,
        }
    finally:
        env.close()


def test_env_real_ee_delta_action_supports_local_frame_and_position_clipping(tmp_path: Path) -> None:
    fake_gz = tmp_path / "fake_gz.py"
    world = tmp_path / "world.sdf"
    _write_fake_gz_bridge_script(fake_gz)
    _write_world(world)

    env = GazeboEnv(
        runtime=GazeboRuntime(
            GazeboRuntimeConfig(
                world_path=str(world),
                world_name="test_world",
                timeout=0.2,
                executable=str(fake_gz),
                source_entity_name="robot",
                target_entity_name="task_board",
            )
        )
    )

    try:
        env.reset(seed=44, options={"mode": "env-ee-local"})
        orientation_observation, _, _, _, orientation_info = env.step(
            {
                "ee_delta_action": {
                    "orientation_delta": [0.0, 0.0, 2.0, 2.0],
                },
                "multi_step": 1,
            }
        )
        assert orientation_observation["entities_by_name"]["robot"]["orientation"] == [
            0.0,
            0.0,
            0.7071067811865476,
            0.7071067811865476,
        ]
        assert orientation_info["applied_action"] == {
            "delta_source_pose": {
                "orientation_delta": [0.0, 0.0, 0.7071067811865475, 0.7071067811865475],
            },
            "multi_step": 1,
        }

        local_observation, local_reward, local_terminated, local_truncated, local_info = env.step(
            {
                "ee_delta_action": {
                    "position_delta": [3.0, 4.0, 0.0],
                    "frame": "local",
                    "max_position_delta_norm": 2.0,
                },
                "multi_step": 1,
            }
        )
        assert local_observation["entities_by_name"]["robot"]["position"] == [-0.6000000000000005, 3.2000000000000006, 3.0]
        assert local_observation["task_geometry"]["tracked_entity_pair"]["relative_position"] == [
            4.6000000000000005,
            1.7999999999999994,
            3.0,
        ]
        assert local_reward == -5.779273310719955
        assert local_terminated is False
        assert local_truncated is False
        assert local_info["applied_action"] == {
            "delta_source_pose": {
                "position_delta": [-1.6000000000000005, 1.2000000000000006, 0.0],
            },
            "multi_step": 1,
        }
    finally:
        env.close()


def test_runtime_real_instances_share_underlying_world_state(tmp_path: Path) -> None:
    fake_gz = tmp_path / "fake_gz.py"
    world = tmp_path / "world.sdf"
    _write_fake_gz_bridge_script(fake_gz)
    _write_world(world)

    writer_runtime = GazeboRuntime(
        GazeboRuntimeConfig(
            world_path=str(world),
            world_name="test_world",
            timeout=0.2,
            executable=str(fake_gz),
            source_entity_name="robot",
            target_entity_name="task_board",
            orientation_success_threshold=0.2,
        )
    )
    reader_runtime = GazeboRuntime(
        GazeboRuntimeConfig(
            world_path=str(world),
            world_name="test_world",
            timeout=0.2,
            executable=str(fake_gz),
            source_entity_name="robot",
            target_entity_name="task_board",
        )
    )

    writer_runtime.start()
    reader_runtime.start()
    try:
        writer_runtime.reset(seed=10, options={"mode": "shared-world"})
        step_observation, step_reward, step_terminated, step_truncated, step_info = writer_runtime.step(
            {
                "set_entity_pose": {
                    "name": "robot",
                    "position": [5.0, 5.0, 6.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
                "multi_step": 1,
            }
        )
        assert step_observation["task_geometry"]["tracked_entity_pair"]["orientation_success_threshold"] == 0.2
        assert step_observation["task_geometry"]["tracked_entity_pair"]["orientation_success"] is False
        assert step_observation["task_geometry"]["tracked_entity_pair"]["success"] is False
        assert step_reward == -1.0
        assert step_terminated is False
        assert step_truncated is False
        assert step_info["orientation_success_threshold"] == 0.2

        reader_observation, reader_info = reader_runtime.get_observation()
        assert reader_observation["entities_by_name"]["robot"]["position"] == [5.0, 5.0, 6.0]
        assert reader_observation["entities_by_name"]["robot"]["orientation"] == [0.0, 0.0, 0.0, 1.0]
        assert reader_observation["task_geometry"]["tracked_entity_pair"]["orientation_success_threshold"] is None
        assert reader_observation["task_geometry"]["tracked_entity_pair"]["orientation_success"] is True
        assert reader_observation["task_geometry"]["tracked_entity_pair"]["success"] is True
        assert reader_info["observation_topic"] == "/world/test_world/state"
    finally:
        writer_runtime.stop()
        reader_runtime.stop()


def test_gymnasium_wrapper_real_step_preserves_env_behavior(tmp_path: Path) -> None:
    fake_gz = tmp_path / "fake_gz.py"
    world = tmp_path / "world.sdf"
    _write_fake_gz_bridge_script(fake_gz)
    _write_world(world)

    wrapper = GymnasiumGazeboEnv(
        runtime=GazeboRuntime(
            GazeboRuntimeConfig(
                world_path=str(world),
                world_name="test_world",
                timeout=0.2,
                executable=str(fake_gz),
                source_entity_name="robot",
                target_entity_name="task_board",
            )
        )
    )

    try:
        observation, info = wrapper.reset(seed=55, options={"mode": "gym-wrapper"})
        assert observation["world_name"] == "test_world"
        assert observation["step_count"] == 0
        assert observation["entity_count"] == 4
        assert observation["task_geometry"]["tracked_entity_pair"]["relative_position"] == [3.0, 3.0, 3.0]
        assert observation["task_geometry"]["tracked_entity_pair"]["source_orientation"] == [0.0, 0.0, 0.0, 1.0]
        assert wrapper.observation_space.spaces["step_count"].shape == (1,)
        assert wrapper.observation_space.spaces["tracked_entity_pair"].spaces["relative_position"].shape == (3,)
        assert info["seed"] == 55

        first_observation, first_reward, first_terminated, first_truncated, first_info = wrapper.step(
            {
                "position_delta": [1.0, 0.0, 0.0],
            }
        )
        assert first_observation["entities_by_name"]["robot"]["position"] == [2.0, 2.0, 3.0]
        assert first_observation["step_count"] == 1
        assert first_observation["entity_count"] == 4
        assert first_observation["task_geometry"]["tracked_entity_pair"]["relative_position"] == [2.0, 3.0, 3.0]
        assert first_observation["task_geometry"]["tracked_entity_pair"]["distance"] == -first_reward
        assert first_reward == -4.69041575982343
        assert first_terminated is False
        assert first_truncated is False
        assert first_info["applied_action"] == {
            "delta_source_pose": {
                "position_delta": [1.0, 0.0, 0.0],
            },
            "multi_step": 1,
        }

        second_observation, second_reward, second_terminated, second_truncated, second_info = wrapper.step(
            {
                "position_delta": [0.0, 1.0, 0.0],
                "orientation_delta": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
            }
        )
        assert second_observation["entities_by_name"]["robot"]["position"] == [2.0, 3.0, 3.0]
        assert second_observation["entities_by_name"]["robot"]["orientation"] == [
            0.0,
            0.0,
            0.7071067811865476,
            0.7071067811865476,
        ]
        assert second_observation["task_geometry"]["tracked_entity_pair"]["source_orientation"] == [
            0.0,
            0.0,
            0.7071067811865476,
            0.7071067811865476,
        ]
        assert second_observation["task_geometry"]["tracked_entity_pair"]["orientation_error"] == 0.0
        assert second_reward == -4.123105625617661
        assert second_terminated is False
        assert second_truncated is False
        assert second_info["applied_action"] == {
            "delta_source_pose": {
                "position_delta": [0.0, 1.0, 0.0],
                "orientation_delta": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
            },
            "multi_step": 1,
        }
    finally:
        wrapper.close()


def test_gymnasium_wrapper_real_flattened_mode_matches_raw_observation(tmp_path: Path) -> None:
    fake_gz = tmp_path / "fake_gz.py"
    world = tmp_path / "world.sdf"
    _write_fake_gz_bridge_script(fake_gz)
    _write_world(world)

    wrapper = GymnasiumGazeboEnv(
        runtime=GazeboRuntime(
            GazeboRuntimeConfig(
                world_path=str(world),
                world_name="test_world",
                timeout=0.2,
                executable=str(fake_gz),
                source_entity_name="robot",
                target_entity_name="task_board",
            )
        ),
        flatten_observation=True,
    )

    try:
        reset_observation, reset_info = wrapper.reset(seed=56, options={"mode": "gym-wrapper-flat"})
        assert isinstance(reset_observation, list)
        assert len(reset_observation) == 18
        assert wrapper.observation_space.shape == (18,)
        assert reset_observation == wrapper.unwrapped_env.flatten_observation(reset_info["raw_observation"])

        step_observation, step_reward, step_terminated, step_truncated, step_info = wrapper.step(
            {
                "position_delta": [1.0, 0.0, 0.0],
                "orientation_delta": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
            }
        )
        raw_observation = step_info["raw_observation"]
        assert step_observation == wrapper.unwrapped_env.flatten_observation(raw_observation)
        assert step_observation == [
            1.0,
            4.0,
            2.0,
            3.0,
            3.0,
            4.69041575982343,
            0.0,
            0.0,
            0.7071067811865476,
            0.7071067811865476,
            0.0,
            0.0,
            0.707,
            0.707,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        assert step_reward == -4.69041575982343
        assert step_terminated is False
        assert step_truncated is False
        assert step_info["applied_action"] == {
            "delta_source_pose": {
                "position_delta": [1.0, 0.0, 0.0],
                "orientation_delta": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
            },
            "multi_step": 1,
        }
    finally:
        wrapper.close()


def test_runtime_real_joint_delta_bridge_updates_joint_positions_and_tcp_pose(
    tmp_path: Path,
) -> None:
    fake_gz = tmp_path / "fake_gz.py"
    world = tmp_path / "world.sdf"
    _write_fake_gz_bridge_script(fake_gz)
    _write_world(world)

    runtime = GazeboRuntime(
        GazeboRuntimeConfig(
            world_path=str(world),
            world_name="test_world",
            timeout=0.2,
            executable=str(fake_gz),
            source_entity_name="gripper/tcp",
            target_entity_name="task_board",
        )
    )

    runtime.start()
    try:
        reset_observation, _ = runtime.reset(seed=91, options={"mode": "joint-bridge"})
        assert reset_observation["joint_positions"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert reset_observation["entities_by_name"]["gripper/tcp"]["position"] == [1.0, 2.0, 3.0]

        step_observation, reward, terminated, truncated, step_info = runtime.step(
            {
                "joint_position_delta": [0.2, -0.1, 0.3, 0.0, 0.0, 0.0],
                "multi_step": 2,
            }
        )

        assert step_observation["joint_names"] == [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        assert step_observation["joint_positions"] == [0.2, -0.1, 0.3, 0.0, 0.0, 0.0]
        assert step_observation["entities_by_name"]["gripper/tcp"]["position"] == [1.15, 2.3, 3.0]
        assert step_observation["entities_by_name"]["gripper/tcp"]["orientation"] == [0.0, 0.0, 0.0, 1.0]
        assert step_observation["task_geometry"]["tracked_entity_pair"]["source"] == "gripper/tcp"
        assert step_observation["task_geometry"]["tracked_entity_pair"]["relative_position"] == [2.85, 2.7, 3.0]
        assert step_observation["step_count"] == 2
        assert reward == -4.940900727600181
        assert terminated is False
        assert truncated is False
        assert step_info["applied_action"] == {
                "set_joint_positions": {
                    "model_name": "ur5e",
                    "joint_names": [
                        "shoulder_pan_joint",
                        "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ],
                "positions": [0.2, -0.1, 0.3, 0.0, 0.0, 0.0],
            },
            "multi_step": 2,
        }
        assert step_info["joint_target_service"] is not None
        assert '"/world/test_world/joint_target"' in step_info["joint_target_service"]
        assert step_info["pose_service"] is None

        updated_observation, _ = runtime.get_observation()
        assert updated_observation["joint_positions"] == [0.2, -0.1, 0.3, 0.0, 0.0, 0.0]
        assert updated_observation["entities_by_name"]["gripper/tcp"]["position"] == [1.15, 2.3, 3.0]
    finally:
        runtime.stop()
