#!/usr/bin/env python3
"""Focused smoke test for the real Gazebo joint target bridge path."""

from __future__ import annotations

from pathlib import Path
import tempfile

from aic_gazebo_env import GazeboRuntime, GazeboRuntimeConfig


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
            "            'task_board': {",
            "                'id': 202,",
            "                'position': [4.0, 5.0, 6.0],",
            "                'orientation': [0.0, 0.0, 0.707, 0.707],",
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
            "def _read_state():",
            "    return json.loads(state_file.read_text(encoding='utf-8'))",
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
            "    state = _read_state()",
            "    print(",
            "        f'''world: \"test_world\"",
            "step_count: {state['step_count']}",
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
            "topic: \"{sys.argv[-1]}\"''',",
            "        flush=True,",
            "    )",
            "    sys.exit(0)",
            "",
            "if sys.argv[1:3] == ['service', '-s']:",
            "    service = sys.argv[3]",
            "    request_payload = sys.argv[sys.argv.index('--req') + 1]",
            "    state = _read_state()",
            "    if 'multi_step:' in request_payload:",
            "        state['step_count'] += int(request_payload.split('multi_step:', 1)[1].strip())",
            "    if 'reset:' in request_payload:",
            "        state = _base_state()",
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
            "    print(json.dumps({'service': service, 'request': request_payload, 'success': True}), flush=True)",
            "    sys.exit(0)",
            "",
            "print(f'unexpected argv: {sys.argv}', file=sys.stderr, flush=True)",
            "sys.exit(3)",
        ]
    )
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def _write_world(path: Path) -> None:
    path.write_text(
        "<sdf version='1.9'><world name='test_world'></world></sdf>\n",
        encoding="utf-8",
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="aic_gz_joint_bridge_smoke_") as tmp_dir:
        tmp_path = Path(tmp_dir)
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
            reset_observation, _ = runtime.reset(seed=123, options={"mode": "joint-bridge"})
            print(
                "joint_bridge.reset:",
                {
                    "joint_positions": reset_observation["joint_positions"],
                    "tcp_pose": reset_observation["entities_by_name"]["gripper/tcp"]["pose"],
                },
            )

            step_observation, reward, terminated, truncated, step_info = runtime.step(
                {
                    "joint_position_delta": [0.2, -0.1, 0.3, 0.0, 0.0, 0.0],
                    "multi_step": 2,
                }
            )
            print(
                "joint_bridge.step:",
                {
                    "joint_positions": step_observation["joint_positions"],
                    "tcp_pose": step_observation["entities_by_name"]["gripper/tcp"]["pose"],
                    "tracked_entity_pair": step_observation["task_geometry"]["tracked_entity_pair"],
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "joint_target_service": step_info["joint_target_service"],
                    "pose_service": step_info["pose_service"],
                },
            )
            print("smoke_gazebo_joint_bridge: OK")
        finally:
            runtime.stop()


if __name__ == "__main__":
    main()
