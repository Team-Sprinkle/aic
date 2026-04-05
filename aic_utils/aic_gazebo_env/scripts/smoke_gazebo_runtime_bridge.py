#!/usr/bin/env python3
"""Manual smoke test for the real Gazebo runtime bridge path."""

from __future__ import annotations

from pathlib import Path
import tempfile

from aic_gazebo_env import GazeboEnv, GazeboRuntime, GazeboRuntimeConfig, GymnasiumGazeboEnv


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
            "if not state_file.exists():",
            "    state_file.write_text(",
            "        json.dumps(",
            "            {",
            "                'step_count': 0,",
            "                'entities': {",
            "                    'robot': {",
            "                        'id': 101,",
            "                        'position': [1.0, 2.0, 3.0],",
            "                        'orientation': [0.0, 0.0, 0.0, 1.0],",
            "                    },",
            "                    'task_board': {",
            "                        'id': 202,",
            "                        'position': [4.0, 5.0, 6.0],",
            "                        'orientation': [0.0, 0.0, 0.707, 0.707],",
            "                    },",
            "                    'inspection_target': {",
            "                        'id': 303,",
            "                        'position': [10.0, 12.0, 14.0],",
            "                        'orientation': [0.0, 0.0, 0.0, 1.0],",
            "                    },",
            "                },",
            "            }",
            "        ),",
            "        encoding='utf-8',",
            "    )",
            "",
            "def _read_state():",
            "    return json.loads(state_file.read_text(encoding='utf-8'))",
            "",
            "def _write_state(state):",
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
            "if sys.argv[1:3] == ['topic', '-e']:",
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
            "topic: \"{sys.argv[-1]}\"''',",
            "        flush=True,",
            "    )",
            "    sys.exit(0)",
            "",
            "if sys.argv[1:3] == ['service', '-s']:",
            "    request_payload = sys.argv[sys.argv.index('--req') + 1]",
            "    state = _read_state()",
            "    if 'multi_step:' in request_payload:",
            "        increment = int(request_payload.split('multi_step:', 1)[1].strip())",
            "        state['step_count'] += increment",
            "    if 'reset:' in request_payload:",
            "        state = {",
            "            'step_count': 0,",
            "            'entities': {",
            "                'robot': {",
            "                    'id': 101,",
            "                    'position': [1.0, 2.0, 3.0],",
            "                    'orientation': [0.0, 0.0, 0.0, 1.0],",
            "                },",
            "                'task_board': {",
            "                    'id': 202,",
            "                    'position': [4.0, 5.0, 6.0],",
            "                    'orientation': [0.0, 0.0, 0.707, 0.707],",
            "                },",
            "                'inspection_target': {",
            "                    'id': 303,",
            "                    'position': [10.0, 12.0, 14.0],",
            "                    'orientation': [0.0, 0.0, 0.0, 1.0],",
            "                },",
            "            },",
            "        }",
            "    if sys.argv[3].endswith('/set_pose'):",
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
            "    _write_state(state)",
            "    print(",
            "        json.dumps(",
            "            {",
            "                'service': sys.argv[3],",
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


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="aic_gz_runtime_smoke_") as tmp_dir:
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
                source_entity_name="robot",
                target_entity_name="task_board",
                max_episode_steps=4,
            )
        )

        runtime.start()
        try:
            observation, info = runtime.get_observation()
            print(
                "default_runtime.observation:",
                {
                    "configured_pair": {
                        "source": runtime.config.source_entity_name,
                        "target": runtime.config.target_entity_name,
                    },
                    "max_episode_steps": runtime.config.max_episode_steps,
                    "world_name": observation["world_name"],
                    "step_count": observation["step_count"],
                    "entity_count": observation["entity_count"],
                    "entity_names": observation["entity_names"],
                    "robot_pose": observation["entities_by_name"]["robot"]["pose"],
                    "task_board_pose": observation["entities_by_name"]["task_board"]["pose"],
                    "tracked_entity_pair": observation["task_geometry"]["tracked_entity_pair"],
                    "orientation_success_threshold": runtime.config.orientation_success_threshold,
                    "observation_topic": info["observation_topic"],
                },
            )

            reset_observation, reset_info = runtime.reset(
                seed=123,
                options={"mode": "smoke"},
            )
            print(
                "default_runtime.reset:",
                {
                    "configured_pair": {
                        "source": runtime.config.source_entity_name,
                        "target": runtime.config.target_entity_name,
                    },
                    "max_episode_steps": runtime.config.max_episode_steps,
                    "world_name": reset_observation["world_name"],
                    "step_count": reset_observation["step_count"],
                    "entity_count": reset_observation["entity_count"],
                    "robot_pose": reset_observation["entities_by_name"]["robot"]["pose"],
                    "tracked_entity_pair": reset_observation["task_geometry"]["tracked_entity_pair"],
                    "orientation_success_threshold": runtime.config.orientation_success_threshold,
                    "seed": reset_info["seed"],
                    "options": reset_info["options"],
                    "reset_service": reset_info["reset_service"],
                },
            )

            step_observation, reward, terminated, truncated, step_info = runtime.step(
                {
                    "set_entity_position": {
                        "name": "robot",
                        "position": [7.0, 8.0, 9.0],
                    },
                    "multi_step": 2,
                }
            )
            print(
                "default_runtime.step_far:",
                {
                    "configured_pair": {
                        "source": runtime.config.source_entity_name,
                        "target": runtime.config.target_entity_name,
                    },
                    "max_episode_steps": runtime.config.max_episode_steps,
                    "world_name": step_observation["world_name"],
                    "step_count": step_observation["step_count"],
                    "entity_count": step_observation["entity_count"],
                    "entity_names": step_observation["entity_names"],
                    "robot_pose": step_observation["entities_by_name"]["robot"]["pose"],
                    "task_board_pose": step_observation["entities_by_name"]["task_board"]["pose"],
                    "tracked_entity_pair": step_observation["task_geometry"]["tracked_entity_pair"],
                    "orientation_success_threshold": runtime.config.orientation_success_threshold,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "multi_step": step_info["multi_step"],
                    "pose_service": step_info["pose_service"],
                },
            )

            success_observation, success_reward, success_terminated, success_truncated, success_info = runtime.step(
                {
                    "set_entity_position": {
                        "name": "robot",
                        "position": [4.0, 5.0, 6.0],
                    },
                    "multi_step": 1,
                }
            )
            print(
                "default_runtime.step_success:",
                {
                    "configured_pair": {
                        "source": runtime.config.source_entity_name,
                        "target": runtime.config.target_entity_name,
                    },
                    "max_episode_steps": runtime.config.max_episode_steps,
                    "world_name": success_observation["world_name"],
                    "step_count": success_observation["step_count"],
                    "entity_count": success_observation["entity_count"],
                    "entity_names": success_observation["entity_names"],
                    "robot_pose": success_observation["entities_by_name"]["robot"]["pose"],
                    "task_board_pose": success_observation["entities_by_name"]["task_board"]["pose"],
                    "tracked_entity_pair": success_observation["task_geometry"]["tracked_entity_pair"],
                    "orientation_success_threshold": runtime.config.orientation_success_threshold,
                    "reward": success_reward,
                    "terminated": success_terminated,
                    "truncated": success_truncated,
                    "multi_step": success_info["multi_step"],
                    "pose_service": success_info["pose_service"],
                },
            )

            orientation_runtime = GazeboRuntime(
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

            orientation_runtime.start()
            try:
                orientation_runtime.reset(seed=222, options={"mode": "orientation"})
                orientation_observation, orientation_reward, orientation_terminated, orientation_truncated, orientation_info = orientation_runtime.step(
                    {
                        "set_entity_pose": {
                            "name": "robot",
                            "position": [4.0, 5.0, 6.0],
                            "orientation": [0.0, 0.0, 0.707, 0.707],
                        },
                        "multi_step": 1,
                    }
                )
                print(
                    "orientation_runtime.step_pose_success:",
                    {
                        "configured_pair": {
                            "source": orientation_runtime.config.source_entity_name,
                            "target": orientation_runtime.config.target_entity_name,
                        },
                        "orientation_success_threshold": orientation_runtime.config.orientation_success_threshold,
                        "world_name": orientation_observation["world_name"],
                        "step_count": orientation_observation["step_count"],
                        "robot_pose": orientation_observation["entities_by_name"]["robot"]["pose"],
                        "task_board_pose": orientation_observation["entities_by_name"]["task_board"]["pose"],
                        "tracked_entity_pair": orientation_observation["task_geometry"]["tracked_entity_pair"],
                        "reward": orientation_reward,
                        "terminated": orientation_terminated,
                        "truncated": orientation_truncated,
                        "pose_service": orientation_info["pose_service"],
                    },
                )
            finally:
                orientation_runtime.stop()

            delta_runtime = GazeboRuntime(
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

            delta_runtime.start()
            try:
                delta_runtime.reset(seed=333, options={"mode": "delta"})
                delta_runtime.step(
                    {
                        "set_entity_pose": {
                            "name": "robot",
                            "position": [4.0, 5.0, 6.0],
                            "orientation": [0.0, 0.0, 0.707, 0.707],
                        },
                        "multi_step": 1,
                    }
                )
                delta_observation, delta_reward, delta_terminated, delta_truncated, delta_info = delta_runtime.step(
                    {
                        "delta_source_pose": {
                            "position_delta": [1.0, 0.0, 0.0],
                            "orientation_delta": [0.0, 0.0, -0.7071067811865476, 0.7071067811865476],
                        },
                        "multi_step": 1,
                    }
                )
                print(
                    "delta_runtime.step_delta:",
                    {
                        "configured_pair": {
                            "source": delta_runtime.config.source_entity_name,
                            "target": delta_runtime.config.target_entity_name,
                        },
                        "orientation_success_threshold": delta_runtime.config.orientation_success_threshold,
                        "world_name": delta_observation["world_name"],
                        "step_count": delta_observation["step_count"],
                        "robot_pose": delta_observation["entities_by_name"]["robot"]["pose"],
                        "task_board_pose": delta_observation["entities_by_name"]["task_board"]["pose"],
                        "tracked_entity_pair": delta_observation["task_geometry"]["tracked_entity_pair"],
                        "reward": delta_reward,
                        "terminated": delta_terminated,
                        "truncated": delta_truncated,
                        "pose_service": delta_info["pose_service"],
                    },
                )

                delta_updated_observation, delta_updated_info = delta_runtime.get_observation()
                print(
                    "delta_runtime.get_observation_after_delta:",
                    {
                        "configured_pair": {
                            "source": delta_runtime.config.source_entity_name,
                            "target": delta_runtime.config.target_entity_name,
                        },
                        "orientation_success_threshold": delta_runtime.config.orientation_success_threshold,
                        "world_name": delta_updated_observation["world_name"],
                        "step_count": delta_updated_observation["step_count"],
                        "robot_pose": delta_updated_observation["entities_by_name"]["robot"]["pose"],
                        "task_board_pose": delta_updated_observation["entities_by_name"]["task_board"]["pose"],
                        "tracked_entity_pair": delta_updated_observation["task_geometry"]["tracked_entity_pair"],
                        "matches_delta_step": delta_updated_observation == delta_observation,
                        "observation_topic": delta_updated_info["observation_topic"],
                    },
                )
            finally:
                delta_runtime.stop()

            policy_env = GazeboEnv(
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
                policy_env.reset(seed=444, options={"mode": "policy"})
                print(
                    "policy_env.action_spec:",
                    {
                        "format": policy_env.action_spec["format"],
                        "required_any_of": policy_env.action_spec["required_any_of"],
                        "multi_step_default": policy_env.action_spec["optional_fields"]["multi_step"]["default"],
                    },
                )
                print(
                    "policy_env.observation_spec:",
                    {
                        "format": policy_env.observation_spec["format"],
                        "top_level_fields": list(policy_env.observation_spec["stable_top_level_fields"].keys()),
                        "flattened_field_order": policy_env.observation_spec["flattened_view"]["field_order"],
                        "tracked_pair_fields": [
                            "relative_position",
                            "distance",
                            "source_orientation",
                            "target_orientation",
                            "orientation_error",
                            "distance_success",
                            "orientation_success",
                            "success",
                        ],
                    },
                )
                position_only_observation, position_only_reward, position_only_terminated, position_only_truncated, position_only_info = policy_env.step(
                    {
                        "position_delta": [1.0, 0.0, 0.0],
                    }
                )
                print(
                    "policy_env.step_position_only_action:",
                    {
                        "configured_pair": {
                            "source": "robot",
                            "target": "task_board",
                        },
                        "world_name": position_only_observation["world_name"],
                        "step_count": position_only_observation["step_count"],
                        "robot_pose": position_only_observation["entities_by_name"]["robot"]["pose"],
                        "task_board_pose": position_only_observation["entities_by_name"]["task_board"]["pose"],
                        "tracked_entity_pair": position_only_observation["task_geometry"]["tracked_entity_pair"],
                        "reward": position_only_reward,
                        "terminated": position_only_terminated,
                        "truncated": position_only_truncated,
                        "applied_action": position_only_info["applied_action"],
                        "pose_service": position_only_info["pose_service"],
                        "flattened_observation": policy_env.flatten_observation(position_only_observation),
                    },
                )
                policy_observation, policy_reward, policy_terminated, policy_truncated, policy_info = policy_env.step(
                    {
                        "position_delta": [0.0, 1.0, 0.0],
                        "orientation_delta": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
                        "multi_step": 1,
                    }
                )
                print(
                    "policy_env.step_position_and_orientation_action:",
                    {
                        "configured_pair": {
                            "source": "robot",
                            "target": "task_board",
                        },
                        "world_name": policy_observation["world_name"],
                        "step_count": policy_observation["step_count"],
                        "robot_pose": policy_observation["entities_by_name"]["robot"]["pose"],
                        "task_board_pose": policy_observation["entities_by_name"]["task_board"]["pose"],
                        "tracked_entity_pair": policy_observation["task_geometry"]["tracked_entity_pair"],
                        "reward": policy_reward,
                        "terminated": policy_terminated,
                        "truncated": policy_truncated,
                        "applied_action": policy_info["applied_action"],
                        "pose_service": policy_info["pose_service"],
                        "flattened_observation": policy_env.flatten_observation(policy_observation),
                    },
                )

                ee_adapter_observation, ee_adapter_reward, ee_adapter_terminated, ee_adapter_truncated, ee_adapter_info = policy_env.step(
                    {
                        "ee_delta_action": {
                            "position_delta": [0.0, 0.0, 0.0],
                            "orientation_delta": [0.0, 0.0, 2.0, 2.0],
                            "frame": "world",
                        },
                        "multi_step": 1,
                    }
                )
                print(
                    "policy_env.step_ee_delta_action:",
                    {
                        "configured_pair": {
                            "source": "robot",
                            "target": "task_board",
                        },
                        "frame": "world",
                        "world_name": ee_adapter_observation["world_name"],
                        "step_count": ee_adapter_observation["step_count"],
                        "robot_pose": ee_adapter_observation["entities_by_name"]["robot"]["pose"],
                        "task_board_pose": ee_adapter_observation["entities_by_name"]["task_board"]["pose"],
                        "tracked_entity_pair": ee_adapter_observation["task_geometry"]["tracked_entity_pair"],
                        "reward": ee_adapter_reward,
                        "terminated": ee_adapter_terminated,
                        "truncated": ee_adapter_truncated,
                        "applied_action": ee_adapter_info["applied_action"],
                        "flattened_observation": policy_env.flatten_observation(ee_adapter_observation),
                    },
                )

                ee_local_observation, ee_local_reward, ee_local_terminated, ee_local_truncated, ee_local_info = policy_env.step(
                    {
                        "ee_delta_action": {
                            "position_delta": [3.0, 4.0, 0.0],
                            "frame": "local",
                            "max_position_delta_norm": 2.0,
                        },
                        "multi_step": 1,
                    }
                )
                print(
                    "policy_env.step_ee_delta_action_local_clipped:",
                    {
                        "configured_pair": {
                            "source": "robot",
                            "target": "task_board",
                        },
                        "frame": "local",
                        "max_position_delta_norm": 2.0,
                        "world_name": ee_local_observation["world_name"],
                        "step_count": ee_local_observation["step_count"],
                        "robot_pose": ee_local_observation["entities_by_name"]["robot"]["pose"],
                        "task_board_pose": ee_local_observation["entities_by_name"]["task_board"]["pose"],
                        "tracked_entity_pair": ee_local_observation["task_geometry"]["tracked_entity_pair"],
                        "reward": ee_local_reward,
                        "terminated": ee_local_terminated,
                        "truncated": ee_local_truncated,
                        "applied_action": ee_local_info["applied_action"],
                        "flattened_observation": policy_env.flatten_observation(ee_local_observation),
                    },
                )
            finally:
                policy_env.close()

            updated_observation, updated_info = runtime.get_observation()
            print(
                "default_runtime.get_observation_after_cross_runtime_mutation:",
                {
                    "configured_pair": {
                        "source": runtime.config.source_entity_name,
                        "target": runtime.config.target_entity_name,
                    },
                    "max_episode_steps": runtime.config.max_episode_steps,
                    "world_name": updated_observation["world_name"],
                    "step_count": updated_observation["step_count"],
                    "entity_count": updated_observation["entity_count"],
                    "entity_names": updated_observation["entity_names"],
                    "robot_pose": updated_observation["entities_by_name"]["robot"]["pose"],
                    "task_board_pose": updated_observation["entities_by_name"]["task_board"]["pose"],
                    "tracked_entity_pair": updated_observation["task_geometry"]["tracked_entity_pair"],
                    "orientation_success_threshold": runtime.config.orientation_success_threshold,
                    "shared_world_note": "default_runtime is reading the shared world after policy_env mutated it",
                    "observation_topic": updated_info["observation_topic"],
                    "state_text_contains_step_count": "step_count: 3" in updated_info["state_text"],
                },
            )

            gym_wrapper = GymnasiumGazeboEnv(
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
                gym_wrapper_observation, gym_wrapper_info = gym_wrapper.reset(
                    seed=555,
                    options={"mode": "gym-wrapper"},
                )
                print(
                    "gym_wrapper.reset:",
                    {
                        "world_name": gym_wrapper_observation["world_name"],
                        "step_count": gym_wrapper_observation["step_count"],
                        "action_space_keys": list(gym_wrapper.action_space.spaces.keys()),
                        "observation_space_keys": list(gym_wrapper.observation_space.spaces.keys()),
                        "tracked_pair_observation_space_keys": list(
                            gym_wrapper.observation_space.spaces["tracked_entity_pair"].spaces.keys()
                        ),
                        "info": gym_wrapper_info,
                    },
                )
                gym_wrapper_step_observation, gym_wrapper_reward, gym_wrapper_terminated, gym_wrapper_truncated, gym_wrapper_step_info = gym_wrapper.step(
                    {
                        "position_delta": [1.0, 0.0, 0.0],
                        "orientation_delta": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
                    }
                )
                print(
                    "gym_wrapper.step_clean_action:",
                    {
                        "world_name": gym_wrapper_step_observation["world_name"],
                        "step_count": gym_wrapper_step_observation["step_count"],
                        "robot_pose": gym_wrapper_step_observation["entities_by_name"]["robot"]["pose"],
                        "task_board_pose": gym_wrapper_step_observation["entities_by_name"]["task_board"]["pose"],
                        "tracked_entity_pair": gym_wrapper_step_observation["task_geometry"]["tracked_entity_pair"],
                        "reward": gym_wrapper_reward,
                        "terminated": gym_wrapper_terminated,
                        "truncated": gym_wrapper_truncated,
                        "applied_action": gym_wrapper_step_info["applied_action"],
                    },
                )
            finally:
                gym_wrapper.close()

            gym_wrapper_flat = GymnasiumGazeboEnv(
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
                gym_wrapper_flat_observation, gym_wrapper_flat_info = gym_wrapper_flat.reset(
                    seed=556,
                    options={"mode": "gym-wrapper-flat"},
                )
                print(
                    "gym_wrapper_flat.reset:",
                    {
                        "observation_space_shape": gym_wrapper_flat.observation_space.shape,
                        "flattened_observation": gym_wrapper_flat_observation,
                        "raw_observation_world_name": gym_wrapper_flat_info["raw_observation"]["world_name"],
                    },
                )
                gym_wrapper_flat_step_observation, gym_wrapper_flat_reward, gym_wrapper_flat_terminated, gym_wrapper_flat_truncated, gym_wrapper_flat_step_info = gym_wrapper_flat.step(
                    {
                        "position_delta": [1.0, 0.0, 0.0],
                        "orientation_delta": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
                    }
                )
                print(
                    "gym_wrapper_flat.step_clean_action:",
                    {
                        "observation_space_shape": gym_wrapper_flat.observation_space.shape,
                        "flattened_observation": gym_wrapper_flat_step_observation,
                        "reward": gym_wrapper_flat_reward,
                        "terminated": gym_wrapper_flat_terminated,
                        "truncated": gym_wrapper_flat_truncated,
                        "raw_observation": gym_wrapper_flat_step_info["raw_observation"]["task_geometry"]["tracked_entity_pair"],
                        "applied_action": gym_wrapper_flat_step_info["applied_action"],
                    },
                )
            finally:
                gym_wrapper_flat.close()
        finally:
            runtime.stop()

        truncation_runtime = GazeboRuntime(
            GazeboRuntimeConfig(
                world_path=str(world),
                world_name="test_world",
                timeout=0.2,
                executable=str(fake_gz),
                source_entity_name="robot",
                target_entity_name="task_board",
                max_episode_steps=2,
            )
        )

        truncation_runtime.start()
        try:
            truncation_runtime.reset(seed=456, options={"mode": "truncate"})
            truncation_observation, truncation_reward, truncation_terminated, truncation_truncated, truncation_info = truncation_runtime.step(
                {
                    "set_entity_position": {
                        "name": "robot",
                        "position": [7.0, 8.0, 9.0],
                    },
                    "multi_step": 2,
                }
            )
            print(
                "truncation_runtime.step_budget_limit:",
                {
                    "configured_pair": {
                        "source": truncation_runtime.config.source_entity_name,
                        "target": truncation_runtime.config.target_entity_name,
                    },
                    "max_episode_steps": truncation_runtime.config.max_episode_steps,
                    "world_name": truncation_observation["world_name"],
                    "step_count": truncation_observation["step_count"],
                    "tracked_entity_pair": truncation_observation["task_geometry"]["tracked_entity_pair"],
                    "reward": truncation_reward,
                    "terminated": truncation_terminated,
                    "truncated": truncation_truncated,
                    "multi_step": truncation_info["multi_step"],
                },
            )
        finally:
            truncation_runtime.stop()

    print("smoke_gazebo_runtime_bridge: OK")


if __name__ == "__main__":
    main()
