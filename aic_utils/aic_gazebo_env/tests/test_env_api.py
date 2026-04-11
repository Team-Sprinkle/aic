"""Tests for the public Gazebo environment API."""

from typing import Any

from aic_gazebo_env import GazeboEnv
from aic_gazebo_env.runtime import Runtime


class RecordingRuntime(Runtime):
    """Runtime test double that records the last action passed through the env."""

    def __init__(self) -> None:
        self.last_action: dict[str, Any] | None = None
        self.is_started = False
        self.current_observation: dict[str, Any] = {
            "runtime": "recording",
            "world_name": "test_world",
            "step_count": 0,
            "entity_count": 2,
            "entity_names": ["robot", "task_board"],
            "task_geometry": {
                "tracked_entity_pair": {
                    "source_orientation": [0.0, 0.0, 0.0, 1.0],
                }
            },
        }

    def start(self) -> None:
        self.is_started = True

    def stop(self) -> None:
        self.is_started = False

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del seed, options
        return dict(self.current_observation), {}

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.last_action = dict(action)
        return {"last_action": dict(action)}, 0.0, False, False, {"applied_action": dict(action)}

    def get_observation(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return dict(self.current_observation), {}


def test_reset_returns_observation_and_info() -> None:
    env = GazeboEnv()

    observation, info = env.reset(seed=123, options={"episode": "smoke"})

    assert isinstance(observation, dict)
    assert isinstance(info, dict)
    assert observation["runtime"] == "stub"
    assert observation["step_count"] == 0
    assert info["seed"] == 123
    assert info["options"] == {"episode": "smoke"}


def test_step_returns_gymnasium_shape() -> None:
    env = GazeboEnv()
    env.reset()

    result = env.step({"command": [0.1, -0.2, 0.3]})

    assert isinstance(result, tuple)
    assert len(result) == 5
    observation, reward, terminated, truncated, info = result
    assert isinstance(observation, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert observation["last_action"] == [0.1, -0.2, 0.3]


def test_invalid_action_shape_raises_clear_error() -> None:
    env = GazeboEnv()
    env.reset()

    try:
        env.step({"command": "bad-shape"})
    except ValueError as exc:
        assert "command" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid action shape.")


def test_env_accepts_canonical_policy_action_and_forwards_it() -> None:
    runtime = RecordingRuntime()
    env = GazeboEnv(runtime=runtime)

    observation, reward, terminated, truncated, info = env.step(
        {
            "policy_action": {
                "position_delta": [0.1, 0.0, -0.1],
                "orientation_delta": [0.0, 0.0, 0.0, 1.0],
            },
            "multi_step": 2,
        }
    )

    assert runtime.last_action == {
        "policy_action": {
            "position_delta": [0.1, 0.0, -0.1],
            "orientation_delta": [0.0, 0.0, 0.0, 1.0],
        },
        "multi_step": 2,
    }
    assert observation["last_action"] == runtime.last_action
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["applied_action"] == runtime.last_action


def test_env_accepts_clean_tracked_source_action_and_normalizes_to_policy_action() -> None:
    runtime = RecordingRuntime()
    env = GazeboEnv(runtime=runtime)

    observation, reward, terminated, truncated, info = env.step(
        {
            "position_delta": [0.1, 0.0, -0.1],
            "orientation_delta": [0.0, 0.0, 0.0, 1.0],
            "multi_step": 2,
        }
    )

    assert runtime.last_action == {
        "policy_action": {
            "position_delta": [0.1, 0.0, -0.1],
            "orientation_delta": [0.0, 0.0, 0.0, 1.0],
        },
        "multi_step": 2,
    }
    assert observation["last_action"] == runtime.last_action
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["applied_action"] == runtime.last_action


def test_env_action_spec_describes_preferred_rl_action_contract() -> None:
    env = GazeboEnv(runtime=RecordingRuntime())

    action_spec = env.action_spec

    assert action_spec["format"] == "tracked_source_delta"
    assert action_spec["robot_adapter"]["field"] == "ee_delta_action"
    assert action_spec["robot_adapter"]["optional_fields"]["frame"]["default"] == "world"
    assert action_spec["robot_adapter"]["optional_fields"]["frame"]["allowed_values"] == ["world", "local"]
    assert action_spec["robot_adapter"]["optional_fields"]["max_position_delta_norm"]["minimum"] == 0.0
    assert action_spec["required_any_of"] == ["position_delta", "orientation_delta"]
    assert action_spec["optional_fields"]["position_delta"]["length"] == 3
    assert action_spec["optional_fields"]["orientation_delta"]["length"] == 4
    assert action_spec["optional_fields"]["multi_step"]["default"] == 1


def test_env_observation_spec_describes_stable_real_observation_contract() -> None:
    env = GazeboEnv(runtime=RecordingRuntime())

    observation_spec = env.observation_spec

    assert observation_spec["format"] == "structured_real_observation"
    assert observation_spec["stable_top_level_fields"]["world_name"]["type"] == "str"
    assert observation_spec["stable_top_level_fields"]["step_count"]["type"] == "int"
    assert observation_spec["flattened_view"]["length"] == 18
    assert observation_spec["flattened_view"]["field_order"][0] == "step_count"
    assert observation_spec["flattened_view"]["field_order"][-1] == "tracked_entity_pair.success"
    assert observation_spec["stable_nested_fields"]["task_geometry.tracked_entity_pair.relative_position"]["length"] == 3
    assert observation_spec["stable_nested_fields"]["task_geometry.tracked_entity_pair.orientation_error"]["type"] == "float"
    assert observation_spec["gymnasium_space_fields"]["tracked_entity_pair.success"]["shape"] == (1,)


def test_env_flatten_observation_uses_deterministic_field_order_and_bool_encoding() -> None:
    env = GazeboEnv(runtime=RecordingRuntime())

    observation = {
        "world_name": "test_world",
        "step_count": 2,
        "entity_count": 3,
        "entities_by_name": {
            "robot": {
                "pose": {
                    "position": [2.0, 3.0, 3.0],
                    "orientation": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
                }
            },
            "task_board": {
                "pose": {
                    "position": [4.0, 5.0, 6.0],
                    "orientation": [0.0, 0.0, 0.707, 0.707],
                }
            },
        },
        "task_geometry": {
            "tracked_entity_pair": {
                "relative_position": [2.0, 2.0, 3.0],
                "distance": 4.123105625617661,
                "source_orientation": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
                "target_orientation": [0.0, 0.0, 0.707, 0.707],
                "orientation_error": 0.0,
                "distance_success": False,
                "orientation_success": True,
                "success": False,
            }
        },
    }

    assert env.flatten_observation(observation) == [
        2.0,
        3.0,
        2.0,
        2.0,
        3.0,
        4.123105625617661,
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


def test_env_clean_action_omits_orientation_and_defaults_multi_step() -> None:
    runtime = RecordingRuntime()
    env = GazeboEnv(runtime=runtime)

    observation, reward, terminated, truncated, info = env.step(
        {
            "position_delta": [1, 0, -1],
        }
    )

    assert runtime.last_action == {
        "policy_action": {
            "position_delta": [1.0, 0.0, -1.0],
        },
        "multi_step": 1,
    }
    assert observation["last_action"] == runtime.last_action
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["applied_action"] == runtime.last_action


def test_env_accepts_joint_position_delta_action_and_forwards_it() -> None:
    runtime = RecordingRuntime()
    env = GazeboEnv(runtime=runtime)

    observation, reward, terminated, truncated, info = env.step(
        {
            "joint_position_delta": [0.1, -0.1, 0.2, 0.0, 0.0, 0.0],
            "multi_step": 3,
        }
    )

    assert runtime.last_action == {
        "joint_position_delta": [0.1, -0.1, 0.2, 0.0, 0.0, 0.0],
        "multi_step": 3,
    }
    assert observation["last_action"] == runtime.last_action
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["applied_action"] == runtime.last_action


def test_env_clean_action_accepts_orientation_only() -> None:
    runtime = RecordingRuntime()
    env = GazeboEnv(runtime=runtime)

    env.step(
        {
            "orientation_delta": [0, 0, 0, 1],
        }
    )

    assert runtime.last_action == {
        "policy_action": {
            "orientation_delta": [0.0, 0.0, 0.0, 1.0],
        },
        "multi_step": 1,
    }


def test_env_accepts_ee_delta_action_and_normalizes_to_clean_policy_path() -> None:
    runtime = RecordingRuntime()
    env = GazeboEnv(runtime=runtime)

    observation, reward, terminated, truncated, info = env.step(
        {
            "ee_delta_action": {
                "position_delta": [0.1, 0.0, -0.1],
                "orientation_delta": [0.0, 0.0, 2.0, 2.0],
            },
            "multi_step": 2,
        }
    )

    assert runtime.last_action == {
        "policy_action": {
            "position_delta": [0.1, 0.0, -0.1],
            "orientation_delta": [0.0, 0.0, 0.7071067811865475, 0.7071067811865475],
        },
        "multi_step": 2,
    }
    assert observation["last_action"] == runtime.last_action
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["applied_action"] == runtime.last_action


def test_env_accepts_local_frame_ee_delta_action_and_rotates_to_world_delta() -> None:
    runtime = RecordingRuntime()
    runtime.current_observation["task_geometry"]["tracked_entity_pair"]["source_orientation"] = [
        0.0,
        0.0,
        0.7071067811865476,
        0.7071067811865476,
    ]
    env = GazeboEnv(runtime=runtime)

    env.step(
        {
            "ee_delta_action": {
                "position_delta": [1.0, 0.0, 0.0],
                "frame": "local",
            }
        }
    )

    assert runtime.last_action == {
        "policy_action": {
            "position_delta": [0.0, 1.0, 0.0],
        },
        "multi_step": 1,
    }


def test_env_clips_ee_delta_action_position_norm_before_normalization() -> None:
    runtime = RecordingRuntime()
    env = GazeboEnv(runtime=runtime)

    env.step(
        {
            "ee_delta_action": {
                "position_delta": [3.0, 4.0, 0.0],
                "max_position_delta_norm": 2.0,
            }
        }
    )

    assert runtime.last_action == {
        "policy_action": {
            "position_delta": [1.2000000000000002, 1.6, 0.0],
        },
        "multi_step": 1,
    }


def test_env_rejects_invalid_ee_delta_action_frame() -> None:
    runtime = RecordingRuntime()
    env = GazeboEnv(runtime=runtime)

    try:
        env.step(
            {
                "ee_delta_action": {
                    "position_delta": [0.1, 0.0, 0.0],
                    "frame": "tool",
                }
            }
        )
    except ValueError as exc:
        assert "ee_delta_action.frame" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid ee_delta_action frame.")


def test_env_rejects_zero_quaternion_in_ee_delta_action() -> None:
    runtime = RecordingRuntime()
    env = GazeboEnv(runtime=runtime)

    try:
        env.step(
            {
                "ee_delta_action": {
                    "orientation_delta": [0.0, 0.0, 0.0, 0.0],
                }
            }
        )
    except ValueError as exc:
        assert "zero quaternion" in str(exc)
    else:
        raise AssertionError("Expected ValueError for zero ee_delta_action quaternion.")


def test_env_rejects_invalid_ee_delta_action_clip_value() -> None:
    runtime = RecordingRuntime()
    env = GazeboEnv(runtime=runtime)

    try:
        env.step(
            {
                "ee_delta_action": {
                    "position_delta": [1.0, 0.0, 0.0],
                    "max_position_delta_norm": -1.0,
                }
            }
        )
    except ValueError as exc:
        assert "max_position_delta_norm" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid ee_delta_action clip value.")


def test_env_rejects_non_dict_policy_action() -> None:
    runtime = RecordingRuntime()
    env = GazeboEnv(runtime=runtime)

    try:
        env.step({"policy_action": "bad-shape"})
    except ValueError as exc:
        assert "policy_action" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid policy_action shape.")


def test_env_rejects_invalid_clean_action_position_delta_shape() -> None:
    runtime = RecordingRuntime()
    env = GazeboEnv(runtime=runtime)

    try:
        env.step({"position_delta": [0.1, 0.0]})
    except ValueError as exc:
        assert "position_delta" in str(exc)
        assert "exactly 3" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid position_delta shape.")


def test_env_rejects_invalid_clean_action_multi_step() -> None:
    runtime = RecordingRuntime()
    env = GazeboEnv(runtime=runtime)

    try:
        env.step({"position_delta": [0.1, 0.0, 0.0], "multi_step": 0})
    except ValueError as exc:
        assert "multi_step" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid multi_step.")


def test_env_rejects_mixing_clean_deltas_with_policy_action() -> None:
    runtime = RecordingRuntime()
    env = GazeboEnv(runtime=runtime)

    try:
        env.step(
            {
                "position_delta": [0.1, 0.0, 0.0],
                "policy_action": {"position_delta": [0.2, 0.0, 0.0]},
            }
        )
    except ValueError as exc:
        assert "either" in str(exc)
        assert "policy_action" in str(exc)
    else:
        raise AssertionError("Expected ValueError for ambiguous action shape.")
