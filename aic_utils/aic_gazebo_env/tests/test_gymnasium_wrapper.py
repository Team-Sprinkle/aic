"""Tests for the thin Gymnasium-compatible Gazebo wrapper."""

from typing import Any

from aic_gazebo_env import GazeboEnv, GymnasiumGazeboEnv
from aic_gazebo_env.runtime import Runtime


class RecordingRuntime(Runtime):
    """Runtime test double that records wrapper-forwarded actions."""

    def __init__(self) -> None:
        self.last_action: dict[str, Any] | None = None
        self.source_orientation = [0.0, 0.0, 0.0, 1.0]

    def _observation(self) -> dict[str, Any]:
        return {
            "runtime": "wrapper-recording",
            "world_name": "test_world",
            "step_count": 0,
            "entity_count": 2,
            "entity_names": ["robot", "task_board"],
            "entities_by_name": {
                "robot": {
                    "pose": {
                        "position": [1.0, 2.0, 3.0],
                        "orientation": [0.0, 0.0, 0.0, 1.0],
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
                    "relative_position": [3.0, 3.0, 3.0],
                    "distance": 5.196152422706632,
                    "source_orientation": list(self.source_orientation),
                    "target_orientation": [0.0, 0.0, 0.707, 0.707],
                    "orientation_error": 1.5707963267948966,
                    "distance_success": False,
                    "orientation_success": True,
                    "success": False,
                }
            },
        }

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._observation(), {"seed": seed, "options": options}

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.last_action = dict(action)
        observation = self._observation()
        observation["last_action"] = dict(action)
        return observation, 1.5, False, False, {"applied_action": dict(action)}

    def get_observation(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._observation(), {}


def test_gymnasium_wrapper_reset_and_step_delegate_to_env() -> None:
    wrapper = GymnasiumGazeboEnv(env=GazeboEnv(runtime=RecordingRuntime()))

    observation, info = wrapper.reset(seed=7, options={"mode": "wrapper"})
    step_observation, reward, terminated, truncated, step_info = wrapper.step(
        {
            "position_delta": [1.0, 0.0, 0.0],
        }
    )

    assert observation["runtime"] == "wrapper-recording"
    assert info == {"seed": 7, "options": {"mode": "wrapper"}}
    assert step_observation["last_action"] == {
        "policy_action": {"position_delta": [1.0, 0.0, 0.0]},
        "multi_step": 1,
    }
    assert reward == 1.5
    assert terminated is False
    assert truncated is False
    assert step_info["applied_action"] == step_observation["last_action"]
    wrapper.close()


def test_gymnasium_wrapper_action_space_matches_preferred_env_contract() -> None:
    wrapper = GymnasiumGazeboEnv(env=GazeboEnv(runtime=RecordingRuntime()))

    assert wrapper.action_space.spaces["position_delta"].shape == (3,)
    assert wrapper.action_space.spaces["orientation_delta"].shape == (4,)
    assert wrapper.action_space.spaces["multi_step"].shape == (1,)
    assert wrapper.observation_space.spaces["step_count"].shape == (1,)
    assert wrapper.observation_space.spaces["entity_count"].shape == (1,)
    assert wrapper.observation_space.spaces["tracked_entity_pair"].spaces["relative_position"].shape == (3,)
    assert wrapper.observation_space.spaces["tracked_entity_pair"].spaces["source_orientation"].shape == (4,)
    assert wrapper.unwrapped_env.observation_spec["format"] == "structured_real_observation"
    wrapper.close()


def test_gymnasium_wrapper_flattened_mode_returns_numeric_vector_and_flat_space() -> None:
    wrapper = GymnasiumGazeboEnv(env=GazeboEnv(runtime=RecordingRuntime()), flatten_observation=True)

    observation, info = wrapper.reset(seed=7, options={"mode": "wrapper"})
    step_observation, reward, terminated, truncated, step_info = wrapper.step(
        {
            "position_delta": [1.0, 0.0, 0.0],
        }
    )

    assert isinstance(observation, list)
    assert len(observation) == 18
    assert wrapper.observation_space.shape == (18,)
    assert len(step_observation) == 18
    assert reward == 1.5
    assert terminated is False
    assert truncated is False
    assert "raw_observation" in info
    assert "raw_observation" in step_info
    assert step_info["raw_observation"]["last_action"] == {
        "policy_action": {"position_delta": [1.0, 0.0, 0.0]},
        "multi_step": 1,
    }
    wrapper.close()


def test_gymnasium_wrapper_accepts_ee_delta_action_adapter() -> None:
    wrapper = GymnasiumGazeboEnv(env=GazeboEnv(runtime=RecordingRuntime()))

    step_observation, reward, terminated, truncated, step_info = wrapper.step(
        {
            "ee_delta_action": {
                "position_delta": [1.0, 0.0, 0.0],
                "orientation_delta": [0.0, 0.0, 2.0, 2.0],
            }
        }
    )

    assert step_observation["last_action"] == {
        "policy_action": {
            "position_delta": [1.0, 0.0, 0.0],
            "orientation_delta": [0.0, 0.0, 0.7071067811865475, 0.7071067811865475],
        },
        "multi_step": 1,
    }
    assert reward == 1.5
    assert terminated is False
    assert truncated is False
    assert step_info["applied_action"] == step_observation["last_action"]
    wrapper.close()


def test_gymnasium_wrapper_accepts_local_frame_ee_delta_action_adapter() -> None:
    runtime = RecordingRuntime()
    runtime.source_orientation = [0.0, 0.0, 0.7071067811865476, 0.7071067811865476]
    wrapper = GymnasiumGazeboEnv(env=GazeboEnv(runtime=runtime))

    step_observation, reward, terminated, truncated, step_info = wrapper.step(
        {
            "ee_delta_action": {
                "position_delta": [1.0, 0.0, 0.0],
                "frame": "local",
            }
        }
    )

    assert step_observation["last_action"] == {
        "policy_action": {
            "position_delta": [0.0, 1.0, 0.0],
        },
        "multi_step": 1,
    }
    assert reward == 1.5
    assert terminated is False
    assert truncated is False
    assert step_info["applied_action"] == step_observation["last_action"]
    wrapper.close()
