"""Tests for the stable RL-facing Gazebo wrapper."""

from __future__ import annotations

from typing import Any

from aic_gazebo_env import StableRLEnvConfig, StableRLGazeboEnv, training_api_report
from aic_gazebo_env.rl import EpisodeMonitor
from aic_gazebo_env.runtime import Runtime


class RecordingRuntime(Runtime):
    def __init__(self) -> None:
        self.step_count = 0
        self.last_action: dict[str, Any] | None = None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def _observation(self) -> dict[str, Any]:
        return {
            "world_name": "test_world",
            "step_count": self.step_count,
            "entity_count": 2,
            "joint_count": 6,
            "task_geometry": {
                "tracked_entity_pair": {
                    "relative_position": [1.0, 2.0, 3.0],
                    "distance": 3.7416573867739413,
                    "source_orientation": [0.0, 0.0, 0.0, 1.0],
                    "target_orientation": [0.0, 0.0, 0.0, 1.0],
                    "orientation_error": 0.0,
                    "distance_success": False,
                    "orientation_success": True,
                    "success": self.step_count >= 2,
                }
            },
        }

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.step_count = 0
        return self._observation(), {"seed": seed, "options": options}

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.last_action = dict(action)
        self.step_count += 1
        return self._observation(), 1.25, False, False, {"applied_action": dict(action)}

    def get_observation(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._observation(), {}


def test_stable_rl_env_reset_and_step_return_flat_observation() -> None:
    env = StableRLGazeboEnv(runtime=RecordingRuntime())

    observation, info = env.reset(seed=3, options={"mode": "rl"})
    step_obs, reward, terminated, truncated, step_info = env.step([0.5, 0.0, -0.5])

    assert len(observation) == 18
    assert len(step_obs) == 18
    assert reward == 1.25
    assert terminated is False
    assert truncated is False
    assert info["rl_api"] == "position_delta_3d"
    assert step_info["translated_action"] == {
        "position_delta": [0.005, 0.0, -0.005],
        "multi_step": 1,
    }
    env.close()


def test_stable_rl_env_clips_and_scales_actions() -> None:
    runtime = RecordingRuntime()
    env = StableRLGazeboEnv(
        runtime=runtime,
        config=StableRLEnvConfig(action_clip=1.0, max_position_delta=0.02, multi_step=4),
    )

    env.reset()
    env.step([2.0, -3.0, 0.5])

    assert runtime.last_action == {
        "policy_action": {"position_delta": [0.02, -0.02, 0.01]},
        "multi_step": 4,
    }
    env.close()


def test_stable_rl_env_time_limit_sets_truncation_flag() -> None:
    env = StableRLGazeboEnv(
        runtime=RecordingRuntime(),
        config=StableRLEnvConfig(episode_step_limit=2),
    )

    env.reset()
    env.step([0.0, 0.0, 0.0])
    _, _, _, truncated, info = env.step([0.0, 0.0, 0.0])

    assert truncated is True
    assert info["time_limit_reached"] is True
    env.close()


def test_episode_monitor_finishes_with_summary() -> None:
    monitor = EpisodeMonitor()
    monitor.step(reward=1.0, success=False)
    monitor.step(reward=2.0, success=True)

    summary = monitor.finish()

    assert summary["episode_reward"] == 3.0
    assert summary["episode_length"] == 2
    assert summary["episode_success"] is True
    assert summary["steps_per_second"] is not None


def test_training_api_report_matches_frozen_contract() -> None:
    report = training_api_report(StableRLEnvConfig(multi_step=8, episode_step_limit=128))

    assert report["action_mode"] == "position_delta_3d"
    assert report["observation_mode"] == "flat_tracked_pair_v1"
    assert report["action_shape"] == (3,)
    assert report["flattened_observation_length"] == 18
    assert report["multi_step"] == 8
    assert report["episode_step_limit"] == 128
