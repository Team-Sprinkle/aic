"""Tests for official-score-aligned helpers used by the Gazebo bridge."""

from aic_gazebo_env import OfficialTier3TrackedPairScorer
from aic_gazebo_env.gazebo_client import GazeboCliClient, GazeboCliClientConfig


def test_official_tier3_tracked_pair_score_matches_proximity_shape() -> None:
    scorer = OfficialTier3TrackedPairScorer()

    score, details = scorer.score(
        tracked_pair={"distance": 0.0, "success": True},
        initial_distance=2.0,
    )

    assert score == 25.0
    assert details["max_distance"] == 1.0
    assert details["mode"] == "official_tier3_tracked_pair"


def test_official_tier3_tracked_pair_score_falls_to_zero_outside_radius() -> None:
    scorer = OfficialTier3TrackedPairScorer()

    score, details = scorer.score(
        tracked_pair={"distance": 1.25, "success": False},
        initial_distance=2.0,
    )

    assert score == 0.0
    assert details["distance"] == 1.25


def test_cli_client_can_use_official_tier3_reward_mode() -> None:
    client = GazeboCliClient(
        GazeboCliClientConfig(
            executable="/bin/true",
            world_path="/tmp/fake_world.sdf",
            timeout=0.1,
            reward_mode="official_tier3",
        )
    )
    observation = {
        "task_geometry": {
            "tracked_entity_pair": {
                "distance": 0.25,
                "success": False,
            }
        }
    }
    client._initial_tracked_distance = 2.0

    reward, terminated, truncated, details = client._compute_step_outcome(observation)

    assert reward == 18.75
    assert terminated is False
    assert truncated is False
    assert details["mode"] == "official_tier3_tracked_pair"


def test_cli_client_defaults_to_heuristic_reward_mode() -> None:
    client = GazeboCliClient(
        GazeboCliClientConfig(
            executable="/bin/true",
            world_path="/tmp/fake_world.sdf",
            timeout=0.1,
        )
    )
    observation = {
        "task_geometry": {
            "tracked_entity_pair": {
                "distance": 0.25,
                "success": True,
            }
        }
    }

    reward, terminated, truncated, details = client._compute_step_outcome(observation)

    assert reward == 9.75
    assert terminated is True
    assert truncated is False
    assert details["mode"] == "heuristic"
