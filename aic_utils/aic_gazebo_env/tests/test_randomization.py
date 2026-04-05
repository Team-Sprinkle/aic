"""Tests for config-driven task randomization."""

from __future__ import annotations

from aic_gazebo_env import MinimalTask, MinimalTaskSamplerConfig, PoseRandomizationConfig


def test_sampled_configs_are_valid() -> None:
    task = MinimalTask(
        sampler_config=MinimalTaskSamplerConfig(
            board_pose=PoseRandomizationConfig(
                x_range=(-0.02, 0.02),
                y_range=(-0.01, 0.01),
                z_range=(0.0, 0.0),
            ),
            object_pose=PoseRandomizationConfig(
                x_range=(-0.03, 0.03),
                y_range=(-0.04, 0.04),
                z_range=(-0.01, 0.01),
            ),
        )
    )

    state = task.reset(seed=7)

    assert len(state.board_pose["position"]) == 3
    assert len(state.object_pose["position"]) == 3
    assert len(state.target_pose["position"]) == 3
    assert state.board_pose["orientation"] == [0.0, 0.0, 0.0, 1.0]
    assert state.object_pose["orientation"] == [0.0, 0.0, 0.0, 1.0]


def test_distribution_is_bounded() -> None:
    task = MinimalTask(
        sampler_config=MinimalTaskSamplerConfig(
            board_pose=PoseRandomizationConfig(
                x_range=(-0.02, 0.02),
                y_range=(-0.01, 0.01),
                z_range=(0.0, 0.0),
            ),
            object_pose=PoseRandomizationConfig(
                x_range=(-0.03, 0.03),
                y_range=(-0.04, 0.04),
                z_range=(-0.01, 0.01),
            ),
        )
    )

    for seed in range(25):
        state = task.reset(seed=seed)
        board_x, board_y, board_z = state.board_pose["position"]
        object_x, object_y, object_z = state.object_pose["position"]

        assert -0.02 <= board_x <= 0.02
        assert -0.01 <= board_y <= 0.01
        assert board_z == 0.0
        assert board_x - 0.03 <= object_x <= board_x + 0.03
        assert board_y - 0.04 <= object_y <= board_y + 0.04
        assert board_z + 0.49 <= object_z <= board_z + 0.51


def test_seed_reproducibility_works() -> None:
    task = MinimalTask(
        sampler_config=MinimalTaskSamplerConfig(
            board_pose=PoseRandomizationConfig(
                x_range=(-0.05, 0.05),
                y_range=(-0.02, 0.02),
                z_range=(0.0, 0.01),
            ),
            object_pose=PoseRandomizationConfig(
                x_range=(-0.02, 0.02),
                y_range=(-0.02, 0.02),
                z_range=(-0.01, 0.01),
            ),
        )
    )

    first = task.reset(seed=123)
    second = task.reset(seed=123)
    third = task.reset(seed=124)

    assert first == second
    assert first != third
