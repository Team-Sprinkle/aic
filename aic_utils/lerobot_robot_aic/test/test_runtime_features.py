from __future__ import annotations

import numpy as np
import pytest

from lerobot_robot_aic.runtime_features import (
    AICRuntimeFeatureAssembler,
    base_state_from_gazebo_observation,
)
from lerobot_robot_aic.task_encoding import encode_task_vector


def _obs(force: list[float] | None = None) -> dict:
    return {
        "controller": {
            "current_tcp_pose": {"position": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1]},
            "tcp_velocity": {"linear": [4, 5, 6], "angular": [7, 8, 9]},
            "tcp_error": [10, 11, 12, 13, 14, 15],
        },
        "joints": {"position": [16, 17, 18, 19, 20, 21, 22]},
        "wrist_wrench": {"force": force or [23, 24, 25], "torque": [26, 27, 28]},
    }


def test_runtime_feature_assembler_builds_82d_base_contact_task_order() -> None:
    task = encode_task_vector(task_family="sfp_to_nic", target_port_index=1, target_card_index=3)
    assembler = AICRuntimeFeatureAssembler(82, task_vector=task)

    state0 = assembler.assemble_gazebo(_obs(force=[0, 0, 0]))
    state1 = assembler.assemble_gazebo(_obs(force=[2, 0, 0]))

    assert state0.shape == (82,)
    assert np.allclose(state0[:32], base_state_from_gazebo_observation(_obs(force=[0, 0, 0])))
    assert np.allclose(state0[-10:], task)
    assert state0[32] == pytest.approx(-1.0)
    assert state1[32] == pytest.approx(0.0)
    assert state1[36] == pytest.approx(2.0)


def test_runtime_feature_assembler_requires_task_for_task_conditioned_state() -> None:
    assembler = AICRuntimeFeatureAssembler(42)
    with pytest.raises(ValueError, match="task vector"):
        assembler.assemble_gazebo(_obs())
