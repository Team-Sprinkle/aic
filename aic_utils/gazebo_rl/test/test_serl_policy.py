from __future__ import annotations

import numpy as np

from gazebo_rl.serl_policy import lowdim_state_from_gazebo_observation


def test_lowdim_state_from_gazebo_observation_matches_dataset_order():
    obs = {
        "controller": {
            "current_tcp_pose": {
                "position": [1, 2, 3],
                "orientation_xyzw": [0, 0, 0, 1],
            },
            "tcp_velocity": {"linear": [4, 5, 6], "angular": [7, 8, 9]},
            "tcp_error": [10, 11, 12, 13, 14, 15],
        },
        "joints": {"position": [16, 17, 18, 19, 20, 21, 22]},
        "wrist_wrench": {"force": [23, 24, 25], "torque": [26, 27, 28]},
    }
    state = lowdim_state_from_gazebo_observation(obs)
    assert state.shape == (32,)
    assert np.allclose(state[:7], [1, 2, 3, 0, 0, 0, 1])
    assert np.allclose(state[-6:], [23, 24, 25, 26, 27, 28])
