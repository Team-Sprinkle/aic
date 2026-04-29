import math

import numpy as np
import pytest

from gazebo_rl.action import delta_tcp_action_from_array, rotation_vector_to_quaternion_xyzw


def test_action_clipping():
    action = delta_tcp_action_from_array([1, -1, 0.1, 1, -1, 0.1])
    assert np.allclose(action.delta_position_xyz, [0.003, -0.003, 0.003])
    assert np.allclose(action.delta_rotation_xyz, [0.03, -0.03, 0.03])


def test_action_shape_validation():
    with pytest.raises(ValueError):
        delta_tcp_action_from_array([0.0, 1.0])


def test_quaternion_output_sanity():
    quat = rotation_vector_to_quaternion_xyzw([0.0, 0.0, 0.03])
    assert np.isclose(np.linalg.norm(quat), 1.0)
    assert quat[3] > 0.0
    assert math.isclose(quat[2], math.sin(0.015), rel_tol=1e-6)
