from __future__ import annotations

import math
import unittest

import numpy as np

from aic_gym_gz.runtime import (
    _angular_delta_to_quaternion,
    _parse_sim_time_seconds,
    _pose_dict_to_array,
)


class LiveRuntimeHelpersTest(unittest.TestCase):
    def test_parse_sim_time_seconds_from_state_text(self) -> None:
        payload = """
stats {
  sim_time {
    sec: 12
    nsec: 345000000
  }
}
"""
        self.assertAlmostEqual(_parse_sim_time_seconds(payload), 12.345)

    def test_pose_dict_to_array_extracts_yaw(self) -> None:
        pose = {
            "position": [1.0, 2.0, 3.0],
            "orientation": [0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)],
        }
        pose_array = _pose_dict_to_array(pose)
        np.testing.assert_allclose(pose_array[:3], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(pose_array[5], math.pi / 2.0, places=5)
        self.assertAlmostEqual(pose_array[6], math.cos(math.pi / 4.0), places=5)

    def test_angular_delta_to_quaternion_identity_for_zero_rotation(self) -> None:
        quaternion = _angular_delta_to_quaternion(np.zeros(3, dtype=np.float64))
        np.testing.assert_allclose(quaternion, [0.0, 0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
