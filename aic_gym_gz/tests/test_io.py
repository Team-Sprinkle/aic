from __future__ import annotations

import types
import unittest

import numpy as np

from aic_gym_gz.io import _ros_image_to_array


class IoTest(unittest.TestCase):
    def test_ros_image_to_array_rgb8(self) -> None:
        message = types.SimpleNamespace(
            height=2,
            width=2,
            encoding="rgb8",
            data=bytes(
                [
                    255,
                    0,
                    0,
                    0,
                    255,
                    0,
                    0,
                    0,
                    255,
                    10,
                    20,
                    30,
                ]
            ),
        )
        array = _ros_image_to_array(message, expected_shape=(2, 2, 3))
        self.assertEqual(array.shape, (2, 2, 3))
        self.assertTrue(np.array_equal(array[0, 0], np.array([255, 0, 0], dtype=np.uint8)))


if __name__ == "__main__":
    unittest.main()
