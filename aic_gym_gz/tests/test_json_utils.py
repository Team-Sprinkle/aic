from __future__ import annotations

import unittest

import numpy as np

from aic_gym_gz.utils import to_jsonable


class JsonUtilsTest(unittest.TestCase):
    def test_to_jsonable_converts_numpy_containers_and_scalars(self) -> None:
        payload = {
            "array": np.array([[1, 2], [3, 4]], dtype=np.int32),
            "float_scalar": np.float32(1.5),
            "int_scalar": np.int64(7),
            "nested": [
                {"vector": np.array([0.1, 0.2], dtype=np.float64)},
                (np.bool_(True), np.float32(2.5)),
            ],
        }

        converted = to_jsonable(payload)

        self.assertEqual(converted["array"], [[1, 2], [3, 4]])
        self.assertEqual(converted["float_scalar"], 1.5)
        self.assertEqual(converted["int_scalar"], 7)
        self.assertEqual(converted["nested"][0]["vector"], [0.1, 0.2])
        self.assertEqual(converted["nested"][1], [True, 2.5])


if __name__ == "__main__":
    unittest.main()
