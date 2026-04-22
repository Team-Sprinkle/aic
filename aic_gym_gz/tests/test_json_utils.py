from __future__ import annotations

import json
import unittest

import numpy as np

from aic_gym_gz.utils import to_jsonable


class JsonUtilsTest(unittest.TestCase):
    def test_to_jsonable_converts_numpy_nested_values(self) -> None:
        payload = {
            "array": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "scalar_float": np.float64(1.25),
            "scalar_int": np.int32(7),
            "scalar_bool": np.bool_(True),
            "nested": {
                "tuple": (np.array([5, 6], dtype=np.int64), np.float32(2.5)),
                "list": [np.bool_(False), {"value": np.array([8], dtype=np.int32)}],
            },
        }

        converted = to_jsonable(payload)

        self.assertEqual(converted["array"], [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(converted["scalar_float"], 1.25)
        self.assertEqual(converted["scalar_int"], 7)
        self.assertIs(converted["scalar_bool"], True)
        self.assertEqual(converted["nested"]["tuple"], [[5, 6], 2.5])
        self.assertEqual(converted["nested"]["list"], [False, {"value": [8]}])
        self.assertIn('"array": [[1.0, 2.0], [3.0, 4.0]]', json.dumps(converted, sort_keys=True))
