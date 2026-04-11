from __future__ import annotations

import unittest

import numpy as np

from aic_gym_gz.policies import deterministic_policy_actions


class PoliciesTest(unittest.TestCase):
    def test_deterministic_policy_shape(self) -> None:
        actions = deterministic_policy_actions()
        self.assertGreaterEqual(len(actions), 1)
        for action in actions:
            vector = action.as_env_action()
            self.assertEqual(vector.shape, (6,))
            self.assertEqual(vector.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
