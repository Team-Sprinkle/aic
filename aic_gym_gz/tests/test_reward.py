from __future__ import annotations

import unittest

import numpy as np

from aic_gym_gz.reward import AicScoreCalculator


class RewardTest(unittest.TestCase):
    def test_official_like_summary_has_expected_keys(self) -> None:
        calculator = AicScoreCalculator()
        episode = {
            "initial_distance": 0.2,
            "sim_time": [0.0, 0.1, 0.2],
            "tcp_positions": [np.zeros(3), np.array([0.01, 0.0, 0.0]), np.array([0.02, 0.0, 0.0])],
            "tcp_linear_velocity": [np.zeros(3), np.array([0.1, 0.0, 0.0]), np.array([0.1, 0.0, 0.0])],
            "distances": [0.2, 0.15, 0.1],
            "force_magnitudes": [0.0, 0.0, 0.0],
            "off_limit_contacts": [False, False, False],
            "success": False,
            "wrong_port": False,
        }
        summary = calculator.evaluate(episode)
        self.assertIn("duration", summary.tier2)
        self.assertIn("score", summary.tier3)


if __name__ == "__main__":
    unittest.main()
