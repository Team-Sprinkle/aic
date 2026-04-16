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
            "plug_positions": [np.array([0.2, 0.0, 0.0]), np.array([0.1, 0.0, 0.0]), np.array([0.1, 0.0, 0.0])],
            "target_port_pose": np.zeros(7),
            "target_port_entrance_pose": np.array([0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 1.0]),
            "force_magnitudes": [0.0, 0.0, 0.0],
            "wrench_time": [0.0, 0.1, 0.2],
            "wrench_samples": [np.zeros(6), np.zeros(6), np.zeros(6)],
            "off_limit_contacts": [False, False, False],
            "success": False,
            "wrong_port": False,
        }
        summary = calculator.evaluate(episode)
        self.assertIn("duration", summary.tier2)
        self.assertIn("score", summary.tier3)
        self.assertIn("parity_notes", summary.__dict__)

    def test_partial_insertion_score_uses_port_entrance_geometry(self) -> None:
        calculator = AicScoreCalculator()
        episode = {
            "initial_distance": 0.15,
            "sim_time": [0.0, 0.1, 0.2, 0.3, 0.4],
            "tcp_positions": [np.zeros(3) for _ in range(5)],
            "tcp_linear_velocity": [np.array([0.02, 0.0, -0.01]) for _ in range(5)],
            "distances": [0.15, 0.10, 0.05, 0.02, 0.005],
            "plug_positions": [
                np.array([0.03, 0.0, 0.03]),
                np.array([0.02, 0.0, 0.02]),
                np.array([0.01, 0.0, 0.015]),
                np.array([0.002, 0.001, 0.008]),
                np.array([0.001, 0.001, 0.005]),
            ],
            "target_port_pose": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            "target_port_entrance_pose": np.array([0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 1.0]),
            "wrench_time": [0.0, 0.1, 0.2, 0.3, 0.4],
            "wrench_samples": [np.zeros(6) for _ in range(5)],
            "off_limit_contacts": [False] * 5,
            "success": False,
            "wrong_port": False,
        }
        summary = calculator.evaluate(episode)
        self.assertGreater(float(summary.tier3["score"]), 38.0)
        self.assertIn("Partial insertion detected", summary.message)


if __name__ == "__main__":
    unittest.main()
