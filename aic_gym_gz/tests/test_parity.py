from __future__ import annotations

import unittest

from aic_gym_gz.parity import AicParityHarness


class ParityTest(unittest.TestCase):
    def test_parity_report_structure(self) -> None:
        harness = AicParityHarness()
        rollout = [
            {
                "tcp_x": 0.0,
                "tcp_y": 0.0,
                "tcp_z": 0.0,
                "plug_x": 0.1,
                "plug_y": 0.0,
                "plug_z": 0.0,
                "classification": "timeout",
            }
        ]
        report = harness.compare_rollouts(reference_steps=rollout, candidate_steps=rollout)
        self.assertTrue(report["final_task_classification_match"])
        self.assertEqual(report["num_steps"], 1)


if __name__ == "__main__":
    unittest.main()
