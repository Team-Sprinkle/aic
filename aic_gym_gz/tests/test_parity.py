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

    def test_score_parity_report_structure(self) -> None:
        harness = AicParityHarness()
        trace = {
            "initial_native": {"distance_to_target": 0.2},
            "records": [
                {
                    "step_idx": 0,
                    "native": {
                        "sim_time": 0.0,
                        "tcp_position": [0.0, 0.0, 0.0],
                        "plug_position": [0.2, 0.0, 0.0],
                        "target_position": [0.0, 0.0, 0.0],
                        "distance_to_target": 0.2,
                        "orientation_error": 0.0,
                        "success_like": False,
                        "joint_positions": [0.0] * 6,
                    },
                },
                {
                    "step_idx": 1,
                    "native": {
                        "sim_time": 0.1,
                        "tcp_position": [0.01, 0.0, 0.0],
                        "plug_position": [0.1, 0.0, 0.0],
                        "target_position": [0.0, 0.0, 0.0],
                        "distance_to_target": 0.1,
                        "orientation_error": 0.0,
                        "success_like": False,
                        "joint_positions": [0.0] * 6,
                    },
                },
            ],
        }
        report = harness.compare_score_json(reference_report=trace, candidate_report=trace)
        self.assertIn("reference", report)
        self.assertIn("candidate", report)
        self.assertIn("deltas", report)
        self.assertEqual(report["deltas"]["total_score_abs_error"], 0.0)

    def test_image_parity_report_structure(self) -> None:
        harness = AicParityHarness()
        trace = {
            "records": [
                {
                    "step_idx": 0,
                    "images": {
                        "left": {
                            "shape": [64, 64, 3],
                            "dtype": "uint8",
                            "pixel_sum": 100,
                            "mean": 1.0,
                            "std": 0.5,
                            "timestamp": 1.0,
                            "present": True,
                        },
                        "center": {
                            "shape": [64, 64, 3],
                            "dtype": "uint8",
                            "pixel_sum": 200,
                            "mean": 2.0,
                            "std": 0.5,
                            "timestamp": 1.0,
                            "present": True,
                        },
                        "right": {
                            "shape": [64, 64, 3],
                            "dtype": "uint8",
                            "pixel_sum": 300,
                            "mean": 3.0,
                            "std": 0.5,
                            "timestamp": 1.0,
                            "present": True,
                        },
                    },
                }
            ]
        }
        report = harness.compare_image_trace_json(reference_report=trace, candidate_report=trace)
        self.assertEqual(report["num_steps"], 1)
        self.assertTrue(report["cameras"]["left"]["reference_present_all_steps"])


if __name__ == "__main__":
    unittest.main()
