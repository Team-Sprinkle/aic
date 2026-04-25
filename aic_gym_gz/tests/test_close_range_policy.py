from __future__ import annotations

import unittest

import numpy as np

from aic_gym_gz.teacher.close_range import CloseRangeInsertionPolicy


class CloseRangePolicyTest(unittest.TestCase):
    def test_close_range_action_corrects_yaw_toward_target(self) -> None:
        policy = CloseRangeInsertionPolicy()
        observation = {
            "plug_pose": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            "target_port_pose": np.array([0.0, 0.0, 0.9, 0.0, 0.0, 3.0, 1.0], dtype=np.float64),
            "target_port_entrance_pose": np.array([0.0, 0.0, 0.95, 0.0, 0.0, 3.0, 1.0], dtype=np.float64),
            "score_geometry": {
                "distance_to_entrance": [0.04],
                "distance_to_target": [0.1],
                "lateral_misalignment": [0.0],
                "insertion_progress": [0.0],
            },
            "wrench": np.zeros(6, dtype=np.float64),
        }
        action = policy.action(observation)
        self.assertGreater(action[5], 0.0)
        self.assertLessEqual(action[5], 2.0)

    def test_insert_latches_at_corridor_threshold(self) -> None:
        policy = CloseRangeInsertionPolicy()
        observation = {
            "plug_pose": np.array([0.0, 0.0, 0.964, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            "target_port_pose": np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            "target_port_entrance_pose": np.array([0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            "score_geometry": {
                "distance_to_entrance": [0.014],
                "distance_to_target": [0.064],
                "lateral_misalignment": [0.001],
                "insertion_progress": [0.2],
            },
            "wrench": np.zeros(6, dtype=np.float64),
        }
        action = policy.action(observation)
        self.assertEqual(policy.phase, "insert")
        self.assertLess(action[2], 0.0)

    def test_insert_ramps_down_near_final_depth(self) -> None:
        policy = CloseRangeInsertionPolicy()
        observation = {
            "plug_pose": np.array([0.0, 0.0, 0.945, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            "target_port_pose": np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            "target_port_entrance_pose": np.array([0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            "score_geometry": {
                "distance_to_entrance": [0.006],
                "distance_to_target": [0.010],
                "lateral_misalignment": [0.001],
                "insertion_progress": [0.82],
            },
            "wrench": np.zeros(6, dtype=np.float64),
        }
        action = policy.action(observation)
        self.assertEqual(policy.phase, "insert")
        self.assertLess(action[2], 0.0)
        self.assertLessEqual(abs(float(action[2])), 0.006 + 1e-6)

    def test_progress_latches_to_port_frame_axial_insert_when_aligned(self) -> None:
        policy = CloseRangeInsertionPolicy()
        observation = {
            "plug_pose": np.array([0.001, 0.0, 0.918, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            "target_port_pose": np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            "target_port_entrance_pose": np.array([0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            "score_geometry": {
                "distance_to_entrance": [0.032],
                "distance_to_target": [0.018],
                "lateral_misalignment": [0.001],
                "insertion_progress": [0.64],
            },
            "wrench": np.zeros(6, dtype=np.float64),
        }
        action = policy.action(observation)
        self.assertEqual(policy.phase, "insert")
        self.assertLess(action[2], 0.0)
        self.assertLessEqual(abs(float(action[0])), 1e-6)

    def test_insert_relieves_force_spike(self) -> None:
        policy = CloseRangeInsertionPolicy()
        observation = {
            "plug_pose": np.array([0.0, 0.0, 0.945, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            "target_port_pose": np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            "target_port_entrance_pose": np.array([0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            "score_geometry": {
                "distance_to_entrance": [0.006],
                "distance_to_target": [0.010],
                "lateral_misalignment": [0.001],
                "insertion_progress": [0.82],
            },
            "wrench": np.array([0.0, 0.0, 45.0, 0.0, 0.0, 0.0], dtype=np.float64),
        }
        action = policy.action(observation)
        self.assertEqual(policy.phase, "insert_force_relief")
        self.assertGreater(action[2], 0.0)

    def test_insert_prioritizes_large_yaw_error_before_translation(self) -> None:
        policy = CloseRangeInsertionPolicy()
        observation = {
            "plug_pose": np.array([0.0, 0.0, 0.945, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            "target_port_pose": np.array([0.0, 0.0, 0.9, 0.0, 0.0, 1.0, 1.0], dtype=np.float64),
            "target_port_entrance_pose": np.array([0.0, 0.0, 0.95, 0.0, 0.0, 1.0, 1.0], dtype=np.float64),
            "score_geometry": {
                "distance_to_entrance": [0.006],
                "distance_to_target": [0.010],
                "lateral_misalignment": [0.001],
                "insertion_progress": [0.82],
            },
            "wrench": np.zeros(6, dtype=np.float64),
        }
        action = policy.action(observation)
        self.assertEqual(policy.phase, "insert")
        self.assertAlmostEqual(float(np.linalg.norm(action[:3])), 0.0, places=6)
        self.assertGreater(action[5], 0.0)

    def test_handoff_waits_for_near_port_lateral_and_yaw_alignment(self) -> None:
        policy = CloseRangeInsertionPolicy()
        observation = {
            "score_geometry": {
                "distance_to_entrance": [0.045],
                "distance_to_target": [0.055],
                "lateral_misalignment": [0.025],
                "orientation_error": [0.25],
            },
        }

        self.assertFalse(policy.should_handoff(observation))

        observation["score_geometry"]["lateral_misalignment"] = [0.004]
        observation["score_geometry"]["orientation_error"] = [0.04]
        self.assertTrue(policy.should_handoff(observation))


if __name__ == "__main__":
    unittest.main()
