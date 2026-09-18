"""CPU checks for diagnostic semantics; no policy construction or training."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from analyze_task_act import (absolute_targets, alternative_task_vector,
                              balanced_samples, image_donors, pose_errors, summarize)


def episode(index, family="sfp_to_nic", nic_count=1, sc_count=0, port=1):
    return {"episode_index": index, "nic_count": nic_count, "sc_count": sc_count,
            "task": {"task_family": family, "target_card_index": 0 if family == "sfp_to_nic" else -1,
                     "target_port_index": port, "target_card_valid": int(family == "sfp_to_nic")}}


class DiagnosticTests(unittest.TestCase):
    def test_absolute_target_rotates_translation(self):
        state = np.zeros((1, 32)); state[0, :3] = [1, 2, 3]
        state[0, 3:7] = Rotation.from_euler("z", 90, degrees=True).as_quat()
        command = np.array([[.01, 0, 0, 0, 0, 0]])
        target = absolute_targets(state, command)
        np.testing.assert_allclose(target[0, :3], [1, 2.01, 3], atol=1e-6)
        np.testing.assert_allclose(Rotation.from_rotvec(target[:, 3:]).as_matrix(),
                                   Rotation.from_quat(state[:, 3:7]).as_matrix(), atol=1e-6)

    def test_geodesic_rotation_does_not_penalize_equivalent_angles(self):
        first = np.array([[0., 0, 0, 0, 0, 0]])
        second = np.array([[.001, 0, 0, 2 * np.pi, 0, 0]])
        mm, degrees = pose_errors(first, second)
        self.assertAlmostEqual(mm[0], 1.)
        self.assertLess(degrees[0], 1e-10)

    def test_episode_balancing_never_uses_training(self):
        episodes = np.array([0] * 100 + [1] * 8 + [2] * 20)
        selected = balanced_samples(episodes, {1, 2}, 4, np.random.default_rng(4))
        self.assertEqual(len(np.unique(selected)), 8)
        self.assertEqual((episodes[selected] == 1).sum(), 4)
        self.assertEqual((episodes[selected] == 2).sum(), 4)

    def test_shuffle_preserves_time_phase_and_task_and_skips_unmatched(self):
        episodes = np.repeat([0, 1, 2], 4)
        times = np.tile(np.array([0., 1., 2., 3.]), 3)
        metadata = {0: episode(0), 1: episode(1), 2: episode(2, nic_count=5)}
        selected = np.array([0, 1, 8])
        donors = image_donors(selected, episodes, times, metadata, {0, 1, 2},
                             4, 1., np.random.default_rng(1))
        np.testing.assert_array_equal(donors, [4, 5, -1])

    def test_task_swap_does_not_invent_second_sc_port(self):
        self.assertIsNone(alternative_task_vector(episode(0, family="sc_to_sc", sc_count=1, port=0)))
        vector = alternative_task_vector(episode(0, family="sc_to_sc", sc_count=2, port=0))
        np.testing.assert_array_equal(vector, [0, 1, 0, 1, 0, 0, 0, 0, 0, 0])
        vector = alternative_task_vector(episode(0))
        np.testing.assert_array_equal(vector, [1, 0, 1, 0, 1, 0, 0, 0, 0, 1])

    def test_summary_task_change_is_not_accuracy(self):
        records = [{"episode_index": 0, "baseline_first_error_mm": 2.,
                    "task_swap_first_change_mm": 7.}]
        result = summarize(records, {0: episode(0)})
        self.assertEqual(result["all"]["metrics"]["task_swap_first_change_mm"]["mean"], 7.)
        self.assertFalse(any("task_swap" in key and "error" in key for key in result["all"]["metrics"]))


if __name__ == "__main__":
    unittest.main()
