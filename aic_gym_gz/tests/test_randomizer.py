from __future__ import annotations

import unittest

from aic_gym_gz.randomizer import AicEnvRandomizer


class RandomizerTest(unittest.TestCase):
    def test_randomizer_is_seeded(self) -> None:
        randomizer = AicEnvRandomizer(enable_randomization=True)
        scenario_a = randomizer.sample(seed=9)
        scenario_b = randomizer.sample(seed=9)
        self.assertEqual(
            scenario_a.task_board.nic_rails["nic_rail_0"].translation,
            scenario_b.task_board.nic_rails["nic_rail_0"].translation,
        )

    def test_randomizer_varies_board_pose_and_task_target(self) -> None:
        randomizer = AicEnvRandomizer(enable_randomization=True)
        scenario = randomizer.sample(seed=5)
        task = next(iter(scenario.tasks.values()))
        self.assertNotEqual(scenario.task_board.pose_xyz_rpy[:2], (0.15, -0.2))
        self.assertTrue(task.target_module_name.startswith(("nic_card_mount_", "sc_port_")))


if __name__ == "__main__":
    unittest.main()
