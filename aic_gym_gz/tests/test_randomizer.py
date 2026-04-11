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


if __name__ == "__main__":
    unittest.main()
