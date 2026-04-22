from __future__ import annotations

import unittest

import numpy as np

from aic_gym_gz.randomizer import AicEnvRandomizer
from aic_gym_gz.runtime import AicGazeboRuntime, MockStepperBackend


class RuntimeCheckpointTest(unittest.TestCase):
    def test_mock_backend_checkpoint_round_trip_is_exact(self) -> None:
        runtime = AicGazeboRuntime(backend=MockStepperBackend(), ticks_per_step=4)
        scenario = AicEnvRandomizer(enable_randomization=False).sample(seed=7)
        runtime.reset(seed=7, scenario=scenario)
        first = runtime.step(np.array([0.02, -0.01, 0.0, 0.0, 0.0, 0.05], dtype=np.float64))
        checkpoint = runtime.export_checkpoint()
        advanced = runtime.step(np.array([-0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
        restored = runtime.restore_checkpoint(checkpoint)
        self.assertTrue(checkpoint.exact)
        self.assertEqual(checkpoint.mode, "mock_exact")
        np.testing.assert_allclose(restored.tcp_pose, first.tcp_pose)
        np.testing.assert_allclose(restored.plug_pose, first.plug_pose)
        self.assertEqual(
            restored.score_geometry.get("insertion_progress"),
            first.score_geometry.get("insertion_progress"),
        )
        self.assertEqual(
            restored.auxiliary_force_contact_summary.sample_count,
            first.auxiliary_force_contact_summary.sample_count,
        )
        self.assertEqual(restored.sim_tick, first.sim_tick)
        self.assertNotEqual(advanced.sim_tick, restored.sim_tick)
        runtime.close()


if __name__ == "__main__":
    unittest.main()
