from __future__ import annotations

import unittest

import numpy as np

from aic_gym_gz.env import make_default_env


class EnvTest(unittest.TestCase):
    def test_reset_is_deterministic(self) -> None:
        env = make_default_env(enable_randomization=True)
        obs_a, info_a = env.reset(seed=123)
        obs_b, info_b = env.reset(seed=123)
        np.testing.assert_allclose(obs_a["joint_positions"], obs_b["joint_positions"])
        np.testing.assert_allclose(obs_a["target_port_pose"], obs_b["target_port_pose"])
        self.assertEqual(info_a["trial_id"], info_b["trial_id"])
        env.close()

    def test_step_advances_exact_ticks(self) -> None:
        env = make_default_env(ticks_per_step=5)
        obs, _ = env.reset(seed=1)
        self.assertEqual(obs["sim_tick"], 0)
        obs, _, _, _, _ = env.step(np.zeros(6, dtype=np.float32))
        self.assertEqual(obs["sim_tick"], 5)
        env.close()

    def test_reward_terms_and_final_evaluation_present(self) -> None:
        env = make_default_env()
        env.reset(seed=2)
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
            if terminated or truncated:
                break
        self.assertIn("evaluation", info)
        self.assertIn("reward_terms", info)
        self.assertIn("total_score", info["evaluation"])
        env.close()


if __name__ == "__main__":
    unittest.main()
