from __future__ import annotations

import unittest

import numpy as np

from aic_gym_gz.env import live_env_health_check, make_default_env


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
        self.assertEqual(info["reward_label"], "rl_step_reward")
        self.assertIn("reward_terms", info)
        self.assertIn("reward_metrics", info)
        self.assertIn("final_evaluation", info)
        self.assertIn("evaluation", info)
        self.assertEqual(info["final_evaluation"]["score_label"], "gym_final_score")
        self.assertEqual(info["evaluation"]["score_label"], "gym_final_score")
        self.assertIn("gym_final_score", info["final_evaluation"])
        self.assertIn("training_reward_total", info["final_evaluation"])
        env.close()

    def test_image_schema_present_when_enabled(self) -> None:
        env = make_default_env(include_images=True, allow_mock_images=True)
        obs, _ = env.reset(seed=3)
        self.assertEqual(obs["images"]["left"].shape, (256, 256, 3))
        self.assertEqual(obs["images"]["left"].dtype, np.uint8)
        self.assertEqual(obs["image_timestamps"].shape, (3,))
        env.close()

    def test_default_env_rejects_real_image_requests(self) -> None:
        with self.assertRaises(RuntimeError):
            make_default_env(include_images=True)

    def test_observation_contains_explicit_scalar_and_score_geometry_fields(self) -> None:
        env = make_default_env()
        obs, _ = env.reset(seed=4)
        self.assertIn("step_count", obs)
        self.assertIn("sim_tick", obs)
        self.assertIn("sim_time", obs)
        self.assertIn("distance_to_entrance", obs["score_geometry"])
        self.assertIn("lateral_misalignment", obs["score_geometry"])
        self.assertIn("orientation_error", obs["score_geometry"])
        self.assertIn("insertion_progress", obs["score_geometry"])
        env.close()

    def test_live_health_check_signature_works_for_mock_free_import(self) -> None:
        self.assertTrue(callable(live_env_health_check))


if __name__ == "__main__":
    unittest.main()
