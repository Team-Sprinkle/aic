from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from aic_gym_gz.env import AicInsertionEnv
from aic_gym_gz.io import MockGazeboIO
from aic_gym_gz.randomizer import AicEnvRandomizer
from aic_gym_gz.runtime import (
    AicGazeboRuntime,
    MockStepperBackend,
    MockTransientContactConfig,
)
from aic_gym_gz.task import AicInsertionTask
from aic_gym_gz.validate_force_transients import run_force_transient_validation


def _make_env(*, ticks_per_step: int, contact_band_z: tuple[float, float] | None) -> AicInsertionEnv:
    return AicInsertionEnv(
        runtime=AicGazeboRuntime(
            backend=MockStepperBackend(
                transient_contact_config=MockTransientContactConfig(contact_band_z=contact_band_z)
            ),
            ticks_per_step=ticks_per_step,
        ),
        task=AicInsertionTask(hold_action_ticks=ticks_per_step, include_images=False, max_episode_steps=32),
        io=MockGazeboIO(),
        randomizer=AicEnvRandomizer(enable_randomization=False),
    )


class ForceTransientTest(unittest.TestCase):
    def test_mock_auxiliary_summary_captures_hidden_transient(self) -> None:
        runtime = AicGazeboRuntime(
            backend=MockStepperBackend(
                transient_contact_config=MockTransientContactConfig(contact_band_z=(1.292, 1.294))
            ),
            ticks_per_step=20,
        )
        scenario = AicEnvRandomizer(enable_randomization=False).sample(seed=7)
        runtime.reset(seed=7, scenario=scenario)
        state = runtime.step(np.array([0.0, 0.0, -0.25, 0.0, 0.0, 0.0], dtype=np.float64))
        summary = state.auxiliary_force_contact_summary
        self.assertEqual(summary.source, "mock_substeps_exact")
        self.assertEqual(summary.sample_count, 20)
        self.assertTrue(summary.had_contact_recent)
        self.assertGreater(summary.wrench_max_force_abs_recent, 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(state.wrench[:3])), 0.0, places=6)
        runtime.close()

    def test_env_keeps_auxiliary_summary_out_of_public_observation(self) -> None:
        env = _make_env(ticks_per_step=20, contact_band_z=(1.292, 1.294))
        env.reset(seed=11)
        obs, _, _, _, info = env.step(np.array([0.0, 0.0, -0.25, 0.0, 0.0, 0.0], dtype=np.float32))
        self.assertNotIn("auxiliary_force_contact_summary", obs)
        self.assertIn("auxiliary_force_contact_summary", info)
        self.assertTrue(info["official_compatible_observation_semantics"]["wrench_is_current_sample_only"])
        aux = info["auxiliary_force_contact_summary"]
        np.testing.assert_allclose(obs["wrench"], aux["wrench_current"])
        self.assertTrue(aux["had_contact_recent"])
        self.assertGreater(aux["wrench_max_force_abs_recent"], float(np.linalg.norm(obs["wrench"][:3])))
        env.close()

    def test_force_transient_validation_report_structure(self) -> None:
        report = run_force_transient_validation(ticks_per_step=20, seed=123)
        scenarios = {item["name"]: item for item in report["scenarios"]}
        self.assertIn("scenario_a_obstacle_contact_transient", scenarios)
        self.assertIn("scenario_b_no_contact_control", scenarios)
        self.assertIn("scenario_c_repeated_coarse_boundary_crossing", scenarios)
        self.assertGreater(
            scenarios["scenario_a_obstacle_contact_transient"]["summary"]["aliasing_detected_step_count"],
            0,
        )
        self.assertEqual(
            scenarios["scenario_b_no_contact_control"]["summary"]["had_contact_recent_step_count"],
            0,
        )
        self.assertFalse(report["isaac_lab_style_check"]["direct_isaac_lab_parity_tested"])
        self.assertFalse(report["official_path_check"]["direct_official_parity_tested"])

    def test_force_transient_validation_writes_json_with_numpy_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_force_transient_validation(
                ticks_per_step=20,
                seed=123,
                output_dir=Path(tmpdir),
            )
            output_path = Path(tmpdir) / "force_transient_validation_report.json"
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(written["observation_contract"], report["observation_contract"])
        first_record = written["scenarios"][0]["records"][0]
        self.assertIsInstance(first_record["current_wrench"], list)
        self.assertIsInstance(first_record["wrench_max_abs_recent"], list)
        self.assertIsInstance(first_record["time_of_peak_within_step"], float)


if __name__ == "__main__":
    unittest.main()
