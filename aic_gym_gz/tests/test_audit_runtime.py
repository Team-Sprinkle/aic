from __future__ import annotations

import unittest

from aic_gym_gz.audit_runtime import generate_runtime_audit


class RuntimeAuditTest(unittest.TestCase):
    def test_audit_has_required_sections(self) -> None:
        report = generate_runtime_audit()
        self.assertIn("observation_parity", report)
        self.assertIn("observation_dependencies", report)
        self.assertIn("scoring_parity", report)
        self.assertIn("replay_support", report)
        self.assertIn("score_labels", report)
        self.assertIn("rl_step_reward", report["score_labels"])
        self.assertIn("gym_final_score", report["score_labels"])


if __name__ == "__main__":
    unittest.main()
