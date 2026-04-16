from __future__ import annotations

import tempfile
import unittest

from aic_gym_gz.env import make_default_env
from aic_gym_gz.planners.mock import DeterministicMockPlannerBackend
from aic_gym_gz.teacher.search import TeacherCandidateSearch, TeacherSearchConfig


class TeacherSearchTest(unittest.TestCase):
    def test_candidate_search_ranks_multiple_rollouts(self) -> None:
        search = TeacherCandidateSearch(
            env_factory=lambda: make_default_env(enable_randomization=True, include_images=False),
            planner_factory=lambda: DeterministicMockPlannerBackend(),
            config=TeacherSearchConfig(
                planner_candidate_count=2,
                local_perturbation_count=2,
                top_k=2,
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = search.run(seed=123, output_path=f"{tmpdir}/search.json")
        self.assertGreaterEqual(len(result.payload["ranked_candidates"]), 4)
        self.assertEqual(len(result.payload["top_candidates"]), 2)
        top_score = result.payload["top_candidates"][0]["official_style_score"]["total_score"]
        bottom_score = result.payload["ranked_candidates"][-1]["official_style_score"]["total_score"]
        self.assertGreaterEqual(top_score, bottom_score)


if __name__ == "__main__":
    unittest.main()
