from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import json

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
        self.assertIn("ranking_metrics", result.payload["top_candidates"][0])
        self.assertIn("data_quality", result.payload["top_candidates"][0]["ranking_metrics"])

    def test_candidate_search_output_serializes_numpy_values(self) -> None:
        search = TeacherCandidateSearch(
            env_factory=lambda: make_default_env(enable_randomization=True, include_images=False),
            planner_factory=lambda: DeterministicMockPlannerBackend(),
            config=TeacherSearchConfig(
                planner_candidate_count=1,
                local_perturbation_count=0,
                top_k=1,
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "search.json"
            search.run(seed=123, output_path=output_path)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertIn("ranked_candidates", payload)
        self.assertEqual(payload["metadata"]["seed"], 123)

    def test_quality_adjustment_penalizes_missing_wrench_and_controller(self) -> None:
        search = TeacherCandidateSearch(
            env_factory=lambda: make_default_env(enable_randomization=True, include_images=False),
            planner_factory=lambda: DeterministicMockPlannerBackend(),
        )
        base_artifact = {
            "metadata": {
                "final_metrics": {"gym_final_score": 50.0, "rl_step_reward_total": 10.0},
                "data_quality": {
                    "wrench": {"is_real": True, "is_missing": False},
                    "controller_state": {"is_real": True},
                    "camera_info": {"available": False, "is_real": False},
                    "partial_insertion_depth": {"is_real": False},
                    "tier1_validity": {"is_real": False},
                },
            }
        }
        degraded_artifact = {
            "metadata": {
                "final_metrics": {"gym_final_score": 50.0, "rl_step_reward_total": 10.0},
                "data_quality": {
                    "wrench": {"is_real": False, "is_missing": True},
                    "controller_state": {"is_real": False},
                    "camera_info": {"available": False, "is_real": False},
                    "partial_insertion_depth": {"is_real": False},
                    "tier1_validity": {"is_real": False},
                },
            }
        }
        teacher_score = {"total_score": 60.0}
        base_metrics = search._ranking_metrics(artifact=base_artifact, teacher_score=teacher_score)
        degraded_metrics = search._ranking_metrics(artifact=degraded_artifact, teacher_score=teacher_score)
        self.assertGreater(base_metrics["composite_score"], degraded_metrics["composite_score"])


if __name__ == "__main__":
    unittest.main()
