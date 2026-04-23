from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import json

from aic_gym_gz.env import make_default_env
from aic_gym_gz.planners.mock import DeterministicMockPlannerBackend
from aic_gym_gz.teacher.search import (
    TeacherCandidateGenerator,
    TeacherCandidateSearch,
    TeacherSearchConfig,
    _candidate_signature,
)


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
        self.assertEqual(
            len({item["candidate_spec"]["name"] for item in result.payload["top_candidates"]}),
            len(result.payload["top_candidates"]),
        )
        self.assertIn("duplicate_penalty", result.payload["ranked_candidates"][0]["ranking_metrics"])
        self.assertIn("local_trajectory_score_summary", result.payload["ranked_candidates"][0]["ranking_metrics"])

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
                "auxiliary_summary_metadata": {
                    "auxiliary_summary_available": True,
                    "hidden_contact_event_count_recent": 0,
                    "repeated_contact_rich_steps": 0,
                    "current_wrench_force_l2_norm": 5.0,
                    "auxiliary_wrench_max_recent": 5.0,
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
                "auxiliary_summary_metadata": {
                    "auxiliary_summary_available": True,
                    "hidden_contact_event_count_recent": 2,
                    "repeated_contact_rich_steps": 3,
                    "current_wrench_force_l2_norm": 0.1,
                    "auxiliary_wrench_max_recent": 12.0,
                },
            }
        }
        teacher_score = {"total_score": 60.0}
        base_metrics = search._ranking_metrics(artifact=base_artifact, teacher_score=teacher_score)
        degraded_metrics = search._ranking_metrics(artifact=degraded_artifact, teacher_score=teacher_score)
        self.assertGreater(base_metrics["composite_score"], degraded_metrics["composite_score"])
        self.assertLess(degraded_metrics["auxiliary_adjustment"], 0.0)

    def test_candidate_signature_uses_segment_endpoints(self) -> None:
        candidate = {
            "candidate_spec": {"name": "candidate_0", "mode": "planner_waypoint"},
            "artifact": {
                "trajectory_segments": [
                    {
                        "phase": "free_space_approach",
                        "points": [
                            {"target_tcp_pose": [0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 1.0]},
                            {"target_tcp_pose": [0.0, 0.0, 0.91, 0.0, 0.0, 0.0, 1.0]},
                        ],
                    }
                ]
            },
        }
        signature = _candidate_signature(candidate)
        self.assertEqual(signature["waypoint_signature"], [[0.0, 0.0, 0.91]])

    def test_refinement_specs_follow_best_planner_seed(self) -> None:
        generator = TeacherCandidateGenerator(
            TeacherSearchConfig(
                planner_candidate_count=3,
                local_perturbation_count=2,
                candidate_segment_limit=6,
                refinement_segment_limit=8,
            )
        )
        planner_ranked = [
            {
                "candidate_spec": {
                    "name": "obstacle_clearance_1",
                    "mode": "planner_waypoint",
                    "family": "obstacle_clearance",
                    "planner_candidate_index": 1,
                },
                "artifact": {
                    "trajectory_segments": [{"phase": "obstacle_avoidance"}],
                    "final_info": {"distance_to_target": 0.68},
                },
                "ranking_metrics": {"composite_score": 1.0},
            },
            {
                "candidate_spec": {
                    "name": "alignment_first_2",
                    "mode": "planner_waypoint",
                    "family": "alignment_first",
                    "planner_candidate_index": 2,
                },
                "artifact": {
                    "trajectory_segments": [{"phase": "pre_insert_align"}],
                    "final_info": {"distance_to_target": 0.72},
                },
                "ranking_metrics": {"composite_score": 0.8},
            },
        ]
        specs = generator.refinement_specs(planner_ranked)
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].source_candidate_name, "obstacle_clearance_1")
        self.assertEqual(specs[0].segment_limit_override, 8)


if __name__ == "__main__":
    unittest.main()
