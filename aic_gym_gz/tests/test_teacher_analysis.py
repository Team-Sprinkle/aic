from __future__ import annotations

import unittest

from aic_gym_gz.teacher.analysis import (
    analyze_replay_comparison,
    analyze_rollout_artifact,
    analyze_search_payload,
    classify_replay_fidelity,
)
from aic_gym_gz.teacher.replay import TeacherReplayArtifact


def _rollout_artifact() -> dict:
    return {
        "metadata": {
            "trial_id": "trial_0",
            "task_id": "task_0",
            "planner_backend": "openai",
            "final_metrics": {
                "rl_step_reward_total": 10.0,
                "gym_final_score": 25.0,
                "official_eval_score": None,
            },
            "teacher_official_style_score": {
                "total_score": 22.0,
                "tier2": {"jerk_mps3": 1.5},
                "tier3": {"status": "proximity", "message": "No insertion detected."},
            },
            "data_quality": {
                "wrench": {"is_real": False},
                "controller_state": {"is_real": False},
            },
        },
        "trajectory_segments": [{"phase": "free_space_approach", "points": [{"target_tcp_pose": [0, 0, 0, 0, 0, 0, 1]}]}],
        "planner_candidates": [{"name": "candidate_0"}],
        "step_logs": [
            {
                "phase": "free_space_approach",
                "sim_time": 0.0,
                "reward": 1.0,
                "trajectory_point": {"action": [0, 0, 0, 0, 0, 0]},
                "observation_summary": {
                    "tcp_pose": [0, 0, 0, 0, 0, 0, 1],
                    "plug_to_port_relative": [0, 0, 0.1, 0.1],
                    "off_limit_contact": False,
                },
            },
            {
                "phase": "free_space_approach",
                "sim_time": 1.0,
                "reward": 1.0,
                "trajectory_point": {"action": [0, 0, 0, 0, 0, 0]},
                "observation_summary": {
                    "tcp_pose": [0, 0, 0.01, 0, 0, 0, 1],
                    "plug_to_port_relative": [0, 0, 0.09, 0.09],
                    "off_limit_contact": False,
                },
            },
        ],
        "final_info": {"success": False, "wrong_port": False},
    }


class TeacherAnalysisTest(unittest.TestCase):
    def test_rollout_analysis_surfaces_warnings(self) -> None:
        result = analyze_rollout_artifact(_rollout_artifact())
        self.assertEqual(result.summary["counts"]["planner_calls"], 1)
        self.assertTrue(result.summary["warnings"])
        self.assertIn("teacher_rollout", result.summary["artifact_type"])

    def test_search_analysis_reports_diversity_and_rank_changes(self) -> None:
        candidate = {
            "candidate_spec": {"name": "planner_candidate_0", "mode": "planner_waypoint"},
            "rank": 1,
            "artifact": _rollout_artifact(),
            "ranking_metrics": {
                "composite_score": 10.0,
                "teacher_official_style_score": 9.0,
                "gym_final_score": 8.0,
                "rl_step_reward_total": 7.0,
                "quality_adjustment": -2.0,
            },
        }
        other = {
            "candidate_spec": {"name": "local_perturbation_0", "mode": "local_perturbation"},
            "rank": 2,
            "artifact": _rollout_artifact(),
            "ranking_metrics": {
                "composite_score": 9.5,
                "teacher_official_style_score": 9.4,
                "gym_final_score": 8.0,
                "rl_step_reward_total": 7.0,
                "quality_adjustment": 0.0,
            },
        }
        payload = {
            "metadata": {"planner_backend": "openai", "top_k": 1},
            "ranked_candidates": [candidate, other],
        }
        result = analyze_search_payload(payload)
        self.assertEqual(result.summary["top_candidates"][0]["name"], "planner_candidate_0")
        self.assertIn("diversity_analysis", result.summary)
        self.assertIn(
            result.summary["ranking_analysis"]["metric_dominance"]["dominant_metric"],
            {"teacher_official_style_score", "quality_adjustment"},
        )

    def test_search_metric_dominance_ignores_constant_quality_adjustment(self) -> None:
        candidate = {
            "candidate_spec": {"name": "planner_candidate_0", "mode": "planner_waypoint"},
            "rank": 1,
            "artifact": _rollout_artifact(),
            "ranking_metrics": {
                "composite_score": 10.0,
                "teacher_official_style_score": 9.0,
                "gym_final_score": 8.0,
                "rl_step_reward_total": 7.0,
                "quality_adjustment": -2.0,
            },
        }
        other = {
            "candidate_spec": {"name": "planner_candidate_1", "mode": "planner_waypoint"},
            "rank": 2,
            "artifact": _rollout_artifact(),
            "ranking_metrics": {
                "composite_score": 9.9,
                "teacher_official_style_score": 9.4,
                "gym_final_score": 8.0,
                "rl_step_reward_total": 7.0,
                "quality_adjustment": -2.0,
            },
        }
        payload = {
            "metadata": {"planner_backend": "openai", "top_k": 1},
            "ranked_candidates": [candidate, other],
        }
        result = analyze_search_payload(payload)
        self.assertEqual(
            result.summary["ranking_analysis"]["metric_dominance"]["dominant_metric"],
            "teacher_official_style_score",
        )

    def test_replay_analysis_classifies_approximate_match(self) -> None:
        artifact = TeacherReplayArtifact(
            metadata={"final_metrics": {"gym_final_score": 20.0}},
            trajectory_segments=[],
            probe_results=[],
            planner_candidates=[],
            step_logs=[
                {
                    "reward": 1.0,
                    "observation_summary": {
                        "tcp_pose": [0, 0, 0, 0, 0, 0, 1],
                        "plug_to_port_relative": [0, 0, 0.1, 0.1],
                        "off_limit_contact": False,
                    },
                }
            ],
            final_info={},
            limitations=[],
        )
        replayed = {
            "records": [
                {
                    "reward": 1.5,
                    "tcp_pose": [0, 0, 0.03, 0, 0, 0, 1],
                    "plug_to_port_relative": [0, 0, 0.12, 0.12],
                    "off_limit_contact": False,
                }
            ],
            "final_info": {"final_evaluation": {"gym_final_score": 24.0}},
        }
        result = analyze_replay_comparison(original=artifact, replayed=replayed)
        self.assertEqual(result.summary["fidelity"]["label"], "approximately faithful")

    def test_replay_fidelity_thresholds(self) -> None:
        self.assertEqual(
            classify_replay_fidelity(
                step_delta=1,
                final_tcp_pose_delta=0.01,
                final_plug_target_delta=0.01,
                reward_total_delta=1.0,
                gym_final_score_delta=1.0,
            ),
            "faithful",
        )
        self.assertEqual(
            classify_replay_fidelity(
                step_delta=100,
                final_tcp_pose_delta=1.0,
                final_plug_target_delta=1.0,
                reward_total_delta=100.0,
                gym_final_score_delta=100.0,
            ),
            "poor replay match",
        )


if __name__ == "__main__":
    unittest.main()
