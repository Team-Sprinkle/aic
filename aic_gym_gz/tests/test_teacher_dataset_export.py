from __future__ import annotations

import json
import tempfile
import unittest

import numpy as np

from aic_gym_gz.teacher.dataset_export import export_teacher_jsonl_dataset


class TeacherDatasetExportTest(unittest.TestCase):
    def test_jsonl_export_writes_records_and_metadata(self) -> None:
        candidate = {
            "candidate_spec": {"name": "candidate_0", "mode": "planner_waypoint"},
            "rank": 1,
            "selected_top_k": True,
            "near_perfect": False,
            "official_style_score": {"total_score": 50.0},
            "ranking_metrics": {"data_quality": {"wrench": {"is_real": False}}},
            "artifact": {
                "metadata": {
                    "trial_id": "trial_1",
                    "task_id": "task_1",
                    "data_quality": {"wrench": {"is_real": False}},
                },
                "step_logs": [
                    {
                        "sim_time": 0.1,
                        "planner_rationale": "r",
                        "trajectory_point": {"action": [0, 0, 0, 0, 0, 0]},
                        "dynamics_summary": {"cable_settling_score": 1.0},
                        "observation_summary": {"sim_tick": 1},
                        "history_summary": {"window_size": 1},
                        "data_quality": {"wrench": {"is_real": False}},
                        "auxiliary_summary_available": True,
                        "auxiliary_force_contact_summary": {"had_contact_recent": True},
                        "auxiliary_contact_metrics": {"hidden_contact_recent": True},
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_teacher_jsonl_dataset(candidate, output_dir=tmpdir)
            lines = result.dataset_path.read_text(encoding="utf-8").strip().splitlines()
            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            record = json.loads(lines[0])
        self.assertEqual(len(lines), 1)
        self.assertEqual(metadata["rank"], 1)
        self.assertIn("data_quality", record)
        self.assertTrue(record["auxiliary_summary_available"])

    def test_jsonl_export_serializes_numpy_values(self) -> None:
        candidate = {
            "candidate_spec": {"name": "candidate_0", "mode": "planner_waypoint"},
            "rank": 1,
            "selected_top_k": True,
            "near_perfect": False,
            "official_style_score": {"total_score": np.float64(50.0)},
            "ranking_metrics": {
                "data_quality": {"wrench": {"is_real": np.bool_(False)}},
                "score_geometry": np.array([1.0, 2.0], dtype=np.float32),
            },
            "artifact": {
                "metadata": {
                    "trial_id": "trial_1",
                    "task_id": "task_1",
                    "data_quality": {"wrench": {"is_real": np.bool_(False)}},
                    "auxiliary_summary": {"pose": np.array([0.1, 0.2, 0.3], dtype=np.float32)},
                    "auxiliary_summary_metadata": {"hidden_contact_event_count_recent": np.int32(2)},
                },
                "step_logs": [
                    {
                        "sim_time": 0.1,
                        "planner_rationale": "r",
                        "trajectory_point": {"action": np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)},
                        "dynamics_summary": {"cable_settling_score": np.float32(1.0)},
                        "observation_summary": {"sim_tick": 1, "pose": np.array([1, 2], dtype=np.int32)},
                        "history_summary": {"window_size": np.int32(1)},
                        "data_quality": {"wrench": {"is_real": np.bool_(False)}},
                        "auxiliary_summary_available": np.bool_(True),
                        "auxiliary_force_contact_summary": {"wrench_max_force_abs_recent": np.float32(4.0)},
                        "auxiliary_contact_metrics": {"hidden_contact_recent": np.bool_(True)},
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_teacher_jsonl_dataset(candidate, output_dir=tmpdir)
            record = json.loads(result.dataset_path.read_text(encoding="utf-8").splitlines()[0])
            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(record["action"], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(record["observation"]["pose"], [1, 2])
        pose = metadata["artifact_metadata"]["auxiliary_summary"]["pose"]
        self.assertEqual(len(pose), 3)
        self.assertAlmostEqual(pose[0], 0.1, places=6)
        self.assertAlmostEqual(pose[1], 0.2, places=6)
        self.assertAlmostEqual(pose[2], 0.3, places=6)
        self.assertEqual(metadata["auxiliary_summary_metadata"]["hidden_contact_event_count_recent"], 2)


if __name__ == "__main__":
    unittest.main()
