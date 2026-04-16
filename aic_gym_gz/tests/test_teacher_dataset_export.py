from __future__ import annotations

import json
import tempfile
import unittest

from aic_gym_gz.teacher.dataset_export import export_teacher_jsonl_dataset


class TeacherDatasetExportTest(unittest.TestCase):
    def test_jsonl_export_writes_records_and_metadata(self) -> None:
        candidate = {
            "candidate_spec": {"name": "candidate_0", "mode": "planner_waypoint"},
            "rank": 1,
            "selected_top_k": True,
            "near_perfect": False,
            "official_style_score": {"total_score": 50.0},
            "artifact": {
                "metadata": {"trial_id": "trial_1", "task_id": "task_1"},
                "step_logs": [
                    {
                        "sim_time": 0.1,
                        "planner_rationale": "r",
                        "trajectory_point": {"action": [0, 0, 0, 0, 0, 0]},
                        "dynamics_summary": {"cable_settling_score": 1.0},
                        "observation_summary": {"sim_tick": 1},
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_teacher_jsonl_dataset(candidate, output_dir=tmpdir)
            lines = result.dataset_path.read_text(encoding="utf-8").strip().splitlines()
            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(len(lines), 1)
        self.assertEqual(metadata["rank"], 1)


if __name__ == "__main__":
    unittest.main()
