from __future__ import annotations

import tempfile
import unittest

import numpy as np

from aic_gym_gz.env import make_default_env
from aic_gym_gz.teacher import TeacherReplayRunner, load_teacher_replay, save_teacher_replay
from aic_gym_gz.teacher.replay import TeacherReplayArtifact


class TeacherReplayTest(unittest.TestCase):
    def test_replay_artifact_round_trip(self) -> None:
        artifact = TeacherReplayArtifact(
            metadata={
                "trial_id": "trial_0",
                "task_id": "task_0",
                "seed": 123,
                "scenario_metadata": {"name": "scenario"},
                "task_metadata": {"port_name": "port_0"},
                "data_quality": {"wrench": {"is_real": False}},
            },
            trajectory_segments=[{"points": [{"action": [0, 0, 0, 0, 0, 0]}]}],
            probe_results=[],
            planner_candidates=[],
            step_logs=[],
            final_info={},
            limitations=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/teacher.json"
            save_teacher_replay(artifact, path)
            loaded = load_teacher_replay(path)
        self.assertEqual(loaded.metadata["trial_id"], "trial_0")
        self.assertIn("data_quality", loaded.metadata)

    def test_replay_save_serializes_numpy_values(self) -> None:
        artifact = TeacherReplayArtifact(
            metadata={
                "trial_id": "trial_0",
                "task_id": "task_0",
                "seed": 123,
                "pose": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "flag": np.bool_(True),
            },
            trajectory_segments=[{"points": [{"action": np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)}]}],
            probe_results=[],
            planner_candidates=[],
            step_logs=[],
            final_info={"score": np.float64(1.5)},
            limitations=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/teacher.json"
            save_teacher_replay(artifact, path)
            loaded = load_teacher_replay(path)
        self.assertEqual(len(loaded.metadata["pose"]), 3)
        self.assertAlmostEqual(loaded.metadata["pose"][0], 0.1, places=6)
        self.assertAlmostEqual(loaded.metadata["pose"][1], 0.2, places=6)
        self.assertAlmostEqual(loaded.metadata["pose"][2], 0.3, places=6)
        self.assertIs(loaded.metadata["flag"], True)
        self.assertEqual(loaded.final_info["score"], 1.5)

    def test_replay_runner_reexecutes_dense_segment(self) -> None:
        env = make_default_env(enable_randomization=True)
        artifact = TeacherReplayArtifact(
            metadata={"trial_id": "trial_0", "task_id": "task_0", "seed": 123},
            trajectory_segments=[{"points": [{"action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]} for _ in range(3)]}],
            probe_results=[],
            planner_candidates=[],
            step_logs=[],
            final_info={},
            limitations=[],
        )
        try:
            replay = TeacherReplayRunner(env=env).replay(artifact)
        finally:
            env.close()
        self.assertGreaterEqual(len(replay["records"]), 2)


if __name__ == "__main__":
    unittest.main()
