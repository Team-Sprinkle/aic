from __future__ import annotations

import tempfile
import unittest

from aic_gym_gz.env import make_default_env
from aic_gym_gz.teacher import TeacherReplayRunner, load_teacher_replay, save_teacher_replay
from aic_gym_gz.teacher.replay import TeacherReplayArtifact


class TeacherReplayTest(unittest.TestCase):
    def test_replay_artifact_round_trip(self) -> None:
        artifact = TeacherReplayArtifact(
            metadata={"trial_id": "trial_0", "task_id": "task_0", "seed": 123},
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
