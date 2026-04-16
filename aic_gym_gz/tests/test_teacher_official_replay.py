from __future__ import annotations

import json
import tempfile
import unittest

from aic_gym_gz.teacher.official_replay import load_official_replay_sequence


class TeacherOfficialReplayTest(unittest.TestCase):
    def test_loads_selected_candidate_from_search_artifact(self) -> None:
        payload = {
            "ranked_candidates": [
                {"rank": 1, "artifact": {"metadata": {"trial_id": "trial_1"}, "trajectory_segments": []}}
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/search.json"
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)
            artifact = load_official_replay_sequence(path)
        self.assertEqual(artifact["metadata"]["trial_id"], "trial_1")


if __name__ == "__main__":
    unittest.main()
