from __future__ import annotations

from aic_utils.gazebo_rl.scripts.serl_transfer_validate import classify_rollout


def test_classify_rollout_accepts_score_parser_total_score_key():
    assert classify_rollout({"total_score": 91.0}, success_threshold=90.0) == "success"
    assert classify_rollout({"total_score": 10.0}, success_threshold=90.0) == "transfer_failure"
    assert classify_rollout({"total_score": None}, success_threshold=90.0) == "no_score"
