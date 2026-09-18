from __future__ import annotations

from aic_utils.gazebo_rl.scripts.serl_transfer_validate import classify_rollout


def test_classify_rollout_accepts_score_parser_total_score_key():
    scored = {"trial_count": 1, "scored_trial_count": 1}
    assert classify_rollout({**scored, "total_score": 70.0, "insertion_success": True}, success_threshold=90.0) == "success"
    assert classify_rollout({**scored, "total_score": 150.0, "insertion_success": False}, success_threshold=90.0) == "transfer_failure"
    assert classify_rollout({"total_score": 150.0, "trial_count": 3, "scored_trial_count": 2}, success_threshold=90.0) == "no_score"
    assert classify_rollout({"total_score": 91.0}, success_threshold=90.0) == "no_score"
    assert classify_rollout({"total_score": None}, success_threshold=90.0) == "no_score"
