from pathlib import Path

import pytest
import yaml

from gazebo_rl.score_parser import gazebo_terminal_score, parse_scoring_yaml, score_from_scoring_yaml


def test_parse_engine_style_scoring_yaml(tmp_path: Path):
    scoring = tmp_path / "scoring.yaml"
    scoring.write_text(
        """
total: 42.5
trial_0:
  tier_1:
    score: 10
  tier_2:
    score: 5
    categories:
      force_contact:
        score: -1
  tier_3:
    score: 27.5
""",
        encoding="utf-8",
    )
    parsed = parse_scoring_yaml(scoring)
    assert parsed["total_score"] == 42.5
    assert parsed["tier_scores"]["tier_1"] == 10
    assert parsed["insertion_success"] is False
    assert parsed["insertion_success_rate"] == 0.0
    assert parsed["force_contact_penalty"] == -1


def test_score_from_results_dir_missing_is_zero_like(tmp_path: Path):
    parsed = score_from_scoring_yaml(tmp_path)
    assert parsed["total_score"] is None
    assert gazebo_terminal_score(tmp_path) == 0.0


@pytest.mark.parametrize("tier3,expected", [(0, False), (25, False), (50, False), (-12, False), (75, True), (None, None)])
def test_only_correct_port_full_insertion_is_success(tmp_path, tier3, expected):
    path = tmp_path / "scoring.yaml"
    path.write_text(yaml.safe_dump({"total": 1 + (tier3 or 0), "trial_1": {
        "tier_1": {"score": 1}, "tier_3": {"score": tier3}}}))
    result = parse_scoring_yaml(path)
    assert result["trials"]["trial_1"]["insertion_success"] is expected
    assert result["trials"]["trial_1"]["model_valid"] is True
    assert result["insertion_success"] is (expected is True)


def test_multiple_trials_preserve_denominator_and_unknown_results(tmp_path):
    path = tmp_path / "scoring.yaml"
    path.write_text(yaml.safe_dump({"total": 150, "trial_1": {"tier_3": {"score": 75}},
                                    "trial_2": {"tier_3": {"score": 50}}, "trial_3": {}}))
    result = parse_scoring_yaml(path)
    assert result["trial_count"] == 3
    assert result["scored_trial_count"] == 2
    assert result["successful_trial_count"] == 1
    assert result["insertion_success_rate"] == 0.5
    assert result["insertion_success"] is False


def test_arbitrary_engine_trial_names_are_scored(tmp_path):
    path = tmp_path / "scoring.yaml"
    path.write_text(yaml.safe_dump({"total": 101,
        "ep011_heldout": {"tier_1": {"score": 1}, "tier_2": {"score": 0}, "tier_3": {"score": 75}},
        "another_scene": {"tier_1": {"score": 1}, "tier_3": {"score": 25}},
        "metadata": {"seed": 42}}))
    result = parse_scoring_yaml(path)
    assert set(result["trials"]) == {"ep011_heldout", "another_scene"}
    assert result["successful_trial_count"] == 1
    assert result["insertion_success_rate"] == .5


def test_missing_results_root_does_not_search_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AIC_RESULTS_DIR", raising=False)
    (tmp_path / "scoring.yaml").write_text("total: 99\n")
    assert score_from_scoring_yaml()["total_score"] is None


@pytest.mark.parametrize("contents", ["total: .nan\n", "total: .inf\n", "trial_1:\n  total: 99\n"])
def test_invalid_or_missing_aggregate_is_not_a_score(tmp_path, contents):
    path = tmp_path / "scoring.yaml"
    path.write_text(contents)
    assert parse_scoring_yaml(path)["total_score"] is None
