from pathlib import Path

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
    assert parsed["insertion_success"] is True
    assert parsed["force_contact_penalty"] == -1


def test_score_from_results_dir_missing_is_zero_like(tmp_path: Path):
    parsed = score_from_scoring_yaml(tmp_path)
    assert parsed["total_score"] is None
    assert gazebo_terminal_score(tmp_path) == 0.0
