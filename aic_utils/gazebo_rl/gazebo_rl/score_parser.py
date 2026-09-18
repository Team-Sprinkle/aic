from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Any

import yaml


def _num(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _walk_scores(node: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], float]]:
    scores: list[tuple[tuple[str, ...], float]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"score", "total", "total_score"}:
                numeric = _num(value)
                if numeric is not None:
                    scores.append((path + (str(key),), numeric))
            scores.extend(_walk_scores(value, path + (str(key),)))
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            scores.extend(_walk_scores(value, path + (str(idx),)))
    return scores


def _first_by_path(scores: list[tuple[tuple[str, ...], float]], needles: tuple[str, ...]) -> float | None:
    for path, value in scores:
        joined = "/".join(path).lower()
        if all(needle in joined for needle in needles):
            return value
    return None


def find_scoring_yaml(results_dir: str | os.PathLike[str] | None = None) -> Path | None:
    raw_root = results_dir or os.environ.get("AIC_RESULTS_DIR", "")
    if not raw_root:
        return None
    root = Path(raw_root).expanduser()
    candidate = root / "scoring.yaml"
    if candidate.exists():
        return candidate
    matches = sorted(root.glob("**/scoring.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def parse_scoring_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    score_path = Path(path)
    with score_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Scoring YAML must be a mapping: {score_path}")
    scores = _walk_scores(data)
    total = _num(data.get("total"))
    tier_scores = {
        "tier_1": _first_by_path(scores, ("tier_1",)),
        "tier_2": _first_by_path(scores, ("tier_2",)),
        "tier_3": _first_by_path(scores, ("tier_3",)),
    }
    # ScoringTier2::ComputeTier3Score awards exactly 75 for insertion into the
    # correct port. Proximity/partial insertion is <= 50; wrong-port is -12.
    # Keep the per-trial denominator: summing scores can mask unsuccessful trials.
    trial_nodes = {
        key: value for key, value in data.items()
        if str(key).startswith("trial_") or (
            isinstance(value, dict) and any(tier in value for tier in ("tier_1", "tier_2", "tier_3"))
        )
    }
    if not trial_nodes and any(key in data for key in ("tier_1", "tier_2", "tier_3")):
        trial_nodes = {"trial": data}
    trials = {}
    for name, node in trial_nodes.items():
        node = node if isinstance(node, dict) else {}
        values = {
            tier: _num(node[tier].get("score")) if isinstance(node.get(tier), dict) else None
            for tier in ("tier_1", "tier_2", "tier_3")
        }
        trials[str(name)] = {
            "tier_scores": values,
            "model_valid": values["tier_1"] == 1.0,
            "insertion_success": None if values["tier_3"] is None else values["tier_3"] == 75.0,
            "message": (node.get("tier_3") or {}).get("message") if isinstance(node.get("tier_3"), dict) else None,
        }
    scored_count = sum(t["insertion_success"] is not None for t in trials.values())
    success_count = sum(t["insertion_success"] is True for t in trials.values())
    return {
        "path": str(score_path),
        "raw": data,
        "total_score": total,
        "tier_scores": tier_scores,
        "trials": trials,
        "trial_count": len(trials),
        "scored_trial_count": scored_count,
        "successful_trial_count": success_count,
        "insertion_success_rate": success_count / scored_count if scored_count else None,
        "insertion_success": bool(trials) and success_count == len(trials),
        "insertion_proximity": _first_by_path(scores, ("proximity",)),
        "force_contact_penalty": _first_by_path(scores, ("force",)),
        "all_scores": [{"path": "/".join(path), "score": value} for path, value in scores],
    }


def score_from_scoring_yaml(results_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    path = find_scoring_yaml(results_dir)
    if path is None:
        return {
            "path": None,
            "total_score": None,
            "tier_scores": {},
            "trials": {},
            "trial_count": 0,
            "scored_trial_count": 0,
            "successful_trial_count": 0,
            "insertion_success_rate": None,
            "insertion_success": False,
            "insertion_proximity": None,
            "force_contact_penalty": None,
            "all_scores": [],
        }
    return parse_scoring_yaml(path)


def gazebo_terminal_score(results_dir: str | os.PathLike[str] | None = None) -> float:
    parsed = score_from_scoring_yaml(results_dir)
    total = parsed.get("total_score")
    return 0.0 if total is None else float(total) / 100.0


def dense_training_reward(*, terminal: bool, results_dir: str | os.PathLike[str] | None = None) -> float:
    if terminal:
        return gazebo_terminal_score(results_dir)
    return -0.01
