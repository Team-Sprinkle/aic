from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _num(value: Any) -> float | None:
    try:
        return float(value)
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
    root = Path(results_dir or os.environ.get("AIC_RESULTS_DIR", "")).expanduser()
    if not str(root):
        return None
    candidate = root / "scoring.yaml"
    if candidate.exists():
        return candidate
    matches = sorted(root.glob("**/scoring.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def parse_scoring_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    score_path = Path(path)
    with score_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    scores = _walk_scores(data)
    total = _num(data.get("total") if isinstance(data, dict) else None)
    if total is None:
        total = _first_by_path(scores, ("total",))
    tier_scores = {
        "tier_1": _first_by_path(scores, ("tier_1",)),
        "tier_2": _first_by_path(scores, ("tier_2",)),
        "tier_3": _first_by_path(scores, ("tier_3",)),
    }
    return {
        "path": str(score_path),
        "raw": data,
        "total_score": total,
        "tier_scores": tier_scores,
        "insertion_success": _first_by_path(scores, ("tier_1",)) not in (None, 0.0),
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
