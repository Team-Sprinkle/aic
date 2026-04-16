"""Helpers for converting teacher artifacts into official replay sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_official_replay_sequence(
    artifact_path: Path | str,
    *,
    candidate_rank: int | None = None,
) -> dict[str, Any]:
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    if "trajectory_segments" in payload and "metadata" in payload:
        return payload
    if "ranked_candidates" in payload:
        selected_rank = candidate_rank or 1
        selected = next(
            item for item in payload["ranked_candidates"] if int(item["rank"]) == selected_rank
        )
        return selected["artifact"]
    raise ValueError("Unsupported artifact format for official replay conversion.")
