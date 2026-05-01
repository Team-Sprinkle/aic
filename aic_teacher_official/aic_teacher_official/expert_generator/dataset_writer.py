"""LeRobot-compatible expert metadata sidecar writer."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExpertEpisodeMetadata:
    episode_index: int
    mode: str
    scene_id: str
    candidate_index: int
    trajectory_path: str | None
    validation: dict[str, Any]
    vlm_strategy: dict[str, Any]
    moveit: dict[str, Any]
    phase_labels: list[dict[str, Any]]
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "aic_expert_episode_metadata/v1",
            "episode_index": self.episode_index,
            "mode": self.mode,
            "scene_id": self.scene_id,
            "candidate_index": self.candidate_index,
            "trajectory_path": self.trajectory_path,
            "validation": self.validation,
            "vlm_strategy": self.vlm_strategy,
            "moveit": self.moveit,
            "phase_labels": list(self.phase_labels),
            "extra": dict(self.extra),
        }


class DatasetMetadataWriter:
    """Writes sidecar metadata without changing LeRobot data files."""

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root)
        self.meta_dir = self.dataset_root / "meta"
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def append_episode(self, metadata: ExpertEpisodeMetadata) -> None:
        self._append_jsonl(self.meta_dir / "expert_trajectory_metadata.jsonl", metadata.to_dict())
        self._append_jsonl(
            self.meta_dir / "validation_results.jsonl",
            {
                "episode_index": metadata.episode_index,
                **metadata.validation,
            },
        )
        self._append_jsonl(
            self.meta_dir / "vlm_strategy.jsonl",
            {
                "episode_index": metadata.episode_index,
                **metadata.vlm_strategy,
            },
        )
        self._append_jsonl(
            self.meta_dir / "phase_labels.jsonl",
            {
                "episode_index": metadata.episode_index,
                "phase_labels": metadata.phase_labels,
            },
        )

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
