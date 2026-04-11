"""Parity tooling for comparing rollout traces."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ParityMetric:
    name: str
    max_abs_error: float
    mean_abs_error: float


class AicParityHarness:
    """Compares key trajectories from two open-loop rollout logs."""

    def compare_rollouts(
        self,
        *,
        reference_steps: list[dict[str, Any]],
        candidate_steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if len(reference_steps) != len(candidate_steps):
            raise ValueError("Reference and candidate rollouts must have the same length.")
        metrics = []
        for key in ("tcp_x", "tcp_y", "tcp_z", "plug_x", "plug_y", "plug_z"):
            ref = np.array([row[key] for row in reference_steps], dtype=np.float64)
            cand = np.array([row[key] for row in candidate_steps], dtype=np.float64)
            diff = np.abs(ref - cand)
            metrics.append(
                ParityMetric(
                    name=key,
                    max_abs_error=float(diff.max(initial=0.0)),
                    mean_abs_error=float(diff.mean() if diff.size else 0.0),
                )
            )
        return {
            "num_steps": len(reference_steps),
            "metrics": [metric.__dict__ for metric in metrics],
            "final_task_classification_match": (
                reference_steps[-1].get("classification") == candidate_steps[-1].get("classification")
                if reference_steps
                else True
            ),
        }

    def compare_csv_files(self, reference_csv: Path | str, candidate_csv: Path | str) -> dict[str, Any]:
        reference = self._read_csv(reference_csv)
        candidate = self._read_csv(candidate_csv)
        return self.compare_rollouts(reference_steps=reference, candidate_steps=candidate)

    def write_report(
        self,
        report: dict[str, Any],
        *,
        output_json: Path | str,
    ) -> None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    def _read_csv(self, path: Path | str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                parsed: dict[str, Any] = {}
                for key, value in row.items():
                    try:
                        parsed[key] = float(value)
                    except (TypeError, ValueError):
                        parsed[key] = value
                rows.append(parsed)
        return rows
