"""Replay validation and acceptance filtering."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationCriteria:
    score_threshold: float = 95.0
    max_force_threshold: float | None = None
    max_offlimit_contacts: int = 0
    require_insertion_event: bool = True
    max_tracking_error_m: float | None = None


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    score: float | None
    insertion_event_reached: bool | None
    max_force_n: float | None
    ft_impulse_ns: float | None
    max_tracking_error_m: float | None
    offlimit_contact_count: int | None
    trajectory_duration_s: float | None
    number_of_replans: int
    retry_count: int
    mode: str
    vlm_cable_risk: str | None
    moveit_success: bool
    candidate_index: int | None
    scene_seed: int | None
    phase_labels: list[str]
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "aic_expert_validation/v1",
            "accepted": self.accepted,
            "score": self.score,
            "insertion_event_reached": self.insertion_event_reached,
            "max_force_n": self.max_force_n,
            "ft_impulse_ns": self.ft_impulse_ns,
            "max_tracking_error_m": self.max_tracking_error_m,
            "offlimit_contact_count": self.offlimit_contact_count,
            "trajectory_duration_s": self.trajectory_duration_s,
            "number_of_replans": self.number_of_replans,
            "retry_count": self.retry_count,
            "mode": self.mode,
            "vlm_cable_risk": self.vlm_cable_risk,
            "moveit_success": self.moveit_success,
            "candidate_index": self.candidate_index,
            "scene_seed": self.scene_seed,
            "phase_labels": list(self.phase_labels),
            "reasons": list(self.reasons),
        }


def score_from_summary_csv(path: str | Path) -> float | None:
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            value = row.get("total_score")
            if value not in (None, ""):
                return float(value)
    return None


class TrajectoryValidator:
    def __init__(self, criteria: ValidationCriteria):
        self.criteria = criteria

    def evaluate(self, metrics: dict[str, Any]) -> ValidationResult:
        score = metrics.get("score")
        score = None if score is None else float(score)
        max_force = metrics.get("max_force_n")
        max_force = None if max_force is None else float(max_force)
        offlimit = metrics.get("offlimit_contact_count")
        offlimit = None if offlimit is None else int(offlimit)
        insertion = metrics.get("insertion_event_reached")
        tracking = metrics.get("max_tracking_error_m")
        tracking = None if tracking is None else float(tracking)
        reasons: list[str] = []
        if score is None or score < self.criteria.score_threshold:
            reasons.append("score_below_threshold_or_missing")
        if self.criteria.require_insertion_event and insertion is not True:
            reasons.append("insertion_event_missing")
        if self.criteria.max_force_threshold is not None and (
            max_force is None or max_force > self.criteria.max_force_threshold
        ):
            reasons.append("force_above_threshold_or_missing")
        if offlimit is None or offlimit > self.criteria.max_offlimit_contacts:
            reasons.append("offlimit_contact_threshold")
        if self.criteria.max_tracking_error_m is not None and (
            tracking is None or tracking > self.criteria.max_tracking_error_m
        ):
            reasons.append("tracking_error_threshold")
        return ValidationResult(
            accepted=not reasons,
            score=score,
            insertion_event_reached=insertion,
            max_force_n=max_force,
            ft_impulse_ns=metrics.get("ft_impulse_ns"),
            max_tracking_error_m=tracking,
            offlimit_contact_count=offlimit,
            trajectory_duration_s=metrics.get("trajectory_duration_s"),
            number_of_replans=int(metrics.get("number_of_replans", 0)),
            retry_count=int(metrics.get("retry_count", 0)),
            mode=str(metrics.get("mode")),
            vlm_cable_risk=metrics.get("vlm_cable_risk"),
            moveit_success=bool(metrics.get("moveit_success")),
            candidate_index=metrics.get("candidate_index"),
            scene_seed=metrics.get("scene_seed"),
            phase_labels=list(metrics.get("phase_labels", [])),
            reasons=reasons,
        )
