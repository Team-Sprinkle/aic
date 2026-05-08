"""Force/torque guard state machine for recovery demonstrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
from typing import Any


class RecoveryPhase(StrEnum):
    GUARDED_DESCENT = "guarded_descent"
    SOFT_CONTACT = "soft_contact"
    BACKOFF = "backoff"
    REALIGN = "realign"
    RETRY = "retry"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class FTGuardConfig:
    soft_threshold_n: float = 1.0
    hard_threshold_n: float = 3.0
    backup_distance_m: float = 0.002
    max_retries: int = 3
    probe_pattern: str = "small_cross"


@dataclass
class FTGuardState:
    phase: RecoveryPhase = RecoveryPhase.GUARDED_DESCENT
    retry_count: int = 0
    max_force_n: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)


class FTGuard:
    """Small deterministic recovery state machine.

    Nominal expert generation must not instantiate or call this object.
    """

    def __init__(self, config: FTGuardConfig):
        if config.soft_threshold_n <= 0 or config.hard_threshold_n <= 0:
            raise ValueError("F/T thresholds must be positive")
        if config.soft_threshold_n >= config.hard_threshold_n:
            raise ValueError("soft_threshold_n must be below hard_threshold_n")
        self.config = config
        self.state = FTGuardState()

    @staticmethod
    def force_norm(sample: dict[str, Any] | None) -> float:
        if not sample:
            return 0.0
        if "force_norm_n" in sample:
            return abs(float(sample["force_norm_n"]))
        force = sample.get("force") if isinstance(sample, dict) else None
        if isinstance(force, dict):
            vals = [force.get(axis, 0.0) for axis in ("x", "y", "z")]
        elif isinstance(force, (list, tuple)):
            vals = list(force[:3])
        else:
            vals = [sample.get(axis, 0.0) for axis in ("fx", "fy", "fz")]
        return math.sqrt(sum(float(v) ** 2 for v in vals))

    def update(self, ft_sample: dict[str, Any] | None, *, insertion_succeeded: bool = False) -> RecoveryPhase:
        if self.state.phase in {RecoveryPhase.SUCCESS, RecoveryPhase.FAILURE}:
            return self.state.phase
        norm = self.force_norm(ft_sample)
        self.state.max_force_n = max(self.state.max_force_n, norm)
        if insertion_succeeded:
            self._set_phase(RecoveryPhase.SUCCESS, {"force_norm_n": norm})
        elif norm >= self.config.hard_threshold_n:
            self._set_phase(RecoveryPhase.FAILURE, {"force_norm_n": norm, "reason": "hard_threshold"})
        elif self.state.phase == RecoveryPhase.GUARDED_DESCENT and norm >= self.config.soft_threshold_n:
            self._set_phase(RecoveryPhase.SOFT_CONTACT, {"force_norm_n": norm})
        elif self.state.phase == RecoveryPhase.SOFT_CONTACT:
            self._set_phase(RecoveryPhase.BACKOFF, {"backup_distance_m": self.config.backup_distance_m})
        elif self.state.phase == RecoveryPhase.BACKOFF:
            self._set_phase(RecoveryPhase.REALIGN, {})
        elif self.state.phase == RecoveryPhase.REALIGN:
            if self.state.retry_count >= self.config.max_retries:
                self._set_phase(RecoveryPhase.FAILURE, {"reason": "max_retries"})
            else:
                self.state.retry_count += 1
                self._set_phase(RecoveryPhase.RETRY, {"retry_count": self.state.retry_count})
        elif self.state.phase == RecoveryPhase.RETRY:
            self._set_phase(RecoveryPhase.GUARDED_DESCENT, {"retry_count": self.state.retry_count})
        return self.state.phase

    def _set_phase(self, phase: RecoveryPhase, payload: dict[str, Any]) -> None:
        self.state.phase = phase
        self.state.events.append({"phase": phase.value, **payload})

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": self.state.phase.value,
            "retry_count": self.state.retry_count,
            "max_force_n": self.state.max_force_n,
            "events": list(self.state.events),
            "config": {
                "soft_threshold_n": self.config.soft_threshold_n,
                "hard_threshold_n": self.config.hard_threshold_n,
                "backup_distance_m": self.config.backup_distance_m,
                "max_retries": self.config.max_retries,
                "probe_pattern": self.config.probe_pattern,
            },
        }
