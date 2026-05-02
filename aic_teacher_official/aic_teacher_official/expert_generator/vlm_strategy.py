"""VLM strategy/cable-risk schema.

The VLM is intentionally prohibited from generating executable waypoints in
this module. It returns symbolic approach and cable-risk guidance only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import json
from pathlib import Path
from typing import Any


class ExpertMode(StrEnum):
    NOMINAL = "nominal"
    NOMINAL_RECOVERY = "nominalrecovery"
    RECOVERY = "recovery"


class CableRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def _missing_(cls, value):
        normalized = str(value).strip().lower()
        if normalized in {"moderate", "med"}:
            return cls.MEDIUM
        return None


ALLOWED_APPROACH_SIDES = {
    "above",
    "above_left",
    "above_right",
    "high_clearance_vertical",
    "front",
    "back",
}
APPROACH_SIDE_ALIASES = {
    "left": "above_left",
    "right": "above_right",
    "center": "above",
    "vertical": "high_clearance_vertical",
    "top": "above",
    "robot_front": "front",
    "robot_back": "back",
    "front_left": "above_left",
    "front_right": "above_right",
    "back_left": "above_left",
    "back_right": "above_right",
}
ALLOWED_INSERTION_STRATEGIES = {
    "straight_slow_descent",
    "guarded_descent_with_backoff",
}
ALLOWED_PROBE_PATTERNS = {"none", "small_cross", "small_spiral"}
REQUIRED_FIELDS = {
    "mode",
    "approach_side",
    "cable_risk",
    "reason",
    "mitigation",
    "preferred_clearance_m",
    "avoid_regions",
    "insertion_strategy",
}


@dataclass(frozen=True)
class VLMStrategy:
    mode: ExpertMode
    approach_side: str
    cable_risk: CableRisk
    reason: str
    mitigation: str
    preferred_clearance_m: float
    avoid_regions: list[str]
    insertion_strategy: str
    recovery_allowed: bool = False
    probe_pattern: str = "none"
    backup_distance_m: float = 0.002
    retry_count: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "aic_vlm_strategy/v1",
            "mode": self.mode.value,
            "approach_side": self.approach_side,
            "cable_risk": self.cable_risk.value,
            "reason": self.reason,
            "mitigation": self.mitigation,
            "preferred_clearance_m": self.preferred_clearance_m,
            "avoid_regions": list(self.avoid_regions),
            "insertion_strategy": self.insertion_strategy,
            "recovery_allowed": self.recovery_allowed,
            "probe_pattern": self.probe_pattern,
            "backup_distance_m": self.backup_distance_m,
            "retry_count": self.retry_count,
        }


def _clamp(value: Any, lo: float, hi: float, field: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"{field} must be numeric") from ex
    return max(lo, min(hi, numeric))


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as ex:
        raise ValueError(f"Malformed VLM strategy JSON: {ex}") from ex
    if not isinstance(data, dict):
        raise ValueError("VLM strategy JSON must be an object")
    return data


def _normalize_approach_side(value: Any) -> str:
    normalized = str(value).strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    normalized = APPROACH_SIDE_ALIASES.get(normalized, normalized)
    if normalized in ALLOWED_APPROACH_SIDES:
        return normalized
    if "high" in normalized or "vertical" in normalized:
        return "high_clearance_vertical"
    if "left" in normalized:
        return "above_left"
    if "right" in normalized:
        return "above_right"
    if "front" in normalized or "port_normal" in normalized:
        return "front"
    if "back" in normalized:
        return "back"
    if "above" in normalized or "top" in normalized or "center" in normalized:
        return "above"
    raise ValueError(f"Unsupported approach_side {str(value)!r}")


def _normalize_avoid_regions(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        regions: list[str] = []
        for key, item in value.items():
            if isinstance(item, bool):
                if item:
                    regions.append(str(key))
            elif isinstance(item, str):
                regions.append(item)
            elif item is not None:
                regions.append(f"{key}:{item}")
        return regions
    if isinstance(value, list):
        regions = []
        for item in value:
            if isinstance(item, str):
                if item.strip():
                    regions.append(item)
            elif isinstance(item, dict):
                regions.extend(_normalize_avoid_regions(item))
            elif item is not None:
                regions.append(str(item))
        return regions
    raise ValueError("avoid_regions must be a string, object, or list")


def _normalize_probe_pattern(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, dict):
        value = value.get("type", value.get("pattern", "none"))
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if "spiral" in normalized:
        normalized = "small_spiral"
    if "cross" in normalized:
        normalized = "small_cross"
    if "touch" in normalized or "probe" in normalized:
        normalized = "small_cross"
    if normalized not in ALLOWED_PROBE_PATTERNS:
        normalized = "small_cross"
    return normalized


def parse_vlm_strategy(text_or_data: str | dict[str, Any], *, expected_mode: str | None = None) -> VLMStrategy:
    data = _parse_json_object(text_or_data) if isinstance(text_or_data, str) else dict(text_or_data)
    missing = sorted(REQUIRED_FIELDS - set(data))
    if missing:
        raise ValueError(f"VLM strategy missing required fields: {missing}")

    mode = ExpertMode(str(data["mode"]))
    if expected_mode is not None and mode.value != expected_mode:
        raise ValueError(f"VLM strategy mode {mode.value!r} does not match expected mode {expected_mode!r}")
    approach_side = _normalize_approach_side(data["approach_side"])
    cable_risk = CableRisk(str(data["cable_risk"]))
    insertion_strategy = str(data["insertion_strategy"])
    if insertion_strategy not in ALLOWED_INSERTION_STRATEGIES:
        raise ValueError(f"Unsupported insertion_strategy {insertion_strategy!r}")
    avoid_regions = _normalize_avoid_regions(data["avoid_regions"])
    probe_pattern = _normalize_probe_pattern(data.get("probe_pattern", "none"))

    preferred_clearance_m = _clamp(data["preferred_clearance_m"], 0.04, 0.25, "preferred_clearance_m")
    backup_distance_m = _clamp(data.get("backup_distance_m", 0.002), 0.001, 0.03, "backup_distance_m")
    retry_count = int(_clamp(data.get("retry_count", 0), 0, 10, "retry_count"))

    if mode == ExpertMode.NOMINAL:
        recovery_allowed = False
        retry_count = 0
        probe_pattern = "none"
        if insertion_strategy != "straight_slow_descent":
            raise ValueError("nominal mode must use straight_slow_descent")
    elif mode == ExpertMode.NOMINAL_RECOVERY:
        recovery_allowed = bool(data.get("recovery_allowed", True))
    else:
        recovery_allowed = bool(data.get("recovery_allowed", True))

    return VLMStrategy(
        mode=mode,
        approach_side=approach_side,
        cable_risk=cable_risk,
        reason=str(data["reason"]),
        mitigation=str(data["mitigation"]),
        preferred_clearance_m=preferred_clearance_m,
        avoid_regions=list(avoid_regions),
        insertion_strategy=insertion_strategy,
        recovery_allowed=recovery_allowed,
        probe_pattern=probe_pattern,
        backup_distance_m=backup_distance_m,
        retry_count=retry_count,
        raw=data,
    )


def build_strategy_prompt(scene_summary: dict[str, Any], *, mode: ExpertMode) -> str:
    return (
        "You are choosing a high-level strategy for the AIC cable insertion task.\n"
        "Do not output executable Cartesian waypoints, joint targets, velocities, or trajectory points.\n"
        "Assess cable collision/sweep risk from live images and scene context, then return strict JSON only.\n"
        f"Requested mode: {mode.value}\n"
        "Required schema fields: mode, approach_side, cable_risk, reason, mitigation, "
        "preferred_clearance_m, avoid_regions, insertion_strategy. Recovery mode may also include "
        "probe_pattern, backup_distance_m, retry_count.\n"
        "Use exactly one of these approach_side values: above, above_left, above_right, "
        "high_clearance_vertical, front, back.\n"
        "Use exactly one of these cable_risk values: low, medium, high.\n"
        "Use straight_slow_descent for nominal insertion_strategy.\n"
        f"Scene summary JSON:\n{json.dumps(scene_summary, indent=2, sort_keys=True)}"
    )


def save_strategy_debug(
    output_dir: str | Path,
    *,
    prompt: str,
    raw_response: str,
    strategy: VLMStrategy | None = None,
    error: str | None = None,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vlm_strategy_prompt.md").write_text(prompt, encoding="utf-8")
    (out / "vlm_strategy_response.txt").write_text(raw_response, encoding="utf-8")
    payload = {
        "strategy": strategy.to_dict() if strategy else None,
        "error": error,
    }
    (out / "vlm_strategy.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
