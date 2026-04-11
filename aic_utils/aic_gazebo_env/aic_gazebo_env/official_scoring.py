"""Official-score-aligned helpers for the Gazebo training bridge.

These helpers intentionally model only the stable subset that can be computed
from the current bridge observation surface. The authoritative full trial score
still lives in `aic_engine` + `aic_scoring`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def calculate_inverse_proportional_score(
    *,
    max_score: float,
    min_score: float,
    max_range: float,
    min_range: float,
    measurement: float,
) -> float:
    """Mirror the piecewise-linear interpolation used by `aic_scoring`."""
    if measurement >= max_range:
        return min_score
    if measurement <= min_range:
        return max_score
    return min_score + ((max_range - measurement) / (max_range - min_range)) * (
        max_score - min_score
    )


@dataclass(frozen=True)
class OfficialTier3TrackedPairScorer:
    """Tier-3-style score for the stable tracked-pair observation slice.

    This scorer only uses the currently available tracked distance fields.
    It therefore aligns with the official proximity scoring shape, not the full
    insertion/contact event path.
    """

    closest_task_score: float = 25.0
    furthest_task_score: float = 0.0

    def score(
        self,
        *,
        tracked_pair: dict[str, Any],
        initial_distance: float | None,
    ) -> tuple[float, dict[str, Any]]:
        """Return `(score, details)` for the tracked pair."""
        distance = tracked_pair.get("distance")
        success = tracked_pair.get("success")
        if not isinstance(distance, float):
            return 0.0, {
                "mode": "official_tier3_tracked_pair",
                "reason": "missing_distance",
            }

        if initial_distance is None or initial_distance <= 0.0:
            return 0.0, {
                "mode": "official_tier3_tracked_pair",
                "reason": "missing_initial_distance",
                "distance": distance,
            }

        # The official Tier 3 proximity score uses a max radius equal to half
        # the initial plug-port distance. For the current stable slice we do not
        # have partial-insertion geometry, so only the proximity branch applies.
        max_distance = initial_distance * 0.5
        score = calculate_inverse_proportional_score(
            max_score=self.closest_task_score,
            min_score=self.furthest_task_score,
            max_range=max_distance,
            min_range=0.0,
            measurement=distance,
        )
        return score, {
            "mode": "official_tier3_tracked_pair",
            "distance": distance,
            "initial_distance": initial_distance,
            "max_distance": max_distance,
            "tracked_pair_success": success if isinstance(success, bool) else None,
        }
