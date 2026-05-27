from __future__ import annotations

import torch


def force_delta_penalty_curve(
    delta_norm: torch.Tensor,
    *,
    threshold: float = 10.0,
    reference: float = 20.0,
    knee_penalty_fraction: float = 0.1,
    saturation: float | None = None,
    max_penalty: float = 1.0,
) -> torch.Tensor:
    """Return a non-negative piecewise force-delta penalty curve."""
    threshold = float(threshold)
    reference = max(float(reference), threshold + 1.0e-6)
    saturation = reference + (reference - threshold) if saturation is None else float(saturation)
    saturation = max(saturation, reference + 1.0e-6)
    max_penalty = max(float(max_penalty), 0.0)
    knee_penalty = max_penalty * min(max(float(knee_penalty_fraction), 0.0), 1.0)

    below_knee = ((delta_norm - threshold) / (reference - threshold)).clamp(min=0.0, max=1.0)
    low_penalty = knee_penalty * torch.square(below_knee)

    above_knee = ((delta_norm - reference) / (saturation - reference)).clamp(min=0.0, max=1.0)
    smooth = torch.square(above_knee) * (3.0 - 2.0 * above_knee)
    high_penalty = knee_penalty + (max_penalty - knee_penalty) * smooth

    return torch.where(delta_norm <= reference, low_penalty, high_penalty)
