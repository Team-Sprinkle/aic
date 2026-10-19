"""Inference definitions extracted from correlated_lateral_retry.py; no training entry point."""
import numpy as np

def lateral_retry_direction(retreat_displacement, fallback_axis, rng):
    axis = np.asarray(retreat_displacement, float)
    if np.linalg.norm(axis) < 0.001:
        axis = np.asarray(fallback_axis, float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-08:
        raise ValueError('A measured retreat direction or calibrated visual fallback is required')
    axis = axis / np.linalg.norm(axis)
    for _ in range(20):
        value = rng.normal(size=3)
        value -= axis * np.dot(axis, value)
        if np.linalg.norm(value) > 1e-06:
            return value / np.linalg.norm(value)
    raise RuntimeError('Could not sample a lateral direction')
