"""Choose one coherent retry direction orthogonal to the measured retreat.

This is an explicitly logged supervisor exploration baseline, not an RL actor
or an expert action label. A direction is sampled once, then followed smoothly.
"""
import numpy as np


def lateral_retry_direction(retreat_displacement, fallback_axis, rng):
    axis=np.asarray(retreat_displacement,float)
    if np.linalg.norm(axis)<.001:axis=np.asarray(fallback_axis,float)
    if axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-8:
        raise ValueError('A measured retreat direction or calibrated visual fallback is required')
    axis=axis/np.linalg.norm(axis)
    for _ in range(20):
        value=rng.normal(size=3);value-=axis*np.dot(axis,value)
        if np.linalg.norm(value)>1e-6:return value/np.linalg.norm(value)
    raise RuntimeError('Could not sample a lateral direction')
