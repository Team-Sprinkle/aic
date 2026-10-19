"""Explicit ownership of SC requested-candidate approval.

The old presence predictor does not inspect candidate coordinates. A trained
candidate-association check may replace it in an opt-in ablation; it must not
silently weaken the geometric checks that follow it.
"""
import numpy as np


def candidate_gate(presence, association=None, mode='intersection'):
    if mode not in ('intersection', 'association_replaces_presence'):
        raise ValueError(f'Unknown candidate gate: {mode}')
    presence = np.asarray(presence, float)
    if presence.shape != (3,):
        raise ValueError('Exactly three camera presence probabilities required')
    if association is not None:
        association = np.asarray(association, float)
        if association.shape != (3,):
            raise ValueError('Exactly three camera association probabilities required')
    if mode == 'association_replaces_presence':
        if association is None:
            raise ValueError('Replacement requires a candidate-association checkpoint')
        effective = association
        accepted = bool(np.all(np.isfinite(association)) and np.all(association >= .9))
    else:
        effective = presence
        accepted = bool(np.all(np.isfinite(presence)) and np.all(presence >= .5))
        if association is not None:
            accepted = accepted and bool(np.all(np.isfinite(association)) and np.all(association >= .9))
    return accepted, effective.tolist()
