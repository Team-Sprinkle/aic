import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from requested_candidate_gate import candidate_gate


def test_replacement_removes_unrelated_legacy_veto():
    presence = [.00003, .0003, .1967]
    association = [.9934, .99977, .99729]
    assert not candidate_gate(presence, association)[0]
    accepted, effective = candidate_gate(presence, association, 'association_replaces_presence')
    assert accepted and effective == association


def test_replacement_requires_every_candidate_view_and_finite_values():
    for association in ([.99, .89, 1.], [.99, float('nan'), 1.]):
        assert not candidate_gate([1., 1., 1.], association, 'association_replaces_presence')[0]
    with pytest.raises(ValueError):
        candidate_gate([1., 1., 1.], None, 'association_replaces_presence')


def test_legacy_default_and_strict_shapes_preserved():
    assert candidate_gate([.5, .99, 1.])[0]
    assert not candidate_gate([.49, .99, 1.])[0]
    with pytest.raises(ValueError):
        candidate_gate([1., 1.])
