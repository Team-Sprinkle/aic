import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from correlated_lateral_retry import lateral_retry_direction


def test_retry_is_unit_length_and_orthogonal_to_measured_retreat():
    rng=np.random.default_rng(17)
    for _ in range(100):
        retreat=rng.normal(size=3)*.03
        result=lateral_retry_direction(retreat,[0,0,1],rng)
        assert np.linalg.norm(result)==pytest.approx(1.)
        assert np.dot(retreat,result)==pytest.approx(0.,abs=1e-12)


def test_tiny_retreat_uses_the_visual_normal_without_privileged_geometry():
    normal=np.array([1.,2.,3.]);normal/=np.linalg.norm(normal)
    result=lateral_retry_direction([0,0,0],normal,np.random.default_rng(2))
    assert np.dot(result,normal)==pytest.approx(0.,abs=1e-12)
    with pytest.raises(ValueError):
        lateral_retry_direction([0,0,0],[0,0,0],np.random.default_rng(2))
