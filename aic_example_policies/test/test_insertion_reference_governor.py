"""Physical-reference regression checks, including the observed negative-bias failure."""
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from insertion_reference_governor_v2 import InsertionReferenceGovernor,ReferenceLimits


def test_negative_approach_bias_does_not_create_a_retreat():
    measured=np.eye(4);measured[2,3]=-.025
    reference=measured.copy();reference[2,3]-=.006
    desired=measured.copy();desired[2,3]=.04
    governor=InsertionReferenceGovernor(np.eye(4),np.eye(4),reference,measured,0)
    previous=reference[2,3]
    for i in range(1,101):
        # Even if the measured tool drifts back, forward demand cannot be
        # transformed into a recursively retreating reference.
        measured[2,3]-=.00005
        issued,_=governor.step(desired,measured,i*.05,3.)
        assert issued[2,3]>=previous-1e-10
        previous=issued[2,3]


def test_velocity_acceleration_and_clock_contract():
    initial=np.eye(4);desired=np.eye(4);desired[:3,3]=[.05,-.03,.04]
    desired[:3,:3]=Rotation.from_rotvec([.2,-.1,.3]).as_matrix()
    limits=ReferenceLimits();governor=InsertionReferenceGovernor(np.eye(4),np.eye(4),initial,initial,0,limits)
    previous_v=np.zeros(3);previous_w=np.zeros(3)
    for i in range(1,61):
        issued,d=governor.step(desired,initial,i*.05,0)
        v=np.asarray(d['reference_velocity_mps']);w=np.asarray(d['reference_angular_velocity_rps'])
        assert np.linalg.norm(v)<=limits.speed_mps+1e-10
        assert np.linalg.norm(v-previous_v)<=limits.acceleration_mps2*.05+1e-10
        assert np.linalg.norm(w)<=limits.angular_speed_rps+1e-10
        assert np.linalg.norm(w-previous_w)<=limits.angular_acceleration_rps2*.05+1e-10
        repeated,diagnostic=governor.step(desired,initial,i*.05,0)
        assert np.array_equal(issued,repeated) and diagnostic['repeated_timestamp']
        previous_v=v;previous_w=w
