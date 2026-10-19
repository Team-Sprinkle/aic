"""Frozen initial load reference for observation-only blockage detection.

Raw wrist force includes gravity. Rotate each force to the robot base before
subtracting the initial free-hold reference. This is a load-change estimate,
not an oracle contact-force decomposition. Never retare after contact/retreat.
"""
import numpy as np
from scipy.spatial.transform import Rotation


class InitialLoadReference:
    def __init__(self, *, samples=20, maximum_spread_n=1., sensor_to_tcp_rotation=None, minimum_quiet_span_s=0.):
        if samples<3 or maximum_spread_n<=0:raise ValueError('Invalid load-reference contract')
        self.samples=samples;self.maximum_spread_n=maximum_spread_n
        self.sensor_to_tcp=np.eye(3) if sensor_to_tcp_rotation is None else np.asarray(sensor_to_tcp_rotation)
        if (self.sensor_to_tcp.shape!=(3,3) or not np.allclose(self.sensor_to_tcp.T@self.sensor_to_tcp,np.eye(3),atol=1e-6)
                or np.linalg.det(self.sensor_to_tcp)<.999):raise ValueError('Invalid fixed force-sensor calibration')
        self.reference=None;self.observations=[];self.last_stamp=None
        self.timestamps=[];self.minimum_quiet_span_s=float(minimum_quiet_span_s)
        if self.minimum_quiet_span_s<0:raise ValueError('Invalid quiet-span requirement')

    def force_base(self,force_sensor_n,tcp_quaternion_xyzw):
        force=np.asarray(force_sensor_n,float)
        if force.shape!=(3,) or not np.isfinite(force).all():raise ValueError('Invalid measured force')
        return Rotation.from_quat(tcp_quaternion_xyzw).apply(self.sensor_to_tcp@force)

    def add_initial_free_hold(self,*,stamp_s,force_sensor_n,tcp_quaternion_xyzw,linear_speed_mps,angular_speed_radps):
        if self.reference is not None:raise ValueError('Reference is frozen; contact/recovery must never retare')
        if self.last_stamp is not None and stamp_s<=self.last_stamp:raise ValueError('Duplicate/reset force observation')
        self.last_stamp=stamp_s
        if linear_speed_mps>.005 or angular_speed_radps>np.deg2rad(3):
            self.observations=[];self.timestamps=[];return False
        self.observations.append(self.force_base(force_sensor_n,tcp_quaternion_xyzw))
        self.timestamps.append(float(stamp_s))
        if len(self.observations)<self.samples:return False
        if self.timestamps[-1]-self.timestamps[0]<self.minimum_quiet_span_s:return False
        values=np.array(self.observations[-self.samples:]);centre=np.median(values,axis=0)
        if max(np.linalg.norm(values-centre,axis=-1))>self.maximum_spread_n:return False
        self.reference=centre
        return True

    def measure(self,force_sensor_n,tcp_quaternion_xyzw):
        if self.reference is None:raise ValueError('Initial free-hold calibration is required')
        force=self.force_base(force_sensor_n,tcp_quaternion_xyzw);delta=force-self.reference
        return dict(load_change_norm_n=float(np.linalg.norm(delta)),force_base_n=force.tolist(),
                    load_change_base_n=delta.tolist(),initial_reference_base_n=self.reference.tolist(),
                    reference_frozen=True,oracle_contact_force=False,
                    caveat='Assumes initial hold is free and sensor bias is small; includes inertial/cable loads')
