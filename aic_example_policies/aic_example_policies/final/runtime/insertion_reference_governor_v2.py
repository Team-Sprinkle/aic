"""Inference definitions extracted from insertion_reference_governor_v2.py; no training entry point."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

def limit(vector, maximum):
    return vector * min(1.0, maximum / max(float(np.linalg.norm(vector)), 1e-12))

@dataclass(frozen=True)
class ReferenceLimits:
    speed_mps: float = 0.025
    acceleration_mps2: float = 0.08
    angular_speed_rps: float = 0.15
    angular_acceleration_rps2: float = 0.5
    forward_lead_m: float = 0.006
    alignment_lateral_m: float = 0.001
    alignment_angle_rad: float = np.deg2rad(1.0)
    contact_load_n: float = 2.0

class InsertionReferenceGovernor:

    def __init__(self, port, grasp, reference, measured, stamp, limits=None):
        self.port = np.asarray(port, float).copy()
        self.inverse = np.linalg.inv(self.port)
        self.grasp = np.asarray(grasp, float).copy()
        self.inverse_grasp = np.linalg.inv(self.grasp)
        self.reference = np.asarray(reference, float).copy()
        self.stamp = float(stamp)
        self.limits = limits or ReferenceLimits()
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.bias = (self.inverse @ self.reference @ self.grasp)[:3, 3] - (self.inverse @ measured @ self.grasp)[:3, 3]
        if np.linalg.norm(self.bias) > 0.04:
            raise ValueError('Unreasonable approach compensation')

    def step(self, desired, measured, stamp, load_change_n):
        desired = np.asarray(desired, float).copy()
        measured = np.asarray(measured, float)
        if not np.isfinite(desired).all() or not np.isfinite(measured).all():
            raise ValueError('Nonfinite reference')
        dt = float(stamp) - self.stamp
        if dt <= 0:
            return (self.reference.copy(), dict(repeated_timestamp=True))
        dt = min(dt, 0.1)
        self.stamp = float(stamp)
        current = self.inverse @ measured @ self.grasp
        target = self.inverse @ desired @ self.grasp
        angle = Rotation.from_matrix(current[:3, :3]).magnitude()
        blocked = float(load_change_n) >= self.limits.contact_load_n or np.linalg.norm(current[:2, 3]) > self.limits.alignment_lateral_m or angle > self.limits.alignment_angle_rad
        allowance = 0.0005 if blocked else self.limits.forward_lead_m
        previous_reference = self.inverse @ self.reference @ self.grasp
        maximum = max(float(previous_reference[2, 3]), float(current[2, 3] + self.bias[2] + allowance))
        proposed_z = float(target[2, 3])
        target[2, 3] = min(target[2, 3], maximum)
        desired = self.port @ target @ self.inverse_grasp
        error = desired[:3, 3] - self.reference[:3, 3]
        braking_speed = np.sqrt(2 * self.limits.acceleration_mps2 * np.linalg.norm(error))
        desired_velocity = limit(error / dt, min(self.limits.speed_mps, braking_speed))
        self.velocity += limit(desired_velocity - self.velocity, self.limits.acceleration_mps2 * dt)
        self.reference[:3, 3] += self.velocity * dt
        rotation_error = Rotation.from_matrix(desired[:3, :3] @ self.reference[:3, :3].T).as_rotvec()
        angular_braking = np.sqrt(2 * self.limits.angular_acceleration_rps2 * np.linalg.norm(rotation_error))
        desired_angular = limit(rotation_error / dt, min(self.limits.angular_speed_rps, angular_braking))
        self.angular_velocity += limit(desired_angular - self.angular_velocity, self.limits.angular_acceleration_rps2 * dt)
        self.reference[:3, :3] = Rotation.from_rotvec(self.angular_velocity * dt).as_matrix() @ self.reference[:3, :3]
        actual_reference = self.inverse @ self.reference @ self.grasp
        return (self.reference.copy(), dict(blocked=bool(blocked), load_change_n=float(load_change_n), bias_port_m=self.bias.tolist(), measured_connector_port_m=current[:3, 3].tolist(), proposed_axial_reference_m=proposed_z, limited_axial_reference_m=float(target[2, 3]), issued_axial_lead_above_bias_m=float(actual_reference[2, 3] - current[2, 3] - self.bias[2]), reference_velocity_mps=self.velocity.tolist(), reference_angular_velocity_rps=self.angular_velocity.tolist(), dt_s=dt))
