"""Inference definitions extracted from approach_reference.py; no training entry point."""
import numpy as np

class ApproachReference:

    def __init__(self, point, tracking_bias_port):
        self.reference = np.asarray(point, dtype=float).copy()
        self.integral = np.clip(np.asarray(tracking_bias_port, dtype=float), -0.015, 0.015)

    def step(self, point, velocity_port, dt, axial_speed):
        point = np.asarray(point)
        velocity = np.asarray(velocity_port)
        previous = self.reference.copy()
        lateral = -self.reference[:2]
        distance = float(np.linalg.norm(lateral))
        self.reference[:2] += lateral * min(1.0, 0.03 * dt / max(distance, 1e-12))
        if axial_speed > 0:
            self.reference[2] = min(-0.025, self.reference[2] + axial_speed * dt)
        stationary_reference = np.abs(self.reference - previous) < 1e-10
        quiet = np.abs(velocity) <= 0.005
        integrate = stationary_reference & quiet
        self.integral[integrate] += 0.6 * dt * (self.reference - point)[integrate]
        self.integral = np.clip(self.integral, -0.015, 0.015)
        return (self.reference.copy(), self.integral.copy())
