"""Persistent, bounded angular reference for a frozen free-approach target."""
import numpy as np
from scipy.spatial.transform import Rotation


class ApproachOrientationReference:
    def __init__(self, seed, target, *, speed=.5, acceleration=1., maximum_lead=np.deg2rad(5.)):
        self.seed = np.asarray(seed).copy()
        self.vector = Rotation.from_matrix(self.seed.T@target).as_rotvec()
        self.length = float(np.linalg.norm(self.vector))
        self.progress = 0.
        self.velocity = 0.
        self.reference = self.seed.copy()
        self.speed, self.acceleration, self.maximum_lead = speed, acceleration, maximum_lead

    def step(self, measured, dt):
        remaining = max(0.,self.length-self.progress)
        wanted = min(self.speed,np.sqrt(2*self.acceleration*remaining))
        velocity = self.velocity+np.clip(wanted-self.velocity,-self.acceleration*dt,self.acceleration*dt)
        progress = min(self.length,self.progress+velocity*dt)
        candidate = self.seed@Rotation.from_rotvec(self.vector*(progress/max(self.length,1e-12))).as_matrix()
        lead = float(Rotation.from_matrix(np.asarray(measured).T@candidate).magnitude())
        paused = bool(lead>self.maximum_lead)
        if paused:
            # Stop additional demand; never roll the issued reference backward.
            self.velocity = 0.
        else:
            self.progress, self.reference = progress,candidate
            self.velocity = velocity if progress<self.length else 0.
        return self.reference.copy(),dict(lead_paused=paused,angular_lead_deg=float(np.degrees(
            Rotation.from_matrix(np.asarray(measured).T@self.reference).magnitude())),
            reference_progress_deg=float(np.degrees(self.progress)),
            maximum_lead_deg=float(np.degrees(self.maximum_lead)))
