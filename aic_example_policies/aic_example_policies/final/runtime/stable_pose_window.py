"""Inference definitions extracted from stable_pose_window.py; no training entry point."""
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation

class StablePoseWindow:

    def __init__(self, size=5, minimum=3, position_m=0.001, angle_deg=1.0, max_age_s=1.0):
        self.rows = deque(maxlen=size)
        self.minimum = minimum
        self.position_m = position_m
        self.angle_deg = angle_deg
        self.max_age_s = max_age_s

    def clear(self):
        self.rows.clear()

    def update(self, matrix, stamp):
        matrix = np.asarray(matrix, float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            self.clear()
            return None
        if self.rows and stamp <= self.rows[-1][0]:
            raise ValueError('Pose consensus requires strictly increasing observation times')
        self.rows.append((stamp, matrix.copy()))
        while self.rows and stamp - self.rows[0][0] > self.max_age_s:
            self.rows.popleft()
        while len(self.rows) > 1:
            mats = np.stack([r[1] for r in self.rows])
            points = mats[:, :3, 3]
            distance = float(np.linalg.norm(points[:, None] - points[None, :], axis=-1).max())
            rotations = Rotation.from_matrix(mats[:, :3, :3])
            angle = max((float((a.inv() * b).magnitude() * 180 / np.pi) for a in rotations for b in rotations))
            if distance <= self.position_m and angle <= self.angle_deg:
                break
            self.rows.popleft()
        if len(self.rows) < self.minimum:
            return None
        mats = np.stack([r[1] for r in self.rows])
        result = np.eye(4)
        result[:3, 3] = np.median(mats[:, :3, 3], axis=0)
        result[:3, :3] = Rotation.from_matrix(mats[:, :3, :3]).mean().as_matrix()
        return dict(matrix=result, samples=len(self.rows), position_span_mm=distance * 1000, orientation_span_deg=angle, age_s=stamp - self.rows[0][0])
