"""Inference definitions extracted from close_grasp_consensus.py; no training entry point."""
import numpy as np
from scipy.spatial.transform import Rotation

class CloseGraspConsensus:

    def __init__(self, samples=5, translation_span_mm=0.2, rotation_span_deg=0.5):
        self.samples = samples
        self.translation_span_mm = translation_span_mm
        self.rotation_span_deg = rotation_span_deg
        self.window = []
        self.last_stamps = None

    def add(self, stamps, position, rotation):
        stamps = tuple(stamps)
        if len(stamps) != 3 or (self.last_stamps is not None and any((a <= b for a, b in zip(stamps, self.last_stamps)))):
            return dict(accepted=False, reason='three_distinct_fresh_camera_stamps_required')
        position = np.asarray(position, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        if position.shape != (3,) or rotation.shape != (3, 3) or (not np.isfinite(position).all()) or (not np.isfinite(rotation).all()) or (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-05)) or (not np.isclose(np.linalg.det(rotation), 1, atol=1e-05)):
            return dict(accepted=False, reason='invalid_pose')
        self.last_stamps = stamps
        self.window = (self.window + [(position, rotation)])[-self.samples:]
        if len(self.window) < self.samples:
            return dict(accepted=False, reason='collecting_distinct_frames', samples=len(self.window))
        positions = np.stack([x[0] for x in self.window])
        rotations = np.stack([x[1] for x in self.window])
        pspan = float(np.linalg.norm(positions[:, None] - positions[None, :], axis=-1).max() * 1000)
        rspan = max((float(np.rad2deg(Rotation.from_matrix(a.T @ b).magnitude())) for a in rotations for b in rotations))
        result = dict(accepted=pspan <= self.translation_span_mm and rspan <= self.rotation_span_deg, translation_span_mm=pspan, rotation_span_deg=rspan, samples=len(self.window))
        result['reason'] = 'stable_close_pose' if result['accepted'] else 'inconsistent_close_pose'
        if result['accepted']:
            result.update(position_m=np.median(positions, axis=0).tolist(), rotation=Rotation.from_matrix(rotations).mean().as_matrix().tolist())
        return result

def compare_grasp_update(old_position, old_rotation, candidate):
    """Large disagreement requests observation/reacquisition; never blend blindly."""
    distance = float(np.linalg.norm(np.asarray(candidate['position_m']) - old_position) * 1000)
    angle = float(np.rad2deg(Rotation.from_matrix(np.asarray(old_rotation).T @ np.asarray(candidate['rotation'])).magnitude()))
    return dict(accept_update=distance <= 1 and angle <= 3, translation_change_mm=distance, rotation_change_deg=angle, large_disagreement_requires_second_window_or_reacquisition=distance > 1 or angle > 3, consistency_is_not_absolute_accuracy=True)
