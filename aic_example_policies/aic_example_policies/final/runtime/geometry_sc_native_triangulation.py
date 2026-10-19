"""Inference definitions extracted from evaluate_sc_native_triangulation.py; no training entry point."""
import math
import numpy as np
from .build_sc_native_pose_labels import WIDTH, HEIGHT, HFOV
FOCAL = WIDTH / (2 * math.tan(HFOV / 2))

def triangulate(camera_pixels, base_optical):
    lhs = np.zeros((3, 3), np.float64)
    rhs = np.zeros(3, np.float64)
    for cam, (u, v) in camera_pixels.items():
        transform = base_optical[cam]
        ray = transform[:3, :3] @ np.array([(u - WIDTH / 2) / FOCAL, (v - HEIGHT / 2) / FOCAL, 1.0])
        ray /= np.linalg.norm(ray)
        projection = np.eye(3) - np.outer(ray, ray)
        lhs += projection
        rhs += projection @ transform[:3, 3]
    return np.linalg.solve(lhs + 1e-09 * np.eye(3), rhs)
