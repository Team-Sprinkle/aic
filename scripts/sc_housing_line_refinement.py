"""Fixed long-housing-line orientation diagnostic, not runtime-approved."""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from sc_mesh_edge_refinement import FOCAL


def refine(images, origins, optics, initial, edges):
    # Housing only, long enough to estimate a line angle. The complete mesh's
    # short/internal/occluded edges failed the earlier chamfer screen.
    edges = [(p, n) for p, n in edges if np.linalg.norm(p[1]-p[0]) >= .005]
    data = []
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    for ci, camera in enumerate(['center', 'left', 'right']):
        x, y = origins[ci]
        crop = cv2.warpAffine(images[camera], np.array([[1, 0, -x], [0, 1, -y]], np.float32), (192, 192))
        lines = detector.detect(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY))[0]
        lines = np.empty((0, 2, 2)) if lines is None else lines.reshape(-1, 2, 2)
        length = np.linalg.norm(lines[:, 1]-lines[:, 0], axis=1)
        lines = lines[length >= 10]
        direction = lines[:, 1]-lines[:, 0]
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        data.append((np.linalg.inv(np.asarray(optics[camera])), np.array([576-x, 512-y]), lines, direction))

    def matrix(delta):
        result = initial.copy()
        result[:3, :3] = initial[:3, :3]@Rotation.from_rotvec(np.deg2rad(delta)).as_matrix()
        return result

    def project(points, pose, ci):
        transform, center, _, _ = data[ci]
        q = (points@pose[:3, :3].T+pose[:3, 3])@transform[:3, :3].T+transform[:3, 3]
        return q[:, :2]/q[:, 2:]*FOCAL+center

    delta = np.zeros(3)
    history = []
    for iteration in range(3):
        current = matrix(delta)
        pairs = []
        for ci, (transform, _, lines, directions) in enumerate(data):
            if not len(lines):
                continue
            camera_in_tip = np.linalg.inv(current)@np.linalg.inv(transform)
            for points, normals in edges:
                if np.max(normals@(camera_in_tip[:3, 3]-points.mean(0))) <= 0:
                    continue
                uv = project(points, current, ci)
                tangent = uv[1]-uv[0]
                length = np.linalg.norm(tangent)
                if length < 10:
                    continue
                direction = tangent/length
                normal = np.array([-direction[1], direction[0]])
                angle = np.arccos(np.clip(np.abs(directions@direction), 0, 1))
                midpoint = uv.mean(0)
                distance = np.abs((lines.mean(1)-midpoint)@normal)
                projection = (lines-midpoint)@direction
                overlap = np.maximum(0, np.minimum(projection.max(1), length/2)-np.maximum(projection.min(1), -length/2))
                valid = (angle <= np.deg2rad(10)) & (distance <= 3) & (overlap >= .4*length)
                if not valid.any():
                    continue
                costs = np.where(valid, distance+angle*10, np.inf)
                chosen = np.argmin(costs)
                pairs.append((ci, points, directions[chosen]))
        if len({p[0] for p in pairs}) < 2 or len(pairs) < 6:
            return initial.copy(), dict(accepted=False, reason='insufficient_consistent_long_lines', history=history)
        def residual(value):
            pose = matrix(value)
            errors = []
            for ci, points, observed in pairs:
                uv = project(points, pose, ci)
                predicted = uv[1]-uv[0]
                predicted /= np.linalg.norm(predicted)
                errors.append((predicted[0]*observed[1]-predicted[1]*observed[0])*50)
            return np.r_[errors, .05*value]
        result = least_squares(residual, delta, bounds=(-2, 2), loss='soft_l1', f_scale=.5, max_nfev=40)
        delta = result.x
        history.append(dict(iteration=iteration, pairs=len(pairs), camera_count=len({p[0] for p in pairs}),
                            rms=float(np.sqrt(np.mean(residual(delta)**2)))))
    return matrix(delta), dict(accepted=True, rotation_delta_deg=delta.tolist(), history=history,
                               runtime_true_geometry=False, diagnostic_only=True,
                               translation_retained=True)
