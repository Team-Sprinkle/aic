"""Inference definitions extracted from observed_board_marker.py; no training entry point."""
import time
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .geometry_sc_native_triangulation import FOCAL
from .shared_port_tcp_geometry import matrix_from_xyz_xyzw
MARKER_CORNERS_BOARD_M = np.array([[-0.125, 0.1, 0.013], [-0.125, 0.2, 0.013], [-0.025, 0.2, 0.013], [-0.025, 0.1, 0.013]], dtype=float)

def _box(rgb):
    bgr = rgb[..., ::-1]
    blue, green, red = cv2.split(bgr)
    mask = ((blue > 85) & (red > 85) & (green < 70) & (red > green * 1.7) & (blue > green * 1.7)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 2000:
        return None
    box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(float)
    if box[:, 0].min() < 12 or box[:, 0].max() > rgb.shape[1] - 13 or box[:, 1].min() < 12 or (box[:, 1].max() > rgb.shape[0] - 13):
        return None
    sides = np.linalg.norm(box - np.roll(box, -1, axis=0), axis=1)
    if sides.min() < 70 or sides.max() / sides.min() > 1.8:
        return None
    return box

def _camera_candidates(box, camera_base, initial):
    inverse = np.linalg.inv(camera_base)

    def project(pose):
        points = MARKER_CORNERS_BOARD_M @ Rotation.from_euler('z', pose[2]).as_matrix().T
        points += np.array([pose[0], pose[1], 0.0])
        optical = (inverse @ np.c_[points, np.ones(4)].T).T
        return optical[:, :2] / optical[:, 2, None] * FOCAL + [576.0, 512.0]
    candidates = []
    for shift in range(4):
        target = np.roll(box, -shift, axis=0)
        fitted = least_squares(lambda pose: (project(pose) - target).ravel(), initial, bounds=([-0.9, -0.4, -3.15], [0.1, 0.9, 3.15]), max_nfev=50)
        error = np.linalg.norm(project(fitted.x) - target, axis=1)
        if fitted.success and error.mean() <= 12.0 and (error.max() <= 18.0):
            candidates.append(dict(pose=fitted.x, error_px=float(error.mean()), shift=shift))
    return candidates

def estimate_board_marker(images_rgb, state, optics, nominal_board, nominal_port):
    """Return a uniquely selected visual board pose or None; no truth TF."""
    started = time.perf_counter()
    tcp = matrix_from_xyz_xyzw(state[:7])
    nominal = np.asarray(nominal_board, float)
    port = np.asarray(nominal_port, float)
    local_port = np.linalg.inv(nominal) @ port
    initial = [float(nominal[0, 3]), float(nominal[1, 3]), float(np.arctan2(nominal[1, 0], nominal[0, 0]))]
    detections = []
    for camera in ('center', 'left', 'right'):
        box = _box(images_rgb[camera])
        if box is None:
            continue
        area = float(abs(np.cross(box[1] - box[0], box[2] - box[1])))
        detections.append((area, camera, box))
    for _, camera, box in sorted(detections, reverse=True):
        candidates = []
        for fitted in _camera_candidates(box, tcp @ np.asarray(optics[camera]), initial):
            pose = fitted['pose']
            board = np.eye(4)
            board[:3, :3] = Rotation.from_euler('z', pose[2]).as_matrix()
            board[:2, 3] = pose[:2]
            predicted = board @ local_port
            prior_error = float(np.linalg.norm(predicted[:3, 3] - port[:3, 3]))
            candidates.append((prior_error, board, fitted['error_px']))
        if not candidates:
            continue
        candidates.sort(key=lambda row: row[0])
        prior_error, board, reprojection = candidates[0]
        if prior_error > 0.08 or (len(candidates) > 1 and candidates[1][0] - prior_error < 0.02):
            continue
        return dict(board=board, cameras=[camera], mean_reprojection_px=reprojection, camera_disagreement_mm=None, camera_yaw_disagreement_deg=None, selected_port_prior_error_mm=prior_error * 1000, inference_ms=(time.perf_counter() - started) * 1000, geometry_input=False)
    return None
