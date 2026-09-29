#!/usr/bin/env python3
"""Estimate fixed SC board yaw from visible rail lines in native RGB views.

The orientation landmark prediction only gives a coarse search prior. Long
board-rail edges provide an observation-only yaw refinement. Training-scene
TF calibrates the fixed tool cameras; scored per-scene TF is never consulted
by the estimator. Pose labels are read only to compute errors afterward.
"""

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from audit_sc_port_targets import matrix, pose_matrix
from build_sc_native_pose_labels import chain
from evaluate_sc_native_triangulation import triangulate
from evaluate_sc_orientation_landmarks import POINTS, orientation


CAMERAS = ('center', 'left', 'right')
WIDTH, HEIGHT, HFOV = 1152, 1024, .8718
FOCAL = WIDTH / (2 * math.tan(HFOV / 2))


def circular_line_error(a, b):
    return abs((a - b + math.pi / 2) % math.pi - math.pi / 2)


def weighted_median(values, weights):
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum() / 2)])


def projected_axis_angle(yaw, midpoint, base_optical):
    base_direction = Rotation.from_euler('z', yaw).as_matrix()[:, 0]
    optical_direction = base_optical[:3, :3].T @ base_direction
    if abs(optical_direction[2]) < 1e-8:
        ray = optical_direction[:2]
    else:
        vanishing = FOCAL * optical_direction[:2] / optical_direction[2] + np.array([WIDTH / 2, HEIGHT / 2])
        ray = vanishing - midpoint
    return math.atan2(ray[1], ray[0]) % math.pi


def camera_yaw(image_path, target_pixel, base_optical, prior):
    image = cv2.imread(image_path)
    if image is None or image.shape[:2] != (HEIGHT, WIDTH):
        raise ValueError(f'Missing or wrong native RGB: {image_path}')
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=90,
                            minLineLength=160, maxLineGap=25)
    if lines is None:
        return None
    target = np.asarray(target_pixel, dtype=float)
    grid = np.linspace(prior - .07, prior + .07, 141)
    base_directions = np.stack((np.cos(grid), np.sin(grid), np.zeros_like(grid)), axis=1)
    optical_directions = base_directions @ base_optical[:3, :3]
    z = optical_directions[:, 2]
    z = np.where(z >= 0, np.maximum(z, 1e-8), np.minimum(z, -1e-8))
    vanishing = FOCAL * optical_directions[:, :2] / z[:, None] + np.array([WIDTH / 2, HEIGHT / 2])
    estimates, weights, boundary_weights = [], [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        midpoint = np.array([(int(x1) + int(x2)) / 2, (int(y1) + int(y2)) / 2])
        length = math.hypot(int(x2) - int(x1), int(y2) - int(y1))
        if length < 240 or np.linalg.norm(midpoint - target) > 500:
            continue
        observed = math.atan2(int(y2) - int(y1), int(x2) - int(x1)) % math.pi
        prior_ray = vanishing[len(grid) // 2] - midpoint
        prior_angle = math.atan2(prior_ray[1], prior_ray[0]) % math.pi
        if math.degrees(circular_line_error(observed, prior_angle)) > 5:
            continue
        rays = vanishing - midpoint[None, :]
        angles = np.mod(np.arctan2(rays[:, 1], rays[:, 0]), math.pi)
        residuals = np.abs((observed - angles + math.pi / 2) % math.pi - math.pi / 2)
        best = int(np.argmin(residuals))
        estimates.append(float(grid[best]))
        weights.append(length)
        boundary_weights.append(length if best in (0, len(grid) - 1) else 0.)
    if len(estimates) < 3:
        return None
    estimate = weighted_median(estimates, weights)
    spread = np.median(np.abs(np.asarray(estimates) - estimate))
    return {'yaw_rad': estimate, 'lines': len(estimates), 'mad_deg': math.degrees(spread),
            'boundary_weight_fraction': sum(boundary_weights) / sum(weights)}


def stat(values):
    return {'count': len(values), 'median': float(np.median(values)) if len(values) else None,
            'p95': float(np.percentile(values, 95)) if len(values) else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--orientation-predictions', type=Path, required=True)
    p.add_argument('--position-predictions', type=Path, required=True)
    p.add_argument('--training-calibration-edges', type=Path, required=True)
    p.add_argument('--scored-edges', type=Path, required=True,
                   help='Scored scene TF for error calculation only; never passed to the estimator')
    p.add_argument('--board-port-calibration-json', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trace-output', type=Path,
                   help='Optional full per-frame estimate trace for development diagnosis')
    p.add_argument('--split', default='validation')
    p.add_argument('--window', type=int, default=40)
    p.add_argument('--max-camera-mad-deg', type=float, default=1.)
    p.add_argument('--max-view-spread-deg', type=float, default=1.)
    p.add_argument('--max-history-jump-deg', type=float, default=180.,
                   help='Optional development ablation; 180 disables this gate')
    p.add_argument('--reject-search-boundary', action='store_true',
                   help='Reject camera candidates whose local yaw search is edge-saturated')
    args = p.parse_args()
    rows = [json.loads(x) for x in args.labels.open()]
    orientations = [json.loads(x) for x in args.orientation_predictions.open()]
    positions = [json.loads(x) for x in args.position_predictions.open()]
    keys = lambda records: [(r['trial'], r['frame']) for r in records]
    if keys(rows) != keys(orientations) or keys(rows) != keys(positions):
        raise ValueError('Label/prediction identity mismatch')
    calibration = json.loads(args.training_calibration_edges.read_text())['rows'][0]['edges']
    tool_tcp = chain(calibration, ['tool0', 'cam_mount/cam_mount_link', 'ati/base_link',
                                   'ati/tool_link', 'gripper/hande_base_link', 'gripper/tcp'])
    tcp_optical = {}
    for camera in CAMERAS:
        tool_optical = chain(calibration, ['tool0', 'cam_mount/cam_mount_link',
            f'{camera}_camera/camera_link', f'{camera}_camera/sensor_link', f'{camera}_camera/optical'])
        tcp_optical[camera] = np.linalg.inv(tool_tcp) @ tool_optical
    mount = Rotation.from_quat(json.loads(args.board_port_calibration_json.read_text())['board_to_port_xyzw'])
    opening_offset = matrix(json.loads(Path('configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json').read_text())['base_to_opening'])
    scored_rows = {r['trial']: r for r in json.loads(args.scored_edges.read_text())['rows']}
    true_yaw_by_trial = {}
    for trial, scored in scored_rows.items():
        edges = scored['edges']
        port = scored['target_module_name'].rsplit('_', 1)[-1]
        parent = f'task_board/sc_port_{port}'
        world_base = chain(edges, ['world', 'tabletop', 'base_link'])
        world_port = chain(edges, ['aic_world', 'task_board', parent,
                                   parent + '/sc_port_base_link'])
        base_opening = np.linalg.inv(world_base) @ world_port @ opening_offset
        true_yaw_by_trial[trial] = (Rotation.from_matrix(base_opening[:3, :3]) * mount.inv()).as_euler('xyz')[2]
    history = {}; last_update = {}; records = []
    for row, pred, position_pred in zip(rows, orientations, positions):
        if row['split'] != args.split:
            continue
        tcp = pose_matrix(row['state'][:7])
        cameras = {cam: tcp @ offset for cam, offset in tcp_optical.items()}
        points = {name: triangulate({cam: pred['cameras'][cam][name] for cam in CAMERAS}, cameras)
                  for name in POINTS}
        prior = (orientation(points) * mount.inv()).as_euler('xyz')[2]
        camera_results = {}
        for cam in CAMERAS:
            camera_results[cam] = camera_yaw(row['images'][cam],
                position_pred['cameras'][cam]['opening_xy'], cameras[cam], prior)
        good = {cam: result for cam, result in camera_results.items()
                if result is not None and result['mad_deg'] <= args.max_camera_mad_deg
                and (not args.reject_search_boundary or result['boundary_weight_fraction'] < .5)}
        candidate = float(np.median([r['yaw_rad'] for r in good.values()])) if good else None
        spread = (math.degrees(max(r['yaw_rad'] for r in good.values()) -
                               min(r['yaw_rad'] for r in good.values())) if good else None)
        accepted = len(good) >= 2 and spread <= args.max_view_spread_deg
        episode = row['episode_id']
        stack = history.setdefault(episode, deque(maxlen=args.window))
        if accepted and len(stack) >= 2:
            prior_history = float(np.median(stack))
            jump = math.degrees(abs(math.atan2(math.sin(candidate-prior_history),
                                               math.cos(candidate-prior_history))))
            accepted = jump <= args.max_history_jump_deg
        elapsed = float(row['sim_time']) - last_update.get(episode, -float('inf'))
        if accepted and elapsed >= .5:
            stack.append(candidate)
            last_update[episode] = float(row['sim_time'])
        estimate = float(np.median(stack)) if len(stack) >= 2 else (candidate if candidate is not None else prior)
        # Use the scored scene transform only here, after all estimator
        # decisions. The physical SC grasp varies by reset, so reconstructing
        # this truth via a fixed TCP-to-tip proxy biases the metric.
        truth = true_yaw_by_trial[row['trial']]
        yaw_error = math.degrees(abs(math.atan2(math.sin(estimate - truth), math.cos(estimate - truth))))
        near = abs(row['observed_sc_tip_pose_opening_frame'][2]) < .03
        records.append({'trial': row['trial'], 'frame': row['frame'], 'near': near,
            'yaw_error_deg': yaw_error, 'raw_prior_error_deg': math.degrees(abs(prior - truth)),
            'estimated_yaw_deg': math.degrees(estimate),
            'true_yaw_deg': math.degrees(truth),
            'candidate_yaw_deg': math.degrees(candidate) if candidate is not None else None,
            'history_length': len(stack), 'accepted': bool(accepted),
            'view_count': len(good), 'view_spread_deg': spread,
            'camera_lines': {cam: result['lines'] if result else 0 for cam, result in camera_results.items()}})
    near = [r for r in records if r['near']]
    per_episode = defaultdict(list)
    for row in near:
        per_episode[row['trial']].append(row)
        report = {'schema': 'sc_board_lines_yaw/v1',
        'runtime_inputs': 'native RGB rails, RGB target pixels, orientation-landmark prior, measured TCP, fixed training camera/mount calibration',
        'true_geometry_runtime': False,
        'scored_truth_source': 'selected port opening TF from scored bag, used only after estimation',
        'training_calibration_edges': str(args.training_calibration_edges),
        'settings': {'window': args.window, 'max_camera_mad_deg': args.max_camera_mad_deg,
                     'max_view_spread_deg': args.max_view_spread_deg,
                     'max_history_jump_deg': args.max_history_jump_deg,
                     'reject_search_boundary': args.reject_search_boundary},
        'near_count': len(near), 'near_episodes': len(per_episode),
        'raw_prior_error_deg': stat([r['raw_prior_error_deg'] for r in near]),
        'yaw_error_deg': stat([r['yaw_error_deg'] for r in near]),
        'history_initialized_rate': sum(r['history_length'] >= 2 for r in near) / len(near) if near else None,
        'per_episode': {trial: {'count': len(group),
            'yaw_error_deg': stat([r['yaw_error_deg'] for r in group]),
            'history_initialized_rate': sum(r['history_length'] >= 2 for r in group) / len(group),
            'first_near_history_length': group[0]['history_length'],
            'near_acceptance_rate': sum(r['accepted'] for r in group) / len(group)}
            for trial, group in per_episode.items()},
        'worst_near': sorted(near, key=lambda r: r['yaw_error_deg'], reverse=True)[:20]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if args.trace_output:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        with args.trace_output.open('w') as stream:
            for record in records:
                stream.write(json.dumps(record) + '\n')
    print(json.dumps({'near_count': len(near), 'near_episodes': len(per_episode),
                      'yaw_p95_deg': report['yaw_error_deg']['p95'],
                      'history_initialized_rate': report['history_initialized_rate']}))


if __name__ == '__main__':
    main()
