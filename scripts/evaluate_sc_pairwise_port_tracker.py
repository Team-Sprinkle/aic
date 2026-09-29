#!/usr/bin/env python3
"""Score a causal, observation-only SC port tracker with a fixed camera pair.

This is a development ablation. The selected port chooses a fixed view pair;
neither the scored port pose nor projected label pixels enter tracking.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from audit_sc_port_targets import matrix, pose_matrix
from build_sc_native_pose_labels import chain
from evaluate_sc_native_triangulation import reprojection_residual_px, stats, triangulate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--edges', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trajectory-output', type=Path,
                   help='Write per-frame observation-only port positions for a separate plug/port fusion audit')
    p.add_argument('--acquire-distance-m', type=float, default=.04)
    p.add_argument('--stable-distance-m', type=float, default=.002)
    p.add_argument('--accept-distance-m', type=float, default=.0005)
    p.add_argument('--reset-distance-m', type=float, default=.005)
    p.add_argument('--window', type=int, default=40)
    p.add_argument('--three-view-gate-px', type=float, default=1.5)
    a = p.parse_args()
    labels = [json.loads(x) for x in a.labels.open()]
    predictions = [json.loads(x) for x in a.predictions.open()]
    if [(r['trial'], r['frame']) for r in labels] != [(r['trial'], r['frame']) for r in predictions]:
        raise ValueError('Prediction/label join mismatch')
    edge_rows = {r['trial']: r for r in json.loads(a.edges.read_text())['rows']}
    first = next(iter(edge_rows.values()))['edges']
    tool_tcp = chain(first, ['tool0', 'cam_mount/cam_mount_link', 'ati/base_link',
                             'ati/tool_link', 'gripper/hande_base_link', 'gripper/tcp'])
    offsets = {cam: np.linalg.inv(tool_tcp) @ chain(first, ['tool0', 'cam_mount/cam_mount_link',
               f'{cam}_camera/camera_link', f'{cam}_camera/sensor_link', f'{cam}_camera/optical'])
               for cam in ('center', 'left', 'right')}
    tip_offset = matrix(json.loads(Path('configs/hierarchical_recovery/sc_tcp_tip_gazebo_tf_251.json').read_text())['tcp_to_sc_tip'])
    opening_offset = matrix(json.loads(Path('configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json').read_text())['base_to_opening'])
    truth = {}
    for trial, row in edge_rows.items():
        port = row['target_module_name'].rsplit('_', 1)[-1]
        parent = f'task_board/sc_port_{port}'
        base = chain(row['edges'], ['world', 'tabletop', 'base_link'])
        world_port = chain(row['edges'], ['aic_world', 'task_board', parent,
                                          parent + '/sc_port_base_link'])
        truth[trial] = np.linalg.inv(base) @ world_port @ opening_offset
    histories = {}
    pending = {}
    last_update = {}
    results = []
    for row, pred in zip(labels, predictions):
        episode = row['episode_id']
        selected_port = int(np.argmax(row['task_vector'][2:4]))
        pair = ('left', 'right') if selected_port == 0 else ('center', 'right')
        base_tcp = pose_matrix(row['state'][:7])
        all_cameras = {cam: base_tcp @ offset for cam, offset in offsets.items()}
        all_pixels = {cam: pred['cameras'][cam]['opening_xy'] for cam in offsets}
        three_view = triangulate(all_pixels, all_cameras)
        three_view_residual = reprojection_residual_px(three_view, all_pixels, all_cameras)
        use_three = three_view_residual <= a.three_view_gate_px
        candidate = three_view if use_three else triangulate(
            {cam: all_pixels[cam] for cam in pair},
            {cam: all_cameras[cam] for cam in pair})
        tip_base = (base_tcp @ tip_offset)[:3, 3]
        history = histories.setdefault(episode, deque(maxlen=a.window))
        elapsed = float(row['sim_time']) - last_update.get(episode, -float('inf'))
        accepted = False
        if elapsed >= .5 and np.linalg.norm(candidate - tip_base) < a.acquire_distance_m:
            last_update[episode] = float(row['sim_time'])
            if history:
                center = np.median(np.stack(history), axis=0)
                jump = np.linalg.norm(candidate - center)
                if jump <= a.accept_distance_m:
                    history.append(candidate)
                    accepted = True
                    pending.pop(episode, None)
                elif jump >= a.reset_distance_m:
                    # Fixed targets cannot jump with the camera. Start a new
                    # hypothesis only after two stable, independent samples.
                    old = pending.get(episode)
                    pending[episode] = candidate
                    if old is not None and np.linalg.norm(candidate - old) <= a.stable_distance_m:
                        history.clear()
                        history.extend((old, candidate))
                        pending.pop(episode, None)
                        accepted = True
            else:
                old = pending.get(episode)
                pending[episode] = candidate
                if old is not None and np.linalg.norm(candidate - old) <= a.stable_distance_m:
                    history.extend((old, candidate))
                    pending.pop(episode, None)
                    accepted = True
        estimate = np.median(np.stack(history), axis=0) if len(history) >= 2 else None
        true_frame = truth[row['trial']]
        near = abs(row['observed_sc_tip_pose_opening_frame'][2]) < .03
        if estimate is None:
            lateral = axial = None
        else:
            local = true_frame[:3, :3].T @ (estimate - true_frame[:3, 3]) * 1000
            lateral, axial = float(np.linalg.norm(local[:2])), float(abs(local[2]))
        results.append({'trial': row['trial'], 'frame': row['frame'], 'near': near,
                        'selected_port': selected_port, 'camera_pair': pair,
                        'used_three_views': use_three,
                        'three_view_reprojection_px': three_view_residual,
                        'tip_candidate_distance_mm': float(np.linalg.norm(candidate - tip_base) * 1000),
                        'accepted': accepted, 'history_length': len(history),
                        'estimated_base_xyz_m': estimate.tolist() if estimate is not None else None,
                        'lateral_mm': lateral, 'axial_mm': axial})
    groups = defaultdict(list)
    for result in results:
        if result['near']:
            groups[result['trial']].append(result)
    observed = [r for group in groups.values() for r in group if r['lateral_mm'] is not None]
    all_near = [r for group in groups.values() for r in group]
    report = {'schema': 'sc_pairwise_port_tracker/v1',
              'runtime_inputs': 'RGB opening pixels, selected port task bit, observed TCP, fixed training camera calibration',
              'scored_geometry_used_in_estimator': False,
              'settings': {'port_0_fallback_cameras': ['left', 'right'],
                           'port_1_fallback_cameras': ['center', 'right'],
                           'three_view_gate_px': a.three_view_gate_px,
                           'acquire_distance_m': a.acquire_distance_m,
                           'stable_distance_m': a.stable_distance_m,
                           'accept_distance_m': a.accept_distance_m,
                           'reset_distance_m': a.reset_distance_m, 'window': a.window},
              'near_frames': len(all_near), 'near_episodes': len(groups),
              'initialized_rate': len(observed) / len(all_near) if all_near else None,
              'lateral_mm': stats([r['lateral_mm'] for r in observed]),
              'axial_mm': stats([r['axial_mm'] for r in observed]),
              'per_episode': {trial: {'near_frames': len(items),
                   'initialized_rate': sum(r['lateral_mm'] is not None for r in items) / len(items),
                   'lateral_mm': stats([r['lateral_mm'] for r in items if r['lateral_mm'] is not None]),
                   'axial_mm': stats([r['axial_mm'] for r in items if r['axial_mm'] is not None])}
                   for trial, items in groups.items()},
              'worst_near': sorted(observed, key=lambda r: r['lateral_mm'], reverse=True)[:20]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    if a.trajectory_output:
        a.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
        with a.trajectory_output.open('w') as stream:
            for row in results:
                stream.write(json.dumps(row) + '\n')
    print(json.dumps({'near_episodes': len(groups), 'initialized_rate': report['initialized_rate'],
                      'lateral_p95_mm': report['lateral_mm']['p95']}))


if __name__ == '__main__':
    main()
