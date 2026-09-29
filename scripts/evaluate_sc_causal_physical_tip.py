#!/usr/bin/env python3
"""Estimate a stable TCP-to-physical-plug offset from RGB observations.

Scored physical TF labels appear only in the scoring branch. The causal
history is built from predicted pixels and ordinary measured TCP poses.
"""

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from audit_sc_port_targets import matrix, pose_matrix
from build_sc_native_pose_labels import chain
from evaluate_sc_native_triangulation import triangulate, reprojection_residual_px, stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--edges', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--window', type=int, default=40)
    p.add_argument('--reprojection-gate-px', type=float, default=2.)
    p.add_argument('--max-offset-m', type=float, default=.05)
    p.add_argument('--max-jump-m', type=float, default=.01)
    args = p.parse_args()
    labels = [json.loads(x) for x in args.labels.open()]
    preds = [json.loads(x) for x in args.predictions.open()]
    if [(r['trial'], r['frame']) for r in labels] != [(r['trial'], r['frame']) for r in preds]:
        raise ValueError('Label/prediction frame mismatch')
    edge_rows = json.loads(args.edges.read_text())['rows']
    source = edge_rows[0]['edges']
    tool_tcp = chain(source, ['tool0', 'cam_mount/cam_mount_link', 'ati/base_link',
                              'ati/tool_link', 'gripper/hande_base_link', 'gripper/tcp'])
    tcp_optical = {}
    for cam in ('center', 'left', 'right'):
        optical = chain(source, ['tool0', 'cam_mount/cam_mount_link',
            f'{cam}_camera/camera_link', f'{cam}_camera/sensor_link', f'{cam}_camera/optical'])
        tcp_optical[cam] = np.linalg.inv(tool_tcp) @ optical
    opening_offset = matrix(json.loads(Path('configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json').read_text())['base_to_opening'])
    true_openings = {}
    for item in edge_rows:
        port = item['target_module_name'].rsplit('_', 1)[-1]
        parent = f'task_board/sc_port_{port}'
        world_base = chain(item['edges'], ['world', 'tabletop', 'base_link'])
        world_port = chain(item['edges'], ['aic_world', 'task_board', parent, parent + '/sc_port_base_link'])
        true_openings[item['trial']] = np.linalg.inv(world_base) @ world_port @ opening_offset
    history = {}; last_update = {}; records = []
    for row, pred in zip(labels, preds):
        tcp = pose_matrix(row['state'][:7])
        cameras = {cam: tcp @ offset for cam, offset in tcp_optical.items()}
        pixels = {cam: pred['cameras'][cam]['tip_xy'] for cam in cameras}
        observed_base = triangulate(pixels, cameras)
        offset = (np.linalg.inv(tcp) @ np.r_[observed_base, 1.])[:3]
        residual = reprojection_residual_px(observed_base, pixels, cameras)
        stack = history.setdefault(row['episode_id'], deque(maxlen=args.window))
        gap = float(row['sim_time']) - last_update.get(row['episode_id'], -float('inf'))
        jump = np.linalg.norm(offset - np.median(np.stack(stack), axis=0)) if stack else 0.
        accepted = (np.linalg.norm(offset) <= args.max_offset_m and
                    residual <= args.reprojection_gate_px and
                    jump <= args.max_jump_m and gap >= .5)
        if accepted:
            stack.append(offset)
            last_update[row['episode_id']] = float(row['sim_time'])
        estimate = np.median(np.stack(stack), axis=0) if len(stack) >= 2 else offset
        predicted = (tcp @ np.r_[estimate, 1.])[:3]
        truth = (true_openings[row['trial']] @ pose_matrix(row['observed_sc_tip_pose_opening_frame']))[:3, 3]
        local = true_openings[row['trial']][:3, :3].T @ (predicted - truth) * 1000
        near = abs(row['observed_sc_tip_pose_opening_frame'][2]) < .03
        records.append({'trial': row['trial'], 'split': row['split'], 'near': near,
                        'lateral_mm': float(np.linalg.norm(local[:2])),
                        'axial_mm': float(abs(local[2])), 'accepted': accepted,
                        'history_length': len(stack), 'reprojection_residual_px': residual})
    by_split = {}
    for split in ('train', 'validation'):
        group = [r for r in records if r['split'] == split and r['near']]
        episodes = defaultdict(list)
        for row in group:
            episodes[row['trial']].append(row)
        by_split[split] = {'near_frames': len(group), 'near_episodes': len(episodes),
            'lateral_mm': stats([r['lateral_mm'] for r in group]),
            'axial_mm': stats([r['axial_mm'] for r in group]),
            'history_initialized_rate': sum(r['history_length'] >= 2 for r in group) / len(group) if group else None,
            'near_acceptance_rate': sum(r['accepted'] for r in group) / len(group) if group else None,
            'per_episode': {key: {'frames': len(rows),
                'lateral_mm': stats([r['lateral_mm'] for r in rows]),
                'axial_mm': stats([r['axial_mm'] for r in rows]),
                'first_near_history_length': rows[0]['history_length']}
                for key, rows in episodes.items()}}
    report = {'schema': 'sc_causal_physical_tip/v1',
              'runtime_inputs': 'RGB-predicted physical-tip pixels, measured TCP, fixed training camera calibration, causal history',
              'true_tip_tf_runtime': False,
              'settings': {'window': args.window, 'reprojection_gate_px': args.reprojection_gate_px,
                           'max_offset_m': args.max_offset_m, 'max_jump_m': args.max_jump_m},
              'by_split': by_split}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'validation_lateral_p95_mm': by_split['validation']['lateral_mm']['p95'],
                      'validation_history_initialized_rate': by_split['validation']['history_initialized_rate']}))


if __name__ == '__main__':
    main()
