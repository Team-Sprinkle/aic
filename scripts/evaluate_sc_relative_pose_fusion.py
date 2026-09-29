#!/usr/bin/env python3
"""Score physical plug-to-opening translation from two causal RGB estimates.

Scored TF supplies only the evaluation target and local error axes. The
estimator reads predicted RGB pixels, measured TCP, and fixed calibration.
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
    p.add_argument('--opening-predictions', type=Path, required=True)
    p.add_argument('--tip-predictions', type=Path, required=True)
    p.add_argument('--edges', type=Path, required=True)
    p.add_argument('--port-track', type=Path,
                   help='Precomputed observation-only causal port positions; skip the legacy three-view port tracker')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    labels = [json.loads(x) for x in args.labels.open()]
    openings = [json.loads(x) for x in args.opening_predictions.open()]
    tips = [json.loads(x) for x in args.tip_predictions.open()]
    keys = lambda data: [(r['trial'], r['frame']) for r in data]
    if keys(labels) != keys(openings) or keys(labels) != keys(tips):
        raise ValueError('Label/opening/tip identity mismatch')
    port_track = [json.loads(x) for x in args.port_track.open()] if args.port_track else None
    if port_track is not None and keys(labels) != keys(port_track):
        raise ValueError('Label/port-track identity mismatch')
    edge_rows = json.loads(args.edges.read_text())['rows']
    source = edge_rows[0]['edges']
    tool_tcp = chain(source, ['tool0', 'cam_mount/cam_mount_link', 'ati/base_link',
                              'ati/tool_link', 'gripper/hande_base_link', 'gripper/tcp'])
    tcp_optical = {}
    for cam in ('center', 'left', 'right'):
        optical = chain(source, ['tool0', 'cam_mount/cam_mount_link',
            f'{cam}_camera/camera_link', f'{cam}_camera/sensor_link', f'{cam}_camera/optical'])
        tcp_optical[cam] = np.linalg.inv(tool_tcp) @ optical
    fixed_tip = matrix(json.loads(Path('configs/hierarchical_recovery/sc_tcp_tip_gazebo_tf_251.json').read_text())['tcp_to_sc_tip'])
    opening_offset = matrix(json.loads(Path('configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json').read_text())['base_to_opening'])
    truth = {}
    for item in edge_rows:
        port = item['target_module_name'].rsplit('_', 1)[-1]
        parent = f'task_board/sc_port_{port}'
        world_base = chain(item['edges'], ['world', 'tabletop', 'base_link'])
        world_port = chain(item['edges'], ['aic_world', 'task_board', parent, parent + '/sc_port_base_link'])
        truth[item['trial']] = np.linalg.inv(world_base) @ world_port @ opening_offset
    port_hist = {}; tip_hist = {}; last_port = {}; last_tip = {}; records = []
    for index, (row, port_pred, tip_pred) in enumerate(zip(labels, openings, tips)):
        tcp = pose_matrix(row['state'][:7])
        cameras = {cam: tcp @ offset for cam, offset in tcp_optical.items()}
        episode = row['episode_id']; sim_time = float(row['sim_time'])
        if port_track is not None:
            track_row = port_track[index]
            port_base = (np.asarray(track_row['estimated_base_xyz_m'], dtype=float)
                         if track_row['estimated_base_xyz_m'] is not None else None)
            port_history_len = int(track_row['history_length'])
        else:
            port_pixels = {cam: port_pred['cameras'][cam]['opening_xy'] for cam in cameras}
            port_base = triangulate(port_pixels, cameras)
            port_residual = reprojection_residual_px(port_base, port_pixels, cameras)
            pstack = port_hist.setdefault(episode, deque(maxlen=40))
            pjump = np.linalg.norm(port_base - np.median(np.stack(pstack), axis=0)) if pstack else 0.
            proxy_tip = (tcp @ fixed_tip)[:3, 3]
            if (np.linalg.norm(port_base - proxy_tip) < .1 and port_residual <= 1.5
                    and pjump < .005 and sim_time - last_port.get(episode, -float('inf')) >= .5):
                pstack.append(port_base); last_port[episode] = sim_time
            if len(pstack) >= 2:
                port_base = np.median(np.stack(pstack), axis=0)
            port_history_len = len(pstack)
        tip_pixels = {cam: tip_pred['cameras'][cam]['tip_xy'] for cam in cameras}
        observed_tip_base = triangulate(tip_pixels, cameras)
        tip_offset = (np.linalg.inv(tcp) @ np.r_[observed_tip_base, 1.])[:3]
        tip_residual = reprojection_residual_px(observed_tip_base, tip_pixels, cameras)
        tstack = tip_hist.setdefault(episode, deque(maxlen=40))
        tjump = np.linalg.norm(tip_offset - np.median(np.stack(tstack), axis=0)) if tstack else 0.
        if (np.linalg.norm(tip_offset) <= .05 and tip_residual <= 2.
                and tjump <= .01 and sim_time - last_tip.get(episode, -float('inf')) >= .5):
            tstack.append(tip_offset); last_tip[episode] = sim_time
        estimate_offset = np.median(np.stack(tstack), axis=0) if len(tstack) >= 2 else tip_offset
        tip_base = (tcp @ np.r_[estimate_offset, 1.])[:3]
        actual_port = truth[row['trial']]
        actual_tip = (actual_port @ pose_matrix(row['observed_sc_tip_pose_opening_frame']))[:3, 3]
        local_error = (actual_port[:3, :3].T @ ((tip_base - port_base) -
                       (actual_tip - actual_port[:3, 3])) * 1000) if port_base is not None else None
        near = abs(row['observed_sc_tip_pose_opening_frame'][2]) < .03
        records.append({'trial': row['trial'], 'frame': row['frame'],
                        'split': row['split'], 'near': near,
                        'lateral_mm': float(np.linalg.norm(local_error[:2])) if local_error is not None else None,
                        'axial_mm': float(abs(local_error[2])) if local_error is not None else None,
                        'port_history': port_history_len, 'tip_history': len(tstack)})
    by_split = {}
    for split in ('train', 'validation'):
        group = [r for r in records if r['split'] == split and r['near']]
        episodes = defaultdict(list)
        for r in group:
            episodes[r['trial']].append(r)
        by_split[split] = {'near_frames': len(group), 'near_episodes': len(episodes),
            'relative_lateral_mm': stats([r['lateral_mm'] for r in group if r['lateral_mm'] is not None]),
            'relative_axial_mm': stats([r['axial_mm'] for r in group if r['axial_mm'] is not None]),
            'both_histories_initialized_rate': sum(r['port_history'] >= 2 and r['tip_history'] >= 2
                                                    for r in group) / len(group) if group else None,
            'per_episode': {key: {'frames': len(rows),
                'relative_lateral_mm': stats([r['lateral_mm'] for r in rows if r['lateral_mm'] is not None]),
                'relative_axial_mm': stats([r['axial_mm'] for r in rows if r['axial_mm'] is not None]),
                'first_near_port_history': rows[0]['port_history'],
                'first_near_tip_history': rows[0]['tip_history']}
                for key, rows in episodes.items()}}
    report = {'schema': 'sc_causal_physical_plug_to_opening/v1',
              'runtime_inputs': 'RGB-predicted port and tip pixels, measured TCP, fixed training camera calibration, causal histories',
              'scored_tf_runtime': False, 'by_split': by_split,
              'per_frame': records,
              'port_track': str(args.port_track) if args.port_track else None,
              'orientation_note': 'Position error components use true port axes only to score; board-yaw estimation is evaluated separately.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'validation_relative_lateral_p95_mm':
                      by_split['validation']['relative_lateral_mm']['p95'],
                      'both_histories_initialized_rate':
                      by_split['validation']['both_histories_initialized_rate']}))


if __name__ == '__main__':
    main()
