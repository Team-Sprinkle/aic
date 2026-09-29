#!/usr/bin/env python3
"""Score RGB-predicted SC plug-tip keypoints against scored physical-tip labels."""

import argparse
import json
from collections import defaultdict
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
    args = p.parse_args()
    labels = [json.loads(x) for x in args.labels.open()]
    preds = [json.loads(x) for x in args.predictions.open()]
    if [(r['trial'], r['frame']) for r in labels] != [(r['trial'], r['frame']) for r in preds]:
        raise ValueError('Label/prediction frame mismatch')
    edge_rows = json.loads(args.edges.read_text())['rows']
    source = edge_rows[0]['edges']
    tool_tcp = chain(source, ['tool0', 'cam_mount/cam_mount_link', 'ati/base_link',
                              'ati/tool_link', 'gripper/hande_base_link', 'gripper/tcp'])
    cameras = {}
    for cam in ('center', 'left', 'right'):
        optical = chain(source, ['tool0', 'cam_mount/cam_mount_link',
            f'{cam}_camera/camera_link', f'{cam}_camera/sensor_link', f'{cam}_camera/optical'])
        cameras[cam] = np.linalg.inv(tool_tcp) @ optical
    opening_offset = matrix(json.loads(Path('configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json').read_text())['base_to_opening'])
    true_opening = {}
    for item in edge_rows:
        port = item['target_module_name'].rsplit('_', 1)[-1]
        parent = f'task_board/sc_port_{port}'
        world_base = chain(item['edges'], ['world', 'tabletop', 'base_link'])
        world_port = chain(item['edges'], ['aic_world', 'task_board', parent, parent + '/sc_port_base_link'])
        true_opening[item['trial']] = np.linalg.inv(world_base) @ world_port @ opening_offset
    records = []
    for row, pred in zip(labels, preds):
        tcp = pose_matrix(row['state'][:7])
        optical = {cam: tcp @ offset for cam, offset in cameras.items()}
        pixels = {cam: pred['cameras'][cam]['tip_xy'] for cam in cameras}
        predicted = triangulate(pixels, optical)
        actual = (true_opening[row['trial']] @ pose_matrix(row['observed_sc_tip_pose_opening_frame']))[:3, 3]
        local = true_opening[row['trial']][:3, :3].T @ (predicted - actual) * 1000
        near = abs(row['observed_sc_tip_pose_opening_frame'][2]) < .03
        records.append({'trial': row['trial'], 'frame': row['frame'], 'split': row['split'], 'near': near,
            'lateral_mm': float(np.linalg.norm(local[:2])), 'axial_mm': float(abs(local[2])),
            'translation_mm': float(np.linalg.norm(local)),
            'reprojection_residual_px': reprojection_residual_px(predicted, pixels, optical)})
    near = [r for r in records if r['near']]
    groups = defaultdict(list)
    for row in near:
        groups[row['trial']].append(row)
    report = {'schema': 'sc_physical_tip_keypoint_evaluation/v1',
              'runtime_inputs': 'RGB-predicted tip pixels, measured TCP, fixed training camera calibration',
              'scored_tf_runtime': False, 'labels': str(args.labels),
              'near_count': len(near), 'near_episodes': len(groups),
              'near_lateral_mm': stats([r['lateral_mm'] for r in near]),
              'near_axial_mm': stats([r['axial_mm'] for r in near]),
              'near_translation_mm': stats([r['translation_mm'] for r in near]),
              'near_reprojection_residual_px': stats([r['reprojection_residual_px'] for r in near]),
              'by_split': {split: {'near_frames': len(group),
                  'near_episodes': len({r['trial'] for r in group}),
                  'lateral_mm': stats([r['lateral_mm'] for r in group]),
                  'axial_mm': stats([r['axial_mm'] for r in group])}
                  for split, group in {split: [r for r in near if r['split'] == split]
                                       for split in ('train', 'validation')}.items()},
              'by_episode': {key: {'frames': len(group),
                  'lateral_mm': stats([r['lateral_mm'] for r in group]),
                  'axial_mm': stats([r['axial_mm'] for r in group])}
                  for key, group in groups.items()},
              'worst_near': sorted(near, key=lambda r: r['translation_mm'], reverse=True)[:20]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'episodes': len(groups), 'lateral_p95_mm': report['near_lateral_mm']['p95'],
                      'axial_p95_mm': report['near_axial_mm']['p95']}))


if __name__ == '__main__':
    main()
