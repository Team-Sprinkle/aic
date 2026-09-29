#!/usr/bin/env python3
"""Replace a fixed SC plug-tip proxy with episode-specific scored TF labels.

The scored transform is used only to create training/evaluation labels. The
runtime model must infer the plug from images and measured robot state.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_sc_port_targets import matrix, pose_list, pose_matrix
from build_sc_native_pose_labels import chain, project


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--edges', type=Path, required=True)
    p.add_argument('--calibrations', type=Path, nargs='+', required=True)
    p.add_argument('--fixed-tcp-tip-calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--summary', type=Path, required=True)
    args = p.parse_args()
    source = json.loads(args.edges.read_text())['rows'][0]['edges']
    tool_tcp = chain(source, ['tool0', 'cam_mount/cam_mount_link', 'ati/base_link',
                              'ati/tool_link', 'gripper/hande_base_link', 'gripper/tcp'])
    cameras = {}
    for cam in ('center', 'left', 'right'):
        tool_optical = chain(source, ['tool0', 'cam_mount/cam_mount_link',
            f'{cam}_camera/camera_link', f'{cam}_camera/sensor_link', f'{cam}_camera/optical'])
        cameras[cam] = np.linalg.inv(tool_tcp) @ tool_optical
    old_tip = matrix(json.loads(args.fixed_tcp_tip_calibration.read_text())['tcp_to_sc_tip'])
    physical = {}
    source_by_trial = {}
    for calibration_file in args.calibrations:
        for episode in json.loads(calibration_file.read_text())['episodes']:
            trial = episode['trial']
            value = pose_matrix(episode['samples'][0]['tcp_to_physical_sc_tip'])
            if trial in physical and not np.allclose(physical[trial], value, atol=1e-7):
                raise ValueError(f'Conflicting physical grasp for {trial}')
            physical[trial] = value
            source_by_trial[trial] = str(calibration_file)
    counts = {}
    misses = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as out:
        for line in args.labels.open():
            row = json.loads(line)
            trial = row['trial']
            if trial not in physical:
                misses.add(trial)
                continue
            tcp = pose_matrix(row['state'][:7])
            old_base_tip = tcp @ old_tip
            old_relative = pose_matrix(row['observed_sc_tip_pose_opening_frame'])
            base_opening = old_base_tip @ np.linalg.inv(old_relative)
            physical_base_tip = tcp @ physical[trial]
            physical_relative = np.linalg.inv(base_opening) @ physical_base_tip
            row['fixed_proxy_sc_tip_pose_opening_frame'] = row['observed_sc_tip_pose_opening_frame']
            row['observed_sc_tip_pose_opening_frame'] = pose_list(physical_relative)
            for camera, tcp_optical in cameras.items():
                row['projected_training_pixels'][camera]['tip'] = project(
                    physical_base_tip, tcp @ tcp_optical)
            row['tip_label_source'] = 'scored physical plug TF sampled at episode start; training/evaluation only'
            row['runtime_physical_tf_input'] = False
            out.write(json.dumps(row, separators=(',', ':')) + '\n')
            counts[trial] = counts.get(trial, 0) + 1
    if misses:
        raise ValueError(f'Missing physical tip calibration for {sorted(misses)}')
    report = {'schema': 'sc_physical_tip_relabel/v1', 'source_labels': str(args.labels),
              'source_labels_sha256': digest(args.labels), 'edges': str(args.edges),
              'edges_sha256': digest(args.edges),
              'calibrations': {str(x): digest(x) for x in args.calibrations},
              'fixed_proxy_calibration_sha256': digest(args.fixed_tcp_tip_calibration),
              'rows': sum(counts.values()), 'episodes': len(counts),
              'rows_by_trial': counts, 'calibration_by_trial': source_by_trial,
              'runtime_physical_tf_input': False,
              'caveat': 'Episode-start physical grasp is nearly rigid in audited scenes; hold-out policy must predict plug from RGB, not read this calibration.'}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'episodes': len(counts), 'rows': sum(counts.values())}))


if __name__ == '__main__':
    main()
