#!/usr/bin/env python3
"""Convert verified SFP teacher TCP-body commands to port-frame TCP targets.

Scene YAML and bag-derived fixed port geometry are training labels only.
The actor receives images, ordinary state, task encoding, and an independent
observation-only port/plug estimator at inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from audit_sc_port_targets import matrix, pose_list, pose_matrix


def transform(xyz, rpy):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    out[:3, 3] = xyz
    return out


def action_transform(action):
    out = np.eye(4)
    out[:3, 3] = action[:3]
    out[:3, :3] = Rotation.from_rotvec(action[3:]).as_matrix()
    return out


def scene_port_pose(episode, calibration):
    scene_path = Path(episode['trial_yaml'])
    config = yaml.safe_load(scene_path.read_text())
    trial = config['trials'][episode['trial_id']]
    task = trial['tasks']['task_1']
    if task['target_module_name'] != f"nic_card_mount_{episode['task']['target_card_index']}":
        raise ValueError(f'Task/manifest card mismatch: {scene_path}')
    if task['port_name'] != f"sfp_port_{episode['task']['target_port_index']}":
        raise ValueError(f'Task/manifest port mismatch: {scene_path}')
    board = trial['scene']['task_board']
    bp = board['pose']
    world_board = transform([bp[k] for k in ('x', 'y', 'z')],
                            [bp[k] for k in ('roll', 'pitch', 'yaw')])
    card = int(episode['task']['target_card_index'])
    rail = board[f'nic_rail_{card}']
    if not rail['entity_present'] or rail['entity_name'] != f'nic_card_{card}':
        raise ValueError(f'Missing selected NIC card: {scene_path}')
    rp = rail['entity_pose']; mount = calibration['board_to_nic_mount']
    board_nic = transform([mount['x_nominal_m'] + rp['translation'],
                           mount['y_rail0_m'] + mount['y_step_per_rail_m'] * card,
                           mount['z_m']], [rp[k] for k in ('roll', 'pitch', 'yaw')])
    port = calibration['nic_mount_to_port_opening'][str(episode['task']['target_port_index'])]
    world_port = world_board @ board_nic @ matrix(port)
    world_base = matrix(calibration['world_to_base'])
    return np.linalg.inv(world_base) @ world_port, scene_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--horizon', type=int, default=4)
    p.add_argument('--step-seconds', type=float, default=.05)
    p.add_argument('--time-tolerance-seconds', type=float, default=.01)
    a = p.parse_args()
    if a.horizon < 1:
        raise ValueError('Positive waypoint horizon required')
    info_path = a.cache / 'cache.json'
    info = json.loads(info_path.read_text())
    calibration = json.loads(a.calibration.read_text())
    arrays = {name: np.load(a.cache / (name + '.npy'), mmap_mode='r')
              for name in ('states', 'actions', 'timestamps', 'task_vectors')}
    reports = []; total = 0; max_translation = max_rotation = 0.; splits = Counter()
    train_scenes = set(); other_scenes = set()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w') as stream:
        for episode in info['episodes']:
            if episode['task']['task_family'] != 'sfp_to_nic':
                continue
            start, stop = int(episode['cache_from_index']), int(episode['cache_to_index'])
            if stop - start != int(episode['frames']):
                raise ValueError(f'Cache episode length mismatch: {episode["episode_index"]}')
            if episode['split'] == 'train':
                train_scenes.add(episode['scene_sha256'])
            else:
                other_scenes.add(episode['scene_sha256'])
            base_port, scene_yaml = scene_port_pose(episode, calibration)
            port_base = np.linalg.inv(base_port)
            count = 0; invalid_timing = 0
            for index in range(start, stop - a.horizon + 1):
                times = np.asarray(arrays['timestamps'][index:index+a.horizon], dtype=float)
                if np.any(np.abs(np.diff(times) - a.step_seconds) > a.time_tolerance_seconds):
                    invalid_timing += 1
                    continue
                target_poses = []
                for j in range(index, index + a.horizon):
                    state = np.asarray(arrays['states'][j], dtype=float)
                    action = np.asarray(arrays['actions'][j], dtype=float)
                    tcp = pose_matrix(state[:7])
                    tcp_target = tcp @ action_transform(action)
                    target_port = port_base @ tcp_target
                    recovered = np.linalg.inv(tcp) @ base_port @ target_port
                    max_translation = max(max_translation, float(np.linalg.norm(recovered[:3, 3] - action[:3])))
                    max_rotation = max(max_rotation, float(np.degrees((Rotation.from_matrix(recovered[:3, :3]).inv() * Rotation.from_rotvec(action[3:])).magnitude())))
                    target_poses.append(pose_list(target_port))
                stream.write(json.dumps({
                    'schema': 'sfp_port_tcp_waypoints/v1',
                    'episode_index': int(episode['episode_index']),
                    'scene_sha256': episode['scene_sha256'], 'split': episode['split'],
                    'frame_index': int(index-start), 'cache_frame_index': index,
                    'timestamp_s': float(times[0]), 'source_scene_yaml': str(scene_yaml),
                    'source_dataset': episode.get('dataset') or episode.get('collection'),
                    'image_cache_root': str(a.cache),
                    'state': np.asarray(arrays['states'][index], dtype=float).tolist(),
                    'task_vector': np.asarray(arrays['task_vectors'][index], dtype=float).tolist(),
                    'selected_port_frame': f"task_board/nic_card_mount_{episode['task']['target_card_index']}/sfp_port_{episode['task']['target_port_index']}_link_entrance",
                    'observed_tcp_pose_port_frame': pose_list(port_base @ pose_matrix(arrays['states'][index][:7])),
                    'teacher_tcp_target_poses_port_frame': target_poses,
                    'target_source': 'verified recorded teacher TCP-body commands composed with their aligned observed TCP states',
                }, separators=(',', ':')) + '\n')
                count += 1
            total += count; splits[episode['split']] += count
            reports.append({'episode_index': int(episode['episode_index']), 'split': episode['split'],
                            'scene_sha256': episode['scene_sha256'], 'chunks': count,
                            'invalid_timing': invalid_timing,
                            'source_scene_yaml_sha256': hashlib.sha256(scene_yaml.read_bytes()).hexdigest()})
    if train_scenes & other_scenes:
        raise ValueError('SFP scene leakage across splits')
    summary = {'schema': 'sfp_port_tcp_waypoint_summary/v1',
               'source_cache': str(a.cache), 'source_cache_sha256': hashlib.sha256(info_path.read_bytes()).hexdigest(),
               'calibration': str(a.calibration), 'calibration_sha256': hashlib.sha256(a.calibration.read_bytes()).hexdigest(),
               'output': str(a.output), 'output_sha256': hashlib.sha256(a.output.read_bytes()).hexdigest(),
               'training_label_only_geometry': True,
               'episode_count': len(reports), 'chunks': total, 'chunks_by_split': dict(splits),
               'train_scene_groups': len(train_scenes), 'validation_scene_groups': len(other_scenes),
               'max_action_target_roundtrip_m': max_translation,
               'max_action_rotation_roundtrip_deg': max_rotation,
               'target_frame': 'selected NIC SFP-port entrance, full TCP pose, XYZW quaternion',
               'reports': reports}
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'episodes': len(reports), 'chunks': total,
                      'max_roundtrip_m': max_translation, 'train_val_scene_overlap': 0}))


if __name__ == '__main__':
    main()
