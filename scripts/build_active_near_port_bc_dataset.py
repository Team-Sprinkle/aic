#!/usr/bin/env python3
"""Select meaningful near-port commands from verified predicted-frame BC rows.

The selected action is still the recorded full teacher target. This filter
removes settling holds, never fabricates an insertion command, and uses no
simulator geometry at actor inference.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def pose_rotation(pose):
    x = np.asarray(pose[3:6], dtype=float)
    y = np.asarray(pose[6:9], dtype=float)
    x /= np.linalg.norm(x)
    y -= x * np.dot(x, y)
    y /= np.linalg.norm(y)
    return Rotation.from_matrix(np.column_stack((x, y, np.cross(x, y))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--axial-start-m', type=float, default=-0.07)
    p.add_argument('--axial-end-m', type=float, default=0.025)
    p.add_argument('--minimum-position-m', type=float, default=0.0007)
    p.add_argument('--minimum-orientation-deg', type=float, default=0.7)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output already exists; preserve previous dataset')
    a.output.mkdir(parents=True)
    counts = Counter()
    episodes = defaultdict(set)
    depths = defaultdict(list)
    with (a.input / 'rows.jsonl').open() as source, (a.output / 'rows.jsonl').open('w') as target:
        for line in source:
            row = json.loads(line)
            key = (row['family'], row['split'])
            counts[(key, 'source')] += 1
            pose = np.asarray(row['predicted_port_tcp_pose9'], dtype=float)
            z = float(pose[2])
            if not a.axial_start_m <= z <= a.axial_end_m:
                counts[(key, 'outside_band')] += 1
                continue
            targets = np.asarray(row['teacher_tcp_targets_predicted_port_pose9'], dtype=float)
            motion = np.linalg.norm(targets[:, :3] - pose[:3], axis=1)
            rotation = pose_rotation(pose)
            angular = np.asarray([(rotation.inv() * pose_rotation(t)).magnitude() * 180 / np.pi for t in targets])
            # At deployment the first chunk member is always considered.
            # A moving fourth member cannot rescue an almost static first
            # target when the policy replans before that member executes.
            if motion[0] < a.minimum_position_m and angular[0] < a.minimum_orientation_deg:
                counts[(key, 'stationary_hold')] += 1
                continue
            target.write(line)
            counts[(key, 'kept')] += 1
            episodes[key].add(row['episode'])
            depths[key].append(z)
    summary = {
        'schema': 'aic_active_near_port_bc_dataset/v1',
        'source': str(a.input),
        'selection': vars(a) | {'input': str(a.input), 'output': str(a.output)},
        'counts': {f'{family}/{split}/{category}': count for ((family, split), category), count in sorted(counts.items())},
        'episodes': {f'{family}/{split}': len(groups) for (family, split), groups in sorted(episodes.items())},
        'axial_range_m': {f'{family}/{split}': [float(min(values)), float(max(values))]
                          for (family, split), values in sorted(depths.items())},
        'targets': 'recorded complete teacher TCP poses in RGB-estimated port frame',
        'true_geometry_actor_input': False,
        'heldout_scene_groups_unchanged': True,
    }
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
