#!/usr/bin/env python3
"""Add supervised SFP one-second route and safe-standoff labels to a frozen cache.

Ground-truth selected-port geometry is read only to create offline labels.
The runtime receives no target transform. The coordinate prior is fitted from
training positives only and can veto, never trigger, a handoff proposal.
"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from train_shared_port_tcp_bc import pose9


def task_key(row):
    task = np.asarray(row['task'], float)
    if row['family'] == 'sfp':
        return f'sfp_card_{int(task[4:9].argmax())}_port_{int(task[2:4].argmax())}'
    return f'sc_port_{int(task[2:4].argmax())}'


def future_pose9(current, future):
    a = Rotation.from_quat(current[3:7])
    b = Rotation.from_quat(future[3:7])
    rel = np.r_[a.inv().apply(np.asarray(future[:3]) - current[:3]),
                (a.inv() * b).as_quat()]
    return pose9(rel).tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--sfp-labels', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--future-seconds', type=float, default=1.0)
    p.add_argument('--prior-margin-mm', type=float, default=60.0)
    p.add_argument('--sfp-axial-min-m',type=float,default=-.100)
    p.add_argument('--sfp-axial-max-m',type=float,default=-.070)
    p.add_argument('--sfp-lateral-max-m',type=float,default=.010)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve previous cache')
    if a.future_seconds <= 0 or a.prior_margin_mm <= 0:
        p.error('Positive future horizon and prior margin required')
    if not (a.sfp_axial_min_m<a.sfp_axial_max_m<0 and a.sfp_lateral_max_m>0):
        p.error('Invalid SFP proposal label window')
    labels = {}
    for line in a.sfp_labels.open():
        row = json.loads(line)
        key = (int(row['episode_index']), int(row['frame_index']))
        if key in labels:
            raise ValueError(f'Duplicate SFP label {key}')
        labels[key] = row['observed_tcp_pose_port_frame']
    rows = [json.loads(line) for line in (a.cache / 'rows.jsonl').open()]
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[(row['family'], row['episode'])].append(i)
    counts = Counter()
    for (family, _), indices in grouped.items():
        if family != 'sfp':
            continue
        times = [float(rows[i]['time_s']) for i in indices]
        for j, i in enumerate(indices):
            row = rows[i]
            rel = np.asarray(labels[(int(row['episode']), int(row['frame']))], float)
            row['handoff'] = int(a.sfp_axial_min_m <= rel[2] <= a.sfp_axial_max_m and
                                 np.linalg.norm(rel[:2]) <= a.sfp_lateral_max_m)
            wanted = times[j] + a.future_seconds
            k = min(len(indices) - 1, bisect_left(times, wanted))
            if k > j and k > 0 and abs(times[k-1] - wanted) < abs(times[k] - wanted):
                k -= 1
            row['subgoal'] = future_pose9(row['state'], rows[indices[k]]['state'])
            row['sfp_route_future_seconds_actual'] = times[k] - times[j]
            counts[(row['split'], 'sfp', row['handoff'])] += 1
    for row in rows:
        if row['family'] == 'sc':
            counts[(row['split'], 'sc', row['handoff'])] += 1
    if not all(counts[(split, 'sfp', 1)] > 0 and counts[(split, 'sc', 1)] > 0
               for split in ('train', 'validation')):
        raise ValueError('Missing grouped handoff positives')

    positives = defaultdict(list)
    for row in rows:
        if row['split'] == 'train' and row['handoff'] == 1:
            positives[task_key(row)].append(row['state'][:3])
    required = {f'sfp_card_{card}_port_{port}' for card in range(5) for port in range(2)}
    required |= {f'sc_port_{port}' for port in range(2)}
    if set(positives) != required:
        raise ValueError(f'Missing train-only prior keys: {required - set(positives)}')
    margin = a.prior_margin_mm / 1000
    prior = {'schema': 'aic_handoff_tcp_base_region/v1',
             'meaning': 'train-positive TCP base AABB plus fixed margin; veto only',
             'margin_mm': a.prior_margin_mm, 'regions': {},
             'training_geometry_runtime_input': False,
             'final_scenes_opened': False}
    for key, values in positives.items():
        x = np.asarray(values, float)
        prior['regions'][key] = {'min_base_m': (x.min(0)-margin).tolist(),
                                 'max_base_m': (x.max(0)+margin).tolist(),
                                 'train_positive_rows': len(values)}
    validation = Counter()
    for row in rows:
        if row['split'] != 'validation':
            continue
        region = prior['regions'][task_key(row)]
        xyz = np.asarray(row['state'][:3], float)
        inside = bool(np.all(xyz >= region['min_base_m']) and
                      np.all(xyz <= region['max_base_m']))
        validation[(row['family'], row['handoff'], inside)] += 1
    a.output.mkdir(parents=True)
    for name in ('features.npy', 'camera_offsets.npy'):
        os.link(a.cache / name, a.output / name)
    with (a.output / 'rows.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, separators=(',', ':')) + '\n')
    (a.output / 'handoff_prior.json').write_text(json.dumps(prior, indent=2) + '\n')
    manifest = {'schema': 'aic_temporal_route_handoff_cache/v1',
                'source_cache': str(a.cache), 'sfp_labels': str(a.sfp_labels),
                'rows': len(rows), 'future_seconds': a.future_seconds,
                'prior_margin_mm': a.prior_margin_mm,
                'sfp_axial_min_m':a.sfp_axial_min_m,
                'sfp_axial_max_m':a.sfp_axial_max_m,
                'sfp_lateral_max_m':a.sfp_lateral_max_m,
                'counts': {str(key): value for key, value in sorted(counts.items())},
                'validation_region_counts': {str(key): value
                                             for key, value in sorted(validation.items())},
                'port_geometry_runtime_input': False,
                'final_scenes_opened': False}
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'rows': len(rows), 'counts': manifest['counts'],
                      'validation_region_counts': manifest['validation_region_counts']}))


if __name__ == '__main__':
    main()
