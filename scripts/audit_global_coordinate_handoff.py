#!/usr/bin/env python3
"""Replay an observation-only nominal-coordinate arrival rule on SFP traces.

Truth geometry is read only to fit training-set nominal positions and to score
the frozen rule after its decisions. No evaluation-scene geometry is an input.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def groups(path, split=None):
    out = defaultdict(list)
    for line in path.open():
        row = json.loads(line)
        if split is None or row['split'] == split:
            out[row['scene_group']].append(row)
    return {key: sorted(rows, key=lambda row: row['time_s'])
            for key, rows in out.items()}


def target(row, identity=None):
    identity = identity or row['selected_identity']
    return next(port for port in row['targets'] if port['identity'] == identity)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--dev', type=Path, required=True)
    p.add_argument('--diagnostic', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    train = groups(a.train, 'train')
    dev = groups(a.dev, 'validation')
    diagnostic = groups(a.diagnostic, 'validation')
    train_positions = defaultdict(list)
    for rows in train.values():
        row = rows[0]
        train_positions[row['selected_identity']].append(
            target(row)['position_base_m'])
    nominal = {name: np.median(points, axis=0)
               for name, points in train_positions.items()}

    result = {'schema': 'aic_nominal_global_coordinate_replay/v1',
              'runtime_inputs': ['measured_tcp_base_position',
                                 'public_requested_card_and_port'],
              'evaluation_geometry_used_for_decisions': False,
              'train_reset_groups': len(train),
              'nominal_positions_base_m': {name: xyz.tolist()
                                           for name, xyz in nominal.items()},
              'decision': 'first observed row with measured TCP within the fixed radius of the training-only per-request nominal port location',
              'radii_mm': [60, 80, 100, 120],
              'near_truth_thresholds_mm': [80, 100, 120],
              'splits': {}}
    for split, episodes in [('train', train), ('dev', dev),
                            ('inspected_diagnostic', diagnostic)]:
        records = []
        for scene, rows in episodes.items():
            identity = rows[0]['selected_identity']
            true = np.asarray(target(rows[0])['position_base_m'])
            center = nominal[identity]
            record = {'scene_group': scene, 'episode': rows[0]['episode'],
                      'selected_identity': identity, 'sampled_rows': len(rows),
                      'target_nominal_error_mm': float(np.linalg.norm(center-true)*1000),
                      'target_nominal_delta_mm': ((center-true)*1000).tolist(),
                      'initial_tcp_to_actual_port_mm': float(np.linalg.norm(
                          np.asarray(rows[0]['state'][:3])-true)*1000),
                      'first_crossing': {}}
            for radius in result['radii_mm']:
                selected = None
                for row in rows:
                    tcp = np.asarray(row['state'][:3])
                    if np.linalg.norm(tcp-center)*1000 <= radius:
                        selected = row
                        break
                if selected is None:
                    record['first_crossing'][str(radius)] = None
                    continue
                tcp = np.asarray(selected['state'][:3])
                port = target(selected)
                relative = Rotation.from_quat(port['orientation_base_xyzw']).inv().apply(
                    tcp-np.asarray(port['position_base_m']))*1000
                distances = {port['identity']: float(np.linalg.norm(
                    tcp-np.asarray(port['position_base_m']))*1000)
                    for port in selected['targets']}
                record['first_crossing'][str(radius)] = {
                    'time_s': selected['time_s'],
                    'nominal_tcp_distance_mm': float(np.linalg.norm(tcp-center)*1000),
                    'actual_tcp_distance_mm': distances[identity],
                    'actual_port_frame_tcp_mm': relative.tolist(),
                    'actual_lateral_mm': float(np.linalg.norm(relative[:2])),
                    'actual_axial_mm': float(relative[2]),
                    'actual_standoff_gate': bool(np.linalg.norm(relative[:2])<=10
                                                 and -100<=relative[2]<=-70),
                    'nearest_actual_port': min(distances, key=distances.get),
                    'tcp_base_m': tcp.tolist()}
            records.append(record)
        metrics = {'episodes': len(records), 'records': records}
        for radius in result['radii_mm']:
            detections = [r['first_crossing'][str(radius)] for r in records]
            present = [x for x in detections if x is not None]
            metrics[str(radius)] = {
                'detected_episodes': len(present),
                'actual_distance_mm_p50_p90_max': np.percentile(
                    [x['actual_tcp_distance_mm'] for x in present],
                    [50, 90, 100]).tolist() if present else None,
                'nearest_actual_port_is_wrong': sum(
                    x['nearest_actual_port'] != r['selected_identity']
                    for r, x in zip(records, detections) if x is not None),
                'actual_standoff_gate_count': sum(x['actual_standoff_gate']
                                                   for x in present),
                'actual_lateral_mm_p50_p90_max': np.percentile(
                    [x['actual_lateral_mm'] for x in present],
                    [50, 90, 100]).tolist() if present else None,
                'actual_within_threshold_counts': {str(threshold): sum(
                    x['actual_tcp_distance_mm'] <= threshold
                    for x in present) for threshold in result['near_truth_thresholds_mm']}}
        errors = [r['target_nominal_error_mm'] for r in records]
        metrics['target_nominal_error_mm_p50_p90_max'] = np.percentile(
            errors, [50, 90, 100]).tolist() if errors else None
        result['splits'][split] = metrics
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        p.error(f'Output exists: {a.output}')
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    for split, item in result['splits'].items():
        print(json.dumps({'split': split, 'episodes': item['episodes'],
                          'port_error_p50_p90_max': item['target_nominal_error_mm_p50_p90_max'],
                          'radii': {str(r): {k: v for k, v in item[str(r)].items()
                                              if k != 'actual_distance_mm_p50_p90_max'}
                                    for r in result['radii_mm']}}))


if __name__ == '__main__':
    main()
