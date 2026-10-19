#!/usr/bin/env python3
"""Audit saved SC RGB keypoints in TCP 3D on untouched development resets.

The scored port pose is read only for post-run error measurement. This script
does not produce an actor input or a visibility label.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from evaluate_sc_native_triangulation import triangulate


def aggregate(values):
    if not values:
        return {'n': 0}
    values = np.asarray(values, dtype=float)
    return {'n': len(values), 'p50_mm': float(np.median(values)),
            'p95_mm': float(np.percentile(values, 95)),
            'max_mm': float(np.max(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--projection', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve previous result')
    report = json.loads(args.prediction.read_text())
    labels = {(r['camera'], r['sim_ns']): r for r in
              (json.loads(line) for line in args.projection.open())}
    optics = {cam: np.asarray(matrix, dtype=float) for cam, matrix in
              json.loads(args.calibration.read_text())['tcp_to_optical'].items()}
    groups = defaultdict(dict)
    for row in report['rows']:
        groups[(row['checkpoint'], row['sim_ns'])][row['camera']] = row
    records = []
    for (checkpoint, sim_ns), cameras in sorted(groups.items()):
        if set(cameras) != {'center', 'left', 'right'}:
            raise ValueError(f'Incomplete synchronized triplet: {checkpoint} {sim_ns}')
        truth = [np.asarray(labels[(cam, sim_ns)]['relative_opening_tcp_m'], dtype=float)
                 for cam in ('center', 'left', 'right')]
        rotation = np.asarray(labels[('center', sim_ns)]['relative_opening_rotation_tcp'], dtype=float)
        if max(np.linalg.norm(point-truth[0]) for point in truth) > 1e-5:
            raise ValueError(f'Unsynchronized truth poses: {checkpoint} {sim_ns}')
        for subset_name, subset in (('center_right', ('center', 'right')),
                                    ('all_three', ('center', 'left', 'right'))):
            estimate = triangulate({cam: cameras[cam]['predicted_xy'] for cam in subset}, optics)
            error = (estimate-truth[0])*1000
            local_error = rotation.T @ error
            records.append({'checkpoint': checkpoint, 'sim_ns': sim_ns,
                            'subset': subset_name, 'estimate_tcp_m': estimate.tolist(),
                            'truth_tcp_m': truth[0].tolist(),
                            'translation_error_mm': float(np.linalg.norm(error)),
                            'axis_error_mm': np.abs(error).tolist(),
                            'port_local_error_mm': local_error.tolist(),
                            'lateral_error_mm': float(np.linalg.norm(local_error[:2])),
                            'axial_error_mm': float(abs(local_error[2]))})
    checkpoints = sorted({r['checkpoint'] for r in report['rows']})
    metrics = {checkpoint: {subset: {
                   key: aggregate([r[key] for r in records if r['checkpoint'] == checkpoint and
                                   r['subset'] == subset])
                   for key in ('translation_error_mm', 'lateral_error_mm', 'axial_error_mm')}
                            for subset in ('center_right', 'all_three')}
               for checkpoint in checkpoints}
    output = {'schema': 'aic_sc_independent_triangulation/v1',
              'source_prediction': str(args.prediction),
              'source_projection': str(args.projection),
              'calibration': str(args.calibration), 'metrics': metrics,
              'records': records, 'geometry_used_only_for_postrun_scoring': True,
              'visibility_certified': False, 'orientation_estimated': False,
              'final_scenes_opened': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
