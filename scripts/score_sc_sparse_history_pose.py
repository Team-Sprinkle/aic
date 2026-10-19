#!/usr/bin/env python3
"""Posthoc 3D pose error for sparse-history diagnostics; never a runtime input."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def summary(values):
    return {'count': len(values), 'p50_mm': float(np.percentile(values, 50)),
            'p95_mm': float(np.percentile(values, 95))} if values else {'count': 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--track', required=True, type=Path)
    p.add_argument('--identity-audit', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--calibration', type=Path,
                   default=Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json'))
    a = p.parse_args()
    if a.output.exists():
        p.error('Refusing to overwrite prior result')
    tracks = json.loads(a.track.read_text())['rows']
    audit = json.loads(a.identity_audit.read_text())
    standoff_offset = np.asarray(json.loads(a.calibration.read_text())
                                 ['opening_to_transport_tcp_base_offset_m'], float)
    labels = defaultdict(list)
    for path_string in audit['input_rows']:
        path = Path(path_string)
        group = (path.parents[1].name if path.parent.name == 'projection_state_v2'
                 else path.parent.name)
        for line in path.open():
            row = json.loads(line)
            if row['camera'] == 'center':
                labels[group].append(row)
    result = {}
    for group in sorted({r['group'] for r in tracks}):
        if not labels[group]:
            continue
        errors, tracked_errors, single_errors = [], [], []
        near_15, near_30 = [], []
        near_15_lateral, near_15_axial = [], []
        for row in (r for r in tracks if r['group'] == group):
            if row['base_opening_estimate_m'] is None:
                continue
            label = min(labels[group], key=lambda x: abs(x['sim_ns']-row['sim_ns']))
            if abs(label['sim_ns']-row['sim_ns']) > 75_000_000:
                continue
            tcp = np.asarray(label['measured_tcp_base_pose'], float)
            true_base = tcp[:3]+Rotation.from_quat(tcp[3:7]).apply(
                label['relative_opening_tcp_m'])
            error = 1000*np.linalg.norm(np.asarray(row['base_opening_estimate_m'])-true_base)
            port_rotation = Rotation.from_quat(tcp[3:7]).as_matrix() @ np.asarray(
                label['relative_opening_rotation_tcp'], float)
            port_error = 1000*port_rotation.T @ (
                np.asarray(row['base_opening_estimate_m'])-true_base)
            standoff_mm = 1000*np.linalg.norm(true_base+standoff_offset-tcp[:3])
            errors.append(error)
            if standoff_mm <= 15:
                near_15.append(error)
                near_15_lateral.append(float(np.linalg.norm(port_error[:2])))
                near_15_axial.append(float(abs(port_error[2])))
            if standoff_mm <= 30:
                near_30.append(error)
            if row['track_accept']:
                tracked_errors.append(error)
            if row['single_frame_identity']:
                single_errors.append(error)
        result[group] = {'all_geometric': summary(errors),
                         'within_15mm_true_standoff': summary(near_15),
                         'within_15mm_lateral': summary(near_15_lateral),
                         'within_15mm_axial': summary(near_15_axial),
                         'within_30mm_true_standoff': summary(near_30),
                         'track_accepted': summary(tracked_errors),
                         'single_identity_accepted': summary(single_errors)}
    report = {'schema': 'aic_sc_sparse_history_posthoc_pose/v1',
              'label_only_geometry': True, 'track': str(a.track),
              'identity_audit': str(a.identity_audit), 'metrics': result}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
