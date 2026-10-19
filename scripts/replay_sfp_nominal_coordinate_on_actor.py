#!/usr/bin/env python3
"""Score fixed nominal-coordinate arrival rules on saved autonomous SFP runs."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from build_two_family_port_projection_labels import sfp_ports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nominal-summary', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--run', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error(f'Output exists: {a.output}')
    nominal = {k: np.asarray(v) for k, v in json.loads(
        a.nominal_summary.read_text())['nominal_positions_base_m'].items()}
    calibration = json.loads(a.calibration.read_text())
    out = {'schema': 'aic_nominal_coordinate_autonomous_replay/v1',
           'nominal_fit': str(a.nominal_summary),
           'runtime_inputs': ['measured_tcp_base_position', 'public_task_code'],
           'truth_geometry_decision_input': False,
           'posthoc_truth_source': 'matching saved Gazebo configuration',
           'final_scenes_opened': False,
           'radii_mm': [60, 80, 100, 120], 'runs': []}
    for run in a.run:
        cfg = yaml.safe_load((run/'eval_config.yaml').read_text())
        trial = next(iter(cfg['trials']))
        ports = {(card, port): matrix for card, port, matrix in sfp_ports(
            {'trial_yaml': str(run/'eval_config.yaml'), 'trial_id': trial},
            calibration)}
        rows = [json.loads(s) for s in (run/'runtime_trace.jsonl').open()]
        rows = sorted((r for r in rows if r.get('event') == 'proposal'
                       and len(r.get('state', [])) == 32),
                      key=lambda r: r['sim_time'])
        task = np.asarray(rows[0]['task_vector'])
        card = int(task[4:9].argmax())
        port = int(task[2:4].argmax())
        identity = f'sfp_card_{card}_port_{port}'
        center = nominal[identity]
        truth = ports[card, port]
        record = {'run': str(run), 'identity': identity,
                  'actor_proposals': len(rows), 'crossings': {},
                  'nominal_to_true_port_mm': float(np.linalg.norm(center-truth[:3,3])*1000)}
        for radius in out['radii_mm']:
            crossing = next((r for r in rows if np.linalg.norm(
                np.asarray(r['state'][:3])-center)*1000 <= radius), None)
            if crossing is None:
                record['crossings'][str(radius)] = None
                continue
            tcp = np.asarray(crossing['state'][:3])
            relative = truth[:3,:3].T@(tcp-truth[:3,3])*1000
            dist = {f'sfp_card_{c}_port_{p}': float(np.linalg.norm(
                tcp-m[:3,3])*1000) for (c,p),m in ports.items()}
            record['crossings'][str(radius)] = {
                'time_s': crossing['sim_time'],
                'nominal_tcp_distance_mm': float(np.linalg.norm(tcp-center)*1000),
                'actual_tcp_distance_mm': dist[identity],
                'actual_lateral_mm': float(np.linalg.norm(relative[:2])),
                'actual_axial_mm': float(relative[2]),
                'actual_standoff_gate': bool(np.linalg.norm(relative[:2]) <= 10
                                             and -100 <= relative[2] <= -70),
                'nearest_actual_port': min(dist, key=dist.get)}
        out['runs'].append(record)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2)+'\n')
    for r in out['runs']:
        print(json.dumps({'run': r['run'], 'identity': r['identity'],
                          'nominal_to_true_port_mm': r['nominal_to_true_port_mm'],
                          'crossings': r['crossings']}))


if __name__ == '__main__':
    main()
