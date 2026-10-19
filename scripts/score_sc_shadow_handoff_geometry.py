#!/usr/bin/env python3
"""Posthoc scoring of SC shadow decisions against the recorded Gazebo TF."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from analyze_local_actor_scored_tip import record_at
from audit_sc_port_targets import matrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit', type=Path)
    p.add_argument('--opening-calibration', type=Path,
                   default=Path('configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json'))
    p.add_argument('--tcp-prior', type=Path,
                   default=Path('outputs/experiments/2026-10-02_two_family_perception/local_sc_orientation_prior_v1.json'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():p.error('Preserve prior score')
    data = json.loads((a.audit/'scored_stream.json').read_text())
    edges = data['edges']
    selected = int(data['task']['target_module_name'].rsplit('_', 1)[-1])
    prefix = f'task_board/sc_port_{selected}'
    world_base = (record_at(edges['world|tabletop'], 0) @
                  record_at(edges['tabletop|base_link'], 0))
    opening = (np.linalg.inv(world_base) @
               record_at(edges['aic_world|task_board'], 0) @
               record_at(edges[f'task_board|{prefix}'], 0) @
               record_at(edges[f'{prefix}|{prefix}/sc_port_base_link'], 0) @
               matrix(json.loads(a.opening_calibration.read_text())['base_to_opening']))
    true_port = Rotation.from_matrix(opening[:3, :3])
    nominal = Rotation.from_quat(json.loads(a.tcp_prior.read_text())
                                 ['nominal_tcp_in_port_orientation_xyzw'])
    trace = [json.loads(s) for s in (a.audit/'runtime_trace.jsonl').open()]
    holds = [r for r in trace if r['event']=='shadow_handoff_hold']
    checks = [r for r in trace if r['event']=='shadow_handoff_inspection']
    report = []
    for check in checks:
        row = {'sim_time_s': check['sim_time'], 'valid': check['valid'],
               'reason': check['reason'], 'selected_port': selected}
        if 'predicted_opening_base_m' in check:
            predicted = np.asarray(check['predicted_opening_base_m'])
            local = opening[:3, :3].T @ (predicted-opening[:3, 3])
            row.update(opening_3d_error_mm=float(np.linalg.norm(local)*1000),
                       opening_axial_error_mm=float(abs(local[2])*1000),
                       opening_lateral_error_mm=float(np.linalg.norm(local[:2])*1000),
                       port_orientation_error_deg=float(np.rad2deg((
                           true_port.inv()*Rotation.from_quat(
                               check['predicted_port_quat_xyzw'])).magnitude())))
            hold = next((h for h in holds if h['inspection']==check['inspection']), None)
            if hold:
                measured = Rotation.from_quat(hold['measured_tcp_pose'][3:7])
                row['true_tcp_posture_error_deg'] = float(np.rad2deg((
                    (true_port*nominal).inv()*measured).magnitude()))
        report.append(row)
    result = {'schema':'aic_sc_shadow_posthoc_geometry/v1',
              'audit':str(a.audit.resolve()), 'selected_port':selected,
              'ground_truth_used_after_rollout_only':True,
              'checks':report, 'final_scenes_opened':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
