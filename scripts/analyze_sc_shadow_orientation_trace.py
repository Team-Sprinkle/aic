#!/usr/bin/env python3
"""Separate commanded TCP rotation, measured rotation, and RGB board target."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def angular_deg(a,b):
    return float(np.rad2deg((a.inv()*b).magnitude()))


def evaluate(audit,prior):
    rows=[json.loads(s) for s in (audit/'runtime_trace.jsonl').open()]
    proposals=[r for r in rows if r.get('event')=='proposal']
    inspections=[r for r in rows if r.get('event')=='shadow_handoff_inspection']
    targets=[r for r in rows if r.get('event')=='shadow_orientation_target']
    out=[]
    for r in inspections:
        before=[p for p in proposals if p['sim_time']<=r['sim_time']]
        if not before:continue
        proposal=before[-1]
        measured=Rotation.from_quat(proposal['state'][3:7])
        chunk=np.asarray(proposal['targets'][0],float)
        x,y=chunk[3:6],chunk[6:9]
        body=Rotation.from_matrix(np.column_stack((x,y,np.cross(x,y))))
        commanded=measured*body
        record={'inspection_sim_s':r['sim_time'],'proposal_sim_s':proposal['sim_time'],
                'valid':r['valid'],'reason':r['reason'],
                'measured_to_command_deg':angular_deg(measured,commanded),
                'measured_euler_xyz_deg':measured.as_euler('xyz',degrees=True).tolist(),
                'command_euler_xyz_deg':commanded.as_euler('xyz',degrees=True).tolist(),
                'reported_posture_error_deg':r.get('measured_tcp_posture_error_deg'),
                'identity_probabilities':r.get('spatial_identity_probability'),
                'orientation_correction_candidate':r.get('orientation_correction_candidate',False)}
        if 'predicted_port_quat_xyzw' in r:
            board=Rotation.from_quat(r['predicted_port_quat_xyzw'])*prior
            record.update(board_target_euler_xyz_deg=board.as_euler('xyz',degrees=True).tolist(),
                          command_to_board_deg=angular_deg(commanded,board),
                          proposal_measured_to_board_deg=angular_deg(measured,board))
        out.append(record)
    return {'audit':str(audit),'inspections':out,
            'orientation_targets':[{k:r.get(k) for k in
                                    ('sim_time','measured_error_deg','target_tcp_quat_xyzw')}
                                   for r in targets],
            'accepted':any(r['valid'] for r in inspections),
            'final_scenes_opened':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('audits',nargs='+',type=Path)
    p.add_argument('--prior',type=Path,default=Path(
        'outputs/experiments/2026-10-02_two_family_perception/local_sc_orientation_prior_v1.json'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior result')
    prior=Rotation.from_quat(json.loads(a.prior.read_text())['nominal_tcp_in_port_orientation_xyzw'])
    result={'schema':'aic_sc_shadow_orientation_trace/v1',
            'audits':[evaluate(x,prior) for x in a.audits],
            'runtime_uses_oracle_geometry':False,'final_scenes_opened':False}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{'audit':Path(r['audit']).name,
                       'accepted':r['accepted'],
                       'posture_errors_deg':[x['reported_posture_error_deg']
                                             for x in r['inspections']]}
                      for r in result['audits']],indent=2))
if __name__=='__main__':main()
