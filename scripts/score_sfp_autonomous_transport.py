#!/usr/bin/env python3
"""Posthoc selected-port transport scoring; no geometry enters actor runtime."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from build_two_family_port_projection_labels import sfp_ports


def posematrix(d):
    m=np.eye(4);m[:3,3]=[d['position'][k] for k in ('x','y','z')]
    m[:3,:3]=Rotation.from_quat([d['orientation'][k] for k in ('x','y','z','w')]).as_matrix()
    return m


def read_teacher_orientation(audit,port):
    scored=json.loads((audit/'scored_stream.json').read_text())
    for ns,controller in scored['streams']['/aic_controller/controller_state']:
        tcp=posematrix(controller['tcp_pose']);rel=np.linalg.inv(port)@tcp
        if rel[2,3]>=-.070 and np.linalg.norm(rel[:2,3])<=.010:
            return Rotation.from_matrix(rel[:3,:3]),ns/1e9
    raise ValueError(f'Teacher never reached buffered standoff: {audit}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,action='append',required=True)
    p.add_argument('--teacher',type=Path,action='append',required=True,
                   help='Matched successful teacher in the same order as --run')
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous score')
    if len(a.run)!=len(a.teacher):p.error('Pair every run with one matched teacher')
    cal=json.loads(a.calibration.read_text());results=[]
    for run,teacher in zip(a.run,a.teacher):
        cfg=yaml.safe_load((run/'eval_config.yaml').read_text());tid=next(iter(cfg['trials']))
        ports={(c,p):m for c,p,m in sfp_ports({'trial_yaml':str(run/'eval_config.yaml'),
                                              'trial_id':tid},cal)}
        teacher_cfg=yaml.safe_load((teacher/'eval_config.yaml').read_text())
        if cfg!=teacher_cfg:raise ValueError(f'Mismatched scene pair: {run} / {teacher}')
        rows=[json.loads(line) for line in (run/'runtime_trace.jsonl').open()]
        rows=[r for r in rows if r.get('event')=='proposal' and len(r.get('state',[]))==32]
        if not rows:raise ValueError(f'No actor proposals: {run}')
        task=np.asarray(rows[0]['task_vector']);card=int(task[4:9].argmax());port_num=int(task[2:4].argmax())
        port=ports[card,port_num];desired,teacher_time=read_teacher_orientation(teacher,port)
        scored=[]
        for r in rows:
            task_now=np.asarray(r['task_vector'])
            if int(task_now[4:9].argmax())!=card or int(task_now[2:4].argmax())!=port_num:
                raise ValueError('Task request changed within episode')
            state=np.asarray(r['state']);relative=port[:3,:3].T@(state[:3]-port[:3,3])
            current=Rotation.from_matrix(port[:3,:3]).inv()*Rotation.from_quat(state[3:7])
            orientation=float((desired.inv()*current).magnitude()*180/np.pi)
            scored.append(dict(time_s=float(r['sim_time']),axial_mm=float(relative[2]*1000),
                lateral_mm=float(np.linalg.norm(relative[:2])*1000),
                teacher_standoff_orientation_error_deg=orientation,
                inference_ms=float(r.get('inference_ms',float('nan')))))
        band=[r for r in scored if -100<=r['axial_mm']<=-70]
        valid=[r for r in band if r['lateral_mm']<=10]
        official=yaml.safe_load((run/'results/scoring.yaml').read_text())
        trial=next(v for k,v in official.items() if k.startswith('trial_'))
        ms=np.asarray([r['inference_ms'] for r in scored]);ms=ms[np.isfinite(ms)]
        result=dict(run=str(run),teacher=str(teacher),card=card,port=port_num,
            proposals=len(scored),teacher_cutoff_time_s=teacher_time,
            buffered_visit=bool(valid),buffered_visit_count=len(valid),
            best_buffered_visit=min(valid,key=lambda r:r['lateral_mm']) if valid else None,
            closest_lateral_in_axial_band_mm=min((r['lateral_mm'] for r in band),default=None),
            axial_band_rows=len(band),closest_lateral_overall_mm=min(r['lateral_mm'] for r in scored),
            official_total=float(official['total']),
            official_tier3=float(trial.get('tier_3',{}).get('score',0)),
            official_contact_score=float(trial.get('tier_2',{}).get('categories',{}).get('contacts',{}).get('score',0)),
            inference_p50_p95_p99_ms=np.percentile(ms,[50,95,99]).tolist() if len(ms) else None)
        results.append(result);print(json.dumps(result),flush=True)
    a.output.write_text(json.dumps(dict(schema='aic_sfp_autonomous_transport_posthoc/v1',
        calibration=str(a.calibration),geometry_actor_input=False,
        teacher_trajectory_actor_input=False,reserved_final_opened=False,runs=results),indent=2)+'\n')


if __name__=='__main__':main()
