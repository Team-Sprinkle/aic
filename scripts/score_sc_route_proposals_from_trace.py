#!/usr/bin/env python3
"""Score SC transport from causal actor traces and frozen teacher scene geometry.

Selected-port geometry is used after rollout only. No actor scored-stream
extraction is required. Final scenes remain sealed.
"""
import argparse
from bisect import bisect_left
import json
from pathlib import Path

import numpy as np
import yaml

from materialize_sc_scored_route import port_opening
from materialize_sfp_scored_transport import posematrix


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--teacher-streams',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing score')
    plan=json.loads((a.comparison/'plan.json').read_text())
    progress=json.loads((a.comparison/'progress.json').read_text())
    if any(x['family']!='sc' for x in plan['scenes']):p.error('SC only')
    rows=[]
    for spec in plan['scenes']:
        name=spec['name']
        if name not in progress or progress[name]['returncode']:
            raise ValueError(f'Incomplete rollout: {name}')
        actor=Path(progress[name]['audit'])
        teacher=json.loads((a.teacher_streams/(name+'.json')).read_text())
        port=int(teacher['task']['target_module_name'].split('_')[-1])
        opening=port_opening(teacher,port)
        proposals=[]
        for line in (actor/'runtime_trace.jsonl').open():
            record=json.loads(line)
            if record.get('event')=='proposal':proposals.append(record)
        if not proposals:raise ValueError(f'No policy proposals: {name}')
        controls=teacher['streams']['/aic_controller/controller_state']
        timestamps=np.asarray([r[0] for r in controls],np.int64)
        distances=[];axial=[];lateral=[];gaps=[];inference=[];teacher_gap=[]
        for i,row in enumerate(proposals):
            tcp=np.asarray(row['state'][:3],float)
            relative=opening[:3,:3].T@(tcp-opening[:3,3])*1000
            axial.append(float(relative[2]));lateral.append(float(np.linalg.norm(relative[:2])))
            distances.append(float(np.linalg.norm(relative)))
            inference.append(float(row['inference_ms']))
            if i:gaps.append(float(row['sim_time']-proposals[i-1]['sim_time']))
            stamp=int(round(row['sim_time']*1e9));j=bisect_left(timestamps,stamp)
            if j>=len(controls):j=len(controls)-1
            if j and abs(timestamps[j-1]-stamp)<abs(timestamps[j]-stamp):j-=1
            if abs(timestamps[j]-stamp)<=200_000_000:
                teacher_tcp=posematrix(controls[j][1]['tcp_pose'])[:3,3]
                teacher_gap.append(float(np.linalg.norm(tcp-teacher_tcp)*1000))
            else:teacher_gap.append(None)
        departure=next((float(proposals[i]['sim_time']) for i in range(len(proposals)-2)
                        if all(teacher_gap[k] is not None and teacher_gap[k]>20
                               for k in (i,i+1,i+2))),None)
        band=[i for i,v in enumerate(axial) if -100<=v<=-70]
        score=yaml.safe_load((actor/'results/scoring.yaml').read_text())
        trial=next(value for key,value in score.items() if key.startswith('trial_'))
        contact=trial.get('tier_2',{}).get('categories',{}).get('contacts',{}).get('message')
        rows.append(dict(name=name,requested_port=port,proposals=len(proposals),
                         closest_tcp_opening_3d_mm=min(distances),
                         closest_axial_mm=axial[int(np.argmin(distances))],
                         closest_lateral_mm=lateral[int(np.argmin(distances))],
                         axial_approach_band=bool(band),
                         nearest_lateral_in_band_mm=min((lateral[i] for i in band),default=None),
                         first_sustained_20mm_teacher_departure_sim_s=departure,
                         inference_ms_p50_p95_p99=np.quantile(inference,[.5,.95,.99]).tolist(),
                         decision_gap_s_p50_p95=np.quantile(gaps,[.5,.95]).tolist() if gaps else None,
                         official_tier3_score=float(trial.get('tier_3',{}).get('score',0)),
                         official_contact_message=contact,actor_privileged_input=False))
    report=dict(schema='aic_sc_transport_trace_score/v1',comparison=str(a.comparison),
                geometry_posthoc_only=True,final_scenes_opened=False,rows=rows)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    for row in rows:print(json.dumps(row),flush=True)


if __name__=='__main__':main()
