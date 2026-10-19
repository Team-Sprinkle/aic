#!/usr/bin/env python3
"""Audit same-reset SFP teacher actions before training a paired-route model."""
import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def states(stream):
    rows = stream['streams']['/aic_controller/controller_state']
    return [t for t, _ in rows], rows


def causal_pose(times, rows, stamp):
    i = bisect_right(times, stamp)-1
    if i < 0 or stamp-times[i] > 100_000_000:
        return None
    p = rows[i][1]['tcp_pose']
    return np.array([p['position'][v] for v in 'xyz']), Rotation.from_quat(
        [p['orientation'][v] for v in 'xyzw'])


def commands(stream):
    return sorted((c for c in stream['commands'] if c['kind']=='pose'),
                  key=lambda c:c['sim_ns'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    plan=json.loads((a.collection/'plan.json').read_text())
    progress=json.loads((a.collection/'progress.json').read_text())['results']
    details=[];counts=Counter()
    for spec in plan['scenes']:
        name=spec['name']
        if name not in progress:continue
        if progress[name]['official_score'].get('tier3',0)<75:
            counts['alternate_not_full']+=1
            continue
        source=Path(spec['source_config']).parent/'scored_stream.json'
        alternate=Path(progress[name]['audit'])/'scored_stream.json'
        if not source.exists() or not alternate.exists():
            counts['missing_stream']+=1
            continue
        left=json.loads(source.read_text());right=json.loads(alternate.read_text())
        tl,sl=states(left);tr,sr=states(right)
        cl=commands(left);cr=commands(right)
        right_times=[r['sim_ns'] for r in cr]
        scene=[]
        start=max(cl[0]['sim_ns'],cr[0]['sim_ns'])
        for c in cl:
            t=c['sim_ns']
            if t<start-100_000_000:continue
            if t>start+1_500_000_000:break
            j=bisect_right(right_times,t)-1
            candidates=[k for k in (j,j+1) if 0<=k<len(cr)]
            if not candidates:continue
            k=min(candidates,key=lambda q:abs(right_times[q]-t))
            other=cr[k];dt=abs(other['sim_ns']-t)
            if dt>100_000_000:continue
            lp=causal_pose(tl,sl,t);rp=causal_pose(tr,sr,other['sim_ns'])
            if lp is None or rp is None:continue
            tcp_mm=float(np.linalg.norm(lp[0]-rp[0])*1000)
            angle_deg=float((lp[1].inv()*rp[1]).magnitude()*180/np.pi)
            x=np.array([c['message']['pose']['position'][v] for v in 'xyz'])
            y=np.array([other['message']['pose']['position'][v] for v in 'xyz'])
            action_mm=float(np.linalg.norm(x-y)*1000)
            if tcp_mm<=2 and angle_deg<=2:
                scene.append(dict(time_s=t/1e9,time_from_first_command_s=(t-start)/1e9,
                                  dt_s=dt/1e9,tcp_mm=tcp_mm,
                                  orientation_deg=angle_deg,body_command_difference_mm=action_mm,
                                  first_request=spec['original_request'],second_request=spec['alternate_request']))
        strong=[r for r in scene if r['body_command_difference_mm']>=5]
        counts['full_alternates']+=1
        counts['aligned_rows']+=len(scene)
        counts['strong_rows']+=len(strong)
        counts['strong_scenes']+=bool(strong)
        details.append(dict(name=name,source=str(source),alternate=str(alternate),
                            aligned_rows=len(scene),strong_rows=len(strong),
                            maximum_body_command_difference_mm=max((r['body_command_difference_mm'] for r in scene),default=None),
                            rows=scene))
    result=dict(schema='aic_sfp_divergent_pair_stream_audit/v1',
                thresholds=dict(max_time_from_first_command_s=1.5,max_dt_s=.1,max_tcp_mm=2,
                                max_orientation_deg=2,strong_action_mm=5),
                summary=dict(counts),scenes=details,final_scenes_opened=False,
                feature_cosine_not_yet_checked=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary']),flush=True)


if __name__=='__main__':main()
