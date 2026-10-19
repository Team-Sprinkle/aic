#!/usr/bin/env python3
"""Read-only first-10mm-net actor-boundary waypoint coverage and path straightness."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--net-mm',type=float,default=10.)
    p.add_argument('--max-lookahead-s',type=float,default=4.)
    p.add_argument('--max-path-ratio',type=float,default=1.5)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior audit')
    if not 5<=a.net_mm<=20 or not 1<=a.max_lookahead_s<=6 or not 1<=a.max_path_ratio<=2:
        p.error('Bounds exceeded')
    rows=[json.loads(x) for x in (a.cache/'rows.jsonl').open()]
    by=defaultdict(list)
    for i,r in enumerate(rows):
        if r['family']=='sfp':by[(r['split'],r['episode'])].append(i)
    results=[]
    for (split,episode),ids in by.items():
        ids.sort(key=lambda i:rows[i]['time_s'])
        actor=[(k,i) for k,i in enumerate(ids) if rows[i].get('actor_prefix_context_only')]
        if not actor:continue
        k,i=actor[-1];r=rows[i]
        start=np.asarray(r['state'][:3],float);quat=np.asarray(r['state'][3:7],float)
        path=0.;chosen=None;reason='no_teacher_net_within_cap'
        for j in range(k+1,len(ids)):
            x=rows[ids[j]]
            if x['time_s']-r['time_s']>a.max_lookahead_s:break
            if not x.get('corrective_teacher'):
                reason='actor_context_crossed_without_teacher';break
            previous=np.asarray(rows[ids[j-1]]['state'][:3],float)
            current=np.asarray(x['state'][:3],float)
            path+=float(np.linalg.norm(current-previous))
            net=float(np.linalg.norm(current-start))
            if net*1000>=a.net_mm:
                chosen=j;reason='valid';break
        if chosen is None:
            results.append(dict(split=split,episode=int(episode),scene=r['scene'],
                                found=False,reason=reason));continue
        x=rows[ids[chosen]];delta=np.asarray(x['state'][:3],float)-start
        local=Rotation.from_quat(quat).inv().apply(delta)
        net=float(np.linalg.norm(delta)*1000);path_mm=float(path*1000)
        ratio=path_mm/net
        results.append(dict(split=split,episode=int(episode),scene=r['scene'],
                            found=True,frame=int(r['frame']),lookahead_s=float(x['time_s']-r['time_s']),
                            path_mm=path_mm,net_mm=net,path_to_net_ratio=ratio,
                            straight_enough=bool(ratio<=a.max_path_ratio),
                            target_tcp_mm=(local*1000).tolist(),
                            teacher_success_verified_here=False))
    result=dict(schema='aic_sfp_first_net_actor_boundary_audit/v1',net_mm=a.net_mm,
                max_lookahead_s=a.max_lookahead_s,max_path_ratio=a.max_path_ratio,
                train_episodes=sum(x['split']=='train' for x in results),
                development_episodes=sum(x['split']=='validation' for x in results),
                train_found=sum(x['split']=='train' and x['found'] for x in results),
                development_found=sum(x['split']=='validation' and x['found'] for x in results),
                train_straight=sum(x['split']=='train' and x['found'] and x['straight_enough'] for x in results),
                development_straight=sum(x['split']=='validation' and x['found'] and x['straight_enough'] for x in results),
                labels_are_safe_verified=False,policy_training_allowed=False,
                final_scenes_opened=False,records=results)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))

if __name__=='__main__':main()
