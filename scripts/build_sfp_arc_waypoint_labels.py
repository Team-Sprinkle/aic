#!/usr/bin/env python3
"""Build 20 mm measured-path lookahead labels in the current TCP frame."""
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
    p.add_argument('--path-mm',type=float,default=20.)
    p.add_argument('--max-lookahead-s',type=float,default=4.)
    p.add_argument('--min-net-mm',type=float,default=10.)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior labels')
    if not 10<=a.path_mm<=40 or not 1<=a.max_lookahead_s<=6 or not 0<=a.min_net_mm<=a.path_mm:
        p.error('Out-of-budget path label')
    rows=[json.loads(s) for s in (a.cache/'rows.jsonl').open()]
    by_episode=defaultdict(list)
    for i,r in enumerate(rows):
        if r['family']=='sfp':by_episode[(r['split'],r['episode'])].append(i)
    waypoint=np.zeros((len(rows),3),np.float32)
    teacher_valid=np.zeros(len(rows),bool);boundary_valid=np.zeros(len(rows),bool)
    report=[];rejected=defaultdict(int)
    for (split,episode),ids in by_episode.items():
        ids.sort(key=lambda i:rows[i]['time_s'])
        xyz=np.asarray([rows[i]['state'][:3] for i in ids],float)
        time=np.asarray([rows[i]['time_s'] for i in ids],float)
        for k,i in enumerate(ids):
            r=rows[i]
            is_boundary=bool(r.get('actor_prefix_context_only'))
            if is_boundary and (k+1==len(ids) or rows[ids[k+1]].get('actor_prefix_context_only')):
                continue  # Only the last actor state can have a teacher route label.
            distance=0.;chosen=None
            for j in range(k+1,len(ids)):
                if time[j]-time[k]>a.max_lookahead_s:break
                if not is_boundary and bool(rows[ids[j]].get('corrective_teacher'))!=bool(r.get('corrective_teacher')):
                    break
                distance+=float(np.linalg.norm(xyz[j]-xyz[j-1]))
                if distance*1000>=a.path_mm:
                    chosen=j;break
            if chosen is None:
                rejected['no_20mm_path_within_4s']+=1;continue
            if is_boundary and not rows[ids[chosen]].get('corrective_teacher'):
                rejected['actor_boundary_not_teacher']+=1;continue
            delta=Rotation.from_quat(r['state'][3:7]).inv().apply(xyz[chosen]-xyz[k])
            net=float(np.linalg.norm(delta)*1000)
            if net<a.min_net_mm:
                rejected['low_net_displacement']+=1;continue
            waypoint[i]=delta.astype(np.float32)
            (boundary_valid if is_boundary else teacher_valid)[i]=True
            if is_boundary:
                report.append(dict(episode=int(episode),split=split,scene=r['scene'],
                                   frame=int(r['frame']),lookahead_s=float(time[chosen]-time[k]),
                                   path_mm=float(distance*1000),net_mm=net,
                                   target_tcp_mm=(delta*1000).tolist()))
    train_scenes={r['scene'] for r in report if r['split']=='train'}
    val_scenes={r['scene'] for r in report if r['split']=='validation'}
    if train_scenes&val_scenes:raise ValueError('Actor-boundary scene overlap')
    result=dict(schema='aic_sfp_arc20_waypoint_labels/v1',path_mm=a.path_mm,
                max_lookahead_s=a.max_lookahead_s,min_net_mm=a.min_net_mm,
                train_teacher_rows=int(sum(teacher_valid[i] and rows[i]['split']=='train' for i in range(len(rows)))),
                validation_teacher_rows=int(sum(teacher_valid[i] and rows[i]['split']=='validation' for i in range(len(rows)))),
                train_actor_boundary_rows=int(sum(r['split']=='train' for r in report)),
                validation_actor_boundary_rows=int(sum(r['split']=='validation' for r in report)),
                rejected=dict(rejected),actor_boundary=report,
                geometry_actor_input=False,final_scenes_opened=False)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'labels.npz',waypoint_tcp=waypoint,
                        teacher_valid=teacher_valid,actor_boundary_valid=boundary_valid)
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='actor_boundary'}))


if __name__=='__main__':main()
