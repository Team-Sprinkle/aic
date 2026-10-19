#!/usr/bin/env python3
"""One bounded full-phase, 1 s TCP-body waypoint feasibility probe."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from probe_sfp_early_route_branch import fit_ridge, predict


def report(truth, prediction, episode, phase, motion_bin):
    error=np.linalg.norm((prediction-truth)*1000,axis=1)
    distance=np.linalg.norm(truth*1000,axis=1)
    results={}
    for name,mask in [('all',np.ones(len(truth),bool)),
                      ('moving_ge10mm',distance>=10),('moving_ge20mm',distance>=20),
                      *[(f'phase_{p}',phase==p) for p in sorted(set(phase))],
                      *[(f'motion_{m}',motion_bin==m) for m in sorted(set(motion_bin))]]:
        if not mask.any():continue
        results[name]=dict(rows=int(mask.sum()),episodes=len(set(episode[mask])),
                           motion_mm_p50=float(np.median(distance[mask])),
                           error_mm_p50=float(np.median(error[mask])),
                           error_mm_p95=float(np.percentile(error[mask],95)))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--labels-output',type=Path,
                   help='Optional aligned labels/masks for a later bounded nonlinear probe')
    p.add_argument('--max-train',type=int,default=12000)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous result')
    if a.labels_output and a.labels_output.exists():p.error('Preserve previous label cache')
    rows=[json.loads(s) for s in (a.cache/'rows.jsonl').open()]
    features=np.load(a.cache/'features.npy',mmap_mode='r')
    episodes=defaultdict(list)
    for i,r in enumerate(rows):
        if r['family']=='sfp':episodes[(r['split'],int(r['episode']))].append(i)
    labels={};boundary=[]
    for key,ids in episodes.items():
        ids.sort(key=lambda i:rows[i]['time_s'])
        time=np.asarray([rows[i]['time_s'] for i in ids])
        for at,i in enumerate(ids):
            r=rows[i]
            j=ids[int(np.argmin(np.abs(time-(r['time_s']+1.0))))]
            if abs(rows[j]['time_s']-(r['time_s']+1.0))>.20:continue
            if r.get('actor_prefix_context_only'):
                if at!=len(ids)-1 and ids[at+1]!=j and rows[ids[at+1]].get('actor_prefix_context_only'):
                    continue
                if not rows[j].get('corrective_teacher'):continue
                kind='actor_boundary'
            elif bool(r.get('corrective_teacher'))!=bool(rows[j].get('corrective_teacher')):
                continue
            else:kind='teacher'
            state=np.asarray(r['state'],float)
            future=np.asarray(rows[j]['state'],float)
            motion=Rotation.from_quat(state[3:7]).inv().apply(future[:3]-state[:3])
            if not np.isfinite(motion).all():raise ValueError('Nonfinite waypoint')
            task=np.asarray(r['task'],float)
            labels[i]=dict(split=r['split'],episode=int(r['episode']),scene=r['scene'],
                           kind=kind,phase=('corrective' if r.get('corrective_teacher') else
                                            'early' if r['time_s']<3 else
                                            'middle' if r['time_s']<8 else 'late'),
                           y=motion,small=np.r_[state[:7],task],
                           visual=np.r_[np.asarray(features[i],float),state[:7],task])
            if kind=='actor_boundary':boundary.append(i)
    train_all=[i for i,x in labels.items() if x['split']=='train' and x['kind']=='teacher']
    validation=[i for i,x in labels.items() if x['split']=='validation' and x['kind']=='teacher']
    if len(train_all)<5000 or len(validation)<300:
        raise ValueError(f'Insufficient full-phase waypoints: {len(train_all)} / {len(validation)}')
    if {labels[i]['scene'] for i in train_all}&{labels[i]['scene'] for i in validation}:
        raise ValueError('Reset-scene leakage')
    # Deterministic per-episode/phase thinning prevents long routes dominating.
    groups=defaultdict(list)
    for i in train_all:groups[(labels[i]['episode'],labels[i]['phase'])].append(i)
    selected=[]
    per_group=max(1,a.max_train//len(groups))
    for key,ids in sorted(groups.items()):
        ids.sort(key=lambda i:rows[i]['time_s'])
        selected.extend(ids[np.linspace(0,len(ids)-1,min(per_group,len(ids)),dtype=int)[j]]
                        for j in range(min(per_group,len(ids))))
    train=sorted(set(selected))
    if len(train)<3000:raise ValueError('Thin sample unexpectedly small')
    ytrain=np.asarray([labels[i]['y'] for i in train]);yval=np.asarray([labels[i]['y'] for i in validation])
    episode=np.asarray([labels[i]['episode'] for i in validation]);phase=np.asarray([labels[i]['phase'] for i in validation])
    magnitude=np.linalg.norm(yval*1000,axis=1)
    motion_bin=np.where(magnitude<10,'lt10',np.where(magnitude<20,'10to20','ge20'))
    count=defaultdict(int)
    for i in train:count[(labels[i]['episode'],labels[i]['phase'])]+=1
    weight=np.asarray([1/count[(labels[i]['episode'],labels[i]['phase'])] for i in train],float)
    result=dict(schema='aic_sfp_full_route_short_waypoint_probe/v1',horizon_s=1.0,
                label='measured future TCP displacement expressed in current TCP body frame',
                train_rows=len(train),train_episodes=len({labels[i]['episode'] for i in train}),
                validation_rows=len(validation),validation_episodes=len(set(episode)),
                actor_boundary_rows=len(boundary),actor_boundary_train=sum(labels[i]['split']=='train' for i in boundary),
                actor_boundary_validation=sum(labels[i]['split']=='validation' for i in boundary),
                selected_indices=train,models={},final_scenes_opened=False,autonomous_rollouts=0)
    for name in ('small','visual'):
        xtrain=np.asarray([labels[i][name] for i in train])
        xval=np.asarray([labels[i][name] for i in validation])
        model=fit_ridge(xtrain,ytrain,weight,alpha=100)
        pred=predict(model,xval)
        result['models'][name]=report(yval,pred,episode,phase,motion_bin)
        boundary_ids=[i for i in boundary if labels[i]['split']=='validation']
        if boundary_ids:
            bpred=predict(model,np.asarray([labels[i][name] for i in boundary_ids]))
            by=np.asarray([labels[i]['y'] for i in boundary_ids])
            be=np.asarray([labels[i]['episode'] for i in boundary_ids]);bp=np.asarray(['actor_boundary']*len(boundary_ids))
            bm=np.asarray(['unknown']*len(boundary_ids))
            result['models'][name]['actor_boundary']=report(by,bpred,be,bp,bm)['all']
    mean=np.tile(np.average(ytrain,axis=0,weights=weight),(len(validation),1))
    result['models']['constant_mean']=report(yval,mean,episode,phase,motion_bin)
    result['models']['persistence']=report(yval,np.zeros_like(yval),episode,phase,motion_bin)
    visual=result['models']['visual'];state=result['models']['small']
    result['gate']=dict(moving_ge10mm_visual_p95_le15=visual['moving_ge10mm']['error_mm_p95']<=15,
                        moving_ge10mm_improves_state_by_20pct=
                        visual['moving_ge10mm']['error_mm_p95']<=.8*state['moving_ge10mm']['error_mm_p95'],
                        actor_boundary_visual_p95_le20=(visual.get('actor_boundary',{}).get('error_mm_p95',float('inf'))<=20))
    result['gate']['passed']=all(result['gate'].values())
    if a.labels_output:
        waypoint=np.zeros((len(rows),3),np.float32)
        teacher_valid=np.zeros(len(rows),bool)
        boundary_valid=np.zeros(len(rows),bool)
        for i,item in labels.items():
            waypoint[i]=item['y']
            (boundary_valid if item['kind']=='actor_boundary' else teacher_valid)[i]=True
        np.savez_compressed(a.labels_output,waypoint_tcp=waypoint,
                            teacher_valid=teacher_valid,actor_boundary_valid=boundary_valid)
        result['labels_output']=str(a.labels_output)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='selected_indices'},indent=2))


if __name__=='__main__':main()
