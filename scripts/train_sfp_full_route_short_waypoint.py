#!/usr/bin/env python3
"""Matched nonlinear one-second TCP-body waypoint probes; no policy rollout."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from probe_sfp_full_route_short_waypoint import report
from train_temporal_transport_probe import LONG_AGES, TemporalTransport, build_data, predict


def phase(row):
    return ('corrective' if row.get('corrective_teacher') else
            'early' if row['time_s']<3 else 'middle' if row['time_s']<8 else 'late')


@torch.inference_mode()
def evaluate(model,data,truth,indices):
    model.eval()
    predictions=[]
    for part in np.array_split(indices,max(1,(len(indices)+127)//128)):
        if not len(part):continue
        ind=torch.as_tensor(part,device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,_,waypoint,_=predict(model,data,ind,False,force_zero_previous=True)
        predictions.append(waypoint[:,:3].float().cpu().numpy())
    pred=np.concatenate(predictions)
    actual=truth[indices]
    episodes=np.asarray([data['rows'][i]['episode'] for i in indices])
    phases=np.asarray([phase(data['rows'][i]) for i in indices])
    magnitude=np.linalg.norm(actual*1000,axis=1)
    motion=np.where(magnitude<10,'lt10',np.where(magnitude<20,'10to20','ge20'))
    return report(actual,pred,episodes,phases,motion)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','base-checkpoint','prior','labels','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--actor-boundary-mass',type=float,choices=(0.,.15),default=0.)
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20261008)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous result')
    if not 1<=a.updates<=2000:p.error('Cap 2,000 updates')
    a.output.mkdir(parents=True)
    np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(8)
    data=build_data(a.cache,a.base_checkpoint,'cuda',history_ages=LONG_AGES)
    data['prev'].zero_()
    source=np.load(a.labels)
    truth=np.asarray(source['waypoint_tcp'],np.float32)
    teacher=np.asarray(source['teacher_valid'],bool)
    boundary=np.asarray(source['actor_boundary_valid'],bool)
    rows=data['rows']
    if len(truth)!=len(rows):raise ValueError('Waypoint/cache alignment mismatch')
    train=np.asarray([i for i,r in enumerate(rows) if r['split']=='train' and teacher[i]],np.int64)
    val=np.asarray([i for i,r in enumerate(rows) if r['split']=='validation' and teacher[i]],np.int64)
    btrain=np.asarray([i for i,r in enumerate(rows) if r['split']=='train' and boundary[i]],np.int64)
    bval=np.asarray([i for i,r in enumerate(rows) if r['split']=='validation' and boundary[i]],np.int64)
    if len(train)<5000 or len(val)<300 or len(btrain)<10 or len(bval)<3:
        raise ValueError('Unexpected grouped waypoint coverage')
    if {rows[i]['scene'] for i in np.r_[train,btrain]} & {rows[i]['scene'] for i in np.r_[val,bval]}:
        raise ValueError('Scene split overlap')
    counts=Counter((rows[i]['episode'],phase(rows[i])) for i in train)
    weights=np.asarray([1/counts[(rows[i]['episode'],phase(rows[i]))] for i in train],float)
    weights/=weights.sum()
    prior=torch.load(a.prior,map_location='cpu',weights_only=False)
    model=TemporalTransport(data['mean'],data['std'],raw_state_skip=True,
                            feature_dim=data['features'].shape[1],state_task_dim=42,
                            route_conditioning=False).cuda()
    model.load_state_dict(prior['model'])
    for group in (model.action,model.stage,model.handoff):
        for param in group.parameters():param.requires_grad_(False)
    opt=torch.optim.AdamW((param for param in model.parameters() if param.requires_grad),
                          lr=1e-4,weight_decay=1e-4)
    label=torch.as_tensor(truth,device='cuda')
    manifest=dict(schema='aic_sfp_full_route_short_waypoint_neural/v1',
                  train_teacher_rows=len(train),validation_teacher_rows=len(val),
                  train_actor_boundary_rows=len(btrain),validation_actor_boundary_rows=len(bval),
                  train_teacher_episodes=len({rows[i]['episode'] for i in train}),
                  validation_teacher_episodes=len({rows[i]['episode'] for i in val}),
                  actor_boundary_mass=a.actor_boundary_mass,updates=a.updates,seed=a.seed,
                  previous_teacher_commands_used=False,actor_input_true_geometry=False,
                  checkpoint_selection='ordinary grouped validation moving>=10mm p95 only',
                  final_scenes_opened=False,autonomous_rollouts=0)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    best=float('inf');history=[];started=time.monotonic()
    for step in range(1,a.updates+1):
        ordinary_count=int(round(64*(1-a.actor_boundary_mass)))
        chosen=np.random.choice(train,size=ordinary_count,replace=True,p=weights)
        if a.actor_boundary_mass:
            chosen=np.r_[chosen,np.random.choice(btrain,size=64-ordinary_count,replace=True)]
        ind=torch.as_tensor(chosen,device='cuda')
        model.train();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,_,waypoint,_=predict(model,data,ind,False,force_zero_previous=True)
            loss=nn.functional.smooth_l1_loss(waypoint[:,:3]/.05,label[ind]/.05,beta=.1)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if step%250==0:
            teacher_score=evaluate(model,data,truth,val)
            boundary_score=evaluate(model,data,truth,bval)
            selected=teacher_score['moving_ge10mm']['error_mm_p95']
            entry=dict(update=step,elapsed_min=(time.monotonic()-started)/60,
                       train_loss=float(loss),teacher=teacher_score,boundary=boundary_score)
            history.append(entry)
            print(json.dumps(dict(update=step,ordinary_moving_p95_mm=selected,
                                  actor_boundary_p95_mm=boundary_score['all']['error_mm_p95'])),flush=True)
            if selected<best:
                best=selected
                torch.save(dict(model=model.state_dict(),source_manifest=manifest,
                                update=step,teacher=teacher_score,boundary=boundary_score),
                           a.output/'best.pt')
    best_cp=torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)
    result=dict(best_update=best_cp['update'],updates_completed=a.updates,
                teacher=best_cp['teacher'],boundary=best_cp['boundary'],
                offline_gate=dict(teacher_moving_p95_le15=
                                  best_cp['teacher']['moving_ge10mm']['error_mm_p95']<=15,
                                  actor_boundary_p95_le20=
                                  best_cp['boundary']['all']['error_mm_p95']<=20),
                final_scenes_opened=False,autonomous_rollouts=0)
    result['offline_gate']['passed']=all(result['offline_gate'].values())
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps(dict(best_update=result['best_update'],gate=result['offline_gate'])),flush=True)


if __name__=='__main__':main()
