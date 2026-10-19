#!/usr/bin/env python3
"""Bounded DAgger-style supervised correction of one shared transport actor.

Teacher route geometry creates *offline labels* for previously recorded actor
observations. The deployed actor still receives RGB, ordinary state and task.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn

from train_temporal_transport_probe import (LONG_AGES, TemporalTransport,
                                             build_data, predict)
from train_shared_port_tcp_bc import pose9


def actor_data(cache,labels,base,device,near_goal_supervision=False):
    rows=[json.loads(s) for s in (cache/'rows.jsonl').open()]
    features=np.asarray(np.load(cache/'features.npy',mmap_mode='r'),dtype=np.float32)
    if near_goal_supervision:
        targets=np.zeros((len(rows),4,9),dtype=np.float32)
        valid=np.zeros(len(rows),dtype=bool)
        for i,r in enumerate(rows):
            goal=np.asarray(r['goal_relative'],dtype=float)
            distance=float(np.linalg.norm(goal[:3]))
            if not (.02<=distance<=.08 and float(r['sim_time'])>=30):continue
            valid[i]=True
            angle=float(np.linalg.norm(goal[3:]))
            for h in range(4):
                translation=goal[:3]*min(1.,.008*(h+1)/distance)
                rotvec=goal[3:]*min(1.,np.deg2rad(2)*(h+1)/max(angle,1e-8))
                targets[i,h]=pose9(np.r_[translation,Rotation.from_rotvec(rotvec).as_quat()])
    else:
        targets=np.load(labels/'targets.npy');valid=np.load(labels/'valid.npy')
    assert len(rows)==len(features)==len(targets)==len(valid)
    state=np.asarray([r['state'] for r in rows],dtype=np.float32)
    norm=(state-base['model']['state_mean'].numpy())/base['model']['state_std'].numpy()
    norm[:,base.get('masked_state_indices',[])]=0
    task=np.asarray([r['task'] for r in rows],dtype=np.float32)
    state_task=np.c_[norm,task].astype(np.float32)
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['case']].append(i)
    history=np.full((len(rows),len(LONG_AGES)),-1,dtype=np.int64)
    age=np.zeros(history.shape,dtype=np.float32);gap=age.copy()
    motion=np.zeros((len(rows),6),dtype=np.float32)
    clocks=np.zeros((len(rows),4),dtype=np.float32);clocks[:,3]=1
    for i,r in enumerate(rows):
        for j,path in enumerate(r['image_paths']):
            image_s=float(Path(path).stem.split('_')[1])/1e9
            clocks[i,j]=max(0.,float(r['sim_time'])-image_s)/.05
    for indices in groups.values():
        indices.sort(key=lambda i:rows[i]['sim_time'])
        times=np.asarray([rows[i]['sim_time'] for i in indices],dtype=float)
        for k,i in enumerate(indices):
            if k:
                prior=indices[k-1]
                motion[i,:3]=state[i,:3]-state[prior,:3]
                motion[i,3:]=(Rotation.from_quat(state[prior,3:7]).inv()*
                              Rotation.from_quat(state[i,3:7])).as_rotvec()
            chosen=[]
            for wanted in LONG_AGES:
                j=np.searchsorted(times,times[k]-wanted,side='right')-1
                if j>=0 and (wanted==0 or times[k]-times[j]<=60) and indices[j] not in chosen:
                    chosen.append(indices[j])
            chosen.sort(key=lambda q:rows[q]['sim_time'])
            off=len(LONG_AGES)-len(chosen)
            for q,c in enumerate(chosen,start=off):
                history[i,q]=c;age[i,q]=(times[k]-rows[c]['sim_time'])/4
                gap[i,q]=0 if q==off else (rows[c]['sim_time']-rows[history[i,q-1]]['sim_time'])/4
    tensor=lambda v,dtype=None:torch.as_tensor(v,device=device,dtype=dtype)
    return {'rows':rows,'features':tensor(features),'previous':tensor(np.zeros((len(rows),36),dtype=np.float32)),
        'motion':tensor(motion),'state_task':tensor(state_task),'camera_clock':tensor(clocks),
        'history':tensor(np.maximum(history,0)),'mask':tensor(history>=0),
        'age':tensor(age),'gap':tensor(gap),'target':tensor(targets),'valid':valid}


def actor_predict(model,data,indices):
    h=data['history'][indices]
    return model(data['features'][h].float(),data['previous'][h].float(),
        data['motion'][h].float(),data['age'][indices].float(),data['gap'][indices].float(),
        data['camera_clock'][h].float(),data['mask'][indices],data['state_task'][h].float())[0]


@torch.inference_mode()
def metric(model,data,indices):
    model.eval();errors=[]
    for part in np.array_split(indices,max(1,(len(indices)+255)//256)):
        ind=torch.as_tensor(part,device=data['target'].device)
        with torch.autocast('cuda',dtype=torch.bfloat16):prediction=actor_predict(model,data,ind)
        action=prediction[:,0].float()*model.target_std+model.target_mean
        mm=(action[:,:3]-data['target'][ind,0,:3]).norm(dim=-1)*1000
        errors.extend(mm.cpu().tolist())
    return {'rows':len(errors),'first_action_translation_p50_mm':float(np.median(errors)),
            'first_action_translation_p95_mm':float(np.quantile(errors,.95))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--teacher-cache',type=Path,required=True)
    p.add_argument('--onpolicy-cache',type=Path,required=True)
    p.add_argument('--onpolicy-labels',type=Path)
    p.add_argument('--near-goal-supervision',action='store_true')
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=2500)
    p.add_argument('--budget-minutes',type=float,default=30)
    p.add_argument('--onpolicy-loss-weight',type=float,default=1.)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing run')
    a.output.mkdir(parents=True)
    random.seed(20261001);np.random.seed(20261001);torch.manual_seed(20261001);torch.set_num_threads(8)
    device='cuda'
    teacher=build_data(a.teacher_cache,a.base,device,LONG_AGES)
    teacher['prev'].zero_()
    base=torch.load(a.base,map_location='cpu',weights_only=False)
    if not a.near_goal_supervision and a.onpolicy_labels is None:
        p.error('Route supervision requires --onpolicy-labels')
    op=actor_data(a.onpolicy_cache,a.onpolicy_labels,base,device,a.near_goal_supervision)
    initial=torch.load(a.initial,map_location='cpu',weights_only=False)
    manifest=initial['source_manifest']
    if not manifest['zero_previous_command'] or manifest['route_conditioning'] or not manifest['long_history']:
        raise ValueError('Expected prior-free action-only 60s actor')
    model=TemporalTransport(teacher['mean'],teacher['std'],raw_state_skip=True,
                            feature_dim=1536,state_task_dim=42,route_conditioning=False).to(device)
    model.load_state_dict(initial['model'])
    teacher_by_family={f:np.asarray([i for i,r in enumerate(teacher['rows'])
        if r['split']=='train' and r['family']==f]) for f in ('sc','sfp')}
    train=np.asarray([i for i,r in enumerate(op['rows']) if r['split']=='adapt_train' and op['valid'][i]])
    val=np.asarray([i for i,r in enumerate(op['rows']) if r['split']=='adapt_validation' and op['valid'][i]])
    if len(train)<100 or len(val)<20:raise ValueError('Insufficient grouped correction labels')
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=1e-4)
    def report(update):
        result=metric(model,op,val)
        teacher_val={}
        for f in ('sc','sfp'):
            ids=np.asarray([i for i,r in enumerate(teacher['rows']) if r['split']=='validation' and r['family']==f])
            ids=ids[np.linspace(0,len(ids)-1,min(384,len(ids))).astype(int)]
            err=[]
            for part in np.array_split(ids,max(1,(len(ids)+255)//256)):
                ind=torch.as_tensor(part,device=device)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    out=predict(model,teacher,ind,False)[0]
                action=out[:,0].float()*model.target_std+model.target_mean
                err.extend(((action[:,:3]-teacher['target'][ind,0,:3]).norm(dim=-1)*1000).cpu().tolist())
            teacher_val[f]=float(np.quantile(err,.95))
        return {'update':update,'onpolicy_validation':result,'teacher_validation_first_mm_p95':teacher_val}
    history=[report(0)];print(json.dumps(history[-1]),flush=True)
    best=float('inf');best_update=0;bad=0;started=time.monotonic();reason='update_cap'
    for step in range(1,a.updates+1):
        if time.monotonic()-started>a.budget_minutes*60:reason='wall_budget';break
        oi=torch.as_tensor(np.random.choice(train,32),device=device)
        ti=torch.as_tensor(np.r_[np.random.choice(teacher_by_family['sc'],16),
                                  np.random.choice(teacher_by_family['sfp'],16)],device=device)
        model.train();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            op_out=actor_predict(model,op,oi)
            te_out=predict(model,teacher,ti,False)[0]
            op_truth=(op['target'][oi]-model.target_mean)/model.target_std
            te_truth=(teacher['target'][ti]-model.target_mean)/model.target_std
            loss=(a.onpolicy_loss_weight*nn.functional.smooth_l1_loss(op_out,op_truth,beta=.2)+
                  nn.functional.smooth_l1_loss(te_out,te_truth,beta=.2))
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%250==0 or step==a.updates:
            result=report(step);result['loss']=float(loss.detach());history.append(result)
            print(json.dumps(result),flush=True)
            value=result['onpolicy_validation']['first_action_translation_p95_mm']
            teacher_err=result['teacher_validation_first_mm_p95']
            eligible=teacher_err['sc']<=9 and teacher_err['sfp']<=10
            if eligible and value<best-.1:
                best=value;best_update=step;bad=0
                torch.save({'model':model.state_dict(),'update':step,'history_enabled':True,
                    'source_manifest':{**manifest,'onpolicy_route_correction':True},
                    'final_scenes_opened':False,'rl_started':False},a.output/'best.pt')
            else:bad+=1
            if step>=1000 and bad>=5:reason='validation_plateau';break
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    summary={'schema':'aic_sc_onpolicy_route_correction/v1','best_update':best_update,
        'updates_completed':step,'stop_reason':reason,'initial':history[0],
        'selected':next((r for r in history if r['update']==best_update),None),
        'train_cases':sorted({op['rows'][int(i)]['case'] for i in train}),
        'validation_cases':sorted({op['rows'][int(i)]['case'] for i in val}),
        'train_rows':len(train),'validation_rows':len(val),
        'onpolicy_loss_weight':a.onpolicy_loss_weight,
        'near_goal_supervision':a.near_goal_supervision,
        'teacher_plan_runtime_input':False,'final_scenes_opened':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
