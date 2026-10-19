#!/usr/bin/env python3
"""Bounded SC phase-waypoint diagnostic; teacher plan is labels only."""
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


class Probe(nn.Module):
    def __init__(self, width=256, dropout=.1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(1579,width),nn.GELU(),nn.LayerNorm(width),
                               nn.Dropout(dropout),nn.Linear(width,width),nn.GELU(),
                               nn.LayerNorm(width))
        self.stage=nn.Linear(width,6)
        self.waypoints=nn.Linear(width,18)

    def forward(self,x):
        hidden=self.net(x)
        return self.stage(hidden),self.waypoints(hidden).reshape(-1,6,3)


def metrics(logits,waypoints,stage,goal,mean,std):
    chosen=waypoints[torch.arange(len(stage),device=stage.device),stage]*std+mean
    predicted_stage=logits.argmax(-1)
    actual=goal.cpu().numpy();by_stage={}
    error=torch.linalg.norm(chosen-goal,dim=-1).cpu().numpy()*1000
    for s in range(6):
        e=error[stage.cpu().numpy()==s]
        by_stage[str(s)]={'rows':int(len(e)),'p50_mm':float(np.median(e)),
                          'p95_mm':float(np.quantile(e,.95))}
    # The selected-stage target is the clean geometry test. Predicted-stage
    # target additionally tests whether the actor can select that target.
    selected=waypoints[torch.arange(len(stage),device=stage.device),predicted_stage]*std+mean
    joint=torch.linalg.norm(selected-goal,dim=-1).cpu().numpy()*1000
    return {'stage_accuracy':float((predicted_stage==stage).float().mean()),
            'known_stage_p50_mm':float(np.median(error)),
            'known_stage_p95_mm':float(np.quantile(error,.95)),
            'predicted_stage_p95_mm':float(np.quantile(joint,.95)),
            'by_stage':by_stage}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--wall-minutes',type=float,default=45)
    p.add_argument('--width',type=int,default=256)
    p.add_argument('--dropout',type=float,default=.1)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing output')
    a.output.mkdir(parents=True)
    random.seed(20261002);np.random.seed(20261002);torch.manual_seed(20261002)
    torch.set_num_threads(8)
    rows=[json.loads(line) for line in (a.cache/'rows.jsonl').open()]
    features=np.load(a.cache/'features.npy',mmap_mode='r')
    selected=[i for i,r in enumerate(rows) if r['family']=='sc']
    labels=[];inputs=[];split=[];episodes=[]
    for i in selected:
        r=rows[i];state=np.asarray(r['state'],np.float32)
        body=np.asarray(r['subgoal'][:3],np.float32)
        goal=state[:3]+Rotation.from_quat(state[3:7]).apply(body)
        labels.append(goal)
        inputs.append(np.r_[features[i].astype(np.float32),state,
                            np.asarray(r['task'],np.float32),r['time_s']/60.])
        split.append(r['split']);episodes.append(r['episode'])
    inputs=np.asarray(inputs,np.float32);labels=np.asarray(labels,np.float32)
    stages=np.asarray([rows[i]['stage'] for i in selected],np.int64)
    split=np.asarray(split);train=np.flatnonzero(split=='train');val=np.flatnonzero(split=='validation')
    if inputs.shape[1]!=1579 or len(set(episodes[i] for i in train))!=83 or len(set(episodes[i] for i in val))!=21:
        raise ValueError('Unexpected cache shape or episode split')
    # A stable phase waypoint should be invariant within a scene-stage group.
    groups=defaultdict(list)
    for j,i in enumerate(selected):groups[(rows[i]['episode'],int(stages[j]))].append(labels[j])
    spread=np.asarray([np.max(np.linalg.norm(np.asarray(v)-np.mean(v,axis=0),axis=1))*1000
                       for v in groups.values()])
    input_mean=inputs[train].mean(0);input_std=np.maximum(inputs[train].std(0),.01)
    goal_mean=labels[train].mean(0);goal_std=np.maximum(labels[train].std(0),.01)
    normalized=(inputs-input_mean)/input_std
    device='cuda';x=torch.as_tensor(normalized,device=device);y=torch.as_tensor((labels-goal_mean)/goal_std,device=device)
    goal=torch.as_tensor(labels,device=device);stage=torch.as_tensor(stages,device=device)
    mean=torch.as_tensor(goal_mean,device=device);std=torch.as_tensor(goal_std,device=device)
    model=Probe(width=a.width,dropout=a.dropout).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    pools=[train[stages[train]==s] for s in range(6)]
    validation=torch.as_tensor(val,device=device)
    baseline={}
    for s in range(6):
        center=labels[train[stages[train]==s]].mean(0)
        e=np.linalg.norm(labels[val[stages[val]==s]]-center,axis=1)*1000
        baseline[str(s)]={'rows':len(e),'p95_mm':float(np.quantile(e,.95))}
    t0=time.monotonic();history=[];best=float('inf');best_update=0;stale=0
    for update in range(1,a.updates+1):
        indices=np.concatenate([np.random.choice(pool,64) for pool in pools])
        ix=torch.as_tensor(indices,device=device)
        model.train();opt.zero_grad(set_to_none=True)
        logits,waypoints=model(x[ix]);choice=waypoints[torch.arange(len(ix),device=device),stage[ix]]
        loss=nn.functional.smooth_l1_loss(choice,y[ix],beta=.1)+.1*nn.functional.cross_entropy(logits,stage[ix])
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if update%100==0:
            model.eval()
            with torch.inference_mode():
                lg,wp=model(x[validation]);report=metrics(lg,wp,stage[validation],goal[validation],mean,std)
            score=report['known_stage_p95_mm']+max(0.,.9-report['stage_accuracy'])*100
            history.append({'update':update,'loss':float(loss.detach()),**report})
            print(json.dumps({'update':update,'score':score,'known_stage_p95_mm':report['known_stage_p95_mm'],
                              'predicted_stage_p95_mm':report['predicted_stage_p95_mm'],
                              'stage_accuracy':report['stage_accuracy']}),flush=True)
            if score<best-.1:
                best=score;best_update=update;stale=0
                torch.save({'model':model.state_dict(),'input_mean':input_mean,
                            'input_std':input_std,'goal_mean':goal_mean,'goal_std':goal_std,
                            'update':update,'architecture':'current RGB features + measured state/task/time; fixed-base phase waypoints',
                            'teacher_geometry_runtime':False},a.output/'best.pt')
            else:stale+=1
            if stale>=6 or time.monotonic()-t0>a.wall_minutes*60:break
    best_report=next(h for h in history if h['update']==best_update)
    result={'schema':'aic_sc_base_waypoint_probe/v1','best_update':best_update,
            'updates_completed':update,'train_episodes':83,'validation_episodes':21,
            'width':a.width,'dropout':a.dropout,
            'train_rows':len(train),'validation_rows':len(val),
            'phase_waypoint_within_episode_max_spread_mm_p95':float(np.quantile(spread,.95)),
            'constant_stage_mean_baseline':baseline,'best_validation':best_report,
            'offline_geometry_gate_passed':best_report['known_stage_p95_mm']<=20 and
                    best_report['stage_accuracy']>=.9 and best_report['predicted_stage_p95_mm']<=20,
            'checkpoint':str((a.output/'best.pt').resolve()),
            'cache':str(a.cache.resolve()),'final_scenes_opened':False,'rl_started':False}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
