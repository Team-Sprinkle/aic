#!/usr/bin/env python3
"""Requested-port orientation from frozen RGB context, task and measured TCP.

Uses the full grouped native corpus. True orientations supervise training and
scoring only. A state-only arm and a family mean rotation are explicit controls.
The native point refiner itself predicts pixels, not orientation.
"""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.nn import functional as F


def rotation6(x):
    a=F.normalize(x[...,:3],dim=-1);b=F.normalize(x[...,3:]-(a*x[...,3:]).sum(-1,keepdim=True)*a,dim=-1)
    return torch.stack((a,b,torch.cross(a,b,dim=-1)),-1)


def angle(output,target):
    r=output.transpose(-1,-2)@target
    # atan2 avoids the artificial 0.08-degree acos clamp floor near identity.
    skew=torch.stack((r[...,2,1]-r[...,1,2],r[...,0,2]-r[...,2,0],r[...,1,0]-r[...,0,1]),-1)
    sine=torch.linalg.vector_norm(skew,dim=-1)/2;cosine=(r.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
    return torch.rad2deg(torch.atan2(sine,cosine))


def task_code(r):
    x=np.zeros(10,np.float32);x[0 if r['family']=='sfp' else 1]=1
    parts=r['selected_identity'].split('_');x[2+int(parts[-1])]=1
    if r['family']=='sfp':x[4+int(parts[2])]=1;x[9]=1
    return x


class RequestedOrientation(nn.Module):
    def __init__(self,state_only=False):
        super().__init__();self.state_only=state_only
        self.condition=nn.Sequential(nn.Linear(17,128),nn.GELU(),nn.Linear(128,128))
        if not state_only:
            self.visual=nn.Linear(256,128);self.positions=nn.Parameter(torch.randn(72,128)*.01)
            layer=nn.TransformerEncoderLayer(128,4,256,dropout=.1,norm_first=True,batch_first=True)
            self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            self.attend=nn.MultiheadAttention(128,4,batch_first=True)
        self.head=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,128),nn.GELU(),nn.Linear(128,6))
    def forward(self,visual,state,task):
        query=self.condition(torch.cat((state,task),-1))
        if not self.state_only:
            memory=self.encoder(self.visual(visual.flatten(1,2).float())+self.positions[None])
            result,_=self.attend(query[:,None],memory,memory,need_weights=False);query=query+result[:,0]
        return rotation6(self.head(query))


def stats(values):
    v=np.asarray(values);return dict(n=len(v),p50_deg=float(np.median(v)) if len(v) else None,p95_deg=float(np.percentile(v,95)) if len(v) else None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-for-cache-pid',type=int);p.add_argument('--updates',type=int,default=12000);p.add_argument('--minutes-per-arm',type=float,default=45)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    wait_start=time.monotonic()
    while not (a.cache/'cache_summary.json').exists():
        if a.wait_for_cache_pid is not None:os.kill(a.wait_for_cache_pid,0)
        if time.monotonic()-wait_start>7200:raise TimeoutError('Native feature cache never completed')
        time.sleep(30)
    torch.set_num_threads(6);torch.manual_seed(20261015)
    rows=[json.loads(l) for l in (a.cache/'frame_context.jsonl').open()]
    groups={split:{r['scene_group'] for r in rows if r['split']==split} for split in ('train','validation')}
    assert groups['train'].isdisjoint(groups['validation'])
    mmap=np.load(a.cache/'coarse_spatial_8x9.npy',mmap_mode='r')
    features=torch.empty((len(rows),3,24,256),device='cuda',dtype=torch.float16)
    for start in range(0,len(rows),128):
        ids=[r['feature_row'] for r in rows[start:start+128]]
        x=torch.from_numpy(np.asarray(mmap[ids])).cuda().float().flatten(0,1)
        features[start:start+len(ids)]=F.adaptive_avg_pool2d(x,(4,6)).flatten(2).transpose(1,2).reshape(len(ids),3,24,256).half()
    state=torch.tensor([r['state'][:7] for r in rows],device='cuda',dtype=torch.float32)
    state[:,3:7]*=torch.where(state[:,6:7]<0,-1.,1.)
    task=torch.tensor(np.stack([task_code(r) for r in rows]),device='cuda')
    targets=np.stack([Rotation.from_quat(next(t for t in r['targets'] if t['identity']==r['selected_identity'])['orientation_base_xyzw']).as_matrix() for r in rows])
    target=torch.tensor(targets,device='cuda',dtype=torch.float32)
    train=np.array([i for i,r in enumerate(rows) if r['split']=='train']);val=np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
    mean=state[train].mean(0);std=state[train].std(0).clamp_min(.02);state=(state-mean)/std
    by_episode=defaultdict(list)
    for i in val:by_episode[(rows[i]['family'],rows[i]['episode'])].append(i)
    screen=np.array([v[j] for v in by_episode.values() for j in np.linspace(0,len(v)-1,min(8,len(v)),dtype=int)])
    count=Counter((rows[i]['family'],rows[i]['episode']) for i in train);families=Counter(f for f,_ in count)
    weights=np.array([1/count[(rows[i]['family'],rows[i]['episode'])]/families[rows[i]['family']] for i in train]);weights/=weights.sum()
    baseline={}
    for family in ('sc','sfp'):
        ids=[i for i in train if rows[i]['family']==family]
        cells=Counter(rows[i]['scene_group'] for i in ids)
        average=Rotation.from_matrix(targets[ids]).mean(weights=[1/cells[rows[i]['scene_group']] for i in ids])
        vi=[i for i in val if rows[i]['family']==family]
        baseline[family]=stats(np.degrees((average.inv()*Rotation.from_matrix(targets[vi])).magnitude()))
    all_reports=[]
    for variant in ('state_only','RGB_state_task'):
        out=a.output/variant;out.mkdir();torch.manual_seed(20261015);rng=np.random.default_rng(20261015)
        model=RequestedOrientation(variant=='state_only').cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.001)
        @torch.inference_mode()
        def predict(ids):
            model.eval();pred=[]
            for j in range(0,len(ids),128):
                batch=ids[j:j+128];pred.append(model(features[batch],state[batch],task[batch]).cpu())
            return torch.cat(pred)
        best=float('inf');best_step=0;stale=0;reason='update_cap';start=time.monotonic()
        for step in range(1,a.updates+1):
            if time.monotonic()-start>a.minutes_per_arm*60:reason='time_cap_not_proven_plateau';break
            ids=rng.choice(train,128,replace=True,p=weights);model.train();opt.zero_grad(set_to_none=True)
            output=model(features[ids],state[ids],task[ids]);loss=F.mse_loss(output,target[ids]);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            if step%250==0:
                pred=predict(screen);err=angle(pred,target[screen].cpu()).numpy()
                report={f:stats(err[[j for j,i in enumerate(screen) if rows[i]['family']==f]]) for f in ('sc','sfp')}
                score=max(v['p95_deg'] for v in report.values());record=dict(variant=variant,step=step,loss=float(loss.detach()),screen=report,elapsed_s=time.monotonic()-start)
                with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
                checkpoint=dict(schema='native_requested_orientation/v2_public_only',model=model.state_dict(),variant=variant,state_mean=mean.cpu(),state_std=std.cpu(),step=step,geometry_input=False,privileged_scene_setting_input=False,task_code='family2_port2_card5_valid1')
                torch.save(checkpoint,out/'last.pt')
                if score<best-.01:best=score;best_step=step;stale=0;torch.save(checkpoint,out/'best.pt')
                else:stale+=1
                if step>=3000 and stale>=8:reason='validation_plateau';break
        model.load_state_dict(torch.load(out/'best.pt',map_location='cpu',weights_only=False)['model']);pred=predict(val);err=angle(pred,target[val].cpu()).numpy()
        grouped_errors=defaultdict(list)
        with (out/'validation_predictions.jsonl').open('w') as f:
            for j,i in enumerate(val):
                r=rows[i]
                for key in (r['family'],f"{r['family']}/cards{r['card_count']}",f"{r['family']}/{r['selected_identity']}"):grouped_errors[key].append(float(err[j]))
                f.write(json.dumps(dict(episode=r['episode'],scene_group=r['scene_group'],family=r['family'],time_s=r['time_s'],predicted_orientation_base_matrix=pred[j].tolist(),orientation_error_deg=float(err[j]),truth_access='scoring only'))+'\n')
        report=dict(variant=variant,best_step=best_step,updates_completed=step,stop_reason=reason,errors={k:stats(v) for k,v in grouped_errors.items()},parameters=sum(v.numel() for v in model.parameters()),final_scenes_opened=False,promoted=False)
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');all_reports.append(report);del model,opt
    summary=dict(schema='native_requested_orientation_comparison/v1',family_mean_rotation_baseline=baseline,arms=all_reports,
        train_episodes={f:len({rows[i]['episode'] for i in train if rows[i]['family']==f}) for f in ('sc','sfp')},
        validation_episodes={f:len({rows[i]['episode'] for i in val if rows[i]['family']==f}) for f in ('sc','sfp')},
        actual_unique_reset_groups={k:len(v) for k,v in groups.items()},full_pose_not_yet_promoted=True,final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
