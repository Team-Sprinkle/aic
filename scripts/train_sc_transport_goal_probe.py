#!/usr/bin/env python3
"""Bounded SC teacher-standoff pose probe from ordinary images/state/task only."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn

from train_shared_port_tcp_bc import rotation_from_pose9


class GoalProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(1578), nn.Linear(1578,512), nn.GELU(),
                                 nn.Linear(512,256), nn.GELU(), nn.Linear(256,6))
    def forward(self, x): return self.net(x)


def labels(rows):
    final = {}
    for r in rows:
        if r['family']=='sc' and r['stage']==5:
            tcp = Rotation.from_quat(r['state'][3:7])
            target = torch.as_tensor([r['subgoal']], dtype=torch.float32)
            rot = rotation_from_pose9(target)[0].numpy()
            pose = (np.asarray(r['state'][:3])+tcp.apply(r['subgoal'][:3]),
                    tcp*Rotation.from_matrix(rot))
            if r['episode'] in final:
                if np.linalg.norm(final[r['episode']][0]-pose[0]) > 1e-5:
                    raise ValueError('Inconsistent episode target')
            else: final[r['episode']] = pose
    result = np.zeros((len(rows),6),dtype=np.float32)
    for i,r in enumerate(rows):
        if r['family']!='sc': continue
        pos,rot=final[r['episode']]
        tcp=Rotation.from_quat(r['state'][3:7])
        result[i,:3]=tcp.inv().apply(pos-np.asarray(r['state'][:3]))
        result[i,3:]=(tcp.inv()*rot).as_rotvec()
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=5000)
    p.add_argument('--alternate-goals',type=Path,
                   help='Training-only matched-image alternate SC port standoff labels')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing run')
    a.output.mkdir(parents=True)
    torch.set_num_threads(8);random.seed(20261001);np.random.seed(20261001);torch.manual_seed(20261001)
    rows=[json.loads(s) for s in (a.cache/'rows.jsonl').open()]
    feature=np.load(a.cache/'features.npy',mmap_mode='r')
    base=torch.load(a.base,map_location='cpu',weights_only=False)
    state=np.asarray([r['state'] for r in rows],dtype=np.float32)
    mean=base['model']['state_mean'].numpy();std=base['model']['state_std'].numpy()
    state=(state-mean)/std
    state[:,base.get('masked_state_indices',[])]=0
    task=np.asarray([r['task'] for r in rows],dtype=np.float32)
    x=torch.as_tensor(np.concatenate((np.asarray(feature),state,task),axis=1),device='cuda')
    if x.shape[1]!=1578:raise ValueError(x.shape)
    target=labels(rows)
    y=torch.as_tensor(target/np.array([.2,.2,.2,.3,.3,.3],dtype=np.float32),device='cuda')
    alternate=None
    if a.alternate_goals:
        offsets=json.loads(a.alternate_goals.read_text())['episodes']
        chosen_goals={}
        for r in rows:
            if r['family']=='sc' and r['stage']==5:
                tcp=Rotation.from_quat(r['state'][3:7])
                chosen_goals[r['episode']]=np.asarray(r['state'][:3])+tcp.apply(r['subgoal'][:3])
        alternate=target.copy()
        for i,r in enumerate(rows):
            if r['family']!='sc':continue
            spec=offsets[str(r['episode'])]
            selected=int(np.argmax(r['task'][2:4]))
            if selected!=spec['selected_port']:raise ValueError('Selected-port label mismatch')
            delta=np.asarray(spec['port0_minus_port1_standoff_base_m'])
            other=chosen_goals[r['episode']]+(-delta if selected==0 else delta)
            tcp=Rotation.from_quat(r['state'][3:7])
            alternate[i,:3]=tcp.inv().apply(other-np.asarray(r['state'][:3]))
        alternate=torch.as_tensor(alternate/np.array([.2,.2,.2,.3,.3,.3],dtype=np.float32),device='cuda')
    train={stage:np.asarray([i for i,r in enumerate(rows) if r['family']=='sc' and
            r['split']=='train' and r['stage']==stage]) for stage in range(6)}
    near=np.asarray([i for i,r in enumerate(rows) if r['family']=='sc' and r['split']=='train' and
                     np.linalg.norm(target[i,:3])<.05])
    val=np.asarray([i for i,r in enumerate(rows) if r['family']=='sc' and r['split']=='validation'])
    model=GoalProbe().cuda();opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    history=[];best=float('inf');best_update=0;bad=0
    for update in range(1,a.updates+1):
        indices=np.r_[*[np.random.choice(train[s],8) for s in range(6)],np.random.choice(near,16)]
        idx=torch.as_tensor(indices,device='cuda')
        xx=x[idx];yy=y[idx]
        if alternate is not None:
            xx=xx.clone();yy=yy.clone()
            flip=torch.rand(len(idx),device='cuda')<.5
            xx[flip,-8:-6]=xx[flip,-8:-6].flip(-1)
            yy[flip]=alternate[idx[flip]]
        model.train();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=model(xx);loss=nn.functional.smooth_l1_loss(pred,yy,beta=.05)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if update%250==0 or update==a.updates:
            model.eval();preds=[]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for part in np.array_split(val,max(1,(len(val)+511)//512)):
                    preds.append(model(x[torch.as_tensor(part,device='cuda')]).float().cpu().numpy())
            pred=np.concatenate(preds)*np.array([.2,.2,.2,.3,.3,.3])
            truth=target[val]
            err=np.linalg.norm(pred[:,:3]-truth[:,:3],axis=1)*1000
            rot=np.linalg.norm(pred[:,3:]-truth[:,3:],axis=1)*180/np.pi
            near_mask=np.linalg.norm(truth[:,:3],axis=1)<.05
            by_stage={str(stage):float(np.quantile(err[[rows[int(i)]['stage']==stage for i in val]],.95))
                      for stage in range(6)}
            report={'update':update,'loss':float(loss.detach()),
                'all_translation_p95_mm':float(np.quantile(err,.95)),
                'near50_rows':int(near_mask.sum()),
                'near50_translation_median_mm':float(np.median(err[near_mask])),
                'near50_translation_p95_mm':float(np.quantile(err[near_mask],.95)),
                'near50_orientation_p95_deg':float(np.quantile(rot[near_mask],.95)),
                'stage_translation_p95_mm':by_stage}
            history.append(report);print(json.dumps(report),flush=True)
            score=report['near50_translation_p95_mm']
            if score<best-.05:
                best=score;best_update=update;bad=0
                torch.save({'model':model.state_dict(),'update':update,
                    'feature_source':'frozen three-view ImageNet ResNet18',
                    'input':'1536D frozen RGB + normalized masked ordinary 32D state + public 10D task',
                    'output':'TCP-relative final teacher standoff translation and rotation vector',
                    'counterfactual_port_augmentation':bool(a.alternate_goals),
                    'final_scenes_opened':False},a.output/'best.pt')
            else:bad+=1
            if update>=1000 and bad>=6:break
    selected=next(v for v in history if v['update']==best_update)
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps({'best_update':best_update,
        'updates_completed':update,'stop_reason':'validation_plateau' if update<a.updates else 'update_cap',
        'selected_validation':selected,'train_episodes':len({r['episode'] for r in rows if r['family']=='sc' and r['split']=='train'}),
        'validation_episodes':len({r['episode'] for r in rows if r['family']=='sc' and r['split']=='validation'}),
        'alternate_goals':str(a.alternate_goals) if a.alternate_goals else None,
        'final_scenes_opened':False},indent=2)+'\n')


if __name__=='__main__':main()
