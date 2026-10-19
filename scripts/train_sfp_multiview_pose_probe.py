#!/usr/bin/env python3
"""Learn task-selected SFP opening pose from three frozen RGB feature sets.

All scene coordinates and candidate projections are offline labels. Runtime
inputs are RGB-derived frozen features/keypoints, public task identity, and
measured TCP pose. The model does not read the true board or port transform.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler

from train_two_family_port_locator import NAMES


class PoseProbe(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,512),nn.GELU(),
                               nn.Dropout(.10),nn.Linear(512,256),nn.GELU(),nn.Linear(256,3))
    def forward(self,x):return self.net(x)


def scores(model,x,y,indices,device):
    if not len(indices):return None
    model.eval();errors=[];by=defaultdict(list)
    with torch.inference_mode():
        for start in range(0,len(indices),512):
            item=indices[start:start+512]
            pred=model(x[item].to(device)).cpu().numpy();truth=y[item].numpy()
            diff=(pred-truth)*1000
            errors.extend(np.linalg.norm(diff,axis=1).tolist())
    errors=np.asarray(errors)
    return {'n':len(errors),'median_mm':float(np.median(errors)),
            'p95_mm':float(np.quantile(errors,.95)),
            'within_8mm':float(np.mean(errors<=8)),
            'within_1mm':float(np.mean(errors<=1))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=2500)
    p.add_argument('--minutes',type=float,default=45)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--state-only',action='store_true')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing output')
    a.output.mkdir(parents=True)
    random.seed(20261002);np.random.seed(20261002);torch.manual_seed(20261002);torch.set_num_threads(8)
    rows=[json.loads(s) for s in (a.labels/'rows.jsonl').open()]
    feature=np.load(a.features,mmap_mode='r')
    if feature.shape!=(len(rows),3,768):raise ValueError('Visual-feature shape mismatch')
    pred={(q['row'],q['camera']):q for q in map(json.loads,a.predictions.open())}
    inputs=[];targets=[];meta=[]
    for i,r in enumerate(rows):
        identity=r['selected_identity'];port=next(t for t in r['targets'] if t['identity']==identity)
        state=np.asarray(r['state'][:7],np.float32);xyz=np.asarray(port['position_base_m'],np.float32)
        if np.linalg.norm(xyz-state[:3])>.25:continue
        task=np.zeros(10,np.float32);card=int(identity.split('_')[2]);side=int(identity.split('_')[4]);task[card*2+side]=1
        key=[]
        for j,cam in enumerate(('center','left','right')):
            q=pred[(i,cam)]['predicted']
            for name in NAMES[2:]:
                v=q.get(name)
                key.extend((v['xy'][0]/1151,v['xy'][1]/1023,
                            v['projected_inframe_probability']) if v is not None else (0.,0.,0.))
        inputs.append(np.r_[state,task] if a.state_only else
                      np.r_[state,task,np.asarray(key,np.float32),
                            np.asarray(feature[i],np.float32).reshape(-1)])
        targets.append(xyz-state[:3])
        meta.append((r['episode'],r['split'],r['scene_group'],r['image_kind'],identity,float(np.linalg.norm(xyz-state[:3]))))
    raw=np.asarray(inputs,np.float32);target=np.asarray(targets,np.float32)
    selection_names=sorted({e for e,split,*_ in meta if split=='validation'})
    selection={e for i,e in enumerate(selection_names) if ((i//2)+(i%2))%2==0}
    test=set(selection_names)-selection
    train=np.asarray([i for i,m in enumerate(meta) if m[1]=='train'])
    val=np.asarray([i for i,m in enumerate(meta) if m[0] in selection])
    held=np.asarray([i for i,m in enumerate(meta) if m[0] in test])
    scenes={split:{m[2] for m in meta if m[1]==split} for split in ('train','validation')}
    if scenes['train']&scenes['validation']:raise ValueError('Scene/reset leakage')
    input_mean=raw[train].mean(0);input_std=raw[train].std(0).clip(min=.01)
    # Keep the public task one-hot and coordinate channels numerically useful.
    x=torch.from_numpy((raw-input_mean)/input_std)
    y=torch.from_numpy(target)
    episode_count=Counter(meta[i][0] for i in train)
    native='native_1152x1024_rgb_jpeg';source_count=Counter(meta[i][3]==native for i in train)
    weights=[(.5/source_count[meta[i][3]==native])/episode_count[meta[i][0]] for i in train]
    train_set=TensorDataset(x[train],y[train])
    loader=DataLoader(train_set,batch_size=a.batch_size,
                      sampler=WeightedRandomSampler(weights,num_samples=a.updates*a.batch_size,replacement=True))
    model=PoseProbe(x.shape[1]).cuda();opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-3)
    baseline={}
    for name,indices in [('selection',val),('test',held)]:
        mean=target[train].mean(0);error=np.linalg.norm((target[indices]-mean)*1000,axis=1)
        baseline[name]={'n':len(error),'median_mm':float(np.median(error)),
                        'p95_mm':float(np.quantile(error,.95))}
    best=float('inf');best_step=0;stale=0;start=time.monotonic();reason='update_cap';history=[]
    for step,(xb,yb) in enumerate(loader,start=1):
        if step>a.updates:break
        if time.monotonic()-start>a.minutes*60:reason='wall_budget';break
        model.train();opt.zero_grad(set_to_none=True);predicted=model(xb.cuda());truth=yb.cuda()
        loss=F.smooth_l1_loss(predicted,truth,beta=.002)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%100==0:
            result=scores(model,x,y,val,'cuda');history.append({'update':step,'loss':float(loss.detach()),'selection':result})
            print(json.dumps({'update':step,'selection_p95_mm':result['p95_mm'],
                              'selection_median_mm':result['median_mm']}),flush=True)
            if result['p95_mm']<best-.2:
                best=result['p95_mm'];best_step=step;stale=0
                torch.save({'model':model.state_dict(),'input_mean':input_mean,'input_std':input_std,
                            'input_dim':x.shape[1],'state_only':a.state_only,'update':step,
                            'geometry_runtime':False},a.output/'best.pt')
            else:stale+=1
            if step>=700 and stale>=8:reason='validation_plateau';break
    if not best_step:raise RuntimeError('No selected checkpoint')
    model.load_state_dict(torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)['model'])
    all_scores={}
    for name,indices in [('selection',val),('test',held)]:
        all_scores[name]=scores(model,x,y,indices,'cuda')
        for identity in sorted(set(meta[i][4] for i in indices)):
            subset=np.asarray([i for i in indices if meta[i][4]==identity])
            all_scores[name+'_'+identity]=scores(model,x,y,subset,'cuda')
    summary={'schema':'aic_sfp_multiview_pose_probe/v1','best_update':best_step,
             'updates_completed':min(step,a.updates),'stop_reason':reason,
             'state_only':a.state_only,'training_episodes':len(set(meta[i][0] for i in train)),
             'selection_episodes':sorted(selection),'test_episodes':sorted(test),
             'constant_mean_baseline':baseline,'heldout':all_scores,
             'image_features_frozen':not a.state_only,'geometry_runtime':False,
             'final_scenes_opened':False}
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
