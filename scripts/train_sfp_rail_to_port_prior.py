#!/usr/bin/env python3
"""Bounded TCP/learned-rail to requested-port image-offset feasibility probe."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CAMS=('center','left','right')


def read(path):return [json.loads(s) for s in path.open()]
def key(r):return r['scene_group'],r['episode'],round(float(r['time_s']),6)


def feature(pred,label,cam,card,port):
    ci=CAMS.index(cam)
    ends=np.asarray(pred['endpoint_xy_normalized'][ci],np.float32).reshape(-1)
    tcp=np.asarray(label['state'][:7],np.float32)
    one=np.eye(3,dtype=np.float32)[ci]
    return np.r_[ends,tcp,one,np.float32(card/4),np.float32(port)]


def examples(labels,preds,split):
    labels={key(r):r for r in read(labels/'rows.jsonl') if r['split']==split}
    predictions={key(r):r for r in read(preds) if key(r) in labels}
    if set(labels)!=set(predictions):raise ValueError(f'Missing predictions for {len(set(labels)-set(predictions))} rows')
    features=[];targets=[];metadata=[]
    for k,label in labels.items():
        pred=predictions[k]
        choices=label['targets'] if split=='train' else [next(t for t in label['targets'] if t['identity']==label['selected_identity'])]
        for t in choices:
            card=int(t['identity'].split('_')[2]);port=int(t['identity'].split('_')[-1])
            for ci,cam in enumerate(CAMS):
                uv=t['cameras'][cam]
                if not uv[3]:continue
                ends=np.asarray(pred['endpoint_xy_normalized'][ci])*[1151,1023]
                center=ends[0]+card/4*(ends[1]-ends[0])
                features.append(feature(pred,label,cam,card,port))
                targets.append((np.asarray(uv[:2])-center)/300)
                metadata.append({'episode':label['episode'],'scene_group':label['scene_group'],
                                 'time_s':label['time_s'],'requested':t['identity'],
                                 'cam':cam,'center':center.tolist(),
                                 'truth_label_only':uv[:2],
                                 'other_same_port_label_only':[x['cameras'][cam][:2] for x in label['targets']
                                     if x['identity']!=t['identity'] and x['identity'].endswith(f'_port_{port}')
                                     and x['cameras'][cam][3]]})
    return np.asarray(features,np.float32),np.asarray(targets,np.float32),metadata


class Prior(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(16,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,2))
    def forward(self,x):return self.net(x)


@torch.inference_mode()
def score(model,x,y,metadata,save=None):
    model.eval();pred=model(x).cpu().numpy()
    errors=[];nearest=[];records=[]
    for q,truth,m in zip(pred,y.cpu().numpy(),metadata):
        center=np.asarray(m['center']);uv=center+q*300
        err=float(np.linalg.norm((q-truth)*300));errors.append(err)
        other=m['other_same_port_label_only']
        nearest.append(not other or err<min(np.linalg.norm(uv-np.asarray(p)) for p in other))
        records.append({**m,'predicted_xy':uv.tolist(),'error_px_label_only':err,
                        'closer_to_requested_than_neighbor_label_only':bool(nearest[-1])})
    out={'views':len(errors),'median_px':float(np.median(errors)),
         'p95_px':float(np.quantile(errors,.95)),
         'closer_to_requested_fraction':float(np.mean(nearest)),
         'scene_groups':len({m['scene_group'] for m in metadata})}
    if save:
        with save.open('w') as f:
            for r in records:f.write(json.dumps(r,separators=(',',':'))+'\n')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--train-predictions',type=Path,required=True)
    p.add_argument('--development-predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--minutes',type=float,default=10)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous output')
    x,y,trainmeta=examples(a.labels,a.train_predictions,'train')
    vx,vy,valmeta=examples(a.labels,a.development_predictions,'validation')
    assert len({m['scene_group'] for m in trainmeta})==70
    assert len({m['scene_group'] for m in valmeta})==10
    assert not ({m['scene_group'] for m in trainmeta}&{m['scene_group'] for m in valmeta})
    a.output.mkdir(parents=True)
    torch.manual_seed(20261006);torch.set_num_threads(4)
    x=torch.from_numpy(x).cuda();y=torch.from_numpy(y).cuda();vx=torch.from_numpy(vx).cuda();vy=torch.from_numpy(vy).cuda()
    model=Prior().cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    start=time.monotonic();best=float('inf');history=[];best_step=0
    for step in range(1,a.updates+1):
        if time.monotonic()-start>a.minutes*60:break
        idx=torch.randint(len(x),(512,),device='cuda')
        model.train();opt.zero_grad(set_to_none=True)
        loss=F.smooth_l1_loss(model(x[idx]),y[idx],beta=.05)
        loss.backward();opt.step()
        if step%200==0:
            report=score(model,vx,vy,valmeta,a.output/f'development_step{step}.jsonl')
            item={'step':step,'loss':float(loss.detach()),'development':report}
            history.append(item);print(json.dumps(item),flush=True)
            if report['p95_px']<best:
                best=report['p95_px'];best_step=step
                torch.save({'model':model.state_dict(),'step':step},a.output/'best.pt')
    summary={'schema':'sfp_rail_to_port_prior/v1','train_views':len(x),'development_views':len(vx),
             'train_groups':70,'development_groups':10,'updates_completed':step,
             'best_step':best_step,'selected':next(r for r in history if r['step']==best_step),
             'feasibility_gate':{'p95_px_max':30,'closer_to_requested_fraction_min':.95,
                                 'passed':best<=30 and next(r for r in history if r['step']==best_step)['development']['closer_to_requested_fraction']>=.95},
             'runtime_true_geometry_input':False,'final_scenes_opened':False}
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
