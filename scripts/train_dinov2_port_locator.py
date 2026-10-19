#!/usr/bin/env python3
"""Bounded DINOv2-small visual-backbone ablation for all requested ports.

The frozen pretrained encoder sees RGB only. A small spatial head learns
projected port openings and projected-in-frame flags; geometry is label only.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,WeightedRandomSampler

from train_two_family_port_locator import Views,NAMES,WIDTH,HEIGHT


class DinoPortLocator(nn.Module):
    def __init__(self,hub,weights):
        super().__init__()
        self.encoder=torch.hub.load(str(hub),'dinov2_vits14',source='local',pretrained=False)
        self.encoder.load_state_dict(torch.load(weights,map_location='cpu',weights_only=False))
        for p in self.encoder.parameters():p.requires_grad_(False)
        self.decoder=nn.Sequential(nn.Conv2d(384+5,192,3,padding=1),nn.GELU(),
                                   nn.Conv2d(192,96,3,padding=1),nn.GELU(),nn.Conv2d(96,12,1))
        self.inframe=nn.Sequential(nn.Linear(384+5,128),nn.GELU(),nn.Linear(128,12))
        self.register_buffer('mean',torch.tensor((.485,.456,.406))[None,:,None,None])
        self.register_buffer('std',torch.tensor((.229,.224,.225))[None,:,None,None])

    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self

    def forward(self,rgb,meta):
        x=F.interpolate(rgb,size=(252,280),mode='bilinear',align_corners=False)
        x=(x-self.mean)/self.std
        with torch.no_grad():
            tokens=self.encoder.forward_features(x)['x_norm_patchtokens']
        feature=tokens.transpose(1,2).reshape(len(x),384,18,20)
        joined=torch.cat((feature,meta[:,:,None,None].expand(-1,-1,18,20)),dim=1)
        heat=self.decoder(joined)
        b,k,h,w=heat.shape
        probs=F.softmax(heat.flatten(2).float()*4,dim=-1)
        y,x=torch.meshgrid(torch.linspace(0,1,h,device=rgb.device),
                           torch.linspace(0,1,w,device=rgb.device),indexing='ij')
        coords=torch.stack(((probs*x.flatten()).sum(-1),(probs*y.flatten()).sum(-1)),-1)
        flag=self.inframe(torch.cat((tokens.mean(1),meta),-1))
        return coords,flag,heat


@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval();errors=defaultdict(list);rows=0
    for image,meta,points,_,inframe,selected,distance,*_ in loader:
        pred,_,_=model(image.to(device).float()/255.,meta.to(device))
        pred=pred.cpu().numpy();points=points.numpy();inframe=inframe.numpy()
        family=np.argmax(meta.numpy()[:,:2],axis=1);selected=selected.numpy();distance=distance.numpy()
        for i in range(len(image)):
            fam='sc' if family[i]==0 else 'sfp';idx=int(selected[i]);rows+=1
            if not inframe[i,idx]:continue
            error=float(np.linalg.norm((pred[i,idx]-points[i,idx])*[WIDTH-1,HEIGHT-1]))
            errors[(fam,'near' if distance[i]<.2 else 'far')].append(error)
    return {'images':rows,'errors':{str(k):{'n':len(v),'median_px':float(np.median(v)),
                'p95_px':float(np.quantile(v,.95))} for k,v in errors.items() if v}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--extra-labels',type=Path,required=True)
    p.add_argument('--sfp-cache',type=Path,required=True)
    p.add_argument('--hub',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=1800)
    p.add_argument('--minutes',type=float,default=90)
    p.add_argument('--batch-size',type=int,default=16)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing output')
    a.output.mkdir(parents=True)
    random.seed(20261002);np.random.seed(20261002);torch.manual_seed(20261002);torch.set_num_threads(8);cv2.setNumThreads(1)
    rows=[json.loads(s) for s in (a.labels/'rows.jsonl').open()]
    native_start=len(rows);rows.extend(json.loads(s) for s in (a.extra_labels/'rows.jsonl').open())
    if {r['scene_group'] for r in rows if r['split']=='train'} & {r['scene_group'] for r in rows if r['split']=='validation'}:
        raise ValueError('Scene/reset leakage')
    native_val=sorted({r['episode'] for r in rows[native_start:] if r['split']=='validation'})
    selection={e for i,e in enumerate(native_val) if ((i//2)+(i%2))%2==0};test=set(native_val)-selection
    train=[(i,None) for i,r in enumerate(rows) if r['split']=='train']
    old_val=[]
    for family in ('sc','sfp'):
        candidates=[(i,c) for i,r in enumerate(rows[:native_start]) if r['split']=='validation' and r['family']==family for c in range(3)]
        old_val.extend(random.Random(20261002).sample(candidates,min(600,len(candidates))))
    native_select=[(i,c) for i in range(native_start,len(rows)) if rows[i]['episode'] in selection for c in range(3)]
    native_test=[(i,c) for i in range(native_start,len(rows)) if rows[i]['episode'] in test for c in range(3)]
    def source(i):
        if i<native_start:return rows[i]['family']
        return 'agent_sfp' if rows[i]['image_kind']=='agent_288x256_rgb_jpeg' else 'native_sfp'
    count=Counter(source(i) for i,_ in train);mix={'agent_sfp':.35,'native_sfp':.35,'sc':.15,'sfp':.15}
    weights=[mix[source(i)]/count[source(i)] for i,_ in train]
    train_loader=DataLoader(Views(rows,a.sfp_cache,train,True),batch_size=a.batch_size,
        sampler=WeightedRandomSampler(weights,num_samples=a.updates*a.batch_size,replacement=True),
        num_workers=4,pin_memory=True,persistent_workers=True)
    def validation(indices):
        return DataLoader(Views(rows,a.sfp_cache,indices,False),batch_size=48,
            num_workers=4,pin_memory=True,persistent_workers=True)
    old_loader=validation(old_val);select_loader=validation(native_select);test_loader=validation(native_test)
    model=DinoPortLocator(a.hub,a.weights).cuda()
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=2e-4,weight_decay=1e-4)
    start=time.monotonic();best=float('inf');best_step=0;stale=0;reason='update_cap';history=[]
    for step,batch in enumerate(train_loader,start=1):
        if step>a.updates:break
        if time.monotonic()-start>a.minutes*60:reason='wall_budget';break
        image,meta,point,_,inframe,selected,_,_,_=batch
        image=image.cuda(non_blocking=True).float()/255.;meta=meta.cuda(non_blocking=True)
        point=point.cuda(non_blocking=True);inframe=inframe.cuda(non_blocking=True)
        selected=selected.cuda(non_blocking=True)
        model.train();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):pred,flag,heat=model(image,meta)
        weights=inframe.clone();weights[torch.arange(len(image),device='cuda'),selected]*=2
        pixel=F.smooth_l1_loss(pred.float(),point,beta=.01,reduction='none').sum(-1)
        h,w=heat.shape[-2:]
        cell=(point[...,1]*(h-1)).round().long().clamp(0,h-1)*w+(point[...,0]*(w-1)).round().long().clamp(0,w-1)
        spatial=F.cross_entropy((heat.float()*4).flatten(0,1).flatten(1),cell.flatten(),reduction='none').view_as(inframe)
        loss=((pixel+.005*spatial)*weights).sum()/weights.sum().clamp_min(1)
        loss+=.05*(F.binary_cross_entropy_with_logits(flag.float(),inframe,reduction='none')).mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward();nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.);opt.step()
        if step%100==0:
            old_report=evaluate(model,old_loader,'cuda');native_report=evaluate(model,select_loader,'cuda')
            sc=old_report['errors'].get("('sc', 'near')",{}).get('p95_px',1e6)
            sfp=old_report['errors'].get("('sfp', 'near')",{}).get('p95_px',1e6)
            native=native_report['errors'].get("('sfp', 'near')",{}).get('p95_px',1e6)
            score=max(native,8*sc,8*sfp)
            history.append({'update':step,'loss':float(loss.detach()),'selection_score':score,
                            'old_validation':old_report,'native_selection':native_report})
            print(json.dumps({'update':step,'loss':float(loss.detach()),'native_p95_px':native,
                              'sc_p95_px':sc,'sfp_p95_px':sfp}),flush=True)
            if score<best-1:
                best=score;best_step=step;stale=0
                torch.save({'model':model.state_dict(),'update':step,'names':NAMES,
                            'architecture':'frozen_dinov2_small_spatial',
                            'geometry_runtime':False,'literal_visibility_certified':False},a.output/'best.pt')
            else:stale+=1
            if step>=700 and stale>=7:reason='validation_plateau';break
    if not best_step:raise RuntimeError('No selected checkpoint')
    model.load_state_dict(torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)['model'])
    summary={'schema':'aic_dinov2_small_port_locator/v1','best_update':best_step,
             'updates_completed':min(step,a.updates),'stop_reason':reason,
             'train_source_rows':dict(count),'native_selection_episodes':sorted(selection),
             'native_test_episodes':sorted(test),'old_validation':evaluate(model,old_loader,'cuda'),
             'native_selection':evaluate(model,select_loader,'cuda'),
             'native_test':evaluate(model,test_loader,'cuda'),
             'geometry_runtime':False,'projected_inframe_not_visibility':True,'final_scenes_opened':False}
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
