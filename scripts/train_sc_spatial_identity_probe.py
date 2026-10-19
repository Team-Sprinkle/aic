#!/usr/bin/env python3
"""Compare frozen-backbone spatial and pooled heads for selected SC port rejection.

Labels are projected-in-frame supervision, not an occlusion/identifiability claim.
The requested port ID and measured TCP are deployable inputs; no simulator
geometry enters inference. Both heads see identical reset-grouped examples.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_sc_absent_presence_head import load_rows
from train_two_family_port_locator import Locator


CAMERAS=('center','left','right')


class SelectedHead(nn.Module):
    def __init__(self, spatial, channels, condition_dim):
        super().__init__()
        self.spatial=spatial
        if spatial:
            self.conv=nn.Sequential(nn.Conv2d(channels+condition_dim,64,3,padding=1),nn.GELU(),
                                    nn.Conv2d(64,32,3,padding=1),nn.GELU())
            self.final=nn.Sequential(nn.Linear(64+condition_dim,64),nn.GELU(),nn.Linear(64,1))
        else:
            self.final=nn.Sequential(nn.Linear(channels+condition_dim,128),nn.GELU(),
                                     nn.Linear(128,1))

    def forward(self,visual,condition):
        if not self.spatial:return self.final(torch.cat((visual.mean((-1,-2)),condition),-1)).squeeze(-1)
        x=torch.cat((visual,condition[...,None,None].expand(-1,-1,*visual.shape[-2:])),1)
        x=self.conv(x)
        pooled=torch.cat((x.mean((-1,-2)),x.amax((-1,-2)),condition),-1)
        return self.final(pooled).squeeze(-1)


@torch.inference_mode()
def frozen_maps(locator,rows,device):
    locator.eval();result=[]
    for i in range(0,len(rows),24):
        batch=rows[i:i+24];images=[]
        for row in batch:
            image=cv2.imread(row['image'])
            if image is None:raise FileNotFoundError(row['image'])
            images.append(cv2.cvtColor(cv2.resize(image,(288,256),interpolation=cv2.INTER_AREA),
                                       cv2.COLOR_BGR2RGB))
        x=torch.from_numpy(np.stack(images)).permute(0,3,1,2).to(device).float()/255
        x=(x-locator.mean)/locator.std
        x=locator.backbone.maxpool(locator.backbone.relu(locator.backbone.bn1(locator.backbone.conv1(x))))
        x=locator.backbone.layer3(locator.backbone.layer2(locator.backbone.layer1(x)))
        result.append(x.half().cpu())
    return torch.cat(result)


def measures(rows,probabilities,threshold):
    out={};pred=probabilities>=threshold
    for split,source in sorted({(r['split'],r['source']) for r in rows}):
        indices=[i for i,r in enumerate(rows) if (r['split'],r['source'])==(split,source)]
        truth=np.asarray([rows[i]['labels'] for i in indices],bool)
        p=pred[indices]
        out[f'{split}/{source}']={'groups':len({rows[i]['group'] for i in indices}),
            'views':len(indices),'absent_false_positive':int((p&~truth).sum()),
            'absent_total':int((~truth).sum()),
            'present_recalled':int((p&truth).sum()),'present_total':int(truth.sum())}
    for group in sorted({r['group'] for r in rows if r['split']=='development'}):
        indices=[i for i,r in enumerate(rows) if r['group']==group]
        truth=np.asarray([rows[i]['labels'] for i in indices],bool);p=pred[indices]
        out[f'development_group/{group}']={'views':len(indices),
            'absent_false_positive':int((p&~truth).sum()),'absent_total':int((~truth).sum()),
            'present_recalled':int((p&truth).sum()),'present_total':int(truth.sum())}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--two-port-labels',type=Path,required=True)
    p.add_argument('--one-port-train',nargs='+',type=Path,required=True)
    p.add_argument('--one-port-development',nargs='+',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--two-port-frames-per-group',type=int,default=6)
    p.add_argument('--one-port-frames-per-group',type=int,default=24)
    p.add_argument('--updates',type=int,default=800)
    p.add_argument('--near-one-port-boost',type=float,default=1.,
                   help='Training sampling boost for positive one-port views at 150-220 mm TCP distance')
    p.add_argument('--spatial-only',action='store_true')
    p.add_argument('--mask-tcp-orientation',action='store_true',
                   help='Remove measured TCP quaternion from selected-port identity inputs')
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior run')
    a.output.mkdir(parents=True)
    cv2.setNumThreads(1);torch.set_num_threads(4);torch.manual_seed(20261002)
    rows=load_rows(a)
    locator=Locator(None).to(a.device).eval()
    locator.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model'])
    visual=frozen_maps(locator,rows,a.device).to(a.device).float()
    state=torch.tensor([r['state'] for r in rows],dtype=torch.float32,device=a.device)
    train_mask=torch.tensor([r['split']=='train' for r in rows],device=a.device)
    state_mean=state[train_mask].mean(0);state_std=state[train_mask].std(0).clamp_min(.02)
    state=(state-state_mean)/state_std
    if a.mask_tcp_orientation:
        state[:,3:7]=0
    camera=torch.tensor([[float(r['camera']==c) for c in CAMERAS] for r in rows],
                        dtype=torch.float32,device=a.device)
    # Explicit request conditioning: each RGB view is scored once per requested port.
    request=torch.eye(2,device=a.device).repeat(len(rows),1)
    condition=torch.cat((camera.repeat_interleave(2,0),state.repeat_interleave(2,0),request),-1)
    target=torch.tensor([r['labels'] for r in rows],dtype=torch.float32,device=a.device).flatten()
    by_source=defaultdict(list)
    for i,r in enumerate(rows):
        if r['split']=='train':by_source[r['source']].append(i)
    if not {'one_port','two_port'}<=by_source.keys():raise ValueError('Missing training source')
    one_port=np.asarray(by_source['one_port'],dtype=int)
    one_weights=np.ones(len(one_port),dtype=float)
    for j,i in enumerate(one_port):
        row=rows[i]
        relative=row.get('relative_opening_tcp_m')
        if relative is not None and any(row['labels']) and \
                .15<=float(np.linalg.norm(relative))<=.22:
            one_weights[j]=a.near_one_port_boost
    one_weights/=one_weights.sum()
    records=[]
    variants=[('spatial',True)] if a.spatial_only else [('pooled',False),('spatial',True)]
    for variant,spatial in variants:
        torch.manual_seed(20261002);rng=np.random.default_rng(20261002)
        model=SelectedHead(spatial,visual.shape[1],condition.shape[1]).to(a.device)
        opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.01)
        history=[]
        for step in range(1,a.updates+1):
            source_indices=np.r_[rng.choice(one_port,32,replace=True,p=one_weights),
                                 rng.choice(by_source['two_port'],32,replace=True)]
            index=torch.as_tensor((source_indices[:,None]*2+np.arange(2)).flatten(),device=a.device)
            model.train()
            logits=model(visual[index//2],condition[index])
            loss=F.binary_cross_entropy_with_logits(logits,target[index])
            opt.zero_grad();loss.backward();opt.step()
            if step%200==0:history.append({'step':step,'train_loss':float(loss.detach())})
        model.eval()
        with torch.inference_mode():
            logits=torch.cat([model(visual[torch.arange(i,min(i+256,len(target)),device=a.device)//2],
                                    condition[i:i+256])
                              for i in range(0,len(target),256)]).sigmoid().reshape(-1,2).cpu().numpy()
        # No threshold selection on development scenes. 0.5 is fixed in advance.
        metrics=measures(rows,logits,.5)
        ckpt=a.output/(variant+'.pt')
        torch.save({'model':model.state_dict(),'variant':variant,
                    'state_mean':state_mean.cpu(),'state_std':state_std.cpu(),
                    'mask_tcp_orientation':a.mask_tcp_orientation,
                    'frozen_locator':str(a.checkpoint),'projection_not_true_visibility':True},ckpt)
        records.append({'variant':variant,'metrics':metrics,'history':history,'checkpoint':str(ckpt)})
    summary={'schema':'aic_sc_spatial_identity_probe/v1','records':records,
        'label_kind':'projected in-frame; no occlusion assertion',
        'task_request_explicit':True,'measured_tcp_input':True,
        'same_data_updates_and_threshold':True,'updates_each':a.updates,'threshold':.5,
        'near_one_port_boost':a.near_one_port_boost,
        'spatial_only':a.spatial_only,
        'mask_tcp_orientation':a.mask_tcp_orientation,
        'one_port_train':[str(x) for x in a.one_port_train],
        'one_port_development':[str(x) for x in a.one_port_development],
        'two_port_labels':str(a.two_port_labels),'final_scenes_opened':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({x['variant']:x['metrics'] for x in records},indent=2))


if __name__=='__main__':main()
