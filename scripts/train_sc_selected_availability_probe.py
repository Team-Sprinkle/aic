#!/usr/bin/env python3
"""Bounded frozen-RGB probe: can image and measured TCP reject absent SC ports?"""

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


class Availability(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(dimension,128),nn.GELU(),nn.Dropout(.1),
                                  nn.Linear(128,2))

    def forward(self,x):return self.layers(x)


@torch.inference_mode()
def frozen_features(locator,rows,device):
    locator.eval();rgb=[]
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
        x=locator.backbone.layer4(locator.backbone.layer3(locator.backbone.layer2(locator.backbone.layer1(x))))
        rgb.append(x.mean((-1,-2)).cpu())
    return torch.cat(rgb)


def measures(rows,logits):
    out={}
    for split,source in sorted({(r['split'],r['source']) for r in rows}):
        indices=[i for i,r in enumerate(rows) if (r['split'],r['source'])==(split,source)]
        truth=np.asarray([rows[i]['labels'] for i in indices]);pred=logits[indices]>=0
        absent=(truth==0);present=(truth==1)
        out[f'{split}/{source}']={'groups':len({rows[i]['group'] for i in indices}),
            'views':len(indices),'absent_false_positive':int((pred&absent).sum()),
            'absent_total':int(absent.sum()),'present_recalled':int((pred&present).sum()),
            'present_total':int(present.sum())}
    return out


def group_measures(rows,logits):
    out={}
    for group in sorted({r['group'] for r in rows if r['split']=='development'}):
        indices=[i for i,r in enumerate(rows) if r['group']==group]
        truth=np.asarray([rows[i]['labels'] for i in indices]);pred=logits[indices]>=0
        absent=(truth==0);present=(truth==1)
        out[group]={'views':len(indices),'present_port':int(np.argmax(truth.sum(0))),
                    'absent_false_positive':int((pred&absent).sum()),
                    'absent_total':int(absent.sum()),
                    'present_recalled':int((pred&present).sum()),
                    'present_total':int(present.sum())}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--two-port-labels',type=Path,required=True)
    p.add_argument('--one-port-train',nargs='+',type=Path,required=True)
    p.add_argument('--one-port-development',nargs='+',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--two-port-frames-per-group',type=int,default=6)
    p.add_argument('--one-port-frames-per-group',type=int,default=32)
    p.add_argument('--updates',type=int,default=500)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior run')
    a.output.mkdir(parents=True)
    cv2.setNumThreads(1);torch.set_num_threads(4);torch.manual_seed(20261002)
    rows=load_rows(a)
    locator=Locator(None).to(a.device).eval()
    locator.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model'])
    visual=frozen_features(locator,rows,a.device)
    camera=torch.tensor([[float(r['camera']==c) for c in CAMERAS] for r in rows])
    state=torch.tensor([r['state'] for r in rows],dtype=torch.float32)
    train_mask=np.asarray([r['split']=='train' for r in rows])
    state_mean=state[train_mask].mean(0);state_std=state[train_mask].std(0).clamp_min(.02)
    state=(state-state_mean)/state_std
    targets=torch.tensor([r['labels'] for r in rows],dtype=torch.float32)
    train_by_source=defaultdict(list)
    for i,r in enumerate(rows):
        if r['split']=='train':train_by_source[r['source']].append(i)
    if not {'one_port','two_port'}<=train_by_source.keys():raise ValueError('Missing source')
    records=[]
    for variant in ('image_only','image_plus_measured_tcp'):
        torch.manual_seed(20261002)
        rng=np.random.default_rng(20261002)
        x=torch.cat((visual,camera,state if variant=='image_plus_measured_tcp'
                     else torch.empty((len(rows),0))),-1).to(a.device)
        y=targets.to(a.device)
        model=Availability(x.shape[1]).to(a.device)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        history=[]
        for step in range(1,a.updates+1):
            indices=np.r_[rng.choice(train_by_source['one_port'],64,replace=True),
                          rng.choice(train_by_source['two_port'],64,replace=True)]
            index=torch.as_tensor(indices,device=a.device)
            model.train();logits=model(x[index])
            loss=F.binary_cross_entropy_with_logits(logits,y[index])
            opt.zero_grad();loss.backward();opt.step()
            if step%100==0:history.append({'step':step,'loss':float(loss.detach())})
        model.eval()
        with torch.inference_mode():logits=model(x).cpu().numpy()
        torch.save({'model':model.state_dict(),'input_dimension':x.shape[1],
                    'state_mean':state_mean,'state_std':state_std,
                    'variant':variant,'frozen_locator':str(a.checkpoint),
                    'projection_not_true_visibility':True},a.output/(variant+'.pt'))
        records.append({'variant':variant,'metrics':measures(rows,logits),
                        'development_by_group':group_measures(rows,logits),
                        'history':history,
                        'checkpoint':str(a.output/(variant+'.pt'))})
    summary={'schema':'aic_sc_selected_availability_probe/v1',
             'label_kind':'projected in frame, not true visibility',
             'image_encoder_frozen':True,'probe_outputs_two_sc_slots_only':True,
             'two_port_groups_train':len({r['group'] for r in rows if r['source']=='two_port' and r['split']=='train'}),
             'one_port_groups_train':len({r['group'] for r in rows if r['source']=='one_port' and r['split']=='train'}),
             'one_port_groups_development':len({r['group'] for r in rows if r['source']=='one_port' and r['split']=='development'}),
             'two_port_groups_validation':len({r['group'] for r in rows if r['source']=='two_port' and r['split']=='validation'}),
             'updates_each':a.updates,'one_port_train':[str(x) for x in a.one_port_train],
             'one_port_development':[str(x) for x in a.one_port_development],
             'variants':records,'final_scenes_opened':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({r['variant']:r['metrics'] for r in records},indent=2))


if __name__=='__main__':main()
