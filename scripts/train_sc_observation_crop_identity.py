#!/usr/bin/env python3
"""Bounded selected-SC identity test with locator-chosen original-RGB crops.

Projected-in-frame targets are training labels, not certified visibility. No
projection or true point is used to choose a crop, at fit or evaluation time.
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
from train_sc_spatial_identity_probe import CAMERAS, frozen_maps, measures
from train_two_family_port_locator import Locator


def crop(image, xy, size=160):
    """Original 1152x1024 pixels; pad at image boundary, never resize global RGB."""
    x,y=np.rint(xy).astype(int)
    half=size//2
    padded=cv2.copyMakeBorder(image,half,half,half,half,cv2.BORDER_CONSTANT,value=(0,0,0))
    return padded[y:y+size,x:x+size].copy()


@torch.inference_mode()
def locator_crops(locator,rows,device):
    result=[]
    for start in range(0,len(rows),24):
        batch=rows[start:start+24];original=[];small=[]
        for row in batch:
            image=cv2.imread(row['image'])
            if image is None:raise FileNotFoundError(row['image'])
            if image.shape[:2]!=(1024,1152):raise ValueError((row['image'],image.shape))
            original.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            small.append(cv2.resize(original[-1],(288,256),interpolation=cv2.INTER_AREA))
        x=torch.from_numpy(np.stack(small)).permute(0,3,1,2).to(device).float()/255
        meta=np.zeros((len(batch),5),np.float32);meta[:,0]=1
        for j,row in enumerate(batch):meta[j,2+CAMERAS.index(row['camera'])]=1
        xy,_=locator(x,torch.from_numpy(meta).to(device))
        xy=xy[:,:2].float().cpu().numpy()*np.array([1151,1023])
        for j,image in enumerate(original):
            result.append(np.stack([crop(image,xy[j,p]) for p in (0,1)]))
    return np.stack(result)


class CropIdentity(nn.Module):
    def __init__(self,channels,condition_dim):
        super().__init__()
        self.crop=nn.Sequential(
            nn.Conv2d(3,24,5,stride=2,padding=2),nn.GELU(),
            nn.Conv2d(24,48,3,stride=2,padding=1),nn.GELU(),
            nn.Conv2d(48,64,3,stride=2,padding=1),nn.GELU(),
            nn.Conv2d(64,96,3,stride=2,padding=1),nn.GELU(),
            nn.AdaptiveAvgPool2d((1,1)),nn.Flatten())
        self.global_head=nn.Sequential(nn.Conv2d(channels,64,3,padding=1),nn.GELU(),
                                       nn.AdaptiveAvgPool2d((1,1)),nn.Flatten())
        self.final=nn.Sequential(nn.Linear(96+64+condition_dim,128),nn.GELU(),
                                 nn.Dropout(.1),nn.Linear(128,1))

    def forward(self,visual,crops,condition):
        x=torch.cat((self.global_head(visual),self.crop(crops),condition),-1)
        return self.final(x).squeeze(-1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--two-port-labels',type=Path,required=True)
    p.add_argument('--one-port-train',nargs='+',type=Path,required=True)
    p.add_argument('--one-port-development',nargs='+',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--two-port-frames-per-group',type=int,default=6)
    p.add_argument('--one-port-frames-per-group',type=int,default=30)
    p.add_argument('--updates',type=int,default=900)
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args()
    if a.output.exists():p.error('Output exists; preserve prior run')
    a.output.mkdir(parents=True)
    cv2.setNumThreads(1);torch.set_num_threads(4);torch.manual_seed(20261003)
    rows=load_rows(a)
    locator=Locator(None).to(a.device).eval()
    locator.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model'])
    visual=frozen_maps(locator,rows,a.device).float().to(a.device)
    crops=locator_crops(locator,rows,a.device)
    crop_tensor=torch.from_numpy(crops).permute(0,1,4,2,3).to(a.device).float()/255
    state=torch.tensor([r['state'] for r in rows],dtype=torch.float32,device=a.device)
    train_mask=torch.tensor([r['split']=='train' for r in rows],device=a.device)
    mean=state[train_mask].mean(0);std=state[train_mask].std(0).clamp_min(.02)
    state=(state-mean)/std
    cameras=torch.tensor([[float(r['camera']==c) for c in CAMERAS] for r in rows],
                         device=a.device,dtype=torch.float32)
    request=torch.eye(2,device=a.device).repeat(len(rows),1)
    condition=torch.cat((cameras.repeat_interleave(2,0),state.repeat_interleave(2,0),request),-1)
    truth=torch.tensor([r['labels'] for r in rows],device=a.device,dtype=torch.float32).flatten()
    flat_crops=crop_tensor.flatten(0,1)
    by_source=defaultdict(list)
    for i,r in enumerate(rows):
        if r['split']=='train':by_source[r['source']].append(i)
    assert {'one_port','two_port'}<=by_source.keys()
    model=CropIdentity(visual.shape[1],condition.shape[1]).to(a.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
    rng=np.random.default_rng(20261003);history=[]
    for step in range(1,a.updates+1):
        selected=np.r_[rng.choice(by_source['one_port'],24,replace=True),
                       rng.choice(by_source['two_port'],24,replace=True)]
        ids=torch.as_tensor((selected[:,None]*2+np.arange(2)).flatten(),device=a.device)
        model.train()
        logits=model(visual[ids//2],flat_crops[ids],condition[ids])
        loss=F.binary_cross_entropy_with_logits(logits,truth[ids])
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step%100==0:history.append({'step':step,'train_loss':float(loss.detach())})
    model.eval()
    with torch.inference_mode():
        logits=torch.cat([model(visual[torch.arange(i,min(i+64,len(truth)),device=a.device)//2],
                                flat_crops[i:i+64],condition[i:i+64]).sigmoid()
                          for i in range(0,len(truth),64)]).reshape(-1,2).cpu().numpy()
    metrics=measures(rows,logits,.5)
    torch.save({'model':model.state_dict(),'state_mean':mean.cpu(),'state_std':std.cpu(),
                'frozen_locator':str(a.checkpoint),'crop_size_original_px':160,
                'projection_not_true_visibility':True},a.output/'crop_identity.pt')
    summary={'schema':'sc_observation_crop_identity/v1','rows':len(rows),
             'train_groups':len({r['group'] for r in rows if r['split']=='train'}),
             'development_groups':len({r['group'] for r in rows if r['split']=='development'}),
             'updates':a.updates,'threshold':.5,'metrics':metrics,'history':history,
             'one_port_train':[str(x) for x in a.one_port_train],
             'one_port_development':[str(x) for x in a.one_port_development],
             'two_port_labels':str(a.two_port_labels),'locator_checkpoint':str(a.checkpoint),
             'original_rgb_crop_selected_by_observation_only':True,
             'labels_are_projected_inframe_not_certified_visibility':True,
             'final_scenes_opened':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
