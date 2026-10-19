#!/usr/bin/env python3
"""Full local camera preprocessing, shared encoder and head latency benchmark."""
import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torchvision.models import resnet18

from shared_bc_camera_preprocessing import transport_rgb_tensor
from train_diffusion_transport_probe import DiffusionTransport
from train_temporal_transport_probe import build_data,LONG_AGES,TemporalTransport


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--imagenet',type=Path,required=True)
    p.add_argument('--rollout',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    data=build_data(a.cache,a.base,'cuda',LONG_AGES)
    candidate=next(i for i,r in enumerate(data['rows']) if r['family']=='sfp' and
                   r['split']=='validation' and r['time_s']>20)
    idx=torch.tensor([candidate],device='cuda');h=data['history'][idx];mask=data['mask'][idx]
    clock=data['camera_clock'][h].float().clone();clock[...,:3]/=.05
    priors=data['prev'][h].float();motion=data['motion'][h].float()
    ages=data['age'][idx];gaps=data['gap'][idx];state_task=data['state_task_full'][h].float()
    frame_rows=list(csv.DictReader((a.rollout/'frame_manifest.csv').open()))
    raw=[]
    for camera in ('center','left','right'):
        options=[r for r in frame_rows if r['camera']==camera]
        chosen=min(options,key=lambda r:abs(int(r['sim_ns'])/1e9-26.65))
        raw.append(cv2.imread(str(a.rollout/'frames'/chosen['file']))[...,::-1].copy())
    backbone=resnet18(weights=None);backbone.fc=torch.nn.Identity()
    state=torch.load(a.imagenet,map_location='cpu',weights_only=True)
    backbone.load_state_dict({k:v for k,v in state.items() if not k.startswith('fc.')},strict=True)
    backbone=backbone.cuda().eval()
    reports={}
    for name in ('no_history','long_history','route','diffusion'):
        cp=torch.load(a.runs/name/'best.pt',map_location='cpu',weights_only=False)
        if name=='diffusion':
            model=DiffusionTransport(data['features'].shape[1],data['mean'],data['std']).cuda()
        else:
            model=TemporalTransport(data['mean'],data['std'],raw_state_skip=True,
                                    feature_dim=data['features'].shape[1],state_task_dim=42,
                                    route_conditioning=name=='route').cuda()
        model.load_state_dict(cp['model']);model.eval()
        times=[]
        with torch.inference_mode():
            for rep in range(70):
                torch.cuda.synchronize();started=time.perf_counter()
                images=torch.stack([transport_rgb_tensor(rgb,native_jpeg_quality=86,
                                                         native_already_encoded=True)
                                    for rgb in raw]).cuda().float()/255.
                images=((images-images.new_tensor((.485,.456,.406))[None,:,None,None]) /
                        images.new_tensor((.229,.224,.225))[None,:,None,None])
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    current=backbone(images).reshape(1,1536).float()
                    features=data['features'][h].float().clone()
                    features[:,-1]=current
                    if name=='no_history':
                        feat=features[:,-1:];prev=priors[:,-1:];mot=motion[:,-1:]
                        age=ages[:,-1:];gap=gaps[:,-1:];cam=clock[:,-1:]
                        valid=mask[:,-1:];st=state_task[:,-1:]
                    else:
                        feat=features;prev=priors;mot=motion;age=ages;gap=gaps
                        cam=clock;valid=mask;st=state_task
                    if name=='diffusion':
                        pieces=(feat,prev,mot,age[...,None],gap[...,None],cam,st)
                        x=model.history_input(torch.cat(pieces,-1))
                        context=model.history(x,src_key_padding_mask=~valid)[:,-1]
                        output=model.sample(context,rep+12345)
                    else:
                        output,*_=model(feat,prev,mot,age,gap,cam,valid,st)
                torch.cuda.synchronize();elapsed=(time.perf_counter()-started)*1000
                if rep>=10:times.append(elapsed)
        reports[name]=dict(n=len(times),p50_ms=float(np.quantile(times,.5)),
                           p95_ms=float(np.quantile(times,.95)),p99_ms=float(np.quantile(times,.99)),
                           min_ms=float(min(times)),max_ms=float(max(times)))
        print(name,reports[name],flush=True)
    result=dict(schema='aic_transport_five_way_local_latency/v1',device=torch.cuda.get_device_name(0),
                measured_path='three JPEG-decoded RGB views -> transport preprocessing -> ImageNet ResNet18 -> cached causal history -> head',
                excluded='ROS message transport, control thread scheduling; compare with live p95 separately',
                cache_history_reused=True,reports=reports)
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
