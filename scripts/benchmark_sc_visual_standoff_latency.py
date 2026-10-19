#!/usr/bin/env python3
"""Profile actor plus dual SC locator on three recorded native camera frames."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet18

from shared_bc_camera_preprocessing import jpeg_roundtrip_rgb,transport_rgb_tensor
from train_sc_native_keypoints import PretrainedKeypointRefiner
from train_temporal_transport_probe import TemporalTransport


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--actor',type=Path,required=True)
    p.add_argument('--dual',type=Path,required=True)
    p.add_argument('--imagenet',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=80)
    p.add_argument('--direct-native-rgb',action='store_true')
    a=p.parse_args();torch.set_num_threads(8)
    rows=[json.loads(s) for s in a.labels.open()]
    row=next(r for r in rows if r['trial']=='llb_0009' and r['frame']>200)
    rgb=[cv2.imread(row['images'][c])[...,::-1] for c in ('center','left','right')]
    actor_saved=torch.load(a.actor,map_location='cpu',weights_only=False)
    model=TemporalTransport(actor_saved['model']['target_mean'].numpy(),
        actor_saved['model']['target_std'].numpy(),raw_state_skip=True,
        feature_dim=1536,state_task_dim=42,route_conditioning=False).cuda().eval()
    model.load_state_dict(actor_saved['model'])
    backbone=resnet18(weights=None);backbone.fc=torch.nn.Identity()
    weights=torch.load(a.imagenet,map_location='cpu',weights_only=True)
    backbone.load_state_dict({k:v for k,v in weights.items() if not k.startswith('fc.')})
    backbone=backbone.cuda().eval()
    dual=PretrainedKeypointRefiner(None,False,point_count=3).cuda().eval()
    dual.load_state_dict(torch.load(a.dual,map_location='cpu',weights_only=False)['model'])
    values=[]
    for repeat in range(a.repeats+10):
        torch.cuda.synchronize();times=[time.perf_counter()]
        decoded=(rgb if a.direct_native_rgb else [jpeg_roundtrip_rgb(q,86) for q in rgb])
        actor_images=torch.stack([transport_rgb_tensor(q,native_already_encoded=True)
                                  for q in decoded]).cuda().float()/255
        visual=torch.stack([torch.from_numpy(np.asarray(Image.fromarray(q).resize(
            (288,256),Image.Resampling.BILINEAR)).copy()).permute(2,0,1)
            for q in decoded]).cuda().float()/255
        times.append(time.perf_counter())
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            norm=((actor_images-actor_images.new_tensor((.485,.456,.406))[None,:,None,None])/
                  actor_images.new_tensor((.229,.224,.225))[None,:,None,None])
            features=backbone(norm).reshape(1,1536)
        torch.cuda.synchronize();times.append(time.perf_counter())
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            feat=features[:,None,:].expand(1,17,1536)
            zeros=lambda n:torch.zeros((1,17,n),device='cuda')
            model(feat,zeros(36),zeros(6),zeros(1).squeeze(-1),zeros(1).squeeze(-1),
                  zeros(4),torch.ones((1,17),dtype=torch.bool,device='cuda'),zeros(42))
        torch.cuda.synchronize();times.append(time.perf_counter())
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            dual(visual)
        torch.cuda.synchronize();times.append(time.perf_counter())
        if repeat>=10:values.append(np.diff(times)*1000)
    array=np.asarray(values)
    report={'schema':'aic_sc_visual_standoff_latency_breakdown/v1',
            'repeats':len(values),'stages':{},'direct_native_rgb':a.direct_native_rgb,
            'image_inputs':'recorded JPEG decoded to simulated raw RGB; optional second q86 roundtrip'}
    for j,name in enumerate(('shared_camera_preprocessing','resnet18_backbone','temporal_head','dual_port_locator')):
        report['stages'][name]={'p50_ms':float(np.median(array[:,j])),
                                'p95_ms':float(np.quantile(array[:,j],.95))}
    total=array.sum(axis=1)
    report['total']={'p50_ms':float(np.median(total)),'p95_ms':float(np.quantile(total,.95))}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
