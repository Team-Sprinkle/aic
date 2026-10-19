#!/usr/bin/env python3
"""Frozen ResNet-18 layer3 grids for the existing temporal transport row order.

The cache contains observation features only. Scene/port geometry is never read.
SC rows are zeros because this bounded experiment evaluates SFP transport.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet18

from act_cache import load_cached_images
from shared_bc_camera_preprocessing import transport_rgb_tensor


CAMERAS=('center','left','right')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--temporal-cache',type=Path,required=True)
    p.add_argument('--sfp-labels',type=Path,required=True)
    p.add_argument('--sfp-image-cache',type=Path,required=True)
    p.add_argument('--imagenet-resnet',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=24)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous spatial cache')
    a.output.mkdir(parents=True)
    torch.set_num_threads(6)
    rows=[json.loads(line) for line in (a.temporal_cache/'rows.jsonl').open()]
    labels={}
    for line in a.sfp_labels.open():
        r=json.loads(line);key=(int(r['episode_index']),int(r['frame_index']))
        if key in labels:raise ValueError(f'Duplicate image label {key}')
        labels[key]=r
    image_manifest=json.loads((a.sfp_image_cache/'cache.json').read_text())
    images=load_cached_images(a.sfp_image_cache,image_manifest)
    sfp_indices=[i for i,r in enumerate(rows) if r['family']=='sfp']
    if any((rows[i]['episode'],int(rows[i]['frame'])) not in labels for i in sfp_indices):
        raise ValueError('Missing selected SFP image source')
    backbone=resnet18(weights=None)
    weights=torch.load(a.imagenet_resnet,map_location='cpu',weights_only=True)
    backbone.load_state_dict({k:v for k,v in weights.items() if not k.startswith('fc.')},strict=False)
    backbone=backbone.cuda().eval()
    grid=np.lib.format.open_memmap(a.output/'spatial.npy',mode='w+',dtype=np.float16,
                                    shape=(len(rows),3,20,256))
    grid[:]=0
    mean=torch.tensor((.485,.456,.406),device='cuda')[None,:,None,None]
    std=torch.tensor((.229,.224,.225),device='cuda')[None,:,None,None]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,len(sfp_indices),a.batch_size):
            chosen=sfp_indices[start:start+a.batch_size]
            batch=[]
            for i in chosen:
                r=labels[(rows[i]['episode'],int(rows[i]['frame']))]
                for camera in CAMERAS:
                    if r.get('cache_frame_index') is not None:
                        old=images[f'observation.images.{camera}_camera'][int(r['cache_frame_index'])]
                        rgb=np.asarray(Image.fromarray(old,'RGB').resize((224,224),Image.Resampling.BILINEAR))
                        tensor=torch.from_numpy(np.array(rgb,copy=True)).permute(2,0,1)
                    else:
                        with Image.open(r['images'][camera]) as f:rgb=np.asarray(f.convert('RGB'))
                        tensor=transport_rgb_tensor(rgb,native_jpeg_quality=86,native_already_encoded=True)
                    batch.append(tensor)
            x=torch.stack(batch).cuda(non_blocking=True).float()/255.
            x=(x-mean)/std
            x=backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(x))))
            x=backbone.layer3(backbone.layer2(backbone.layer1(x)))
            x=torch.nn.functional.adaptive_avg_pool2d(x,(4,5))
            grid[chosen]=x.float().cpu().numpy().reshape(len(chosen),3,256,20).transpose(0,1,3,2).astype(np.float16)
            if start%480==0:
                grid.flush()
                print(json.dumps({'cached_sfp':min(start+len(chosen),len(sfp_indices)),
                                  'total_sfp':len(sfp_indices)}),flush=True)
    grid.flush()
    manifest=dict(schema='aic_sfp_transport_spatial_grid/v1',rows=len(rows),sfp_rows=len(sfp_indices),
                  shape=list(grid.shape),grid='frozen ImageNet ResNet18 layer3 4x5 per camera',
                  temporal_cache=str(a.temporal_cache),sfp_labels=str(a.sfp_labels),
                  sfp_image_cache=str(a.sfp_image_cache),imagenet_resnet=str(a.imagenet_resnet),
                  actor_geometry_input=False,final_scenes_opened=False)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(a.output),'sfp_rows':len(sfp_indices)}))


if __name__=='__main__':main()
