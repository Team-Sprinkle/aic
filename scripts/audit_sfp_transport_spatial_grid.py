#!/usr/bin/env python3
"""Check frozen spatial cache against pooled cache on old/new SFP image sources."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet18

from act_cache import load_cached_images
from shared_bc_camera_preprocessing import transport_rgb_tensor


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--temporal-cache',type=Path,required=True)
    p.add_argument('--spatial-cache',type=Path,required=True)
    p.add_argument('--sfp-labels',type=Path,required=True)
    p.add_argument('--sfp-image-cache',type=Path,required=True)
    p.add_argument('--imagenet-resnet',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[json.loads(line) for line in (a.temporal_cache/'rows.jsonl').open()]
    labels={(int(x['episode_index']),int(x['frame_index'])):x for x in
            (json.loads(line) for line in a.sfp_labels.open())}
    source=json.loads((a.sfp_image_cache/'cache.json').read_text())
    images=load_cached_images(a.sfp_image_cache,source)
    pooled=np.load(a.temporal_cache/'features.npy',mmap_mode='r')
    spatial=np.load(a.spatial_cache/'spatial.npy',mmap_mode='r')
    assert len(rows)==len(pooled)==len(spatial)
    model=resnet18(weights=None);model.fc=torch.nn.Identity()
    weights=torch.load(a.imagenet_resnet,map_location='cpu',weights_only=True)
    model.load_state_dict({k:v for k,v in weights.items() if not k.startswith('fc.')},strict=True)
    model=model.cuda().eval()
    old=[i for i,r in enumerate(rows) if r['family']=='sfp' and labels[(r['episode'],r['frame'])].get('cache_frame_index') is not None]
    new=[i for i,r in enumerate(rows) if r['family']=='sfp' and labels[(r['episode'],r['frame'])].get('cache_frame_index') is None]
    val=[i for i,r in enumerate(rows) if r['family']=='sfp' and r['split']=='validation']
    selected=[old[k] for k in (0,len(old)//2,len(old)-1)]+[new[k] for k in (0,len(new)//2,len(new)-1)]+[val[0],val[-1]]
    mean=torch.tensor((.485,.456,.406),device='cuda')[None,:,None,None]
    std=torch.tensor((.229,.224,.225),device='cuda')[None,:,None,None]
    report=[]
    for i in selected:
        row=rows[i];label=labels[(row['episode'],row['frame'])]
        views=[]
        for camera in ('center','left','right'):
            if label.get('cache_frame_index') is not None:
                old_image=images[f'observation.images.{camera}_camera'][int(label['cache_frame_index'])]
                rgb=np.asarray(Image.fromarray(old_image,'RGB').resize((224,224),Image.Resampling.BILINEAR))
                tensor=torch.from_numpy(np.array(rgb,copy=True)).permute(2,0,1)
            else:
                with Image.open(label['images'][camera]) as f:rgb=np.asarray(f.convert('RGB'))
                tensor=transport_rgb_tensor(rgb,native_jpeg_quality=86,native_already_encoded=True)
            views.append(tensor)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            x=torch.stack(views).cuda().float()/255.;x=(x-mean)/std
            x=model.maxpool(model.relu(model.bn1(model.conv1(x))))
            x=model.layer3(model.layer2(model.layer1(x)))
            grid=torch.nn.functional.adaptive_avg_pool2d(x,(4,5))
            global_vector=model.avgpool(model.layer4(x)).reshape(1536).float().cpu().numpy()
            grid=grid.float().cpu().numpy().reshape(3,256,20).transpose(0,2,1)
        report.append(dict(row=i,source='cached' if label.get('cache_frame_index') is not None else 'recorded',
                           split=row['split'],pooled_max_abs=float(np.max(abs(global_vector-pooled[i]))),
                           spatial_max_abs=float(np.max(abs(grid-spatial[i]))),
                           spatial_cell_std=float(np.mean(grid.std(axis=1)))))
    result=dict(schema='aic_sfp_spatial_cache_audit/v1',sample_count=len(report),rows=report,
                pooled_max_abs=max(x['pooled_max_abs'] for x in report),
                spatial_max_abs=max(x['spatial_max_abs'] for x in report),
                spatial_cell_std_min=min(x['spatial_cell_std'] for x in report),
                final_scenes_opened=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
