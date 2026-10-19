#!/usr/bin/env python3
"""Audit projected-in-frame logits against independent depth visibility labels.

These labels are geometric visible fraction, not human-certified identifiable
visibility. This script does not train or tune a threshold.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from train_two_family_port_locator import Locator,CAMERAS,LOOKUP


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',nargs='+',type=Path,required=True)
    p.add_argument('--imagenet',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(8)
    rows=[json.loads(l) for path in a.labels for l in path.open()]
    model=Locator(a.imagenet).cuda().eval()
    model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model'])
    records=[];groups=defaultdict(list)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for lo in range(0,len(rows),48):
            batch=rows[lo:lo+48];images=[];meta=[]
            for r in batch:
                image=cv2.imread(r['image'],cv2.IMREAD_COLOR)
                if image is None:raise FileNotFoundError(r['image'])
                image=cv2.cvtColor(cv2.resize(image,(288,256),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
                images.append(torch.from_numpy(image.copy()).permute(2,0,1))
                m=np.zeros(5,np.float32);m[0]=1;m[2+CAMERAS.index(r['camera'])]=1
                meta.append(m)
            image=torch.stack(images).cuda().float()/255.
            m=torch.as_tensor(np.stack(meta),device='cuda')
            xy,flag=model(image,m);prob=flag.float().sigmoid().cpu().numpy()
            xy=xy.float().cpu().numpy()
            for j,r in enumerate(batch):
                candidate=LOOKUP['sc_port_'+str(r['target_port'])]
                p=float(prob[j,candidate]);visible=r['geometric_visible_fraction']>=.7
                inframe=r['in_frame_fraction']>=.7
                kind='visible' if visible else 'occluded_inframe' if inframe else 'offscreen'
                item={'episode':r['episode_id'],'scene_group':r['scene_group_id'],'camera':r['camera'],
                      'target_port':r['target_port'],'geometric_visible_fraction':r['geometric_visible_fraction'],
                      'inframe_fraction':r['in_frame_fraction'],'predicted_projected_inframe_probability':p,
                      'predicted_selected_xy_native':(xy[j,candidate]*[1151,1023]).tolist(),
                      'would_accept_as_visible_at_half':p>=.5,'kind':kind,'image':r['image']}
                records.append(item);groups[kind].append(item)
    summary={k:{'n':len(v),'accept_rate_at_half':float(np.mean([r['would_accept_as_visible_at_half'] for r in v])),
                'median_predicted_inframe_probability':float(np.median([r['predicted_projected_inframe_probability'] for r in v]))}
             for k,v in groups.items()}
    out={'schema':'two_family_occlusion_rgbd_audit/v1','checkpoint':str(a.checkpoint),
         'labels_kind':'geometric visible fraction from synchronized label-only RGBD',
         'identifiable_visibility_certified':False,'episodes':len({r['episode'] for r in records}),
         'summary':summary,'records':records,'reserved_final_opened':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'episodes':out['episodes'],'summary':summary}))


if __name__=='__main__':main()
