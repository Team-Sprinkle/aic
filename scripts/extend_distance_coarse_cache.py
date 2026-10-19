#!/usr/bin/env python3
"""Append new train RGB to a frozen SFP feature cache without re-encoding old rows."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from train_requested_history_crop import CAMERAS, rows_from
from train_target_conditioned_port_locator import ConditionalLocator


def manifest(rows):
    return [{'episode':r['episode'],'time_s':r['time_s'],
             'scene_group':r['scene_group'],'images':r['images']} for r in rows]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-labels',type=Path,required=True)
    ap.add_argument('--added-labels',type=Path,required=True)
    ap.add_argument('--stress-labels',type=Path,required=True)
    ap.add_argument('--base-cache',type=Path,required=True)
    ap.add_argument('--output-labels',type=Path,required=True)
    ap.add_argument('--output-cache',type=Path,required=True)
    a=ap.parse_args()
    if a.output_labels.exists() or a.output_cache.exists():ap.error('Preserve prior outputs')
    base=rows_from(a.base_labels);added=rows_from(a.added_labels);stress=rows_from(a.stress_labels)
    if not added or any(r['split']!='train' for r in added):raise ValueError('Added rows must be training only')
    groups=[{r['scene_group'] for r in part} for part in (base,added,stress)]
    if any(groups[i]&groups[j] for i,j in ((0,1),(1,2))):raise ValueError('Scene overlap')
    old=np.load(a.base_cache/'features.npy',mmap_mode='r')
    if json.loads((a.base_cache/'features_manifest.json').read_text())!=manifest(base+stress):
        raise ValueError('Original cache mismatch')
    a.output_labels.mkdir(parents=True);a.output_cache.mkdir(parents=True)
    all_rows=base+added
    with (a.output_labels/'rows.jsonl').open('w') as f:
        for r in all_rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
    new=np.lib.format.open_memmap(a.output_cache/'features.npy',mode='w+',dtype=np.float16,
                                   shape=(len(base)+len(added)+len(stress),3,256,32,36))
    new[:len(base)]=old[:len(base)]
    new[len(base)+len(added):]=old[len(base):]
    checkpoint=json.loads((a.base_cache/'manifest.json').read_text())['init']
    model=ConditionalLocator(None).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'])
    with torch.inference_mode():
        for start in range(0,len(added),4):
            chunk=added[start:start+4]
            rgb=[]
            for r in chunk:
                for cam in CAMERAS:
                    image=cv2.imread(r['images'][cam],cv2.IMREAD_COLOR)
                    if image is None:raise FileNotFoundError(r['images'][cam])
                    image=cv2.cvtColor(cv2.resize(image,(576,512),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
                    rgb.append(torch.from_numpy(image.copy()).permute(2,0,1))
            x=torch.stack(rgb).cuda().float()/255.
            x=(x-model.mean)/model.std
            backbone=model.backbone
            x=backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(x))))
            x=backbone.layer3(backbone.layer2(backbone.layer1(x)))
            new[len(base)+start:len(base)+start+len(chunk)]=x.cpu().numpy().astype(np.float16).reshape(-1,3,256,32,36)
    new.flush()
    (a.output_cache/'features_manifest.json').write_text(json.dumps(manifest(base+added+stress),indent=2)+'\n')
    (a.output_cache/'manifest.json').write_text(json.dumps({'init':checkpoint,
        'source_cache':str(a.base_cache),'base_rows':len(base),'added_rows':len(added),
        'stress_rows':len(stress),'final_scenes_opened':False},indent=2)+'\n')
    (a.output_labels/'manifest.json').write_text(json.dumps({'base_labels':str(a.base_labels),
        'added_labels':str(a.added_labels),'base_rows':len(base),'added_rows':len(added),
        'train_groups':len({r['scene_group'] for r in all_rows if r['split']=='train'}),
        'development_groups':len({r['scene_group'] for r in all_rows if r['split']=='validation'}),
        'final_scenes_opened':False},indent=2)+'\n')
    print(json.dumps({'base':len(base),'added':len(added),'stress':len(stress)}))

if __name__=='__main__':main()
