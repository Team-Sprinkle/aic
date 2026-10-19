#!/usr/bin/env python3
"""Evaluate frozen original-RGB crop identity on held-out one-port resets."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from train_sc_observation_crop_identity import CropIdentity, locator_crops
from train_sc_spatial_identity_probe import CAMERAS, SelectedHead, frozen_maps
from train_two_family_port_locator import Locator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--projection',nargs='+',type=Path,required=True)
    p.add_argument('--locator',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--crop-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous result')
    cv2.setNumThreads(1);torch.set_num_threads(4)
    rows=[]
    for path in a.projection:
        rows.extend(json.loads(s) for s in path.open())
    for r in rows:
        r.update(group=r['scene_group_id'],state=r['measured_tcp_base_pose'],
                 labels=[float(r['projected_inframe']) if i==r['target_port'] else 0.
                         for i in (0,1)])
    locator=Locator(None).to(a.device).eval()
    locator.load_state_dict(torch.load(a.locator,map_location='cpu',weights_only=False)['model'])
    visual=frozen_maps(locator,rows,a.device).float().to(a.device)
    crops=locator_crops(locator,rows,a.device)
    crop_tensor=torch.from_numpy(crops).permute(0,1,4,2,3).to(a.device).float()/255
    bas=torch.load(a.baseline,map_location='cpu',weights_only=False)
    new=torch.load(a.crop_checkpoint,map_location='cpu',weights_only=False)
    baseline=SelectedHead(True,256,12).to(a.device).eval()
    baseline.load_state_dict(bas['model'])
    crop_model=CropIdentity(256,12).to(a.device).eval()
    crop_model.load_state_dict(new['model'])
    state=torch.tensor([r['state'] for r in rows],device=a.device,dtype=torch.float32)
    camera=torch.tensor([[float(r['camera']==c) for c in CAMERAS] for r in rows],
                        device=a.device,dtype=torch.float32)
    requests=torch.eye(2,device=a.device).repeat(len(rows),1)
    def conditions(saved):
        normalized=(state-saved['state_mean'].to(a.device))/saved['state_std'].to(a.device)
        return torch.cat((camera.repeat_interleave(2,0),
                          normalized.repeat_interleave(2,0),requests),-1)
    with torch.inference_mode():
        old=[];new_values=[]
        old_cond=conditions(bas);new_cond=conditions(new)
        flat_crops=crop_tensor.flatten(0,1)
        for i in range(0,len(rows)*2,64):
            ids=torch.arange(i,min(i+64,len(rows)*2),device=a.device)
            old.append(baseline(visual[ids//2],old_cond[ids]).sigmoid().cpu().numpy())
            new_values.append(crop_model(visual[ids//2],flat_crops[ids],
                                         new_cond[ids]).sigmoid().cpu().numpy())
    old=np.concatenate(old).reshape(-1,2);new_values=np.concatenate(new_values).reshape(-1,2)
    triplets=defaultdict(dict)
    for i,r in enumerate(rows):
        key=(r['group'],r['sim_ns'])
        triplets[key][r['camera']]=(i,r)
    results=[]
    for (group,stamp),views in sorted(triplets.items()):
        if set(views)!=set(CAMERAS):continue
        indices=[views[c][0] for c in CAMERAS]
        r=views['center'][1];port=r['target_port']
        distance_mm=float(np.linalg.norm(r['relative_opening_tcp_m'])*1000)
        real_projected_all=all(rows[i]['projected_inframe'] for i in indices)
        item={'group':group,'sim_ns':stamp,'target_port':port,
              'opening_tcp_distance_mm':distance_mm,
              'real_projected_all_views':real_projected_all}
        for label,pred in (('baseline',old),('crop',new_values)):
            item[label]={'real_probabilities':[float(pred[i,port]) for i in indices],
                         'absent_probabilities':[float(pred[i,1-port]) for i in indices],
                         'real_all_three':bool(np.all(pred[indices,port]>=.5)),
                         'absent_all_three':bool(np.all(pred[indices,1-port]>=.5))}
        results.append(item)
    counts={}
    for label in ('baseline','crop'):
        counts[label]={}
        for scope,subset in (
            ('all',results),
            ('near_220mm',[r for r in results if r['opening_tcp_distance_mm']<=220]),
            ('near_160mm',[r for r in results if r['opening_tcp_distance_mm']<=160])):
            projected=[r for r in subset if r['real_projected_all_views']]
            counts[label][scope]={
                'triplets':len(subset),'projected_all_triplets':len(projected),
                'real_projected_all_recalled':sum(r[label]['real_all_three'] for r in projected),
                'absent_all_three_false_accept':sum(r[label]['absent_all_three'] for r in subset),
                'absent_triplets':len(subset)}
    result={'schema':'sc_crop_identity_independent_development/v1','counts':counts,
            'rows':results,'projection_is_not_certified_visibility':True,
            'threshold_fixed_before_new_development':.5,'final_scenes_opened':False,
            'sources':[str(p) for p in a.projection],
            'baseline':str(a.baseline),'crop_checkpoint':str(a.crop_checkpoint)}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
