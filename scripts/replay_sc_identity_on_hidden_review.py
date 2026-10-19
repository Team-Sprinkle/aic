#!/usr/bin/env python3
"""Replay frozen selected-port identity on the prior SC hidden-target review."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch

from train_sc_spatial_identity_probe import CAMERAS, SelectedHead, frozen_maps
from train_two_family_port_locator import Locator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hidden-review',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--state-projections',nargs='+',type=Path,required=True)
    p.add_argument('--identity',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args()
    old=json.loads(a.hidden_review.read_text())
    audit=json.loads(a.audit.read_text())['records']
    by_key=defaultdict(dict)
    for r in audit:
        sim_ns=int(Path(r['image']).stem.rsplit('_',2)[1])
        by_key[(r['episode'],sim_ns,int(r['target_port']))][r['camera']]=r
    states={}
    for path in a.state_projections:
        for line in path.open():
            r=json.loads(line);states[r['image']]=r
    rows=[];links=[]
    for review in old['rows']:
        key=(review['episode'],review['sim_ns'],review['target_port'])
        group=by_key[key]
        for camera in CAMERAS:
            if camera not in group:continue
            image=group[camera]['image']
            if image not in states:raise ValueError(f'Missing measured TCP: {image}')
            rows.append({'image':image,'state':states[image]['measured_tcp_base_pose'],
                         'camera':camera})
            links.append((review['review_index'],camera,review['target_port']))
    saved=torch.load(a.identity,map_location='cpu',weights_only=False)
    locator=Locator(None).to(a.device).eval()
    locator.load_state_dict(torch.load(saved['frozen_locator'],map_location='cpu',weights_only=False)['model'])
    visual=frozen_maps(locator,rows,a.device).to(a.device).float()
    state=torch.tensor([r['state'] for r in rows],dtype=torch.float32,device=a.device)
    state=(state-saved['state_mean'].to(a.device))/saved['state_std'].to(a.device)
    camera=torch.tensor([[float(r['camera']==c) for c in CAMERAS] for r in rows],
                        device=a.device,dtype=torch.float32)
    request=torch.tensor([[float(port==i) for i in (0,1)] for _,_,port in links],
                         device=a.device)
    condition=torch.cat((camera,state,request),1)
    model=SelectedHead(True,visual.shape[1],condition.shape[1]).to(a.device).eval()
    model.load_state_dict(saved['model'])
    with torch.inference_mode():prob=model(visual,condition).sigmoid().cpu().tolist()
    scores=defaultdict(dict)
    for (idx,camera,_),score in zip(links,prob):scores[idx][camera]=score
    results=[]
    for review in old['rows']:
        s=scores[review['review_index']]
        identity_pass=len(s)==3 and all(x>=.5 for x in s.values())
        old_pass=bool(review['exploratory_all_three_under_2px'])
        results.append({'review_index':review['review_index'],'episode':review['episode'],
            'sim_ns':review['sim_ns'],'target_port':review['target_port'],
            'review_label':review['review_label'],'identity_scores':s,
            'identity_all_three_accept':identity_pass,
            'old_keypoint_reprojection_accept':old_pass,
            'combined_accept':identity_pass and old_pass})
    summary={'schema':'sc_identity_hidden_review_replay/v1',
        'prior_review_provisional_single_reviewer':True,
        'state_from_scored_bag_observation_only':True,
        'geometry_used_as_label_only':True,
        'rows':results,'counts':{'triplets':len(results),
            'identity_all_three_accept':sum(x['identity_all_three_accept'] for x in results),
            'old_keypoint_reprojection_accept':sum(x['old_keypoint_reprojection_accept'] for x in results),
            'combined_accept':sum(x['combined_accept'] for x in results)},
        'final_scenes_opened':False}
    a.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['counts'],indent=2))


if __name__=='__main__':main()
