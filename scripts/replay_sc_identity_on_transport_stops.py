#!/usr/bin/env python3
"""Score a frozen SC requested-port identity head on saved transport proposals."""

import argparse
import csv
import json
from pathlib import Path

import torch

from train_sc_spatial_identity_probe import CAMERAS, SelectedHead, frozen_maps
from train_two_family_port_locator import Locator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--center-s',type=float,required=True)
    p.add_argument('--window-s',type=float,default=2.)
    p.add_argument('--identity',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous replay')
    images={}
    for r in csv.DictReader((a.audit/'frame_manifest.csv').open()):
        if r['camera'] in CAMERAS:
            images[(r['camera'],int(r['sim_ns']))]=str((a.audit/'frames'/r['file']).resolve())
    proposals=[json.loads(x) for x in (a.audit/'runtime_trace.jsonl').open()]
    proposals=[r for r in proposals if r.get('event')=='proposal' and
               abs(float(r['sim_time'])-a.center_s)<=a.window_s]
    if not proposals:raise ValueError('No saved proposals in window')
    rows=[];keys=[];admitted=[];skipped=[]
    for r in proposals:
        port=int(r['task_vector'][3]>r['task_vector'][2])
        candidate=[]
        for camera in CAMERAS:
            key=(camera,int(r['image_sim_ns'][camera]))
            image=images.get(key)
            if image is None:break
            candidate.append({'image':image,'camera':camera,'state':r['state'][:7]})
        if len(candidate)!=3:
            skipped.append({'sim_time':r['sim_time'],'reason':'missing exact RGB frame'})
            continue
        rows.extend(candidate)
        keys.extend([(float(r['sim_time']),port,camera) for camera in CAMERAS])
        admitted.append(r)
    saved=torch.load(a.identity,map_location='cpu',weights_only=False)
    locator=Locator(None).to(a.device).eval()
    locator.load_state_dict(torch.load(saved['frozen_locator'],map_location='cpu',weights_only=False)['model'])
    visual=frozen_maps(locator,rows,a.device).to(a.device).float()
    state=torch.tensor([r['state'] for r in rows],dtype=torch.float32,device=a.device)
    state=(state-saved['state_mean'].to(a.device))/saved['state_std'].to(a.device)
    camera=torch.tensor([[float(r['camera']==c) for c in CAMERAS] for r in rows],
                        dtype=torch.float32,device=a.device)
    request=torch.tensor([[float(port==i) for i in (0,1)] for _,port,_ in keys],
                         dtype=torch.float32,device=a.device)
    condition=torch.cat((camera,state,request),1)
    model=SelectedHead(True,visual.shape[1],condition.shape[1]).to(a.device).eval()
    model.load_state_dict(saved['model'])
    with torch.inference_mode():scores=model(visual,condition).sigmoid().cpu().tolist()
    results=[]
    for i,proposal in enumerate(admitted):
        values={camera:float(scores[3*i+j]) for j,camera in enumerate(CAMERAS)}
        results.append({'sim_time':proposal['sim_time'],
            'requested_port':int(proposal['task_vector'][3]>proposal['task_vector'][2]),
            'scores':values,'all_three_accept':all(x>=.5 for x in values.values()),
            'image_sim_ns':proposal['image_sim_ns'],
            'images':[rows[3*i+j]['image'] for j in range(3)]})
    result={'schema':'sc_identity_saved_transport_stop_replay/v1',
        'audit':str(a.audit),'center_s':a.center_s,'window_s':a.window_s,
        'frozen_identity_checkpoint':str(a.identity),
        'requested_port_from_public_task_vector':True,
        'state_from_saved_runtime_observation':True,
        'geometry_input':False,'skipped':skipped,'rows':results,
        'accepted':sum(x['all_three_accept'] for x in results),
        'total':len(results),'final_scenes_opened':False}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'accepted':result['accepted'],'total':result['total'],
                      'skipped':len(skipped),'last':results[-1] if results else None},indent=2))


if __name__=='__main__':main()
