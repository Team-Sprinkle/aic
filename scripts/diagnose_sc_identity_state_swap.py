#!/usr/bin/env python3
"""Test whether SC identity changes when only measured TCP state changes."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from train_sc_spatial_identity_probe import SelectedHead
from train_two_family_port_locator import CAMERAS, Locator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit',type=Path)
    p.add_argument('--locator',type=Path,required=True)
    p.add_argument('--identity',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve result')
    torch.set_num_threads(4);cv2.setNumThreads(1)
    trace=[json.loads(s) for s in (a.audit/'runtime_trace.jsonl').open()]
    inspections=[r for r in trace if r.get('event')=='shadow_handoff_inspection']
    if len(inspections)<2:raise ValueError('Need two inspections')
    selected=inspections[:2]
    frames={}
    for r in csv.DictReader((a.audit/'frame_manifest.csv').open()):
        if r['camera'] in CAMERAS:
            frames.setdefault(int(r['sim_ns']),{})[r['camera']]=a.audit/'frames'/r['file']
    image_sets=[]
    for r in selected:
        stamp=min((s for s,v in frames.items() if len(v)==3),
                  key=lambda s:abs(s-r['sim_time']*1e9))
        images=[]
        for c in CAMERAS:
            im=cv2.imread(str(frames[stamp][c]))
            images.append(cv2.cvtColor(cv2.resize(im,(288,256),interpolation=cv2.INTER_AREA),
                                       cv2.COLOR_BGR2RGB))
        image_sets.append((stamp,np.stack(images)))
    stream=json.loads((a.audit/'scored_stream.json').read_text())
    controllers=stream['streams']['/aic_controller/controller_state']
    poses=[]
    for stamp,_ in image_sets:
        _,state=min(controllers,key=lambda row:abs(row[0]-stamp))
        pose=state['tcp_pose']
        poses.append([*[pose['position'][k] for k in 'xyz'],
                      *[pose['orientation'][k] for k in 'xyzw']])
    loc=Locator(None).to(a.device).eval()
    loc.load_state_dict(torch.load(a.locator,map_location='cpu',weights_only=False)['model'])
    saved=torch.load(a.identity,map_location='cpu',weights_only=False)
    head=SelectedHead(True,256,12).to(a.device).eval()
    head.load_state_dict(saved['model'])
    results=[]
    for image_index,(stamp,ims) in enumerate(image_sets):
        x=torch.from_numpy(ims.copy()).permute(0,3,1,2).to(a.device).float()/255
        meta=torch.zeros((3,5),device=a.device);meta[:,0]=1
        for i in range(3):meta[i,2+i]=1
        with torch.inference_mode():
            _,_,_,spatial=loc(x,meta,return_features=True,return_spatial=True)
            other=1-image_index
            states=(('matched',poses[image_index]),
                    ('orientation_only_swapped',poses[image_index][:3]+poses[other][3:]),
                    ('position_only_swapped',poses[other][:3]+poses[image_index][3:]),
                    ('entire_pose_swapped',poses[other]))
            for name,pose in states:
                condition=torch.zeros((3,12),device=a.device)
                for i in range(3):condition[i,i]=1
                condition[:,3:10]=(torch.tensor(pose,device=a.device)-
                                    saved['state_mean'].to(a.device))/saved['state_std'].to(a.device)
                if saved.get('mask_tcp_orientation',False):
                    condition[:,6:10]=0
                condition[:,10+selected[image_index]['requested_port']]=1
                probability=head(spatial.float(),condition).sigmoid().cpu().tolist()
                results.append({'image_index':image_index,'state_case':name,
                                'image_sim_ns':stamp,'probabilities':probability,
                                'all_views_pass':min(probability)>=.5})
    output={'schema':'sc_identity_same_image_state_swap/v1',
            'inspection_sim_s':[r['sim_time'] for r in selected],
            'measured_tcp_poses':poses,'results':results,
            'only_tcp_state_swapped_between_identical_RGB':True,
            'final_scenes_opened':False}
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
