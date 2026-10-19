#!/usr/bin/env python3
"""Re-encode recorded autonomous SC observations for pose adaptation.

The target pose is taken from the already recorded teacher transport plan and
is attached offline. Runtime actor observations and images remain untouched.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torchvision.models import resnet18

from shared_bc_camera_preprocessing import transport_rgb_tensor


TRAIN_CASES=('llb_0005','llb_0090','llb_0112','llb_0117')
VALIDATION_CASES=('llb_0074','llb_0109')
CAMERAS=('center','left','right')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rollouts',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--imagenet',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-cases',nargs='+',default=list(TRAIN_CASES))
    p.add_argument('--validation-cases',nargs='+',default=list(VALIDATION_CASES))
    a=p.parse_args()
    if set(a.train_cases)&set(a.validation_cases):p.error('Scene cases overlap')
    if a.output.exists():p.error('Preserve existing output')
    a.output.mkdir(parents=True)
    torch.set_num_threads(8)
    model=resnet18(weights=None);model.fc=torch.nn.Identity()
    weights=torch.load(a.imagenet,map_location='cpu',weights_only=True)
    model.load_state_dict({k:v for k,v in weights.items() if not k.startswith('fc.')},strict=True)
    model=model.cuda().eval()
    source={e['case']:e for e in json.loads(a.dataset.read_text())['episodes']}
    rows=[];images=[];hashes={};scenes=set()
    for case in (*a.train_cases,*a.validation_cases):
        e=source[case]
        if e['split']!='validation':raise ValueError(f'Original actor split changed: {case}')
        if e['scene_reset_group'] in scenes:raise ValueError('Scene/reset leakage')
        scenes.add(e['scene_reset_group'])
        audit=a.rollouts/case
        trace=audit/'runtime_trace.jsonl'
        plan=Path(e['insert_plan'])/'moveit_joint_trajectory.json'
        goal=json.loads(plan.read_text())['waypoints'][-1]['tcp_pose']
        goal_pos=np.asarray(goal['position'],dtype=float)
        goal_rot=Rotation.from_quat(goal['orientation_xyzw'])
        files={}
        for item in csv.DictReader((audit/'frame_manifest.csv').open()):
            if item['camera'] in CAMERAS:
                files[(item['camera'],int(item['sim_ns']))]=audit/'frames'/item['file']
        proposals=[json.loads(line) for line in trace.open() if '"event": "proposal"' in line]
        accepted=0
        for r in proposals:
            paths=[files.get((camera,int(r['image_sim_ns'][camera]))) for camera in CAMERAS]
            if any(path is None for path in paths):
                continue
            decoded=[]
            for path in paths:
                bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
                if bgr is None:raise ValueError(path)
                decoded.append(transport_rgb_tensor(bgr[...,::-1],native_jpeg_quality=86,
                                                    native_already_encoded=True))
            images.append(torch.stack(decoded))
            state=np.asarray(r['state'],dtype=float)
            tcp=Rotation.from_quat(state[3:7])
            relative=np.r_[tcp.inv().apply(goal_pos-state[:3]),
                           (tcp.inv()*goal_rot).as_rotvec()]
            rows.append({'case':case,'split':'adapt_train' if case in a.train_cases else 'adapt_validation',
                'scene_reset_group':e['scene_reset_group'],'card_count':e['card_count'],
                'selected_port':e['selected_port'],'sim_time':r['sim_time'],
                'state':r['state'],'task':r['task_vector'],'goal_relative':relative.tolist(),
                'goal_base_position':goal_pos.tolist(),'image_paths':[str(path) for path in paths],
                'runtime_geometry_input':False,'teacher_geometry_label_only':True})
            accepted+=1
        if accepted<100:raise ValueError(f'Insufficient synchronized images: {case}: {accepted}')
        hashes[case]={'trace_sha256':sha(trace),'frame_manifest_sha256':sha(audit/'frame_manifest.csv'),
                      'teacher_plan_sha256':sha(plan),'accepted_decisions':accepted,
                      'original_bc_validation_scene':True}
        print(case,accepted,flush=True)
    features=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,len(images),32):
            batch=torch.stack(images[start:start+32]).cuda().float()
            n=len(batch);x=batch.reshape(n*3,3,224,224)/255.
            x=((x-x.new_tensor((.485,.456,.406))[None,:,None,None])/
               x.new_tensor((.229,.224,.225))[None,:,None,None])
            features.append(model(x).reshape(n,1536).float().cpu().numpy().astype(np.float16))
    array=np.concatenate(features)
    if array.shape!=(len(rows),1536):raise ValueError(array.shape)
    np.save(a.output/'features.npy',array)
    with (a.output/'rows.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
    manifest={'schema':'aic_sc_onpolicy_goal_features/v1',
        'train_cases':list(a.train_cases),'validation_cases':list(a.validation_cases),
        'rows':len(rows),'feature_shape':list(array.shape),'image_preprocessing':
        'captured native RGB JPEG q86; decoded once; native_already_encoded=True; resize to 288x256, JPEG q95, PIL 224x224 bilinear; ImageNet normalization',
        'source_hashes':hashes,'imagenet_sha256':sha(a.imagenet),
        'labels':'offline teacher-plan final TCP pose relative to measured TCP',
        'final_scenes_opened':False,'runtime_geometry_input':False}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'train_cases':len(a.train_cases),
                      'validation_cases':len(a.validation_cases)}),flush=True)


if __name__=='__main__':main()
