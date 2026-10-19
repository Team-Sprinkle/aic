#!/usr/bin/env python3
"""Compare local actor first commands on one frozen, RGB-only live trace."""
import argparse
import bisect
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from train_shared_port_tcp_bc import SharedPortTCPActor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--checkpoint',action='append',required=True,
                   help='Name=checkpoint.pt, repeat for each model')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-proposals',type=int,default=30)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous replay')
    cv2.setNumThreads(1);torch.set_num_threads(4)
    by_camera={c:[] for c in ('center','left','right')}
    for row in csv.DictReader((a.audit/'frame_manifest.csv').open()):
        if row['camera'] in by_camera:
            by_camera[row['camera']].append((int(row['sim_ns']),a.audit/'frames'/row['file']))
    for rows in by_camera.values():rows.sort()
    trace=[json.loads(line) for line in (a.audit/'runtime_trace.jsonl').open()]
    proposals=[r for r in trace if r['event']=='proposal'][:a.max_proposals]
    perceptions=[r for r in trace if r['event']=='local_perception'][:a.max_proposals]
    if len(perceptions)!=len(proposals):raise ValueError('Perception/proposal count mismatch')
    models={}
    for item in a.checkpoint:
        name,path=item.split('=',1)
        c=torch.load(path,map_location='cpu',weights_only=False)
        m=c['model'];stats={k:m[k].numpy() for k in
            ('state_mean','state_std','pose_mean','pose_std','target_mean','target_std')}
        model=SharedPortTCPActor(stats,pretrained=False,condition_on_pose=True,
            separate_family_orientation_head=bool(c.get('separate_family_orientation_head',False)),
            separate_family_action_head=bool(c.get('separate_family_action_head',False)),
            use_local_history=bool(c.get('use_local_history',False))).cuda().eval()
        model.load_state_dict(m)
        if model.use_local_history:raise ValueError('History actor unsupported by this replay')
        models[name]=model
    records=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for row,perception in zip(proposals,perceptions):
            t=round(row['sim_time'],3)
            images=[];image_age=[]
            for camera in by_camera:
                frames=by_camera[camera];times=[x[0] for x in frames]
                stamp=row['image_sim_ns'][camera]
                j=bisect.bisect_left(times,stamp)
                j=min((max(0,j-1),min(len(times)-1,j)),key=lambda k:abs(times[k]-stamp))
                image_age.append(abs(times[j]-stamp)/1e6)
                if image_age[-1]>100:raise ValueError('Saved RGB frame too far from actor image')
                image=cv2.imread(str(frames[j][1]),cv2.IMREAD_COLOR)
                small=cv2.resize(image,(288,256),interpolation=cv2.INTER_AREA)
                _,encoded=cv2.imencode('.jpg',small,[cv2.IMWRITE_JPEG_QUALITY,95])
                rgb=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
                rgb=np.asarray(Image.fromarray(rgb).resize((224,224),Image.Resampling.BILINEAR)).copy()
                images.append(torch.from_numpy(rgb).permute(2,0,1))
            image=torch.stack(images)[None].cuda()
            state=torch.tensor(row['state'],dtype=torch.float32,device='cuda')[None]
            task=torch.tensor(row['task_vector'],dtype=torch.float32,device='cuda')[None]
            pose=torch.tensor(perception['predicted_tcp_pose_in_port'],dtype=torch.float32,device='cuda')[None]
            rec={'sim_time_s':t,'rgb_frame_age_ms':image_age,
                 'current_tcp_port_xyz_mm':(pose[0,:3].cpu().numpy()*1000).tolist(),
                 'live_old_target_port_xyz_mm':(np.asarray(row['targets'][0][:3])*1000).tolist(),
                 'models':{}}
            for name,model in models.items():
                output,_=model(image,state,task,external_pose=pose,
                    external_mask=torch.ones(1,dtype=torch.bool,device='cuda'))
                target=output[0,0,:3].float().cpu().numpy()*1000
                rec['models'][name]={'first_target_port_xyz_mm':target.tolist(),
                                    'first_delta_port_xyz_mm':(target-pose[0,:3].cpu().numpy()*1000).tolist()}
            records.append(rec)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'schema':'sfp_live_observation_actor_command_replay/v1',
        'audit':str(a.audit),'records':records,'true_geometry_input':False},indent=2)+'\n')
    print(json.dumps({'records':len(records),'first':records[0]},indent=2))

if __name__=='__main__':main()
