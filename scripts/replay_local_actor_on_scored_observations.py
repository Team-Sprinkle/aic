#!/usr/bin/env python3
"""Replay frozen RGB/state/estimated-pose observations through local BC arms.

Scored bag TF is never read; this compares model predictions on the exact
camera frames and ordinary state encountered during an existing rollout.
"""
import argparse
from bisect import bisect_left
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch

from shared_port_tcp_geometry import matrix_from_pose9
from train_shared_port_tcp_bc import SharedPortTCPActor


CAMERAS=('center','left','right')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stream',type=Path,required=True)
    p.add_argument('--actor-trace',type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    p.add_argument('--sc-prior',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(6);cv2.setNumThreads(1)
    stream=json.loads(a.stream.read_text());trace=[json.loads(x) for x in a.actor_trace.open()]
    proposal={round(x['sim_time'],3):x for x in trace if x['event']=='proposal'}
    perception={round(x['sim_time'],3):x for x in trace if x['event']=='local_perception'}
    keys=[t for t in sorted(proposal.keys()&perception.keys()) if 65<=t<=120]
    times={cam:[r[0] for r in stream['cameras'][cam]] for cam in CAMERAS}
    prior=Rotation.from_quat(json.loads(a.sc_prior.read_text())['nominal_tcp_in_port_orientation_xyzw'])
    models=[]
    for path in a.checkpoints:
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        stats={k:ckpt['model'][k].numpy() for k in ('state_mean','state_std','pose_mean','pose_std','target_mean','target_std')}
        model=SharedPortTCPActor(stats,pretrained=False,condition_on_pose=True,
            separate_family_orientation_head=bool(ckpt.get('separate_family_orientation_head',False)),
            separate_family_action_head=bool(ckpt.get('separate_family_action_head',False))).cuda().eval()
        model.load_state_dict(ckpt['model']);models.append((str(path),model))
    records=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for t in keys:
            p=proposal[t];q=perception[t];images=[]
            for cam in CAMERAS:
                stamp=p['image_sim_ns'][cam];array=stream['cameras'][cam]
                j=bisect_left(times[cam],stamp)
                j=min(range(max(0,j-1),min(len(array),j+1)),key=lambda k:abs(times[cam][k]-stamp))
                if abs(times[cam][j]-stamp)>50_000_000:raise ValueError('Stale saved RGB frame')
                path=Path(array[j][1])
                if str(path).startswith('/repo/'):
                    path=Path.cwd()/str(path).removeprefix('/repo/')
                image=cv2.imread(str(path),cv2.IMREAD_COLOR)
                if image is None:raise FileNotFoundError(path)
                rgb=cv2.cvtColor(cv2.resize(image,(288,256),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
                rgb=np.asarray(Image.fromarray(rgb).resize((224,224),Image.Resampling.BILINEAR)).copy()
                images.append(torch.from_numpy(rgb).permute(2,0,1))
            image=torch.stack(images)[None].cuda()
            state=torch.tensor(p['state'],dtype=torch.float32,device='cuda')[None]
            task=torch.tensor(p['task_vector'],dtype=torch.float32,device='cuda')[None]
            pose=torch.tensor(q['predicted_tcp_pose_in_port'],dtype=torch.float32,device='cuda')[None]
            data={'sim_time_s':t,'predicted_tcp_axial_mm':float(pose[0,2]*1000),
                  'recorded_first_target_port_xyz_mm':(np.asarray(p['targets'][0][:3])*1000).tolist(),
                  'models':{}}
            for name,model in models:
                out,_=model(image,state,task,external_pose=pose,
                    external_mask=torch.ones(1,dtype=torch.bool,device='cuda'))
                target=out[0,0].float().cpu().numpy()
                angle=(prior.inv()*Rotation.from_matrix(matrix_from_pose9(target)[:3,:3])).magnitude()*180/np.pi
                data['models'][name]={'first_target_port_xyz_mm':(target[:3]*1000).tolist(),
                                      'sc_prior_orientation_deviation_deg':float(angle)}
            records.append(data)
    summary={'schema':'aic_frozen_sc_onpolicy_replay/v1','rows':len(records),
             'checkpoints':{name:{'near_abs_axial_lt_40mm_orientation_deviation_deg':{
                'median':float(np.median([r['models'][name]['sc_prior_orientation_deviation_deg']
                    for r in records if abs(r['predicted_tcp_axial_mm'])<40])),
                'p95':float(np.quantile([r['models'][name]['sc_prior_orientation_deviation_deg']
                    for r in records if abs(r['predicted_tcp_axial_mm'])<40],.95))}}
                for name,_ in models},
             'actor_input_true_geometry':False,'teacher_prior_for_posthoc_score_only':True}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'summary':summary,'records':records},indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
