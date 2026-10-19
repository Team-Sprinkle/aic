#!/usr/bin/env python3
"""Replay frozen transport-only handoff head beside recorded visual stops.

Both decisions use saved RGB, ordinary state, task, and clock. The MoveIt
teacher standoff is opened only after candidate times have been determined.
"""
import argparse
import bisect
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
from torchvision.models import resnet18
import yaml

from shared_bc_camera_preprocessing import transport_rgb_tensor
from train_temporal_transport_probe import TemporalTransport


CAMERAS=('center','left','right')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def quantiles(values):
    return {k:float(np.quantile(values,q)) for k,q in
            [('p50',.5),('p95',.95),('p99',.99)]} if values else None


def frames(audit):
    out={c:[] for c in CAMERAS}
    for row in csv.DictReader((audit/'frame_manifest.csv').open()):
        if row['camera'] in out:
            out[row['camera']].append((int(row['sim_ns']),audit/'frames'/row['file']))
    for c in out:out[c].sort()
    return out


def nearest(rows,stamp):
    times=[x[0] for x in rows]
    j=bisect.bisect_left(times,stamp)
    j=min((max(0,j-1),min(len(rows)-1,j)),key=lambda k:abs(times[k]-stamp))
    if abs(times[j]-stamp)>100_000_000:raise ValueError('Saved RGB older than 100 ms')
    return rows[j][1]


def route_error_mm(xyz,path):
    values=[]
    for a,b in zip(path[:-1],path[1:]):
        d=b-a
        if d@d<1e-14:continue
        t=np.clip(((xyz-a)@d)/(d@d),0,1)
        values.append(np.linalg.norm(xyz-(a+t[:,None]*d),axis=1)*1000)
    return np.min(np.stack(values),axis=0)


def first_latched(rows,threshold=.95,required=2,min_s=15):
    streak=0
    for row in rows:
        streak=streak+1 if row['sim_time']>=min_s and row['transport_probability']>=threshold else 0
        row['transport_streak']=streak
        if streak>=required:return row['sim_time']
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,action='append',required=True)
    p.add_argument('--handoff',type=Path,required=True)
    p.add_argument('--actor',type=Path,required=True)
    p.add_argument('--imagenet',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous result')
    torch.set_num_threads(4);cv2.setNumThreads(1)
    saved=torch.load(a.handoff,map_location='cpu',weights_only=False)
    manifest=saved['source_manifest']
    if not manifest['zero_previous_command']:raise ValueError('Expected prior-free transport head')
    model=TemporalTransport(saved['model']['target_mean'].numpy(),
        saved['model']['target_std'].numpy(),raw_state_skip=True,
        feature_dim=1536,state_task_dim=42,route_conditioning=False).cuda().eval()
    model.load_state_dict(saved['model'])
    actor=torch.load(a.actor,map_location='cpu',weights_only=False)
    state_mean=actor['model']['state_mean'].numpy();state_std=actor['model']['state_std'].numpy()
    masked=np.array(actor['masked_state_indices'],dtype=int)
    backbone=resnet18(weights=None);backbone.fc=torch.nn.Identity()
    weights=torch.load(a.imagenet,map_location='cpu',weights_only=True)
    backbone.load_state_dict({k:v for k,v in weights.items() if not k.startswith('fc.')})
    backbone=backbone.cuda().eval()
    out=[]
    for audit in a.audit:
        trace=[json.loads(x) for x in (audit/'runtime_trace.jsonl').open()]
        proposals=[x for x in trace if x['event']=='proposal']
        commands=[x for x in trace if x['event']=='command']
        visual=[x for x in trace if x['event']=='visual_standoff_signal']
        complete=next((x for x in trace if x['event']=='visual_standoff_complete'),None)
        diagnostics=json.loads((audit/'teacher_path_diagnostic.json').read_text())
        waypoints=json.loads(Path(diagnostics['plan']).read_text())['waypoints']
        path=np.array([w['tcp_pose']['position'] for w in waypoints]);goal=waypoints[-1]['tcp_pose']
        goal_pos=np.array(goal['position']);goal_rot=Rotation.from_quat(goal['orientation_xyzw'])
        frame_index=frames(audit)
        history=[];predictions=[];previous_state=None
        for prop in proposals:
            t=prop['sim_time'];state=np.array(prop['state'],dtype=np.float32)
            rgb=[]
            for cam in CAMERAS:
                image=cv2.imread(str(nearest(frame_index[cam],prop['image_sim_ns'][cam])))
                if image is None:raise FileNotFoundError(cam)
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                rgb.append(transport_rgb_tensor(image,native_jpeg_quality=86))
            x=torch.stack(rgb).cuda().float()/255
            x=(x-x.new_tensor((.485,.456,.406))[None,:,None,None])/x.new_tensor((.229,.224,.225))[None,:,None,None]
            started=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                feat=backbone(x).reshape(1536).float()
            normalized=(state-state_mean)/state_std;normalized[masked]=0
            state_task=np.r_[normalized,np.array(prop['task_vector'],dtype=np.float32)]
            motion=np.zeros(6,dtype=np.float32)
            if previous_state is not None:
                motion[:3]=state[:3]-previous_state[:3]
                motion[3:]=(Rotation.from_quat(previous_state[3:7]).inv()*
                            Rotation.from_quat(state[3:7])).as_rotvec()
            previous_state=state
            stamps=np.array(list(prop['image_sim_ns'].values()))/1e9
            camera_clock=np.r_[np.maximum(0,t-stamps)/.05,1.].astype(np.float32)
            history.append({'time':t,'feature':feat,'motion':motion,
                            'state_task':state_task,'camera_clock':camera_clock})
            times=np.array([r['time'] for r in history]);chosen=[]
            for age in manifest['history_ages_s']:
                j=np.searchsorted(times,t-age,side='right')-1
                if j>=0 and (age==0 or t-times[j]<=60) and j not in chosen:chosen.append(j)
            selected=[history[j] for j in sorted(chosen)]
            ages=np.array([(t-r['time'])/4 for r in selected],np.float32)
            gaps=np.array([0.]+[(selected[j]['time']-selected[j-1]['time'])/4
                                for j in range(1,len(selected))],np.float32)
            tensor=lambda v:torch.as_tensor(np.asarray(v),device='cuda').float()[None]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                _,_,_,handoff=model(torch.stack([r['feature'] for r in selected])[None],
                    torch.zeros((1,len(selected),36),device='cuda'),
                    tensor([r['motion'] for r in selected]),tensor(ages),tensor(gaps),
                    tensor([r['camera_clock'] for r in selected]),
                    torch.ones((1,len(selected)),dtype=torch.bool,device='cuda'),
                    tensor([r['state_task'] for r in selected]))
                probability=float(handoff.sigmoid()[0])
            predictions.append({'sim_time':t,'transport_probability':probability,
                                'head_replay_ms':(time.perf_counter()-started)*1000})
        transport_time=first_latched(predictions)
        visual_time=next((x['sim_time'] for x in visual if x.get('stop_streak',0)>=3),None)
        if complete is None:visual_time=None
        cmd_times=np.array([x['sim_time'] for x in commands])
        cmd_xyz=np.array([x['state'][:3] for x in commands]);route=route_error_mm(cmd_xyz,path)
        goal_error=np.linalg.norm(cmd_xyz-goal_pos,axis=1)*1000
        angles=np.array([(Rotation.from_quat(x['state'][3:7]).inv()*goal_rot).magnitude()*180/np.pi
                         for x in commands])
        valid=np.where((goal_error<=10)&(angles<=3))[0]
        first_valid=float(cmd_times[valid[0]]) if len(valid) else None
        selected_port=int(np.argmax(proposals[0]['task_vector'][2:4]))
        score=yaml.safe_load((audit/'results/scoring.yaml').read_text())
        trial=next(v for k,v in score.items() if str(k).startswith('trial_'))
        contact=trial['tier_2'].get('categories',{})
        def scored_stop(stop_time,source):
            if stop_time is None:return {'state':'missed','first_stop_sim_s':None,'missed_arrival':first_valid is not None}
            j=int(np.argmin(abs(cmd_times-stop_time)))
            wrong=(source=='visual' and any(x['selected_port']!=selected_port for x in visual
                 if abs(x['sim_time']-stop_time)<.3))
            unsafe=bool(route[:j+1].max()>50)
            correct=bool(goal_error[j]<=10 and angles[j]<=3 and not wrong and not unsafe)
            return {'state':'latch_handoff','first_stop_sim_s':stop_time,
                'selected_port':selected_port,'wrong_port':wrong,'tcp_standoff_position_error_mm':float(goal_error[j]),
                'tcp_standoff_orientation_error_deg':float(angles[j]),
                'route_deviation_at_stop_mm':float(route[j]),
                'route_deviation_max_to_stop_mm':float(route[:j+1].max()),
                'unsafe_route_over_50mm':unsafe,'false_handoff':not correct,
                'first_valid_arrival_sim_s':first_valid,
                'delay_after_first_valid_s':stop_time-first_valid if first_valid is not None else None}
        visual_terminal=next((x for x in trace if x['event']=='terminal_before_return'),None)
        out.append({'audit':str(audit),'episode':Path(diagnostics['plan']).parts[-2],
            'task_port':selected_port,'proposal_count':len(proposals),
            'transport_head':scored_stop(transport_time,'transport'),
            'visual':scored_stop(visual_time,'visual'),
            'first_valid_arrival_sim_s':first_valid,
            'visual_signal_count':len(visual),'visual_wrong_port_signals':sum(x['selected_port']!=selected_port for x in visual),
            'visual_stop_event':complete is not None,
            'visual_latch_state':visual[-1].get('stop_streak') if visual else None,
            'visual_candidate_at_last_signal':visual[-1].get('predicted_distance_mm') if visual else None,
            'force_n_p95_raw_wrist':float(np.quantile([np.linalg.norm(x['state'][27:30]) for x in commands],.95)),
            'official_contact':contact.get('contacts'),'official_excess_force':contact.get('insertion force'),
            'full_runtime_inference_ms':quantiles([x['inference_ms'] for x in proposals]),
            'offline_head_replay_ms':quantiles([x['head_replay_ms'] for x in predictions]),
            'transport_head_trace':predictions,
            'visual_trace':visual,
            'terminal_sim_s':visual_terminal['sim_time'] if visual_terminal else None,
            'posthoc_goal_plan':diagnostics['plan'],'runtime_geometry_input':False})
        print(json.dumps({'episode':audit.name,'transport_stop':transport_time,'visual_stop':visual_time,
                          'valid_arrival':first_valid}),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'schema':'aic_frozen_sc_stop_comparison/v1','episodes':out,
        'transport_head_checkpoint':str(a.handoff),'transport_head_sha256':sha(a.handoff),
        'actor_checkpoint':str(a.actor),'actor_sha256':sha(a.actor),
        'imagenet_sha256':sha(a.imagenet),'transport_threshold':.95,
        'transport_streak':2,'visual_decision_source':'recorded causal visual_standoff_complete',
        'unsafe_route_threshold_mm':50,'reserved_final_opened':False},indent=2)+'\n')


if __name__=='__main__':main()
