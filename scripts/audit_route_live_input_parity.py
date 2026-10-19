#!/usr/bin/env python3
"""Replay frozen early live proposals from saved observations; no control or training."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torchvision.models import resnet18

from shared_bc_camera_preprocessing import transport_rgb_tensor
from train_route_aware_diffusion_transport import RouteDiffusion
from train_temporal_transport_probe import LONG_AGES

ROOT=Path(__file__).resolve().parents[1]
CACHE=Path('/home/nas/chmin/aic_archives/2026-10-12_route_consistent_cache_v1')
BASE=ROOT/'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt'
MODEL=ROOT/'outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/best.pt'
IMAGENET=Path('/var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def choose(records,now,history_ages):
    times=np.asarray([r['time'] for r in records],float)
    selected=[]
    for age in history_ages:
        j=np.searchsorted(times,now-age,side='right')-1
        if j>=0 and (age==0 or now-times[j]<=max(history_ages)) and j not in selected:selected.append(int(j))
    selected.sort()
    age=np.asarray([(now-records[j]['time'])/4 for j in selected],np.float32)
    gap=np.asarray([0.]+[(records[b]['time']-records[a]['time'])/4
                        for a,b in zip(selected,selected[1:])],np.float32)
    return selected,age,gap

def infer(model,records,device,sample_steps=12,history_ages=LONG_AGES):
    selected,age,gap=choose(records,records[-1]['time'],history_ages)
    chosen=[records[i] for i in selected]
    as_tensor=lambda key:torch.as_tensor(np.asarray([r[key] for r in chosen]),device=device).float()
    data=dict(history=torch.arange(len(chosen),device=device)[None],
              mask=torch.ones((1,len(chosen)),device=device,dtype=torch.bool),
              features=as_tensor('feature'),prev=as_tensor('previous'),
              motion=as_tensor('motion'),age=torch.as_tensor(age,device=device)[None],
              gap=torch.as_tensor(gap,device=device)[None],
              camera_clock=as_tensor('camera_clock'),
              state_task_full=as_tensor('state_task'))
    with torch.inference_mode(),torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda'):
        context=model.context(data,torch.zeros(1,device=device,dtype=torch.long))
        action,goal=model.sample(context,seed=20261011,sample_steps=sample_steps)
        target=action*model.target_std+model.target_mean
    return target.float()[0].cpu().numpy(),goal.float()[0].cpu().numpy(),len(chosen)

def state_task(state,task,state_mean,state_std,masked):
    normalized=(state-state_mean)/state_std
    normalized[masked]=0
    return np.r_[normalized,task].astype(np.float32)

def read_cache_episode(cache,episode,family):
    rows=[]
    for i,line in enumerate((cache/'rows.jsonl').open()):
        r=json.loads(line)
        if r['episode']==episode and r['family']==family:rows.append((i,r))
    if not rows:raise ValueError(f'No cache episode {family}/{episode}')
    return rows

def get_frame(camera_series,ns):
    stamps=np.asarray([x[0] for x in camera_series],np.int64)
    j=np.searchsorted(stamps,ns)
    j=min((k for k in (j-1,j) if 0<=k<len(stamps)),key=lambda k:abs(stamps[k]-ns))
    if abs(stamps[j]-ns)>50_000_000:raise ValueError(f'No matching camera frame {ns}')
    return Path(camera_series[j][1]),int(stamps[j]-ns)

def camera_feature(backbone,images,device):
    x=torch.stack(images).to(device).float()/255.
    x=((x-x.new_tensor((.485,.456,.406))[None,:,None,None])/
       x.new_tensor((.229,.224,.225))[None,:,None,None])
    with torch.inference_mode(),torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda'):
        return backbone(x).reshape(1536).float().cpu().numpy()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--episode',type=int,required=True)
    p.add_argument('--family',choices=('sfp','sc'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-proposals',type=int,default=12)
    p.add_argument('--ablation-decisions',type=int,default=0)
    p.add_argument('--checkpoint',type=Path,default=MODEL)
    p.add_argument('--cache',type=Path,default=CACHE,
                   help='Feature cache with the checkpoint decision cadence')
    p.add_argument('--zero-previous',action='store_true')
    p.add_argument('--sample-steps',type=int,default=12)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing report')
    torch.set_num_threads(8);cv2.setNumThreads(2)
    base=torch.load(BASE,map_location='cpu',weights_only=False)
    cp=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    history_ages=tuple(cp['source_manifest'].get('history_ages_s',LONG_AGES))
    if a.zero_previous and cp['source_manifest'].get('prior_dropout')!=1.:
        p.error('Zero-previous replay requires a checkpoint trained without previous actions')
    mean=base['model']['target_mean'].numpy();std=base['model']['target_std'].numpy()
    state_mean=base['model']['state_mean'].numpy();state_std=base['model']['state_std'].numpy()
    masked=base.get('masked_state_indices',[])
    horizon=int(cp['source_manifest'].get('action_horizon',4))
    model=RouteDiffusion(1536,mean,std,action_horizon=horizon).to(a.device).eval()
    model.load_state_dict(cp['model'])
    backbone=resnet18(weights=None);backbone.fc=torch.nn.Identity()
    backbone.load_state_dict({k:v for k,v in torch.load(IMAGENET,map_location='cpu',weights_only=True).items()
                              if not k.startswith('fc.')},strict=True)
    backbone=backbone.to(a.device).eval()
    cache_rows=read_cache_episode(a.cache,a.episode,a.family)
    features=np.load(a.cache/'features.npy',mmap_mode='r')
    clocks=np.load(a.cache/'camera_offsets.npy',mmap_mode='r')
    teacher_records=[];teacher=[];last=None
    for index,r in cache_rows:
        state=np.asarray(r['state'],np.float32);task=np.asarray(r['task'],np.float32)
        previous=np.zeros(36,np.float32);motion=np.zeros(6,np.float32)
        if last is not None:
            previous=((np.asarray(last['target'],np.float32)-mean)/std).reshape(-1)
            if r['time_s']-last['time_s']>4:previous[:]=0
            motion[:3]=state[:3]-np.asarray(last['state'][:3],np.float32)
            motion[3:]=(Rotation.from_quat(last['state'][3:7]).inv()*
                        Rotation.from_quat(state[3:7])).as_rotvec()
        if a.zero_previous:previous[:]=0
        teacher_records.append(dict(time=float(r['time_s']),feature=np.asarray(features[index],np.float32),
            previous=previous,motion=motion,camera_clock=np.asarray(clocks[index],np.float32),
            state_task=state_task(state,task,state_mean,state_std,masked)))
        pred,_,history_len=infer(model,teacher_records,a.device,a.sample_steps,history_ages)
        truth=np.asarray(r['target'],np.float32)
        teacher.append(dict(index=index,time_s=r['time_s'],frame=r['frame'],history_len=history_len,
                            state=state.tolist(),task=task.tolist(),
                            predicted_first_mm=(pred[0,:3]*1000).tolist(),
                            teacher_first_mm=(truth[0,:3]*1000).tolist(),
                            first_error_mm=float(np.linalg.norm(pred[0,:3]-truth[0,:3])*1000),
                            feature=np.asarray(features[index],np.float32)))
        last=r
        if len(teacher)>=a.max_proposals+5:break
    stream=json.loads((a.audit/'scored_stream.json').read_text())
    proposals=[json.loads(line) for line in (a.audit/'runtime_trace.jsonl').open()
               if json.loads(line).get('event')=='proposal'][:a.max_proposals]
    live_records=[];live=[];last=None
    for row in proposals:
        state=np.asarray(row['state'],np.float32);task=np.asarray(row['task_vector'],np.float32)
        previous=np.zeros(36,np.float32);motion=np.zeros(6,np.float32)
        if last is not None:
            previous=((np.asarray(last['targets'][:4],np.float32)-mean)/std).reshape(-1)
            prior=np.asarray(last['state'],np.float32)
            motion[:3]=state[:3]-prior[:3]
            motion[3:]=(Rotation.from_quat(prior[3:7]).inv()*Rotation.from_quat(state[3:7])).as_rotvec()
        if a.zero_previous:previous[:]=0
        images=[];frame_lags=[];image_files=[]
        for camera in ('center','left','right'):
            path,lag=get_frame(stream['cameras'][camera],int(row['image_sim_ns'][camera]))
            bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
            if bgr is None:raise ValueError(f'Unreadable {path}')
            rgb=bgr[...,::-1]
            images.append(transport_rgb_tensor(rgb,native_already_encoded=True))
            frame_lags.append(lag);image_files.append(str(path))
        feature=camera_feature(backbone,images,a.device)
        sim=float(row['sim_time'])
        camera_clock=np.r_[np.maximum(0,sim-np.asarray(list(row['image_sim_ns'].values()))*1e-9),
                           float(a.family=='sc')].astype(np.float32)
        live_records.append(dict(time=sim,feature=feature,previous=previous,motion=motion,
                                 camera_clock=camera_clock,
                                 state_task=state_task(state,task,state_mean,state_std,masked)))
        pred,_,history_len=infer(model,live_records,a.device,a.sample_steps,history_ages)
        actual=np.asarray(row['targets'],np.float32)
        live.append(dict(sim_s=sim,history_len=history_len,state=state.tolist(),task=task.tolist(),
                         recorded_first_mm=(actual[0,:3]*1000).tolist(),
                         replay_first_mm=(pred[0,:3]*1000).tolist(),
                         replay_error_mm=float(np.linalg.norm(pred[0,:3]-actual[0,:3])*1000),
                         frame_stamp_lag_ms=(np.asarray(frame_lags)/1e6).tolist(),
                         images=image_files,feature=feature))
        last=row
    aligned=[]
    teacher_start=teacher[0]['time_s'];live_start=live[0]['sim_s']
    for row in live:
        elapsed=row['sim_s']-live_start
        ref=min(teacher,key=lambda x:abs((x['time_s']-teacher_start)-elapsed))
        delta=np.asarray(row['state'][:3])-np.asarray(ref['state'][:3])
        cosine=float(np.dot(row['feature'],ref['feature'])/
                     (np.linalg.norm(row['feature'])*np.linalg.norm(ref['feature'])+1e-9))
        aligned.append(dict(sim_s=row['sim_s'],teacher_time_s=ref['time_s'],
            teacher_frame=ref['frame'],tcp_position_difference_mm=float(np.linalg.norm(delta)*1000),
            task_equal=bool(np.allclose(row['task'],ref['task'])),feature_cosine=cosine,
            teacher_predicted_first_mm=ref['predicted_first_mm'],
            live_replay_first_mm=row['replay_first_mm'],recorded_first_mm=row['recorded_first_mm']))
    ablations=[]
    for i in range(min(a.ablation_decisions,len(live_records))):
        live_row=live[i];elapsed=live_row['sim_s']-live_start
        teacher_j=min(range(len(teacher)),key=lambda j:abs(
            (teacher[j]['time_s']-teacher_start)-elapsed))
        baseline=np.asarray(live_row['replay_first_mm'])
        variants={}
        for key in ('previous','feature','motion','camera_clock','state_task'):
            hybrid=[dict(r) for r in live_records[:i+1]]
            hybrid[-1][key]=teacher_records[teacher_j][key]
            target,_,_=infer(model,hybrid,a.device,a.sample_steps,history_ages)
            variants['teacher_'+key]=(target[0,:3]*1000).tolist()
        hybrid=[dict(r) for r in live_records[:i+1]]
        hybrid[-1]['previous']=np.zeros_like(hybrid[-1]['previous'])
        target,_,_=infer(model,hybrid,a.device,a.sample_steps,history_ages)
        variants['zero_previous']=(target[0,:3]*1000).tolist()
        ablations.append(dict(sim_s=live_row['sim_s'],teacher_time_s=teacher[teacher_j]['time_s'],
            live_first_mm=baseline.tolist(),teacher_predicted_first_mm=teacher[teacher_j]['predicted_first_mm'],
            teacher_recorded_first_mm=teacher[teacher_j]['teacher_first_mm'],variants=variants))
    for r in teacher:r.pop('feature')
    for r in live:r.pop('feature')
    result=dict(schema='aic_route_live_input_parity/v1',audit=str(a.audit),episode=a.episode,
                family=a.family,checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint),
                cache=str(a.cache),action_horizon=horizon,history_ages_s=history_ages,
                zero_previous=a.zero_previous,sample_steps=a.sample_steps,
                captured_frames_are_jpeg86=True,approximates_live_raw_camera_by_saved_jpeg86=True,
                teacher=teacher,live=live,aligned=aligned,ablations=ablations,
                final_scenes_opened=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(audit=str(a.audit),family=a.family,
        live_replay_error_mm=[round(x['replay_error_mm'],3) for x in live],
        tcp_difference_mm=[round(x['tcp_position_difference_mm'],1) for x in aligned],
        feature_cosine=[round(x['feature_cosine'],4) for x in aligned])),flush=True)

if __name__=='__main__':main()
