#!/usr/bin/env python3
"""Bounded frozen-vision temporal transport candidate and no-history ablation.

All features are observation-only. Stage, subgoal, handoff, and action targets
are offline supervision. This script does not claim autonomous performance.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from train_shared_port_tcp_bc import rotation_from_pose9


AGES = (16., 10., 6., 4., 3., 2.5, 2., 1.6, 1.2, .8, .6, .4, .2, 0.)
LONG_AGES = (60., 45., 30., 20., 16., 12., 8., 6., 4., 3., 2.,
             1.5, 1., .6, .4, .2, 0.)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


class TemporalTransport(nn.Module):
    def __init__(self, target_mean, target_std, raw_state_skip=False,
                 feature_dim=256, state_task_dim=13, route_conditioning=True,
                 task_gated_sfp=False, task_route_film=False,
                 spatial_request_attention=False, route_goal_conditioning=False):
        super().__init__()
        self.raw_state_skip=raw_state_skip
        self.state_task_dim=state_task_dim
        self.route_conditioning=route_conditioning
        self.task_gated_sfp=task_gated_sfp
        self.task_route_film=task_route_film
        self.spatial_request_attention=spatial_request_attention
        self.route_goal_conditioning=route_goal_conditioning
        if route_goal_conditioning:
            self.route_goal_proj=nn.Linear(3,256,bias=False)
            nn.init.zeros_(self.route_goal_proj.weight)
        if spatial_request_attention:
            if not raw_state_skip:raise ValueError('Spatial request attention needs task code')
            self.spatial_request=nn.Embedding(10,256)
            self.spatial_camera=nn.Embedding(3,256)
            self.spatial_position=nn.Embedding(20,256)
            self.spatial_attention=nn.MultiheadAttention(256,4,batch_first=True)
            self.spatial_to_current=nn.Linear(256,256)
            nn.init.zeros_(self.spatial_to_current.weight)
            nn.init.zeros_(self.spatial_to_current.bias)
        if task_route_film:
            if not raw_state_skip:
                raise ValueError('Task route FiLM requires the observed task code')
            self.route_task_embedding=nn.Embedding(10,128)
            self.route_visual_film=nn.Linear(128,feature_dim*2)
            nn.init.zeros_(self.route_visual_film.weight)
            nn.init.zeros_(self.route_visual_film.bias)
        self.register_buffer('target_mean',torch.as_tensor(target_mean,dtype=torch.float32))
        self.register_buffer('target_std',torch.as_tensor(target_std,dtype=torch.float32))
        input_dim=feature_dim+48+(state_task_dim if raw_state_skip else 0)
        self.input=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,256),nn.GELU())
        layer=nn.TransformerEncoderLayer(256,8,512,dropout=.05,
                                         activation='gelu',batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(layer,2)
        self.stage=nn.Linear(256,7)
        self.subgoal=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,9))
        self.handoff=nn.Linear(256,1)
        self.action=nn.Sequential(nn.LayerNorm(256+7+9),nn.Linear(272,256),
                                  nn.GELU(),nn.Linear(256,36))
        if task_gated_sfp:
            if not raw_state_skip:
                raise ValueError('Task-gated SFP head requires explicit state/task input')
            self.sfp_expert_weight=nn.Parameter(torch.empty(10,36,256))
            self.sfp_expert_bias=nn.Parameter(torch.empty(10,36))

    def forward(self,features,previous_action,measured_motion,ages,gaps,camera_clock,mask,
                state_task=None,return_hidden=False,spatial=None,route_goal=None):
        if self.task_route_film:
            task=state_task[...,-10:]
            index=task[...,4:9].argmax(-1)*2+task[...,2:4].argmax(-1)
            gamma,beta=self.route_visual_film(self.route_task_embedding(index)).chunk(2,-1)
            features=features*(1+gamma)+beta
        parts=(features,previous_action,measured_motion,ages[...,None],gaps[...,None],camera_clock)
        if self.raw_state_skip:
            if state_task is None:raise ValueError('Measured TCP/task skip is required')
            parts=(*parts,state_task)
        x=self.input(torch.cat(parts,-1))
        x=self.temporal(x,src_key_padding_mask=~mask)
        # Histories are right aligned; the last token is always current.
        current=x[:,-1]
        if self.spatial_request_attention:
            if spatial is None:raise ValueError('Current spatial camera grid required')
            task=state_task[:,-1,-10:]
            index=task[:,4:9].argmax(-1)*2+task[:,2:4].argmax(-1)
            cam=self.spatial_camera(torch.arange(3,device=x.device))[:,None,:]
            pos=self.spatial_position(torch.arange(20,device=x.device))[None,:,:]
            tokens=(spatial.float()+cam+pos).reshape(len(x),60,256)
            query=(current+self.spatial_request(index))[:,None,:]
            attended,_=self.spatial_attention(query,tokens,tokens,need_weights=False)
            current=current+self.spatial_to_current(attended[:,0])
        if self.route_goal_conditioning:
            if route_goal is None or route_goal.shape!=(len(current),3):
                raise ValueError('Predicted 3D route goal required for this actor')
            current=current+self.route_goal_proj(route_goal.float())
        stage=self.stage(current)
        subgoal=self.subgoal(current)
        handoff=self.handoff(current).squeeze(-1)
        if self.route_conditioning:
            action_input=torch.cat((current,stage.softmax(-1),subgoal),-1)
        elif self.task_route_film:
            action_input=torch.cat((current,stage.softmax(-1),torch.zeros_like(subgoal)),-1)
        else:
            action_input=torch.cat((current,torch.zeros_like(stage),
                                    torch.zeros_like(subgoal)),-1)
        action=self.action(action_input)
        if self.task_gated_sfp:
            task=state_task[:,-1,-10:]
            expert_index=task[:,4:9].argmax(-1)*2+task[:,2:4].argmax(-1)
            hidden=self.action[:3](action_input)
            expert=torch.bmm(self.sfp_expert_weight[expert_index],
                             hidden[...,None]).squeeze(-1)+self.sfp_expert_bias[expert_index]
            action=torch.where(task[:,0:1]>.5,expert,action)
        result=(action.reshape(-1,4,9),stage,subgoal,handoff)
        return (*result,current) if return_hidden else result


def build_data(root,checkpoint,device,history_ages=AGES,spatial_cache=None,
               route_goal_model=None,route_goal_zero_control=False,
               route_goal_progress_gate_mm=None,route_goal_values=None):
    rows=[json.loads(line) for line in (root/'rows.jsonl').open()]
    features=np.load(root/'features.npy',mmap_mode='r')
    spatial=None
    if spatial_cache is not None:
        spatial_source=np.load(spatial_cache/'spatial.npy',mmap_mode='r')
        if spatial_source.shape!=(len(rows),3,20,256):
            raise ValueError('Spatial cache row/order shape mismatch')
        spatial=torch.as_tensor(np.array(spatial_source,copy=True),device=device)
    camera_clock=np.load(root/'camera_offsets.npy',mmap_mode='r')
    assert len(rows)==len(features)
    assert camera_clock.shape==(len(rows),4)
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
    mean=cp['model']['target_mean'].numpy();std=cp['model']['target_std'].numpy()
    ep=defaultdict(list)
    for i,r in enumerate(rows):ep[(r['family'],r['episode'])].append(i)
    n=len(rows);hist=np.full((n,len(history_ages)),-1,dtype=np.int64)
    ages=np.zeros_like(hist,dtype=np.float32);gaps=ages.copy()
    previous=np.zeros((n,36),dtype=np.float32)
    motion=np.zeros((n,6),dtype=np.float32)
    state=np.asarray([r['state'] for r in rows],dtype=np.float32)
    task=np.asarray([r['task'] for r in rows],dtype=np.float32)
    route_goal=None
    if route_goal_values is not None:
        source=np.load(route_goal_values,mmap_mode='r')
        if source.shape!=(len(rows),3):
            raise ValueError('Selected-port route offset row/order shape mismatch')
        route_goal=(np.zeros((len(rows),3),dtype=np.float32) if route_goal_zero_control else
                    np.asarray(source,dtype=np.float32).copy())
        if not np.isfinite(route_goal).all():raise ValueError('Nonfinite selected-port route offsets')
    elif route_goal_model is not None:
        ridge=dict(np.load(route_goal_model))
        sfp=np.asarray([r['family']=='sfp' for r in rows])
        route_goal=np.zeros((len(rows),3),dtype=np.float32)
        if sfp.any() and not route_goal_zero_control:
            public=np.concatenate((np.asarray(features[sfp],dtype=np.float32),
                                   state[sfp,:7],task[sfp]),axis=1)
            if public.shape[1]!=len(ridge['center']):
                raise ValueError('Route goal model input dimension mismatch')
            base=((public-ridge['center'])/ridge['scale'])@ridge['coefficients']+ridge['target_mean']
            route_goal[sfp]=(Rotation.from_quat(state[sfp,3:7]).inv().apply(base)/.05).astype(np.float32)
            if route_goal_progress_gate_mm is not None:
                starts={}
                for i,row in enumerate(rows):
                    if row['family']!='sfp':continue
                    key=(row['split'],row['episode'])
                    starts.setdefault(key,state[i,:3])
                    if np.linalg.norm(state[i,:3]-starts[key])*1000>route_goal_progress_gate_mm:
                        route_goal[i]=0
    normalized_xyz=((state[:,:3]-cp['model']['state_mean'][:3].numpy()) /
                    cp['model']['state_std'][:3].numpy())
    state_task=np.concatenate((normalized_xyz,task),axis=-1)
    normalized_state=((state-cp['model']['state_mean'].numpy()) /
                      cp['model']['state_std'].numpy())
    for masked_index in cp.get('masked_state_indices',[]):
        normalized_state[:,masked_index]=0
    state_task_full=np.concatenate((normalized_state,task),axis=-1)
    target=np.asarray([r['target'] for r in rows],dtype=np.float32)
    subgoal=np.asarray([r['subgoal'] for r in rows],dtype=np.float32)
    times=np.asarray([r['time_s'] for r in rows],dtype=np.float32)
    for indices in ep.values():
        assert all(times[a]<times[b] for a,b in zip(indices,indices[1:]))
        for k,i in enumerate(indices):
            if k:
                prior=indices[k-1]
                previous[i]=((target[prior]-mean)/std).reshape(-1)
                delta=Rotation.from_quat(state[prior,3:7]).inv()*Rotation.from_quat(state[i,3:7])
                motion[i,:3]=state[i,:3]-state[prior,:3]
                motion[i,3:]=delta.as_rotvec()
            if k and times[i]-times[indices[k-1]]>4.:
                # Large image gaps remain visible as a mask and time gap;
                # previous commands are stale after a reset-sized gap.
                previous[i]=0
            chosen=[]
            for age in history_ages:
                limit=times[i]-age
                j=np.searchsorted(times[indices[:k+1]],limit,side='right')-1
                if j>=0 and (age==0 or times[i]-times[indices[j]]<=max(20.,max(history_ages))):
                    candidate=indices[j]
                    if candidate not in chosen:chosen.append(candidate)
            chosen.sort()
            offset=len(history_ages)-len(chosen)
            for q,c in enumerate(chosen,start=offset):
                hist[i,q]=c;ages[i,q]=min(max(history_ages),times[i]-times[c])/4.
                gaps[i,q]=0 if q==offset else min(max(history_ages),times[c]-times[hist[i,q-1]])/4.
    route_stage=np.full(n,6,dtype=np.int64)
    for (family,_episode),indices in ep.items():
        if family!='sfp':continue
        episode_times=times[indices]
        for k,i in enumerate(indices):
            future=min(len(indices)-1,int(np.searchsorted(episode_times,times[i]+1.,side='left')))
            displacement=state[indices[future],:3]-state[i,:3]
            if np.linalg.norm(displacement)<.004:continue
            axis=int(np.argmax(np.abs(displacement)))
            route_stage[i]=2*axis+int(displacement[axis]>0)
    hidx=torch.as_tensor(np.maximum(hist,0),device=device)
    mask=torch.as_tensor(hist>=0,device=device)
    data=dict(rows=rows,features=torch.as_tensor(np.array(features,copy=True),device=device),
              spatial=spatial,
              prev=torch.as_tensor(previous,device=device),motion=torch.as_tensor(motion,device=device),
              camera_clock=torch.as_tensor(np.array(camera_clock,copy=True),device=device),
              state_task=torch.as_tensor(state_task,device=device),
              state_task_full=torch.as_tensor(state_task_full,device=device),
              age=torch.as_tensor(ages,device=device),gap=torch.as_tensor(gaps,device=device),
              history=hidx,mask=mask,target=torch.as_tensor(target,device=device),
              stage=torch.as_tensor([r['stage'] for r in rows],device=device),
              route_stage=torch.as_tensor(route_stage,device=device),
              subgoal=torch.as_tensor(subgoal,device=device),
              handoff=torch.as_tensor([r['handoff'] for r in rows],device=device,dtype=torch.float32),
              route_goal=(torch.as_tensor(route_goal,device=device) if route_goal is not None else None),
              mean=mean,std=std)
    return data


def predict(model,data,indices,no_history,previous_dropout=0.,previous_noise=0.,force_zero_previous=False,
            return_hidden=False):
    h=data['history'][indices];mask=data['mask'][indices]
    if no_history:
        h=h[:,-1:];mask=mask[:,-1:]
    features=data['features'][h].float()
    prior=data['prev'][h].float();motion=data['motion'][h].float()
    if force_zero_previous:
        prior=torch.zeros_like(prior)
    elif previous_dropout or previous_noise:
        if previous_noise:
            prior=prior+torch.randn_like(prior)*previous_noise
        if previous_dropout:
            keep=(torch.rand((*prior.shape[:2],1),device=prior.device)>=previous_dropout)
            prior=prior*keep
    camera_clock=data['camera_clock'][h].float()
    state_task=(data['state_task_full' if model.state_task_dim==42 else 'state_task'][h].float()
                if model.raw_state_skip else None)
    camera_clock[...,:3]/=.05
    ages=data['age'][indices];gaps=data['gap'][indices]
    if no_history:ages=ages[:,-1:];gaps=gaps[:,-1:]
    # Right-aligned causal windows: pad values are ignored by attention.
    return model(features,prior,motion,ages,gaps,camera_clock,mask,state_task,
                 return_hidden=return_hidden,
                 spatial=(data['spatial'][indices].float() if model.spatial_request_attention else None),
                 route_goal=(data['route_goal'][indices] if model.route_goal_conditioning else None))


def paired_counterfactual_first(model,data,left,right):
    """Same observed history twice, with only the public request code changed."""
    h=data['history'][left];mask=data['mask'][left]
    features=data['features'][h].float()
    prior=data['prev'][h].float()
    motion=data['motion'][h].float()
    camera_clock=data['camera_clock'][h].float()
    camera_clock[...,:3]/=.05
    state=data['state_task_full'][h].float()
    alternate=state.clone()
    requested=data['state_task_full'][right,-10:].float()
    alternate[...,-10:]=requested[:,None,:]
    action,*_=model(torch.cat((features,features)),torch.cat((prior,prior)),
                    torch.cat((motion,motion)),
                    torch.cat((data['age'][left],data['age'][left])),
                    torch.cat((data['gap'][left],data['gap'][left])),
                    torch.cat((camera_clock,camera_clock)),torch.cat((mask,mask)),
                    torch.cat((state,alternate)))
    return action[:len(left),0],action[len(left):,0]


@torch.inference_mode()
def evaluate(model,data,indices,no_history,batch=256,force_zero_previous=False):
    model.eval();groups=defaultdict(lambda:defaultdict(list));conf=np.zeros((7,7),dtype=int)
    handoff_by_family={family:dict(tp=0,fp=0,fn=0,tn=0,positives=0,rows=0)
                       for family in ('sc','sfp')}
    stage_predictions={}
    for part in np.array_split(indices,max(1,(len(indices)+batch-1)//batch)):
        if not len(part):continue
        ind=torch.as_tensor(part,device=data['stage'].device)
        action,stage,subgoal,handoff=predict(model,data,ind,no_history,
                                            force_zero_previous=force_zero_previous)
        actual=data['target'][ind];true_sub=data['subgoal'][ind]
        predicted_action=action[:,0]*model.target_std+model.target_mean
        mm=(predicted_action[:,:3]-actual[:,0,:3]).norm(dim=-1)*1e3
        sub_mm=(subgoal[:,:3]-true_sub[:,:3]).norm(dim=-1)*1e3
        def angle_deg(pred,true):
            relative=rotation_from_pose9(pred).transpose(-1,-2)@rotation_from_pose9(true)
            cosine=((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
            return cosine.acos()*180/np.pi
        action_deg=angle_deg(predicted_action,actual[:,0]);sub_deg=angle_deg(subgoal,true_sub)
        pred_stage=stage.argmax(-1)
        for j,i in enumerate(part):
            r=data['rows'][int(i)];family=r['family'];s=r['stage']
            measured_mm=float(data['motion'][int(i),:3].norm()*1e3)
            motion_bin=('still_lt1mm' if measured_mm<1 else
                        'small_1to5mm' if measured_mm<5 else 'large_ge5mm')
            key=(family,r['split'],r['stage'],int(np.argmax(r['task'][2:4])),
                 int(np.argmax(r['task'][4:9])) if family=='sfp' else r.get('card_count'),motion_bin)
            groups[key]['action_mm'].append(float(mm[j]));
            groups[key]['action_deg'].append(float(action_deg[j]));
            if family=='sc':
                groups[key]['subgoal_mm'].append(float(sub_mm[j]));
                groups[key]['subgoal_deg'].append(float(sub_deg[j]));
                groups[key]['stage_correct'].append(int(pred_stage[j]==s));
                stage_predictions[int(i)]=int(pred_stage[j])
            elif model.task_route_film:
                groups[key]['route_stage_correct'].append(
                    int(pred_stage[j]==data['route_stage'][int(i)]))
            elif data['handoff'][int(i)]>=0:
                groups[key]['subgoal_mm'].append(float(sub_mm[j]));
                groups[key]['subgoal_deg'].append(float(sub_deg[j]));
            if data['handoff'][int(i)]>=0:
                truth_handoff=bool(data['handoff'][int(i)]>=.5)
                pred_handoff=bool(handoff[j].sigmoid()>=.5)
                counts=handoff_by_family[family]
                counts['rows']+=1
                counts['positives']+=int(truth_handoff)
                counts[('tp' if truth_handoff else 'fp') if pred_handoff else
                       ('fn' if truth_handoff else 'tn')]+=1
            conf[s,int(pred_stage[j])]+=1
    overall={}
    for family in ('sc','sfp'):
        subset=[v for k,v in groups.items() if k[0]==family]
        def collect(name):return [item for group in subset for item in group.get(name,[])]
        act=collect('action_mm');sub=collect('subgoal_mm');correct=collect('stage_correct')
        actdeg=collect('action_deg');subdeg=collect('subgoal_deg')
        if not act:
            overall[family]=None
            continue
        overall[family]=dict(rows=len(act),action_first_mm_p95=float(np.quantile(act,.95)),
                             action_orientation_deg_p95=float(np.quantile(actdeg,.95)),
                             subgoal_translation_mm_p95=float(np.quantile(sub,.95)) if sub else None,
                             subgoal_orientation_deg_p95=float(np.quantile(subdeg,.95)) if subdeg else None,
                             stage_accuracy=float(np.mean(correct)) if correct else None,
                             route_stage_accuracy=(float(np.mean(collect('route_stage_correct')))
                                                   if collect('route_stage_correct') else None))
    by_group={str(k):{name:(float(np.mean(vals)) if name in ('stage_correct','route_stage_correct') else
                             float(np.quantile(vals,.95)))
                       for name,vals in v.items()} for k,v in groups.items()}
    by_sc_stage={}
    for stage in range(6):
        subset=[values for key,values in groups.items() if key[0]=='sc' and key[2]==stage]
        if not subset:continue
        combined={name:[item for values in subset for item in values.get(name,[])]
                  for name in ('action_mm','action_deg','subgoal_mm','subgoal_deg','stage_correct')}
        by_sc_stage[str(stage)]=dict(rows=len(combined['stage_correct']),
            stage_accuracy=float(np.mean(combined['stage_correct'])),
            action_first_mm_p95=float(np.quantile(combined['action_mm'],.95)),
            subgoal_translation_mm_p95=float(np.quantile(combined['subgoal_mm'],.95)),
            subgoal_orientation_deg_p95=float(np.quantile(combined['subgoal_deg'],.95)))
    episode_stages=defaultdict(list)
    for i,pred in stage_predictions.items():
        row=data['rows'][i];episode_stages[row['episode']].append((row['time_s'],pred))
    stage_skips=[]
    for ep,sequence in episode_stages.items():
        ordered=[stage for _,stage in sorted(sequence)]
        for a,b in zip(ordered,ordered[1:]):
            if b>a+1:stage_skips.append(dict(episode=ep,from_stage=a,to_stage=b))
    for counts in handoff_by_family.values():
        tp,fp,fn=(counts[k] for k in ('tp','fp','fn'))
        counts['precision']=tp/(tp+fp) if tp+fp else 0.
        counts['recall']=tp/(tp+fn) if tp+fn else 0.
    return dict(overall=overall,by_group=by_group,by_sc_stage=by_sc_stage,
                sc_handoff_at_p05=handoff_by_family['sc'],
                sfp_handoff_at_p05=handoff_by_family['sfp'],
                stage_confusion=conf.tolist(),
                stage_skips=stage_skips)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda',
                   help='CPU fallback when shared GPUs are occupied')
    p.add_argument('--no-history',action='store_true')
    p.add_argument('--raw-state-skip',action='store_true',
                   help='Targeted repair: expose measured TCP xyz and 10D task directly to temporal head')
    p.add_argument('--full-state-skip',action='store_true',
                   help='Expose normalized full measured 32D state and 10D task to every history token')
    p.add_argument('--long-history',action='store_true',
                   help='Use causal image tokens up to 60 s instead of the prior 16 s window')
    p.add_argument('--no-route-heads',action='store_true',
                   help='Matched action-only ablation without stage/subgoal/handoff conditioning or losses')
    p.add_argument('--task-family',choices=('both','sfp','sc'),default='both')
    p.add_argument('--balance-sfp-targets',action='store_true',
                   help='Give each requested SFP card/port identity equal training mass')
    p.add_argument('--paired-sfp-mass',type=float,default=0.,
                   help='Within each SFP request, sampler mass for same-board alternate-request scenes')
    p.add_argument('--zero-previous-command',action='store_true',
                   help='Remove teacher-command history from both training and validation')
    p.add_argument('--previous-command-dropout',type=float,default=0.,
                   help='Training-only probability of masking each previous teacher chunk')
    p.add_argument('--previous-command-noise',type=float,default=0.,
                   help='Training-only Gaussian noise in normalized action units')
    p.add_argument('--stage-aux-loss',type=float,default=0.,
                   help='Stage supervision on action-only head; stage prediction is not an actor input')
    p.add_argument('--task-route-film',action='store_true',
                   help='Modulate each visual token by requested SFP identity and route stage')
    p.add_argument('--sfp-route-stage-loss',type=float,default=0.,
                   help='Supervise seven discrete one-second measured transport directions')
    p.add_argument('--spatial-cache',type=Path,
                   help='Frozen 4x5 layer3 SFP camera-grid cache in identical temporal row order')
    p.add_argument('--spatial-request-attention',action='store_true',
                   help='Use requested-card query over current three-camera spatial grid')
    p.add_argument('--route-goal-model',type=Path,
                   help='Frozen RGB+TCP-pose+task prediction of one-second measured TCP motion')
    p.add_argument('--route-goal-values',type=Path,
                   help='Frozen selected-port offsets aligned to temporal cache rows')
    p.add_argument('--route-goal-zero-control',action='store_true',
                   help='Matched same-architecture control with the route goal fixed to zero')
    p.add_argument('--route-goal-progress-gate-mm',type=float,
                   help='Use the early-route goal only within this measured TCP distance from episode start')
    p.add_argument('--handoff-aux-loss',type=float,default=0.,
                   help='SC transport completion supervision; handoff is not an actor input')
    p.add_argument('--handoff-positive-weight',type=float,default=8.)
    p.add_argument('--sfp-route-future-loss',type=float,default=0.,
                   help='Predict the measured TCP pose one second ahead in the current TCP frame')
    p.add_argument('--sfp-handoff-aux-loss',type=float,default=0.,
                   help='Train SFP safe-standoff proposal from offline port-relative labels')
    p.add_argument('--init-action-only',type=Path,
                   help='Warm-start route-conditioned actor from a matched action-only checkpoint')
    p.add_argument('--updates',type=int,default=3000)
    p.add_argument('--min-updates',type=int,default=1000)
    p.add_argument('--budget-minutes',type=float,default=60)
    p.add_argument('--init-temporal',type=Path,
                   help='Warm-start the same temporal architecture from a frozen prior checkpoint')
    p.add_argument('--reset-action-head',action='store_true',
                   help='After temporal warm start, reset only the complete-action decoder')
    p.add_argument('--corrective-sfp-mass',type=float,default=0.,
                   help='Fraction of SFP sampler mass reserved for admitted on-policy teacher corrections')
    p.add_argument('--selection-exclude-corrective',action='store_true',
                   help='Do not use withheld corrective validation rows to select a checkpoint')
    p.add_argument('--selection-corrective-weight',type=float,default=0.,
                   help='Add held-out corrective first-action p95 in mm to checkpoint selection')
    p.add_argument('--learning-rate',type=float,default=2e-4)
    p.add_argument('--seed',type=int,default=20261001)
    p.add_argument('--exclude-train-episode-min',type=int,
                   help='Matched old-data control: exclude new episode IDs at or above this value from optimizer sampling')
    p.add_argument('--excluded-train-indices',type=Path,
                   help='Immutable audited row-index mask; excludes action loss only, preserving causal history')
    p.add_argument('--paired-contrast-manifest',type=Path,
                   help='Preaudited same-observation alternate-request row pairs')
    p.add_argument('--paired-contrast-weight',type=float,default=0.,
                   help='Supervise full first-action difference when only the public task changes')
    p.add_argument('--frozen-reference',type=Path,
                   help='Frozen same-architecture actor for ordinary old-SFP action preservation')
    p.add_argument('--reference-old-sfp-weight',type=float,default=0.,
                   help='Trust-region loss on old unpaired SFP train observations')
    a=p.parse_args()
    if a.output.exists():p.error('Output exists; preserve prior run')
    a.output.mkdir(parents=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(8)
    if a.raw_state_skip and a.full_state_skip:p.error('Choose only one state-skip mode')
    if a.reset_action_head and not a.init_temporal:
        p.error('Action-head reset requires a temporal warm start')
    if a.selection_corrective_weight<0 or (a.selection_corrective_weight and a.selection_exclude_corrective):
        p.error('Corrective selection weight must be nonnegative and include corrective validation')
    if a.paired_contrast_weight<0:p.error('Pair weight must be nonnegative')
    if a.reference_old_sfp_weight<0 or bool(a.frozen_reference)!=bool(a.reference_old_sfp_weight):
        p.error('Reference checkpoint and positive preservation weight must be supplied together')
    if a.reference_old_sfp_weight and (not a.full_state_skip or not a.no_route_heads):
        p.error('Reference preservation supports the same full-state action-only actor only')
    if a.paired_contrast_weight and (not a.paired_contrast_manifest or not a.full_state_skip or
                                     not a.no_route_heads or a.spatial_request_attention or a.task_route_film):
        p.error('Paired contrast needs a manifest and ordinary full-state action-only actor')
    if not 0<=a.previous_command_dropout<=1 or a.previous_command_noise<0:
        p.error('Invalid previous-command augmentation')
    if a.zero_previous_command and (a.previous_command_dropout or a.previous_command_noise):
        p.error('Choose zero previous command or previous-command augmentation')
    if not 0<=a.corrective_sfp_mass<1 or a.learning_rate<=0:
        p.error('Invalid corrective SFP mass or learning rate')
    if not 0<=a.paired_sfp_mass<1 or (a.paired_sfp_mass and not a.balance_sfp_targets):
        p.error('Paired SFP mass must be in [0,1) and needs target balancing')
    if a.paired_sfp_mass and a.corrective_sfp_mass:
        p.error('Select one special SFP sampler mass')
    if (a.stage_aux_loss or a.handoff_aux_loss) and not a.no_route_heads:
        p.error('Auxiliary-only losses require --no-route-heads')
    if (a.sfp_route_future_loss or a.sfp_handoff_aux_loss) and a.no_route_heads:
        p.error('SFP route/handoff supervision requires route conditioning')
    if a.init_action_only and (a.init_temporal or a.no_route_heads):
        p.error('Action-only warm start requires one route-conditioned candidate')
    if a.task_family=='sfp' and not a.no_route_heads:
        p.error('SFP-only training has no SC route labels; use --no-route-heads')
    if a.task_route_film and (not a.full_state_skip or not a.no_route_heads or not a.sfp_route_stage_loss):
        p.error('Task route FiLM needs full state skip, action-only head and positive route loss')
    if a.spatial_request_attention and (not a.spatial_cache or not a.full_state_skip or not a.no_route_heads):
        p.error('Spatial request attention needs spatial cache, full state skip and action-only head')
    if a.route_goal_model and a.route_goal_values:
        p.error('Choose one route-goal source')
    if (a.route_goal_model or a.route_goal_values) and (not a.full_state_skip or not a.no_route_heads):
        p.error('Route goal conditioning requires full-state action-only actor')
    if a.route_goal_zero_control and not (a.route_goal_model or a.route_goal_values):
        p.error('Zero-goal control requires the same frozen route-goal source')
    if a.route_goal_progress_gate_mm is not None and (not a.route_goal_model or
                                                     a.route_goal_progress_gate_mm<=0):
        p.error('Positive progress gate requires the frozen route-goal model')
    device=a.device;data=build_data(a.cache,a.checkpoint,device,
                                  LONG_AGES if a.long_history else AGES,
                                  spatial_cache=a.spatial_cache,
                                  route_goal_model=a.route_goal_model,
                                  route_goal_zero_control=a.route_goal_zero_control,
                                  route_goal_progress_gate_mm=a.route_goal_progress_gate_mm,
                                  route_goal_values=a.route_goal_values)
    if a.zero_previous_command:
        data['prev'].zero_()
    rows=data['rows']
    excluded=set()
    if a.excluded_train_indices:
        exclusion=json.loads(a.excluded_train_indices.read_text())
        if exclusion.get('schema')!='aic_sfp_stall_loss_filter/v1' or exclusion.get('cache')!=str(a.cache):
            p.error('Training-row filter schema/cache mismatch')
        excluded=set(map(int,exclusion['indices']))
        if any(i<0 or i>=len(rows) or rows[i]['split']!='train' or rows[i]['family']!='sfp'
               for i in excluded):
            p.error('Filter contains a nontraining or non-SFP row')
    pair_train=[]
    if a.paired_contrast_manifest:
        pairs=json.loads(a.paired_contrast_manifest.read_text())
        pair_train=pairs['train_pairs']
        if a.paired_contrast_weight and len(pair_train)<30:
            p.error(f'Need at least thirty audited train pairs, got {len(pair_train)}')
        for pair in pair_train:
            left,right=pair['left_index'],pair['right_index']
            if (rows[left]['split']!='train' or rows[right]['split']!='train' or
                rows[left]['scene']!=rows[right]['scene'] or rows[left]['task']==rows[right]['task']):
                p.error('Invalid grouped paired contrast row')
        if excluded & {i for pair in pair_train
                       for i in (pair['left_index'],pair['right_index'])}:
            p.error('A paired contrast target is filtered')
    selected_family=lambda r:a.task_family=='both' or r['family']==a.task_family
    train=np.array([i for i,r in enumerate(rows) if r['split']=='train' and selected_family(r)
                    and i not in excluded
                    and not r.get('actor_prefix_context_only',False)
                    and (a.exclude_train_episode_min is None or
                         int(r['episode'])<a.exclude_train_episode_min)],dtype=np.int64)
    val=np.array([i for i,r in enumerate(rows) if r['split']=='validation' and selected_family(r)
                  and not r.get('actor_prefix_context_only',False)],dtype=np.int64)
    counts=Counter((rows[i]['family'],rows[i]['episode'],rows[i]['stage']) for i in train)
    episodes=Counter((rows[i]['family'],rows[i]['stage'],rows[i]['episode']) for i in train)
    episode_count_by_stage=Counter()
    for family,stage,episode in episodes:episode_count_by_stage[(family,stage)]+=1
    sfp_target_episodes=defaultdict(set)
    for i in train:
        r=rows[i]
        if r['family']=='sfp':
            key=(int(np.argmax(r['task'][4:9])),int(np.argmax(r['task'][2:4])))
            sfp_target_episodes[key].add(r['episode'])
    if a.balance_sfp_targets and len(sfp_target_episodes)!=10:
        p.error(f'Balanced SFP sampler requires all ten requests; found {len(sfp_target_episodes)}')
    def sample_weight(i):
        r=rows[i];denom=counts[(r['family'],r['episode'],r['stage'])]
        if r['family']=='sfp' and a.balance_sfp_targets:
            key=(int(np.argmax(r['task'][4:9])),int(np.argmax(r['task'][2:4])))
            return 1/(2*len(sfp_target_episodes)*len(sfp_target_episodes[key])*denom)
        return 1/(2*(6 if r['family']=='sc' else 1)*
                  episode_count_by_stage[(r['family'],r['stage'])]*denom)
    weights=np.array([sample_weight(i) for i in train],dtype=np.float64)
    weights/=weights.sum()
    paired_episode_keys=set()
    if a.paired_sfp_mass:
        requests_by_scene=defaultdict(set)
        for i in train:
            r=rows[i]
            if r['family']=='sfp':
                requests_by_scene[r['scene']].add((int(np.argmax(r['task'][4:9])),
                                                    int(np.argmax(r['task'][2:4]))))
        paired_scenes={scene for scene,requests in requests_by_scene.items() if len(requests)>1}
        for key in sorted(sfp_target_episodes):
            target_indices=[j for j,i in enumerate(train) if rows[i]['family']=='sfp' and
                            (int(np.argmax(rows[i]['task'][4:9])),
                             int(np.argmax(rows[i]['task'][2:4])))==key]
            paired=[j for j in target_indices if rows[train[j]]['scene'] in paired_scenes]
            unpaired=[j for j in target_indices if rows[train[j]]['scene'] not in paired_scenes]
            if not paired or not unpaired:
                p.error(f'Paired sampler needs paired and unpaired episodes for {key}')
            target_mass=weights[target_indices].sum()
            weights[paired]*=target_mass*a.paired_sfp_mass/weights[paired].sum()
            weights[unpaired]*=target_mass*(1-a.paired_sfp_mass)/weights[unpaired].sum()
            paired_episode_keys.update((rows[train[j]]['scene'],key) for j in paired)
        weights/=weights.sum()
    corrective_indices=[j for j,i in enumerate(train)
                        if rows[i]['family']=='sfp' and rows[i].get('corrective_teacher',False)]
    if a.corrective_sfp_mass:
        sfp_old_indices=[j for j,i in enumerate(train)
                         if rows[i]['family']=='sfp' and not rows[i].get('corrective_teacher',False)]
        sc_indices=[j for j,i in enumerate(train) if rows[i]['family']=='sc']
        if not corrective_indices or not sfp_old_indices or not sc_indices:
            p.error('Corrective mass requires old SFP, corrective SFP and SC training rows')
        for indices,mass in ((sc_indices,.5),
                             (sfp_old_indices,.5*(1-a.corrective_sfp_mass)),
                             (corrective_indices,.5*a.corrective_sfp_mass)):
            weights[indices]*=mass/weights[indices].sum()
        weights/=weights.sum()
    # Validation is episode-grouped; sample every stage in every SC episode.
    val_groups=defaultdict(list)
    for i in val:
        if a.selection_exclude_corrective and rows[i].get('corrective_teacher',False):
            continue
        val_groups[(rows[i]['family'],rows[i]['episode'],rows[i]['stage'])].append(i)
    val_selected=[]
    for key,items in val_groups.items():
        cap=8 if key[0]=='sc' else 12
        val_selected.extend(np.array(items)[np.linspace(0,len(items)-1,min(cap,len(items))).astype(int)].tolist())
    val_selected=np.array(sorted(set(val_selected)),dtype=np.int64)
    model=TemporalTransport(data['mean'],data['std'],
                            raw_state_skip=a.raw_state_skip or a.full_state_skip,
                            feature_dim=data['features'].shape[1],
                            state_task_dim=42 if a.full_state_skip else 13,
                            route_conditioning=not a.no_route_heads,
                            task_route_film=a.task_route_film,
                            spatial_request_attention=a.spatial_request_attention,
                            route_goal_conditioning=bool(a.route_goal_model or a.route_goal_values)).to(device)
    if a.init_temporal:
        initial=torch.load(a.init_temporal,map_location='cpu',weights_only=False)
        old=initial['source_manifest']
        if (old['feature_dim']!=data['features'].shape[1] or
            old['long_history']!=a.long_history or
            old['route_conditioning']!=(not a.no_route_heads) or
            old['full_state_skip']!=a.full_state_skip):
            p.error('Temporal initialization architecture mismatch')
        load_result=model.load_state_dict(initial['model'],strict=not (
            a.task_route_film or a.spatial_request_attention or a.route_goal_model or a.route_goal_values))
        if (a.route_goal_model or a.route_goal_values) and (set(load_result.missing_keys)!={'route_goal_proj.weight'} or
                                   load_result.unexpected_keys):
            p.error(f'Unexpected route-goal warm-start keys: {load_result}')
        if a.task_route_film:
            with torch.no_grad():
                model.action[1].weight[:,256:263].zero_()
        if a.reset_action_head:
            for module in model.action.modules():
                if module is not model.action and hasattr(module,'reset_parameters'):
                    module.reset_parameters()
    reference=None;old_stability_indices=np.empty(0,dtype=np.int64)
    stability_rng=np.random.default_rng(a.seed+881)
    if a.frozen_reference:
        frozen=torch.load(a.frozen_reference,map_location='cpu',weights_only=False)
        old=frozen['source_manifest']
        if (old['feature_dim']!=data['features'].shape[1] or not old['long_history'] or
            not old['full_state_skip'] or old['route_conditioning']):
            p.error('Frozen reference architecture mismatch')
        reference=TemporalTransport(data['mean'],data['std'],raw_state_skip=True,
                                    feature_dim=data['features'].shape[1],state_task_dim=42,
                                    route_conditioning=False).to(device).eval()
        reference.load_state_dict(frozen['model'])
        for param in reference.parameters():param.requires_grad_(False)
        paired_scene_set={pair['scene'] for pair in pair_train}
        old_stability_indices=np.asarray([i for i in train if rows[i]['family']=='sfp' and
            int(rows[i]['episode'])<2400000 and rows[i]['scene'] not in paired_scene_set],dtype=np.int64)
        if len(old_stability_indices)<100:
            p.error('Insufficient old unpaired SFP observations for preservation')
    if a.init_action_only:
        initial=torch.load(a.init_action_only,map_location='cpu',weights_only=False)
        old=initial['source_manifest']
        if (old['feature_dim']!=data['features'].shape[1] or
            old['long_history']!=a.long_history or old['route_conditioning'] or
            old['full_state_skip']!=a.full_state_skip):
            p.error('Action-only initialization architecture mismatch')
        model.load_state_dict(initial['model'],strict=True)
        with torch.no_grad():
            model.action[-1].weight[:,256:].zero_()
    opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-4)
    manifest=dict(schema='aic_temporal_transport_probe/v2',history=not a.no_history,
                  exclude_train_episode_min=a.exclude_train_episode_min,
                  excluded_train_indices_sha256=(sha(a.excluded_train_indices)
                                                  if a.excluded_train_indices else None),
                  excluded_train_rows=len(excluded),
                  training_device=device,
                  paired_contrast_manifest_sha256=(sha(a.paired_contrast_manifest)
                                                   if a.paired_contrast_manifest else None),
                  paired_contrast_weight=a.paired_contrast_weight,
                  paired_contrast_train_pairs=len(pair_train),
                  frozen_reference_sha256=sha(a.frozen_reference) if a.frozen_reference else None,
                  reference_old_sfp_weight=a.reference_old_sfp_weight,
                  reference_old_sfp_train_rows=len(old_stability_indices),
                  raw_state_skip=a.raw_state_skip,
                  full_state_skip=a.full_state_skip,long_history=a.long_history,
                  history_ages_s=list(LONG_AGES if a.long_history else AGES),
                  feature_dim=int(data['features'].shape[1]),
                  task_family=a.task_family,route_conditioning=not a.no_route_heads,
                  balance_sfp_targets=a.balance_sfp_targets,
                  paired_sfp_mass=a.paired_sfp_mass,
                  paired_scene_request_groups=len(paired_episode_keys),
                  corrective_sfp_mass=a.corrective_sfp_mass,
                  corrective_sfp_train_rows=len(corrective_indices),
                  selection_exclude_corrective=a.selection_exclude_corrective,
                  selection_corrective_weight=a.selection_corrective_weight,
                  reset_action_head=a.reset_action_head,
                  withheld_corrective_validation_rows=sum(
                      rows[i].get('corrective_teacher',False) for i in val),
                  init_temporal_sha256=sha(a.init_temporal) if a.init_temporal else None,
                  learning_rate=a.learning_rate,
                  sfp_training_episodes_by_target={str(k):len(v) for k,v in sorted(sfp_target_episodes.items())},
                  zero_previous_command=a.zero_previous_command,
                  previous_command_dropout=a.previous_command_dropout,
                  previous_command_noise=a.previous_command_noise,
                  stage_aux_loss=a.stage_aux_loss,
                  task_route_film=a.task_route_film,
                  sfp_route_stage_loss=a.sfp_route_stage_loss,
                  spatial_request_attention=a.spatial_request_attention,
                  route_goal_conditioning=bool(a.route_goal_model or a.route_goal_values),
                  route_goal_model_sha256=sha(a.route_goal_model) if a.route_goal_model else None,
                  route_goal_values_sha256=sha(a.route_goal_values) if a.route_goal_values else None,
                  route_goal_values_path=str(a.route_goal_values) if a.route_goal_values else None,
                  route_goal_zero_control=a.route_goal_zero_control,
                  route_goal_progress_gate_mm=a.route_goal_progress_gate_mm,
                  spatial_cache=str(a.spatial_cache) if a.spatial_cache else None,
                  handoff_aux_loss=a.handoff_aux_loss,
                  handoff_positive_weight=a.handoff_positive_weight,
                  sfp_route_future_loss=a.sfp_route_future_loss,
                  sfp_handoff_aux_loss=a.sfp_handoff_aux_loss,
                  init_action_only_sha256=sha(a.init_action_only) if a.init_action_only else None,
                  handoff_prior_required=bool(a.sfp_handoff_aux_loss),
                  architecture='frozen 3-camera ResNet18/context; 2-layer 256D temporal transformer; full-action, SC-stage, full-TCP-subgoal, handoff heads',
                  train_episodes=dict(Counter(family for family,_ in
                                              {(r['family'],r['episode']) for r in rows if r['split']=='train'})),
                  validation_episodes=dict(Counter(family for family,_ in
                                                   {(r['family'],r['episode']) for r in rows if r['split']=='validation'})),
                  train_rows=len(train),validation_rows=len(val),validation_sample_rows=len(val_selected),
                  update_cap=a.updates,wall_minutes=a.budget_minutes,min_updates=a.min_updates,
                  early_stop='five 250-update checks without joint validation score improvement after min updates',
                  gate='SC stage accuracy >= .90 and SC subgoal translation p95 <=20mm; no stage skips; live gate separate',
                  cache_manifest_sha256=sha(a.cache/'manifest.json'),checkpoint_sha256=sha(a.checkpoint),
                  camera_offsets_sha256=sha(a.cache/'camera_offsets.npy'),
                  target_inputs='offline labels only; no teacher stage, plan or geometry is supplied to model',
                  final_scenes_opened=False,rl_started=False)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    best=float('inf');best_update=0;bad=0;history=[];started=time.monotonic();reason='update_cap'
    for update in range(1,a.updates+1):
        if time.monotonic()-started>a.budget_minutes*60:reason='wall_budget';break
        selected=np.random.choice(train,size=64,replace=True,p=weights)
        ind=torch.as_tensor(selected,device=device)
        model.train();opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device,dtype=torch.bfloat16,
                            enabled=device=='cuda'):
            action,stage,subgoal,handoff=predict(
                model,data,ind,a.no_history,
                previous_dropout=a.previous_command_dropout,
                previous_noise=a.previous_command_noise)
            truth=(data['target'][ind]-model.target_mean)/model.target_std
            action_loss=nn.functional.smooth_l1_loss(action,truth,beta=.2)
            sc=data['stage'][ind]<6
            stage_loss=nn.functional.cross_entropy(stage,data['stage'][ind])
            sfp=~sc
            route_stage_loss=(nn.functional.cross_entropy(stage[sfp],data['route_stage'][ind][sfp])
                              if sfp.any() and a.sfp_route_stage_loss else action_loss*0)
            if sc.any():
                sub_loss=nn.functional.smooth_l1_loss(subgoal[sc,:3]/.1,
                                                      data['subgoal'][ind][sc,:3]/.1,beta=.1)
                sub_rot=nn.functional.mse_loss(subgoal[sc,3:],data['subgoal'][ind][sc,3:])
                hand_loss=nn.functional.binary_cross_entropy_with_logits(
                    handoff[sc],data['handoff'][ind][sc],
                    pos_weight=torch.tensor(a.handoff_positive_weight,device=device))
            else:sub_loss=sub_rot=hand_loss=action_loss*0
            if sfp.any() and a.sfp_route_future_loss:
                sfp_sub_loss=nn.functional.smooth_l1_loss(
                    subgoal[sfp,:3]/.1,data['subgoal'][ind][sfp,:3]/.1,beta=.1)
                sfp_sub_rot=nn.functional.mse_loss(
                    subgoal[sfp,3:],data['subgoal'][ind][sfp,3:])
            else:sfp_sub_loss=sfp_sub_rot=action_loss*0
            if sfp.any() and a.sfp_handoff_aux_loss:
                if (data['handoff'][ind][sfp]<0).any():
                    raise ValueError('Missing SFP handoff label')
                sfp_hand_loss=nn.functional.binary_cross_entropy_with_logits(
                    handoff[sfp],data['handoff'][ind][sfp])
            else:sfp_hand_loss=action_loss*0
            pair_loss=action_loss*0
            if a.paired_contrast_weight:
                sampled=np.random.choice(len(pair_train),size=16,replace=True)
                left=torch.as_tensor([pair_train[j]['left_index'] for j in sampled],device=device)
                right=torch.as_tensor([pair_train[j]['right_index'] for j in sampled],device=device)
                pred_left,pred_right=paired_counterfactual_first(model,data,left,right)
                truth_delta=((data['target'][left,0]-data['target'][right,0]) /
                             model.target_std)
                pair_loss=nn.functional.smooth_l1_loss(
                    pred_left[:,:3]-pred_right[:,:3],truth_delta[:,:3],beta=.2)
            stability_loss=action_loss*0
            if reference is not None:
                sampled_stability=stability_rng.choice(old_stability_indices,size=32,replace=True)
                stability_ind=torch.as_tensor(sampled_stability,device=device)
                candidate_action,*_=predict(model,data,stability_ind,a.no_history)
                with torch.no_grad():reference_action,*_=predict(
                    reference,data,stability_ind,a.no_history)
                stability_loss=nn.functional.smooth_l1_loss(
                    candidate_action,reference_action,beta=.2)
            loss=(action_loss+a.stage_aux_loss*stage_loss+a.handoff_aux_loss*hand_loss+
                  a.sfp_route_stage_loss*route_stage_loss
                  if a.no_route_heads else
                  action_loss+.5*stage_loss+.7*sub_loss+.1*sub_rot+.1*hand_loss+
                  a.sfp_route_future_loss*(sfp_sub_loss+.1*sfp_sub_rot)+
                  a.sfp_handoff_aux_loss*sfp_hand_loss)
            loss=(loss+a.paired_contrast_weight*pair_loss+
                  a.reference_old_sfp_weight*stability_loss)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if update%250==0 or update==a.updates:
            result=evaluate(model,data,val_selected,a.no_history)
            if a.previous_command_dropout or a.previous_command_noise:
                result_zero=evaluate(model,data,val_selected,a.no_history,
                                     force_zero_previous=True)
                result['zero_previous_validation']=result_zero['overall']
            scv=result['overall']['sc'];sfpv=result['overall']['sfp']
            if a.no_route_heads:
                score=sum(result['overall'][family]['action_first_mm_p95']
                          for family in ('sc','sfp') if a.task_family in ('both',family))
                if a.selection_corrective_weight:
                    corrective_val=np.asarray([i for i in val if rows[i].get('corrective_teacher',False)],
                                              dtype=np.int64)
                    if len(corrective_val)<20:
                        p.error('Not enough corrective validation rows for selection')
                    corrective_score=evaluate(model,data,corrective_val,a.no_history)
                    score+=a.selection_corrective_weight*corrective_score['overall']['sfp']['action_first_mm_p95']
                if a.task_route_film and sfpv:
                    score+=2*(1-sfpv['route_stage_accuracy'])
                if a.previous_command_dropout or a.previous_command_noise:
                    score+=sum(result_zero['overall'][family]['action_first_mm_p95']
                               for family in ('sc','sfp') if a.task_family in ('both',family))
            else:
                score=((1-scv['stage_accuracy'])*50+scv['subgoal_translation_mm_p95']+
                       .3*scv['action_first_mm_p95']+
                       ((sfpv['action_first_mm_p95']+
                         .2*sfpv['subgoal_translation_mm_p95'])
                        if sfpv and a.sfp_route_future_loss else
                        .1*sfpv['action_first_mm_p95'] if sfpv else 0.))
            entry=dict(update=update,elapsed_min=(time.monotonic()-started)/60,
                       score=score,validation=result,train_loss=float(loss))
            history.append(entry)
            print(json.dumps(dict(update=update,elapsed_min=entry['elapsed_min'],
                                  sc=scv,sfp=sfpv,score=score)),flush=True)
            if score<best-1e-3:
                best=score;best_update=update;bad=0
                torch.save(dict(model=model.state_dict(),update=update,validation=result,
                                history_enabled=not a.no_history,source_manifest=manifest,
                                route_goal_ridge=({k:torch.as_tensor(v) for k,v in
                                    dict(np.load(a.route_goal_model)).items() if k in
                                    ('center','scale','target_mean','coefficients')}
                                    if a.route_goal_model else None),
                                final_scenes_opened=False,rl_started=False),a.output/'best.pt')
            else:bad+=1
            if update>=a.min_updates and bad>=5:reason='validation_plateau';break
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    selected_report=next(x['validation'] for x in history if x['update']==best_update)
    summary=dict(best_update=best_update,updates_completed=update,stop_reason=reason,
                 best_validation=selected_report,
                 offline_gate_passed=(not a.no_route_heads and a.task_family!='sfp' and
                                     selected_report['overall']['sc']['stage_accuracy']>=.90 and
                                     selected_report['overall']['sc']['subgoal_translation_mm_p95']<=20. and
                                     not selected_report['stage_skips']),
                 autonomous_gate_passed=False,final_scenes_opened=False,rl_started=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='best_validation'}),flush=True)


if __name__=='__main__':main()
