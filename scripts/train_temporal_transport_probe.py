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


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


class TemporalTransport(nn.Module):
    def __init__(self, target_mean, target_std, raw_state_skip=False):
        super().__init__()
        self.raw_state_skip=raw_state_skip
        self.register_buffer('target_mean',torch.as_tensor(target_mean,dtype=torch.float32))
        self.register_buffer('target_std',torch.as_tensor(target_std,dtype=torch.float32))
        input_dim=304+(13 if raw_state_skip else 0)
        self.input=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,256),nn.GELU())
        layer=nn.TransformerEncoderLayer(256,8,512,dropout=.05,
                                         activation='gelu',batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(layer,2)
        self.stage=nn.Linear(256,7)
        self.subgoal=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,9))
        self.handoff=nn.Linear(256,1)
        self.action=nn.Sequential(nn.LayerNorm(256+7+9),nn.Linear(272,256),
                                  nn.GELU(),nn.Linear(256,36))

    def forward(self,features,previous_action,measured_motion,ages,gaps,camera_clock,mask,
                state_task=None):
        parts=(features,previous_action,measured_motion,ages[...,None],gaps[...,None],camera_clock)
        if self.raw_state_skip:
            if state_task is None:raise ValueError('Measured TCP/task skip is required')
            parts=(*parts,state_task)
        x=self.input(torch.cat(parts,-1))
        x=self.temporal(x,src_key_padding_mask=~mask)
        # Histories are right aligned; the last token is always current.
        current=x[:,-1]
        stage=self.stage(current)
        subgoal=self.subgoal(current)
        handoff=self.handoff(current).squeeze(-1)
        action=self.action(torch.cat((current,stage.softmax(-1),subgoal),-1))
        return action.reshape(-1,4,9),stage,subgoal,handoff


def build_data(root,checkpoint,device):
    rows=[json.loads(line) for line in (root/'rows.jsonl').open()]
    features=np.load(root/'features.npy',mmap_mode='r')
    camera_clock=np.load(root/'camera_offsets.npy',mmap_mode='r')
    assert len(rows)==len(features)
    assert camera_clock.shape==(len(rows),4)
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
    mean=cp['model']['target_mean'].numpy();std=cp['model']['target_std'].numpy()
    ep=defaultdict(list)
    for i,r in enumerate(rows):ep[(r['family'],r['episode'])].append(i)
    n=len(rows);hist=np.full((n,len(AGES)),-1,dtype=np.int64)
    ages=np.zeros_like(hist,dtype=np.float32);gaps=ages.copy()
    previous=np.zeros((n,36),dtype=np.float32)
    motion=np.zeros((n,6),dtype=np.float32)
    state=np.asarray([r['state'] for r in rows],dtype=np.float32)
    task=np.asarray([r['task'] for r in rows],dtype=np.float32)
    normalized_xyz=((state[:,:3]-cp['model']['state_mean'][:3].numpy()) /
                    cp['model']['state_std'][:3].numpy())
    state_task=np.concatenate((normalized_xyz,task),axis=-1)
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
            for age in AGES:
                limit=times[i]-age
                j=np.searchsorted(times[indices[:k+1]],limit,side='right')-1
                if j>=0 and (age==0 or times[i]-times[indices[j]]<=20.):
                    candidate=indices[j]
                    if candidate not in chosen:chosen.append(candidate)
            chosen.sort()
            offset=len(AGES)-len(chosen)
            for q,c in enumerate(chosen,start=offset):
                hist[i,q]=c;ages[i,q]=min(20.,times[i]-times[c])/4.
                gaps[i,q]=0 if q==offset else min(20.,times[c]-times[hist[i,q-1]])/4.
    hidx=torch.as_tensor(np.maximum(hist,0),device=device)
    mask=torch.as_tensor(hist>=0,device=device)
    data=dict(rows=rows,features=torch.as_tensor(np.array(features,copy=True),device=device),
              prev=torch.as_tensor(previous,device=device),motion=torch.as_tensor(motion,device=device),
              camera_clock=torch.as_tensor(np.array(camera_clock,copy=True),device=device),
              state_task=torch.as_tensor(state_task,device=device),
              age=torch.as_tensor(ages,device=device),gap=torch.as_tensor(gaps,device=device),
              history=hidx,mask=mask,target=torch.as_tensor(target,device=device),
              stage=torch.as_tensor([r['stage'] for r in rows],device=device),
              subgoal=torch.as_tensor(subgoal,device=device),
              handoff=torch.as_tensor([r['handoff'] for r in rows],device=device,dtype=torch.float32),
              mean=mean,std=std)
    return data


def predict(model,data,indices,no_history):
    h=data['history'][indices];mask=data['mask'][indices]
    if no_history:
        h=h[:,-1:];mask=mask[:,-1:]
    features=data['features'][h].float()
    prior=data['prev'][h].float();motion=data['motion'][h].float()
    camera_clock=data['camera_clock'][h].float()
    state_task=data['state_task'][h].float() if model.raw_state_skip else None
    camera_clock[...,:3]/=.05
    ages=data['age'][indices];gaps=data['gap'][indices]
    if no_history:ages=ages[:,-1:];gaps=gaps[:,-1:]
    # Right-aligned causal windows: pad values are ignored by attention.
    return model(features,prior,motion,ages,gaps,camera_clock,mask,state_task)


@torch.inference_mode()
def evaluate(model,data,indices,no_history,batch=256):
    model.eval();groups=defaultdict(lambda:defaultdict(list));conf=np.zeros((7,7),dtype=int)
    handoff_counts=dict(tp=0,fp=0,fn=0,tn=0,positives=0,rows=0)
    stage_predictions={}
    for part in np.array_split(indices,max(1,(len(indices)+batch-1)//batch)):
        if not len(part):continue
        ind=torch.as_tensor(part,device=data['stage'].device)
        action,stage,subgoal,handoff=predict(model,data,ind,no_history)
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
                truth_handoff=bool(data['handoff'][int(i)]>=.5)
                pred_handoff=bool(handoff[j].sigmoid()>=.5)
                handoff_counts['rows']+=1
                handoff_counts['positives']+=int(truth_handoff)
                handoff_counts[('tp' if truth_handoff else 'fp') if pred_handoff else
                               ('fn' if truth_handoff else 'tn')]+=1
            conf[s,int(pred_stage[j])]+=1
    overall={}
    for family in ('sc','sfp'):
        subset=[v for k,v in groups.items() if k[0]==family]
        def collect(name):return [item for group in subset for item in group.get(name,[])]
        act=collect('action_mm');sub=collect('subgoal_mm');correct=collect('stage_correct')
        actdeg=collect('action_deg');subdeg=collect('subgoal_deg')
        overall[family]=dict(rows=len(act),action_first_mm_p95=float(np.quantile(act,.95)),
                             action_orientation_deg_p95=float(np.quantile(actdeg,.95)),
                             subgoal_translation_mm_p95=float(np.quantile(sub,.95)) if sub else None,
                             subgoal_orientation_deg_p95=float(np.quantile(subdeg,.95)) if subdeg else None,
                             stage_accuracy=float(np.mean(correct)) if correct else None)
    by_group={str(k):{name:(float(np.mean(vals)) if name=='stage_correct' else
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
    tp,fp,fn=(handoff_counts[k] for k in ('tp','fp','fn'))
    handoff_counts['precision']=tp/(tp+fp) if tp+fp else 0.
    handoff_counts['recall']=tp/(tp+fn) if tp+fn else 0.
    return dict(overall=overall,by_group=by_group,by_sc_stage=by_sc_stage,
                sc_handoff_at_p05=handoff_counts,
                stage_confusion=conf.tolist(),
                stage_skips=stage_skips)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-history',action='store_true')
    p.add_argument('--raw-state-skip',action='store_true',
                   help='Targeted repair: expose measured TCP xyz and 10D task directly to temporal head')
    p.add_argument('--updates',type=int,default=3000)
    p.add_argument('--min-updates',type=int,default=1000)
    p.add_argument('--budget-minutes',type=float,default=60)
    p.add_argument('--seed',type=int,default=20261001)
    a=p.parse_args()
    if a.output.exists():p.error('Output exists; preserve prior run')
    a.output.mkdir(parents=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(8)
    device='cuda';data=build_data(a.cache,a.checkpoint,device)
    rows=data['rows'];train=np.array([i for i,r in enumerate(rows) if r['split']=='train'],dtype=np.int64)
    val=np.array([i for i,r in enumerate(rows) if r['split']=='validation'],dtype=np.int64)
    counts=Counter((r['family'],r['episode'],r['stage']) for r in rows if r['split']=='train')
    episodes=Counter((r['family'],r['stage'],r['episode']) for r in rows if r['split']=='train')
    episode_count_by_stage=Counter()
    for family,stage,episode in episodes:episode_count_by_stage[(family,stage)]+=1
    weights=np.array([1/(2*(6 if rows[i]['family']=='sc' else 1)*
                          episode_count_by_stage[(rows[i]['family'],rows[i]['stage'])]*
                          counts[(rows[i]['family'],rows[i]['episode'],rows[i]['stage'])])
                      for i in train],dtype=np.float64)
    weights/=weights.sum()
    # Validation is episode-grouped; sample every stage in every SC episode.
    val_groups=defaultdict(list)
    for i in val:val_groups[(rows[i]['family'],rows[i]['episode'],rows[i]['stage'])].append(i)
    val_selected=[]
    for key,items in val_groups.items():
        cap=8 if key[0]=='sc' else 12
        val_selected.extend(np.array(items)[np.linspace(0,len(items)-1,min(cap,len(items))).astype(int)].tolist())
    val_selected=np.array(sorted(set(val_selected)),dtype=np.int64)
    model=TemporalTransport(data['mean'],data['std'],raw_state_skip=a.raw_state_skip).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    manifest=dict(schema='aic_temporal_transport_probe/v1',history=not a.no_history,
                  raw_state_skip=a.raw_state_skip,
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
        with torch.autocast('cuda',dtype=torch.bfloat16):
            action,stage,subgoal,handoff=predict(model,data,ind,a.no_history)
            truth=(data['target'][ind]-model.target_mean)/model.target_std
            action_loss=nn.functional.smooth_l1_loss(action,truth,beta=.2)
            sc=data['stage'][ind]<6
            stage_loss=nn.functional.cross_entropy(stage,data['stage'][ind])
            if sc.any():
                sub_loss=nn.functional.smooth_l1_loss(subgoal[sc,:3]/.1,
                                                      data['subgoal'][ind][sc,:3]/.1,beta=.1)
                sub_rot=nn.functional.mse_loss(subgoal[sc,3:],data['subgoal'][ind][sc,3:])
                hand_loss=nn.functional.binary_cross_entropy_with_logits(
                    handoff[sc],data['handoff'][ind][sc],pos_weight=torch.tensor(8.,device=device))
            else:sub_loss=sub_rot=hand_loss=action_loss*0
            loss=action_loss+.5*stage_loss+.7*sub_loss+.1*sub_rot+.1*hand_loss
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if update%250==0 or update==a.updates:
            result=evaluate(model,data,val_selected,a.no_history)
            scv=result['overall']['sc'];sfpv=result['overall']['sfp']
            score=((1-scv['stage_accuracy'])*50+scv['subgoal_translation_mm_p95']+
                   .3*scv['action_first_mm_p95']+.1*sfpv['action_first_mm_p95'])
            entry=dict(update=update,elapsed_min=(time.monotonic()-started)/60,
                       score=score,validation=result,train_loss=float(loss))
            history.append(entry)
            print(json.dumps(dict(update=update,elapsed_min=entry['elapsed_min'],
                                  sc=scv,sfp=sfpv,score=score)),flush=True)
            if score<best-1e-3:
                best=score;best_update=update;bad=0
                torch.save(dict(model=model.state_dict(),update=update,validation=result,
                                history_enabled=not a.no_history,source_manifest=manifest,
                                final_scenes_opened=False,rl_started=False),a.output/'best.pt')
            else:bad+=1
            if update>=a.min_updates and bad>=5:reason='validation_plateau';break
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    selected_report=next(x['validation'] for x in history if x['update']==best_update)
    summary=dict(best_update=best_update,updates_completed=update,stop_reason=reason,
                 best_validation=selected_report,
                 offline_gate_passed=selected_report['overall']['sc']['stage_accuracy']>=.90 and
                                     selected_report['overall']['sc']['subgoal_translation_mm_p95']<=20. and
                                     not selected_report['stage_skips'],
                 autonomous_gate_passed=False,final_scenes_opened=False,rl_started=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='best_validation'}),flush=True)


if __name__=='__main__':main()
