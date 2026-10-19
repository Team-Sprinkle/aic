#!/usr/bin/env python3
"""Bounded full-action SFP task-gated head on the frozen temporal backbone.

The ten requested card/port codes choose ten final linear maps inside one
shared actor. Each map predicts the complete four-target chunk, not a delta.
Only the 50 balanced successful SFP training resets supply action labels.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from train_temporal_transport_probe import LONG_AGES, TemporalTransport, build_data, predict


@torch.inference_mode()
def score(model, data, indices):
    model.eval()
    errors=[];early_errors=[]
    by_task=defaultdict(list)
    for chunk in np.array_split(indices,max(1,(len(indices)+127)//128)):
        if not len(chunk):continue
        idx=torch.as_tensor(chunk,device='cuda:1')
        pred,_,_,_=predict(model,data,idx,False)
        physical=pred.float()*model.target_std+model.target_mean
        truth=data['target'][idx]
        mm=torch.linalg.vector_norm((physical[:,0,:3]-truth[:,0,:3])*1000,dim=-1)
        for index,value in zip(chunk,mm.cpu().numpy()):
            r=data['rows'][int(index)]
            task=(int(np.argmax(r['task'][4:9])),int(np.argmax(r['task'][2:4])))
            by_task[str(task)].append(float(value))
            errors.append(float(value))
            if .3<=r['time_s']<=1.2:early_errors.append(float(value))
    return {'rows':len(errors),'first_target_translation_mm_median':float(np.median(errors)),
            'first_target_translation_mm_p95':float(np.percentile(errors,95)),
            'early_rows':len(early_errors),
            'early_first_target_translation_mm_p95':float(np.percentile(early_errors,95)) if early_errors else None,
            'by_task':{k:{'rows':len(v),'p95_mm':float(np.percentile(v,95))}
                       for k,v in sorted(by_task.items())}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--source-checkpoint',type=Path,required=True)
    p.add_argument('--actor-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--paired-scene-continuation',action='store_true',
                   help='Include the six original and five successful alternate requests in train')
    p.add_argument('--all-train',action='store_true',
                   help='Use all admitted SFP train episodes from an immutable expanded cache')
    p.add_argument('--paired-early-mass',type=float,default=0.,
                   help='Fraction of each batch sampled from 0.3–1.2 s same-board alternate-request windows')
    a=p.parse_args()
    if a.output.exists():p.error(f'Output exists: {a.output}')
    if a.updates>2000:p.error('Predeclared cap is 2,000 updates')
    if not 0<=a.paired_early_mass<1 or (a.paired_early_mass and not a.all_train):
        p.error('Paired early mass must be in [0,1) and needs all admitted training scenes')
    random.seed(20261005);np.random.seed(20261005);torch.manual_seed(20261005)
    torch.set_num_threads(4)
    data=build_data(a.cache,a.source_checkpoint,'cuda:1',LONG_AGES)
    cp=torch.load(a.actor_checkpoint,map_location='cpu',weights_only=False)
    source=cp['source_manifest']
    if not (source['full_state_skip'] and source['long_history']
            and not source['route_conditioning']):
        p.error('Unexpected frozen actor architecture')
    base=TemporalTransport(data['mean'],data['std'],raw_state_skip=True,
                           feature_dim=data['features'].shape[1],
                           state_task_dim=42,route_conditioning=False).to('cuda:1').eval()
    base.load_state_dict(cp['model'])
    model=TemporalTransport(data['mean'],data['std'],raw_state_skip=True,
                            feature_dim=data['features'].shape[1],
                            state_task_dim=42,route_conditioning=False,
                            task_gated_sfp=True).to('cuda:1').eval()
    missing,unexpected=model.load_state_dict(cp['model'],strict=False)
    if set(missing)!={'sfp_expert_weight','sfp_expert_bias'} or unexpected:
        raise ValueError(f'Unexpected initialization keys {missing} {unexpected}')
    with torch.no_grad():
        model.sfp_expert_weight.copy_(base.action[-1].weight[None].expand(10,-1,-1))
        model.sfp_expert_bias.copy_(base.action[-1].bias[None].expand(10,-1))
    for name,param in model.named_parameters():
        param.requires_grad_(name in ('sfp_expert_weight','sfp_expert_bias'))
    train_by_task=defaultdict(list);dev=[];train_groups=set();dev_groups=set()
    for i,r in enumerate(data['rows']):
        if r['family']!='sfp':continue
        task=(int(np.argmax(r['task'][4:9])),int(np.argmax(r['task'][2:4])))
        if r['split']=='validation':dev.append(i);dev_groups.add(r['episode'])
        elif r['split']=='train' and (a.all_train or 1800000<=int(r['episode'])<1800050 or
              (a.paired_scene_continuation and
               (2000000<=int(r['episode'])<2000005 or
                2000010<=int(r['episode'])<2000016))):
            train_by_task[task].append(i);train_groups.add(r['episode'])
    expected_train=(len({r['episode'] for r in data['rows'] if r['family']=='sfp' and
                         r['split']=='train'}) if a.all_train else
                    61 if a.paired_scene_continuation else 50)
    if len(train_groups)!=expected_train or len(dev_groups)!=10 or len(train_by_task)!=10:
        p.error(f'Unexpected complete reset counts: {len(train_groups)}, {len(dev_groups)}')
    dev=np.asarray(dev,dtype=np.int64)
    paired_early_by_task=defaultdict(list)
    if a.paired_early_mass:
        requests_by_scene=defaultdict(set)
        for r in data['rows']:
            if r['family']=='sfp' and r['split']=='train':
                requests_by_scene[r['scene']].add((int(np.argmax(r['task'][4:9])),
                                                   int(np.argmax(r['task'][2:4]))))
        paired={scene for scene,requests in requests_by_scene.items() if len(requests)>1}
        for task,items in train_by_task.items():
            paired_early_by_task[task]=[i for i in items if
                data['rows'][i]['scene'] in paired and .3<=data['rows'][i]['time_s']<=1.2]
            if not paired_early_by_task[task]:
                p.error(f'No paired early branch rows for {task}')
    baseline=score(base,data,dev)
    opt=torch.optim.AdamW([model.sfp_expert_weight,model.sfp_expert_bias],
                          lr=1e-4,weight_decay=1e-4)
    a.output.mkdir(parents=True)
    manifest={'schema':'sfp_task_gated_transport/v1','balanced_train_groups':50,
              'all_train':a.all_train,'admitted_sfp_train_groups':len(train_groups),
              'paired_early_mass':a.paired_early_mass,
              'paired_early_rows_by_task':{str(k):len(v) for k,v in paired_early_by_task.items()},
              'paired_scene_train_groups':11 if a.paired_scene_continuation else 0,
              'paired_scene_groups_with_both_successful_requests':5 if a.paired_scene_continuation else 0,
              'development_groups':10,'train_rows_by_task':{str(k):len(v)
                  for k,v in sorted(train_by_task.items())},
              'source_actor':str(a.actor_checkpoint),'source_cache':str(a.cache),
              'architecture':'frozen shared temporal/image backbone; ten task-selected final linear maps for four complete 9D pose targets (36 output scalars)',
              'teacher_geometry_actor_input':False,'previous_teacher_actions_offline_input':True,
              'only_trainable_parameters':['sfp_expert_weight','sfp_expert_bias'],
              'update_cap':a.updates,'selection':'development first-target translation p95',
              'offline_gate':'at least 20% lower first-action p95 than frozen baseline, no task above 5mm p95; live gate separate',
              'final_scenes_opened':False}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    best=baseline['first_target_translation_mm_p95'];best_step=0;history=[];bad=0
    for step in range(1,a.updates+1):
        broad=int(round(64*(1-a.paired_early_mass)))
        chosen=np.asarray([random.choice(train_by_task[(j//2,j%2)])
                           for j in np.random.choice(10,size=broad)]+[
                           random.choice(paired_early_by_task[(j//2,j%2)])
                           for j in np.random.choice(10,size=64-broad)],dtype=np.int64)
        idx=torch.as_tensor(chosen,device='cuda:1')
        model.eval();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            action,_,_,_=predict(model,data,idx,False)
            truth=(data['target'][idx]-model.target_mean)/model.target_std
            loss=F.smooth_l1_loss(action,truth,beta=.2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([model.sfp_expert_weight,
                                        model.sfp_expert_bias],1.)
        opt.step()
        if step%250==0:
            result=score(model,data,dev)
            current=result['first_target_translation_mm_p95']
            record={'step':step,'loss':float(loss.detach()),'validation':result}
            history.append(record)
            print(json.dumps({'step':step,'loss':record['loss'],
                              'dev_p95_mm':current}),flush=True)
            if current<best:
                best=current;best_step=step;bad=0
                torch.save({'model':model.state_dict(),'source_manifest':{**source,
                            'task_gated_sfp':True},'task_gated_sfp':True,
                            'update':step,'final_scenes_opened':False},
                           a.output/'best.pt')
            else:
                bad+=1
                if bad>=3:break
    selected=next((r['validation'] for r in history if r['step']==best_step),baseline)
    gate=(best_step>0 and selected['first_target_translation_mm_p95']<=
          .8*baseline['first_target_translation_mm_p95'] and
          all(v['p95_mm']<=5 for v in selected['by_task'].values()))
    result={'schema':'sfp_task_gated_transport_result/v1','baseline':baseline,
            'selected':selected,'best_step':best_step,'updates_completed':step,
            'offline_gate_passed':gate,'live_gate_passed':False,
            'final_scenes_opened':False}
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'best_step':best_step,'updates_completed':step,
                      'baseline_p95_mm':baseline['first_target_translation_mm_p95'],
                      'selected_p95_mm':selected['first_target_translation_mm_p95'],
                      'offline_gate_passed':gate}),flush=True)


if __name__=='__main__':main()
