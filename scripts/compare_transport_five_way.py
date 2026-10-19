#!/usr/bin/env python3
"""Matched episode-grouped held-out transport candidate and task-code audit."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from train_shared_port_tcp_bc import rotation_from_pose9
from train_temporal_transport_probe import (build_data,LONG_AGES,
                                           TemporalTransport,predict,sha)
from train_diffusion_transport_probe import DiffusionTransport


def select_rows(rows):
    groups=defaultdict(list)
    for i,row in enumerate(rows):
        if row['split']=='validation':
            groups[(row['family'],row['episode'],row['stage'])].append(i)
    selected=[]
    for (family,episode,stage),items in groups.items():
        cap=8 if family=='sc' else 12
        selected.extend(np.array(items)[np.linspace(0,len(items)-1,min(cap,len(items))).astype(int)])
    return sorted(set(map(int,selected)))


def actor_prediction(model,data,indices,diffusion,no_history,seed):
    idx=torch.as_tensor(indices,device='cuda')
    if diffusion:
        return model.sample(model.context(data,idx),seed)*model.target_std+model.target_mean
    action,*_=predict(model,data,idx,no_history)
    return action*model.target_std+model.target_mean


@torch.inference_mode()
def evaluate_actor(model,data,selected,diffusion,no_history,seed=20261001):
    model.eval();groups=defaultdict(lambda:defaultdict(list))
    for offset in range(0,len(selected),96):
        part=selected[offset:offset+96]
        pred=actor_prediction(model,data,part,diffusion,no_history,seed+offset)
        true=data['target'][part]
        errors=(pred[:,:,:3]-true[:,:,:3]).norm(dim=-1)*1000
        predicted_rot=rotation_from_pose9(pred[:,0]);true_rot=rotation_from_pose9(true[:,0])
        relative=predicted_rot.transpose(-1,-2)@true_rot
        angles=((((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)).acos()*180/np.pi)
        cosine=(pred[:,0,:3]*true[:,0,:3]).sum(-1)
        for j,i in enumerate(part):
            row=data['rows'][i];family=row['family'];task=row['task']
            port=int(np.argmax(task[2:4]));card=int(np.argmax(task[4:9])) if family=='sfp' else -1
            for key in ((family,'all'),(family,'stage',row['stage']),
                        (family,'port',port),(family,'card',card)):
                group=groups[str(key)]
                group['first_mm'].append(float(errors[j,0]))
                group['fourth_mm'].append(float(errors[j,3]))
                group['first_deg'].append(float(angles[j]))
                if float(true[j,0,:3].norm())>=.002:
                    group['direction_correct'].append(float(cosine[j]>0))
    return {key:dict(rows=len(value['first_mm']),
                     first_mm_p50=float(np.quantile(value['first_mm'],.5)),
                     first_mm_p95=float(np.quantile(value['first_mm'],.95)),
                     fourth_mm_p95=float(np.quantile(value['fourth_mm'],.95)),
                     first_deg_p95=float(np.quantile(value['first_deg'],.95)),
                     direction_accuracy_ge2mm=(float(np.mean(value['direction_correct']))
                                                if value['direction_correct'] else None),
                     direction_rows_ge2mm=len(value['direction_correct']))
            for key,value in groups.items()}


@torch.inference_mode()
def task_code_effect(model,data,diffusion,no_history):
    episode=defaultdict(list)
    for i,row in enumerate(data['rows']):
        if row['family']=='sc' and row['split']=='validation':episode[row['episode']].append(i)
    chosen={fraction:[indices[int((len(indices)-1)*fraction)] for indices in episode.values()]
            for fraction in (0.,.5,.9)}
    originals={fraction:actor_prediction(model,data,indices,diffusion,no_history,54321)
               for fraction,indices in chosen.items()}
    backup=data['state_task_full']
    swapped=backup.clone()
    sc_indices=[i for i,row in enumerate(data['rows']) if row['family']=='sc']
    values=swapped[sc_indices,34:36].clone()
    swapped[sc_indices,34:36]=values.flip(-1)
    data['state_task_full']=swapped
    try:
        result={}
        for fraction,indices in chosen.items():
            altered=actor_prediction(model,data,indices,diffusion,no_history,54321)
            effect=(altered[:,0,:3]-originals[fraction][:,0,:3]).norm(dim=-1)*1000
            result[str(fraction)]=dict(episodes=len(indices),first_target_effect_mm_median=float(effect.median()),
                                        first_target_effect_mm_p95=float(torch.quantile(effect,.95)))
        return result
    finally:data['state_task_full']=backup


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(8)
    data=build_data(a.cache,a.base,'cuda',LONG_AGES)
    selected=select_rows(data['rows'])
    reports={}
    for name in ('no_history','long_history','route','diffusion',
                 'sfp_only','sc_only','sfp_only_3000','sc_only_3000'):
        path=a.runs/name/'best.pt'
        if not path.exists():continue
        ck=torch.load(path,map_location='cpu',weights_only=False)
        diffusion=name=='diffusion'
        if diffusion:
            model=DiffusionTransport(data['features'].shape[1],data['mean'],data['std']).cuda()
        else:
            manifest=ck['source_manifest']
            model=TemporalTransport(data['mean'],data['std'],raw_state_skip=True,
                                    feature_dim=data['features'].shape[1],state_task_dim=42,
                                    route_conditioning=manifest['route_conditioning']).cuda()
        model.load_state_dict(ck['model']);model.eval()
        no_history=name=='no_history'
        selected_family=ck['source_manifest'].get('task_family','both')
        part=[i for i in selected if selected_family=='both' or data['rows'][i]['family']==selected_family]
        report=dict(checkpoint=str(path),checkpoint_sha256=sha(path),best_update=ck['update'],
                    heldout=evaluate_actor(model,data,part,diffusion,no_history))
        # Count complete episodes rather than counting rows under a repeated key.
        report['validation_episodes']=dict(Counter(f for f,e in
             {(data['rows'][i]['family'],data['rows'][i]['episode']) for i in part}))
        if selected_family in ('both','sc'):
            report['sc_port_code_effect']=task_code_effect(model,data,diffusion,no_history)
        reports[name]=report
        print(json.dumps(dict(name=name,episodes=report['validation_episodes'],
                              sc=report['heldout'].get("('sc', 'all')"),
                              sfp=report['heldout'].get("('sfp', 'all')"))),flush=True)
    result=dict(schema='aic_transport_five_way_matched_heldout/v1',
                selection='eight rows per SC episode and stage; twelve per SFP episode',
                source_cache_sha256=sha(a.cache/'manifest.json'),
                final_scenes_opened=False,autonomous_gate_passed=False,reports=reports)
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
