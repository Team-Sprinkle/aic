#!/usr/bin/env python3
"""Bounded shared SFP/SC transport BC using successful recorded target commands.

No privileged pose conditioning or invisible-port auxiliary pose loss.
This is offline BC; autonomous transport and insertion require separate rollouts.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader,WeightedRandomSampler
from train_shared_port_tcp_bc import (SharedPortTCPActor,Samples,read_rows,rotation_from_pose9)
from shared_bc_camera_preprocessing import transport_rgb_tensor


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class TransportSamples(Samples):
    def __getitem__(self,index):
        row=self.rows[index]
        if row['family']=='sfp':return super().__getitem__(index)
        views=[]
        for cam in ('center','left','right'):
            with Image.open(row['images'][cam]) as f:
                views.append(transport_rgb_tensor(np.asarray(f.convert('RGB')),native_jpeg_quality=86,native_already_encoded=True))
        return (torch.stack(views),torch.from_numpy(row['state']),torch.from_numpy(row['task']),torch.from_numpy(row['pose']),
                torch.from_numpy(row['targets']),torch.tensor(row['elapsed_s'],dtype=torch.float32),row['family'],torch.from_numpy(row['physical_tip']))


@torch.inference_mode()
def evaluate(model,loader,*,include_elapsed=False):
    model.eval();groups=defaultdict(lambda:defaultdict(list))
    for image,state,task,pose,target,elapsed,families,tip in loader:
        pred,_=model(image.cuda(non_blocking=True),state.cuda(non_blocking=True),
                     task.cuda(non_blocking=True),
                     elapsed.cuda(non_blocking=True) if include_elapsed else None)
        target=target.cuda();mm=(pred[:,:,:3]-target[:,:,:3]).norm(dim=-1)*1000
        r=rotation_from_pose9(pred);truth=rotation_from_pose9(target)
        cosine=(((r.transpose(-1,-2)@truth).diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
        deg=cosine.acos()*180/np.pi
        for i,family in enumerate(families):
            groups[family]['first_target_mm'].append(float(mm[i,0]));groups[family]['last_target_mm'].append(float(mm[i,-1]));groups[family]['orientation_deg'].append(float(deg[i,0]))
    return {family:{key:dict(count=len(v),median=float(np.median(v)),p95=float(np.quantile(v,.95))) for key,v in values.items()} for family,values in groups.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sfp-labels',type=Path,required=True);p.add_argument('--sc-labels',type=Path,required=True)
    p.add_argument('--sfp-cache',type=Path,required=True);p.add_argument('--warm-start',type=Path)
    p.add_argument('--initialization',choices=('mixed_warm_start','imagenet_clean'),default='mixed_warm_start')
    p.add_argument('--imagenet-resnet',type=Path,
                   help='Local ImageNet ResNet18 weights; required for clean initialization')
    p.add_argument('--reset-audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=1500);p.add_argument('--budget-minutes',type=float,default=90)
    p.add_argument('--min-updates',type=int,default=500,
                   help='Do not early-stop a freshly initialized action head before this update')
    p.add_argument('--sc-validation-samples-per-episode',type=int,default=256)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--seed',type=int,default=20260930)
    p.add_argument('--rotation-scale-deg',type=float,default=None,
                   help='Opt-in rotation-matrix loss normalized by this angle; default preserves historical loss.')
    p.add_argument('--mask-controller-error',action='store_true',
                   help='Mask normalized state indices13:19 to test previous-command shortcut')
    p.add_argument('--balance-sc-ports',action='store_true',
                   help='Give each SC selected-port code equal sampling mass')
    p.add_argument('--include-elapsed',action='store_true',
                   help='Bounded route-phase ablation: supply causal elapsed simulation time')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.initialization=='mixed_warm_start' and a.warm_start is None:
        p.error('--warm-start is required for mixed_warm_start')
    if a.initialization=='imagenet_clean' and a.imagenet_resnet is None:
        p.error('--imagenet-resnet is required for imagenet_clean')
    if a.include_elapsed and a.initialization!='imagenet_clean':
        p.error('Elapsed-time ablation requires clean initialization')
    if a.min_updates<1 or a.min_updates>a.updates:
        p.error('--min-updates must lie within the update budget')
    if a.sc_validation_samples_per_episode<16:
        p.error('--sc-validation-samples-per-episode must be at least 16')
    if a.rotation_scale_deg is not None and a.rotation_scale_deg<=0:
        p.error('--rotation-scale-deg must be positive')
    admission_path=a.sc_labels.parent/'manifest.json'
    admission=json.loads(admission_path.read_text())
    assert not admission['final_scenes_opened'] and not admission['actor_geometry_input']
    torch.set_num_threads(8);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    audit=json.loads(a.reset_audit.read_text());assert not audit['leaking_groups'], 'Physical reset split leakage'
    rows=read_rows([('sfp',a.sfp_labels),('sc',a.sc_labels)],target_frame='body')
    assert sum(r['family']=='sc' for r in rows)==admission['counts']['transport'], 'Incomplete SC materialization'
    train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation']
    assert {r['family'] for r in train}=={r['family'] for r in val}=={'sc','sfp'}
    # Use every successful new SC training row. Existing SFP full-success rows
    # retain near-port fallback coverage, although this actor's primary role is transport.
    episode_counts=Counter((r['family'],r['episode']) for r in train);families=Counter(k[0] for k in episode_counts)
    if a.balance_sc_ports:
        port_episode_counts=Counter((int(r['task'][2:4].argmax()),r['episode']) for r in train if r['family']=='sc')
        port_episodes=Counter(k[0] for k in port_episode_counts)
        assert set(port_episodes)=={0,1},'Balanced SC sampling requires both target ports'
        weights=[(1/2/2/port_episodes[int(r['task'][2:4].argmax())]/
                  port_episode_counts[(int(r['task'][2:4].argmax()),r['episode'])]
                  if r['family']=='sc' else
                  1/2/families['sfp']/episode_counts[(r['family'],r['episode'])])
                 for r in train]
    else:
        weights=[1/families[r['family']]/episode_counts[(r['family'],r['episode'])] for r in train]
    sampler=WeightedRandomSampler(weights,a.batch_size*a.updates,replacement=True,generator=torch.Generator().manual_seed(a.seed))
    selected=[]
    for family in ('sfp','sc'):
        pool=defaultdict(list)
        for r in val:
            if r['family']==family:pool[r['episode']].append(r)
        for ep,items in sorted(pool.items()):
            cap=32 if family=='sfp' else a.sc_validation_samples_per_episode
            for i in np.linspace(0,len(items)-1,min(len(items),cap)).astype(int):selected.append(items[int(i)])
    source_paths=[a.sfp_labels,a.sc_labels,a.reset_audit,admission_path]
    if a.warm_start is not None:source_paths.append(a.warm_start)
    if a.imagenet_resnet is not None:source_paths.append(a.imagenet_resnet)
    manifest=dict(schema='aic_clean_transport_bc/v1',hypothesis='clean around-card SC targets plus separate body transport actor correct the old route mixture',
                  train_rows=dict(Counter(r['family'] for r in train)),train_episodes=dict(families),
                  validation_rows=dict(Counter(r['family'] for r in selected)),
                  validation_keys=[dict(family=r['family'],episode=r['episode'],frame=r['frame'],scene=r['scene']) for r in selected],
                  source_sha256={str(path.resolve()):sha(path) for path in source_paths},
                  initialization=a.initialization,
                  inputs=('three RGB cameras, measured 32D AicAdapter observation state, 10D task code' +
                          (', causal elapsed simulation time' if a.include_elapsed else '') +
                          '; no true or predicted target pose'),
                  targets='four full recorded teacher TCP targets relative to frozen chunk-anchor TCP; not residual actions',
                  camera_preprocessing={'sfp':'existing 288x256 quality95 decoded cache -> PIL224 bilinear','sc':'native quality86 JPEG -> 288x256 area -> quality95 JPEG -> PIL224 bilinear'},
                  updates_limit=a.updates,budget_minutes=a.budget_minutes,early_stop=f'four checks without held-out improvement, after at least {a.min_updates} updates',
                  sc_validation_samples_per_episode=a.sc_validation_samples_per_episode,
                  promotion='offline improvement is diagnostic only; full-start observation-only Gazebo comparison required',
                  final_scenes_opened=False,rl_started=False)
    manifest['include_elapsed']=a.include_elapsed
    manifest['loss_contract']=dict(translation_scale_mm=10,
        rotation_scale_deg=a.rotation_scale_deg,
        orientation='normalized rotation-matrix Frobenius components' if a.rotation_scale_deg else 'historical raw6D weight0.1',
        selection='mean per-family (translation p95/10mm + orientation p95/angle)' if a.rotation_scale_deg else 'mean per-family translation p95')
    manifest['masked_state_indices']=list(range(13,19)) if a.mask_controller_error else []
    manifest['sc_port_sampling']='equal probability per selected port, then per episode, then per row' if a.balance_sc_ports else 'equal probability per SC episode'
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    train_loader=DataLoader(TransportSamples(train,a.sfp_cache),batch_size=a.batch_size,sampler=sampler,num_workers=4,pin_memory=True,persistent_workers=True)
    val_loader=DataLoader(TransportSamples(selected,a.sfp_cache),batch_size=a.batch_size,num_workers=4,pin_memory=True,persistent_workers=True)
    keys=('state_mean','state_std','pose_mean','pose_std','target_mean','target_std')
    if a.initialization=='mixed_warm_start':
        prior=torch.load(a.warm_start,map_location='cpu',weights_only=False);assert prior['target_frame']=='body'
        stats={k:prior['model'][k].numpy() for k in keys}
    else:
        state=np.stack([r['state'] for r in train]);pose=np.stack([r['pose'] for r in train]);target=np.concatenate([r['targets'] for r in train])
        stats={}
        for name,array in (('state',state),('pose',pose),('target',target)):
            stats[name+'_mean']=array.mean(axis=0).astype(np.float32)
            stats[name+'_std']=array.std(axis=0).clip(min=1e-3).astype(np.float32)
    model=SharedPortTCPActor(stats,pretrained=False,use_time=a.include_elapsed,condition_on_pose=False).cuda()
    if a.initialization=='mixed_warm_start':
        weights={k:v for k,v in prior['model'].items() if not k.startswith('time_context.')}
        model.load_state_dict(weights,strict=True)
    else:
        backbone=torch.load(a.imagenet_resnet,map_location='cpu',weights_only=True)
        backbone={k:v for k,v in backbone.items() if not k.startswith('fc.')}
        model.backbone.load_state_dict(backbone,strict=True)
    model.masked_state_indices=manifest['masked_state_indices']
    for parameter in model.pose_head.parameters():parameter.requires_grad_(False)
    baseline=evaluate(model,val_loader,include_elapsed=a.include_elapsed);history=[dict(update=0,validation=baseline)]
    (a.output/'baseline.json').write_text(json.dumps(history[0],indent=2)+'\n')
    optimizer=torch.optim.AdamW((v for v in model.parameters() if v.requires_grad),lr=1e-4,weight_decay=1e-4)
    initial=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},update=0,
        architecture=('shared_port_tcp_rgb_resnet18_pose_aux_transformer_time_v2' if a.include_elapsed else
                      'shared_port_tcp_rgb_resnet18_pose_aux_transformer_v1'),include_elapsed=a.include_elapsed,target_frame='body',time_coordinate='sim_elapsed',
        native_jpeg_quality_by_family={'sc':86},validation=baseline,source_sha256=manifest['source_sha256'],
        runtime_inputs=manifest['inputs'],precise_pose_auxiliary_loss=False,reserved_final_opened=False,rl_started=False)
    initial['masked_state_indices']=manifest['masked_state_indices']
    torch.save(initial,a.output/'initial.pt');torch.save(initial,a.output/'best.pt')
    def selection(report):
        return sum(report[f]['first_target_mm']['p95']/10+
                   report[f]['orientation_deg']['p95']/a.rotation_scale_deg
                   for f in ('sfp','sc'))/2 if a.rotation_scale_deg else sum(
                       report[f]['first_target_mm']['p95'] for f in ('sfp','sc'))/2
    best=selection(baseline)
    bad=0;best_update=0;began=time.monotonic();reason='update_budget';update=0
    for update,(image,state,task,pose,target,elapsed,family,tip) in enumerate(train_loader,start=1):
        if time.monotonic()-began>a.budget_minutes*60:reason='wall_time_budget';break
        model.train();target=target.cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred,_=model(image.cuda(non_blocking=True),state.cuda(non_blocking=True),
                         task.cuda(non_blocking=True),
                         elapsed.cuda(non_blocking=True) if a.include_elapsed else None)
            pos=nn.functional.smooth_l1_loss(pred[:,:,:3]/.01,target[:,:,:3]/.01,beta=.1)
            if a.rotation_scale_deg:
                # ||R_pred-R_true||_F ~= sqrt(2)*angle for small errors.
                # Average over nine components is corrected to a per-rotation sum.
                scale=np.sqrt(2)*np.deg2rad(a.rotation_scale_deg)
                orient=nn.functional.smooth_l1_loss(
                    rotation_from_pose9(pred.float())/scale,
                    rotation_from_pose9(target.float())/scale,beta=.1)*9
                loss=pos+orient
            else:
                orient=nn.functional.smooth_l1_loss(pred[:,:,3:],target[:,:,3:],beta=.1)
                loss=pos+.1*orient
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        if update==1 or update%250==0 or update==a.updates:
            report=evaluate(model,val_loader,include_elapsed=a.include_elapsed)
            key=selection(report)
            history.append(dict(update=update,loss=float(loss.detach()),validation=report,selection_score=key))
            improved=key<best*.995
            if improved:best=key;best_update=update;bad=0
            else:bad+=1
            checkpoint=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},update=update,
                architecture=('shared_port_tcp_rgb_resnet18_pose_aux_transformer_time_v2' if a.include_elapsed else
                              'shared_port_tcp_rgb_resnet18_pose_aux_transformer_v1'),include_elapsed=a.include_elapsed,target_frame='body',time_coordinate='sim_elapsed',
                native_jpeg_quality_by_family={'sc':86},validation=report,source_sha256=manifest['source_sha256'],
                runtime_inputs=manifest['inputs'],loss_contract=manifest['loss_contract'],precise_pose_auxiliary_loss=False,reserved_final_opened=False,rl_started=False)
            checkpoint['masked_state_indices']=manifest['masked_state_indices']
            torch.save(checkpoint,a.output/'last.pt')
            if improved:torch.save(checkpoint,a.output/'best.pt')
            (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
            print(json.dumps(dict(update=update,selection_score=key,best_update=best_update)),flush=True)
            if update>=a.min_updates and bad>=4:reason='heldout_plateau';break
        if update>=a.updates:break
    summary=dict(schema='aic_clean_transport_bc_result/v1',completed_updates=update,best_update=best_update,best_selection_score=best,loss_contract=manifest['loss_contract'],stop_reason=reason,
                 elapsed_minutes=(time.monotonic()-began)/60,baseline=baseline,autonomous_rollouts_completed=False,rl_started=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
