#!/usr/bin/env python3
"""Train a separate SFP/SC insertion actor on RGB-estimated port-frame targets.

The input pose and all action targets share the *predicted* port frame. The
recorded complete teacher targets were transformed into that frame at dataset
build time. The actor never receives true port geometry. This is offline BC;
autonomous insertion requires a separate Gazebo rollout.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from PIL import Image

from local_causal_history import sample_robot_history
from act_cache import load_cached_images
from train_shared_port_tcp_bc import SharedPortTCPActor,body_translation_from_poses,rotation_from_pose9
from shared_port_tcp_geometry import matrix_from_pose9,matrix_from_xyz_xyzw
from native_actor_images import NativeActorImageCache,CONTRACT


CAMERAS=('center','left','right')


def near_port_mask(phase_pose,task,threshold,contact_aware=False):
    if not contact_aware:return phase_pose[:,2].abs()<threshold
    upper=torch.where(task[:,0]>.5,phase_pose.new_tensor(.060),phase_pose.new_tensor(threshold))
    return (phase_pose[:,2]>=-threshold)&(phase_pose[:,2]<=upper)


def decoded_tcp_targets(output, pose, tip, task, reference, tip_action_families=('sc',)):
    """Decode full point targets with observation-only body translation.

    Target rotations remain TCP rotations. Legacy checkpoints use SC only;
    new shared checkpoints explicitly declare both control-point families.
    """
    if reference == 'tcp':
        return output
    if tip is None:
        raise ValueError('Plug-tip targets require a predicted body-tip input')
    current_rotation=rotation_from_pose9(pose.float())
    body=(current_rotation.transpose(-1,-2) @
          (tip[:,:3].float()-pose[:,:3].float()).unsqueeze(-1)).squeeze(-1)
    offset=(rotation_from_pose9(output.float()) @ body[:,None,:,None]).squeeze(-1)
    enabled=torch.zeros(len(task),device=task.device,dtype=torch.bool)
    for family in tip_action_families:
        enabled|=task[:,0 if family=='sfp' else 1]>.5
    xyz=output[...,:3].float()-offset*enabled[:,None,None]
    return torch.cat((xyz,output[...,3:].float()),dim=-1)


class LocalRows(Dataset):
    def __init__(self,rows,cache,history_pool=None,include_tip=False,history_offsets=None,history_masked=False,native_image_cache=None):
        self.rows=rows
        self.include_tip=include_tip
        self.native_image_cache=NativeActorImageCache(native_image_cache) if native_image_cache else None
        needs_legacy=any(r['family']=='sfp' and r.get('image_kind')!='native_1152x1024_rgb_jpeg' for r in rows)
        if needs_legacy and cache is None:raise ValueError('Legacy SFP rows require their original cache')
        self.images=load_cached_images(cache,json.loads((cache/'cache.json').read_text())) if needs_legacy else None
        self.history_by_id={}
        groups=defaultdict(list)
        for row in (history_pool if history_pool is not None else rows):
            groups[(row['family'],row['episode'])].append(row)
        for group in groups.values():
            group.sort(key=lambda r:(float(r['time_s']),int(r['frame'])))
            times=[float(r['time_s']) for r in group]
            states=[r['state'] for r in group]
            for j,row in enumerate(group):
                current_base_tcp=matrix_from_xyz_xyzw(row['state'][:7])
                current_port_tcp=matrix_from_pose9(row['predicted_port_tcp_pose9'])
                current_base_port=current_base_tcp@np.linalg.inv(current_port_tcp)
                if history_offsets is not None:
                    self.history_by_id[id(row)]=sample_robot_history(
                        times[:j+1],states[:j+1],times[j],current_base_port,
                        history_offsets,history_masked)
                else:
                    past=[group[max(0,j-3+k)] for k in range(4)]
                    def aligned_pose(p):
                        prior=np.linalg.inv(current_base_port)@matrix_from_xyz_xyzw(p['state'][:7])
                        return [*prior[:3,3],*prior[:3,:2].T.reshape(-1)]
                    self.history_by_id[id(row)]=np.asarray([
                        [*p['state'],*aligned_pose(p),max(0.,min(2.,times[j]-float(p['time_s'])))]
                        for p in past],np.float32)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];images=[]
        for cam in (() if self.native_image_cache is not None else CAMERAS):
            if r['family']=='sfp' and r.get('image_kind')!='native_1152x1024_rgb_jpeg':
                rgb=np.ascontiguousarray(self.images[f'observation.images.{cam}_camera'][r['cache_index']])
            else:
                image=cv2.imread(r['images'][cam],cv2.IMREAD_COLOR)
                if image is None:raise FileNotFoundError(r['images'][cam])
                image=cv2.resize(image,(288,256),interpolation=cv2.INTER_AREA)
                if r['family']=='sfp':
                    ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95])
                    if not ok:raise RuntimeError('Native SFP actor preprocessing failed')
                    image=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
                rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image=Image.fromarray(rgb,'RGB').resize((224,224),Image.Resampling.BILINEAR)
            images.append(torch.from_numpy(np.asarray(image).copy()).permute(2,0,1))
        image_tensor=torch.from_numpy(self.native_image_cache[r]) if self.native_image_cache is not None else torch.stack(images)
        result=(image_tensor,torch.tensor(r['state'],dtype=torch.float32),
                torch.tensor(r['task'],dtype=torch.float32),
                torch.tensor(r['predicted_port_tcp_pose9'],dtype=torch.float32),
                torch.tensor(r['teacher_tcp_targets_predicted_port_pose9'],dtype=torch.float32),
                r['family'],r['episode'],torch.from_numpy(self.history_by_id[id(r)]))
        if self.include_tip:
            result+=(torch.tensor(r['predicted_tip_port_pose9'],dtype=torch.float32),
                     torch.tensor(r['predicted_tip_valid'],dtype=torch.bool),
                     torch.tensor(r.get('recorded_tcp_targets_predicted_port_pose9',
                                        r['teacher_tcp_targets_predicted_port_pose9']),dtype=torch.float32))
        return result


@torch.inference_mode()
def evaluate(model,loader):
    model.eval();errors=defaultdict(lambda:defaultdict(list));eps=defaultdict(set)
    native_episodes={r['episode'] for r in loader.dataset.rows
                     if r['family']=='sfp' and r.get('image_kind')=='native_1152x1024_rgb_jpeg'}
    for batch in loader:
        image,state,task,pose,target,family,episode,history=batch[:8]
        tip=tip_mask=None
        if model.use_tip:
            tip=batch[8].cuda(non_blocking=True);tip_mask=batch[9].cuda(non_blocking=True)
            if model.tip_input_mode=='disabled':tip_mask=torch.zeros_like(tip_mask)
        image=image.cuda(non_blocking=True);state=state.cuda(non_blocking=True)
        task=task.cuda(non_blocking=True);pose=pose.cuda(non_blocking=True)
        target=target.cuda(non_blocking=True)
        history=history.cuda(non_blocking=True)
        if model.tip_input_mode=='constant_translation':
            mean_body=pose.new_tensor(model.constant_body_tip)
            tip=tip.clone()
            tip[:,:3]=(rotation_from_pose9(pose.float())@mean_body[None,:,None]).squeeze(-1)+pose[:,:3]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output,internal=model(image,state,task,external_pose=pose,
                                  external_mask=torch.ones(len(pose),device='cuda',dtype=torch.bool),
                                  local_history=history if model.use_local_history else None,
                                  physical_tip=tip,tip_mask=tip_mask)
        tip_families=getattr(model,'tip_action_families',['sc'])
        decoded=decoded_tcp_targets(output,pose,tip,task,model.sc_action_reference,tip_families)
        phase_pose=(torch.where(tip_mask[:,None],tip,pose) if getattr(model,'phase_reference','tcp')=='predicted_tip' else pose)
        contact_aware=getattr(model,'contact_aware_phase',False)
        near_samples=near_port_mask(phase_pose,task,.03,contact_aware)
        buffer_samples=near_port_mask(phase_pose,task,.08,contact_aware)
        recorded=(batch[10].cuda(non_blocking=True) if model.use_tip else target)
        command=body_translation_from_poses(pose.float(),decoded)
        teacher=body_translation_from_poses(pose.float(),recorded.float())
        predicted_rotation=rotation_from_pose9(output.float())
        teacher_rotation=rotation_from_pose9(target.float())
        angle_cos=((predicted_rotation.transpose(-1,-2)@teacher_rotation).diagonal(
            dim1=-2,dim2=-1).sum(-1)-1)/2
        angle_deg=torch.rad2deg(torch.acos(angle_cos.clamp(-1+1e-6,1-1e-6))).cpu().numpy()
        for j,f in enumerate(family):
            groups=[f]
            if f=='sc':
                groups.append('sc_continuation' if episode[j].startswith('local_causal_continuation_') else 'sc_original')
            if -.19<float(phase_pose[j,2])<-.09:groups.append(f+'_far_approach')
            if f=='sfp' and episode[j] in native_episodes:groups.append('sfp_native')
            reference_z=tip[j,2] if model.sc_action_reference!='tcp' and f in tip_families else pose[j,2]
            if f=='sc' and abs(float(phase_pose[j,2]))<.05 and float(target[j,0,2]-reference_z)>.0005:
                groups.append('sc_advancing')
            for group in groups:
                eps[group].add(episode[j])
                predicted_span=float((output[j,3,2]-output[j,0,2])*1000)
                target_span=float((target[j,3,2]-target[j,0,2])*1000)
                errors[group]['axial_chunk_span_error_mm'].append(abs(predicted_span-target_span))
                if target_span>.05:errors[group]['missed_axial_chunk_advance'].append(float(predicted_span<=.025))
                if abs(target_span)<.025:errors[group]['false_axial_chunk_advance'].append(float(predicted_span>.05))
                errors[group]['first_target_mm'].append(float((output[j,0,:3]-target[j,0,:3]).norm()*1000))
                errors[group]['first_body_command_mm'].append(float((command[j,0]-teacher[j,0]).norm()*1000))
                errors[group]['fourth_target_mm'].append(float((output[j,3,:3]-target[j,3,:3]).norm()*1000))
                errors[group]['first_orientation_deg'].append(float(angle_deg[j,0]))
                if bool(buffer_samples[j]):
                    errors[group]['buffered_lateral_target_mm'].append(float(
                        (output[j,0,:2]-target[j,0,:2]).norm()*1000))
                if bool(near_samples[j]):
                    errors[group]['near_lateral_command_mm'].append(float((command[j,0,:2]-teacher[j,0,:2]).norm()*1000))
                    errors[group]['near_port_lateral_target_mm'].append(float(
                        (output[j,0,:2]-target[j,0,:2]).norm()*1000))
                    errors[group]['near_decoded_tcp_lateral_mm'].append(float(
                        (decoded[j,0,:2]-recorded[j,0,:2]).norm()*1000))
                    errors[group]['near_first_orientation_deg'].append(float(angle_deg[j,0]))
                if (f=='sfp' and -.105<float(pose[j,2])<-.08 and
                    float(torch.linalg.vector_norm(pose[j,:2]))<.005):
                    true_advance=float((target[j,0,2]-pose[j,2])*1000)
                    pred_advance=float((output[j,0,2]-pose[j,2])*1000)
                    errors[group]['aligned_approach_axial_target_mm'].append(abs(pred_advance-true_advance))
                    errors[group]['aligned_approach_sign_correct'].append(float(
                        (pred_advance>.5)==(true_advance>.5)))
    return {f:{'episodes':len(eps[f]),**{name:{'n':len(value),'mean':float(np.mean(value)),
                        'median':float(np.median(value)),
                        'p95':float(np.quantile(value,.95))} for name,value in fields.items() if value}}
            for f,fields in errors.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--extra-rows',type=Path,
                   help='Optional scored on-policy states with offline teacher target labels')
    p.add_argument('--decoder-kind',choices=('transformer','mlp'),default='transformer')
    p.add_argument('--reset-decoder',action='store_true')
    p.add_argument('--coordinate-heads-fp32',action='store_true')
    p.add_argument('--control-path-fp32',action='store_true',
                   help='Keep pose/state conditioning and the four-token decoder in FP32; vision stays mixed precision')
    p.add_argument('--chunk-motion-loss-weight',type=float,default=0.)
    p.add_argument('--balanced-sc-continuation-val',action='store_true')
    p.add_argument('--extra-validation-only',action='store_true',help='Matched control sees the same held-out continuation data without its training episodes')
    p.add_argument('--sfp-native-only',action='store_true',
                   help='Bounded SFP native-only repair on a grouped validation set')
    p.add_argument('--onpolicy-boost',type=float,default=1.,
                   help='Episode-balanced sampling multiplier for matched on-policy rows')
    p.add_argument('--sfp-cache',type=Path)
    p.add_argument('--native-image-cache',type=Path,help='Exact immutable native input cache; avoids repeated original-JPEG decoding')
    p.add_argument('--imagenet',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--warm-transport',type=Path)
    p.add_argument('--warm-visual-only',action='store_true',
                   help='Transfer only camera encoder/attention; keep fresh state/task context with current-data normalization')
    p.add_argument('--resume-local',type=Path,
                   help='Continue all local actor weights, including its target head')
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--selection-metric',choices=('worst_first','two_family_insertion','sc_near','sc_near_orientation','sc_correction','sc_motion','sc_orientation','sc_advancing','sfp_native','sfp_native_near','sfp_aligned_approach'),default='worst_first')
    p.add_argument('--aligned-approach-sample-weight',type=float,default=1.)
    p.add_argument('--aligned-approach-axial-loss-weight',type=float,default=0.)
    p.add_argument('--prefer-near-axial-progress',action='store_true',
                   help='Sample advancing SC teacher actions more than alignment holds')
    p.add_argument('--balanced-native-val',action='store_true',
                   help='Sample native SFP and prior SFP validation episodes separately')
    p.add_argument('--orientation-loss-weight',type=float,default=0.,
                   help='Geodesic rotation loss, with extra near-port weight')
    p.add_argument('--near-lateral-loss-weight',type=float,default=.4,
                   help='Weight for first near-port lateral target error')
    p.add_argument('--near-phase-sample-weight',type=float,default=3.,
                   help='Episode-balanced near-port sampling multiplier at |z|<40 mm')
    p.add_argument('--near-phase-threshold-m',type=float,default=.04)
    p.add_argument('--lateral-buffer-m',type=float,default=.03,
                   help='Apply extra lateral loss before the insertion corridor')
    p.add_argument('--separate-family-orientation-head',action='store_true',
                   help='One shared actor backbone with family-specific rotation output heads')
    p.add_argument('--separate-family-action-head',action='store_true',
                   help='One shared actor backbone with full pose heads for each task family')
    p.add_argument('--stale-evaluations',type=int,default=8)
    p.add_argument('--min-updates',type=int,default=500,
                   help='Minimum optimization opportunity before plateau stopping')
    p.add_argument('--updates',type=int,default=1600)
    p.add_argument('--minutes',type=float,default=90)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--local-history',action='store_true',
                   help='Condition one shared insertion actor on four causal robot/pose samples')
    p.add_argument('--history-offsets-s',type=float,nargs=4,default=None)
    p.add_argument('--history-masked',action='store_true')
    p.add_argument('--mask-command-feedback',action='store_true',
                   help='Mask normalized current-command tracking error indices 13:19')
    p.add_argument('--selection-orientation-tolerance-deg',type=float,default=3.,
                   help='Orientation divisor for sc_near_orientation checkpoint selection')
    p.add_argument('--tip-input-mode',choices=('none','disabled','constant_translation','translation'),default='none',
                   help='Matched architecture with disabled or predicted plug-position input')
    p.add_argument('--sc-action-reference',choices=('tcp','plug_tip_translation_tcp_orientation'),default='tcp')
    p.add_argument('--tip-action-families',choices=('sc','sfp'),nargs='+',default=['sc'],
                   help='Explicit families using the camera-predicted plug control point; legacy default SC')
    p.add_argument('--phase-reference',choices=('tcp','predicted_tip'),default='tcp',
                   help='Near-port loss/sampling use predicted plug distance; TCP default preserves legacy experiments')
    p.add_argument('--contact-aware-phase',action='store_true',
                   help='Include SFP measured insertion depths up to60mm in contact-phase sampling/loss/selection')
    p.add_argument('--reset-action-head',action='store_true',help='Retain representation; reset action decoder output and current-data normalization')
    p.add_argument('--fixed-update-budget',action='store_true',
                   help='Record plateaus but complete the common comparison update budget')
    p.add_argument('--full-validation',action='store_true',
                   help='Evaluate every admitted held-out row, including wider starts')
    p.add_argument('--require-verified-command-contract',action='store_true',
                   help='Require causal RGB locks, recorded targets, completion cuts and matched gains')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing run')
    a.output.mkdir(parents=True)
    torch.set_num_threads(8);torch.manual_seed(20261002);np.random.seed(20261002);random.seed(20261002)
    rows=[json.loads(l) for l in (a.dataset/'rows.jsonl').open()]
    if a.extra_rows:
        extra=[json.loads(l) for l in (a.extra_rows/'rows.jsonl').open()]
        rows.extend(r for r in extra if not a.extra_validation_only or r['split']=='validation')
    if a.sfp_native_only:
        rows=[r for r in rows if r['family']=='sfp' and
              r.get('image_kind')=='native_1152x1024_rgb_jpeg']
    actor_image_contract=CONTRACT if all(r.get('image_kind')=='native_1152x1024_rgb_jpeg' for r in rows) else 'legacy_mixed'
    if a.onpolicy_boost < 1:
        p.error('--onpolicy-boost must be at least one')
    train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation']
    scenes={s:{r['scene'] for r in group} for s,group in [('train',train),('validation',val)]}
    if scenes['train']&scenes['validation']:raise ValueError('Scene group leakage')
    families=('sfp',) if a.sfp_native_only else ('sfp','sc')
    controller_contract=None
    if a.require_verified_command_contract:
        for r in rows:
            if (r.get('actor_training_ready') is not True or
                    r.get('true_geometry_actor_input') is not False or
                    r.get('recorded_complete_targets') is not True or
                    r.get('post_completion_commands_excluded') is not True or
                    r.get('perception_lock_time_s',float('inf'))>r['time_s'] or
                    r.get('action_owner')!='teacher' or
                    r.get('action_target_offsets_s')!=[.05,.1,.15,.2]):
                raise ValueError('Unverified or acausal actor row')
            if (not np.allclose(r['recorded_controller_stiffness'],[90,90,90,50,50,50]) or
                    not np.allclose(r['recorded_controller_damping'],[50,50,50,20,20,20])):
                raise ValueError('Expert/controller mismatch')
            if (not r.get('full_recorded_controller_contract_verified') or
                    not np.allclose(r.get('recorded_controller_wrench_feedback_gains',[]),[.5,.5,.5,0,0,0]) or
                    not np.allclose(r.get('recorded_feedforward_wrench',[]),[0.]*6) or r.get('recorded_trajectory_mode')!=2):
                raise ValueError('Expert force/trajectory contract is not verified')
        controller_contract=dict(stiffness=[90,90,90,50,50,50],
            damping=[50,50,50,20,20,20],action_target_offsets_s=[.05,.1,.15,.2],
            recorded_targets=True,supervisor_reset_actions_excluded=True)
        controller_contract.update(wrench_feedback_gains=[.5,.5,.5,0.,0.,0.],feedforward_wrench=[0.]*6,trajectory_mode=2)
    for family in families:
        if not any(r['family']==family for r in train) or not any(r['family']==family for r in val):
            raise ValueError(f'{family} required in each split')
    stats={}
    for name,values in [('state',np.asarray([r['state'] for r in train])),
                        ('pose',np.asarray([r['predicted_port_tcp_pose9'] for r in train])),
                        ('target',np.concatenate([r['teacher_tcp_targets_predicted_port_pose9'] for r in train]))]:
        stats[name+'_mean']=values.mean(0);stats[name+'_std']=values.std(0).clip(min=.001)
    include_tip=a.tip_input_mode!='none'
    if a.phase_reference=='predicted_tip' and (a.tip_input_mode!='translation' or any(not r.get('predicted_tip_valid') for r in rows)):
        p.error('Plug-based phase sampling requires valid camera-estimated translations for every row')
    if a.sc_action_reference!='tcp':
        if a.tip_input_mode!='translation':p.error('Point targets require camera-estimated body-tip input')
        if any('recorded_tcp_targets_predicted_port_pose9' not in r for r in rows if r['family'] in a.tip_action_families):
            p.error('Missing recorded TCP labels for decoded-command scoring')
        if any(not r.get('predicted_tip_valid') for r in rows if r['family'] in a.tip_action_families):
            p.error('Missing observation-only tip input for a plug-control-point family')
        if not a.reset_action_head:
            previous=(torch.load(a.resume_local,map_location='cpu',weights_only=False) if a.resume_local else {})
            reference=previous.get('sc_action_reference','tcp')
            if reference!=a.sc_action_reference:p.error('Action-reference change requires output-head reset')
            if set(previous.get('tip_action_families',['sc']))!=set(a.tip_action_families):
                p.error('Changing plug-control-point families requires output-head reset')
    constant_body_tip=None
    if include_tip:
        valid=np.asarray([r['predicted_tip_port_pose9'] for r in train if r['predicted_tip_valid']])
        if len(valid)==0:raise ValueError('Missing predicted translation supervision inputs')
        stats['tip_mean']=valid.mean(0);stats['tip_std']=valid.std(0).clip(min=.001)
        # Recover each episode's camera-estimated body offset from its causal
        # inputs. Additional continuation episodes need not exist in the main
        # dataset's lock manifest. Never read label-only grasp calibrations.
        episode_body={}
        for r in train:
            if r['family']=='sc' and r['predicted_tip_valid'] and r['episode'] not in episode_body:
                current=matrix_from_pose9(r['predicted_port_tcp_pose9'])
                episode_body[r['episode']]=current[:3,:3].T@(
                    np.asarray(r['predicted_tip_port_pose9'][:3])-current[:3,3])
        constant_body_tip=np.mean(list(episode_body.values()),axis=0)
    family_episode_count=Counter((r['family'],r['episode']) for r in train)
    family_count=Counter(f for f,_ in family_episode_count)
    weight=[]
    for r in train:
        f=r['family'];signed_z=r['predicted_tip_port_pose9'][2] if a.phase_reference=='predicted_tip' else r['predicted_port_tcp_pose9'][2]
        z=max(0.,-signed_z) if a.contact_aware_phase else abs(signed_z)
        approach=(f=='sfp' and -.105<r['predicted_port_tcp_pose9'][2]<-.08 and
                  np.linalg.norm(r['predicted_port_tcp_pose9'][:2])<.005)
        phase=a.near_phase_sample_weight if z<a.near_phase_threshold_m else 1.5 if z<.09 else 1.
        progress_weight=1.
        if a.prefer_near_axial_progress and f=='sc' and z<.05:
            reference=r['predicted_tip_port_pose9'] if a.sc_action_reference!='tcp' and f in a.tip_action_families else r['predicted_port_tcp_pose9']
            advance=float(r['teacher_tcp_targets_predicted_port_pose9'][0][2]-
                          reference[2])
            progress_weight=5. if advance>.0005 else .2
        weight.append(.5/family_count[f]/family_episode_count[(f,r['episode'])]*phase*progress_weight*
                      (a.aligned_approach_sample_weight if approach else 1.)*
                      (a.onpolicy_boost if r.get('onpolicy_teacher_match') else 1.)*
                      float(r.get('sampling_boost',1.)))
    sampler=WeightedRandomSampler(weight,num_samples=a.updates*a.batch_size,replacement=True)
    loader=DataLoader(LocalRows(train,a.sfp_cache,rows,include_tip=include_tip,history_offsets=a.history_offsets_s,history_masked=a.history_masked,native_image_cache=a.native_image_cache),batch_size=a.batch_size,sampler=sampler,
                      num_workers=a.workers,persistent_workers=a.workers>0,pin_memory=True)
    rng=np.random.default_rng(20261002)
    chosen=[]
    for f in families:
        subset=[r for r in val if r['family']==f]
        if f=='sfp' and a.balanced_native_val:
            groups=[[r for r in subset if r.get('image_kind')=='native_1152x1024_rgb_jpeg'],
                    [r for r in subset if r.get('image_kind')!='native_1152x1024_rgb_jpeg']]
            if not all(groups):p.error('Balanced native validation requires both SFP sources')
        elif f=='sc' and a.balanced_sc_continuation_val:
            groups=[[r for r in subset if not r['episode'].startswith('local_causal_continuation_')],
                    [r for r in subset if r['episode'].startswith('local_causal_continuation_')]]
            if not all(groups):p.error('Matched continuation validation requires both groups')
        else:groups=[subset]
        for group in groups:
            size=(len(group) if a.full_validation or (f=='sc' and a.balanced_sc_continuation_val and group[0]['episode'].startswith('local_causal_continuation_')) else min(512 if len(groups)>1 else 1024,len(group)))
            indices=rng.choice(len(group),size=size,replace=False)
            chosen.extend(group[int(i)] for i in indices)
    manifest=''.join(json.dumps(dict(family=r['family'],episode=r['episode'],scene=r['scene'],
        time_s=r['time_s'],images=r['images']),sort_keys=True)+'\n' for r in chosen)
    (a.output/'validation_manifest.jsonl').write_text(manifest)
    validation_manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest()
    val_loader=DataLoader(LocalRows(chosen,a.sfp_cache,rows,include_tip=include_tip,history_offsets=a.history_offsets_s,history_masked=a.history_masked,native_image_cache=a.native_image_cache),batch_size=a.batch_size,shuffle=False,
                          num_workers=a.workers,persistent_workers=a.workers>0,pin_memory=True)
    model=SharedPortTCPActor(stats,pretrained=False,condition_on_pose=True,use_tip=include_tip,
                            separate_family_orientation_head=a.separate_family_orientation_head,
                            separate_family_action_head=a.separate_family_action_head,
                            use_local_history=a.local_history,decoder_kind=a.decoder_kind).cuda()
    missing,unexpected=model.backbone.load_state_dict(
        torch.load(a.imagenet,map_location='cpu',weights_only=True),strict=False)
    if missing or set(unexpected)!={'fc.weight','fc.bias'}:
        raise ValueError(f'Unexpected ImageNet backbone mapping: {missing}, {unexpected}')
    warm_fields=[]
    if a.warm_transport and a.resume_local:
        p.error('Choose one initialization source')
    if a.warm_visual_only and not a.warm_transport:p.error('Visual warm start needs a representation checkpoint')
    if a.warm_transport:
        prior=torch.load(a.warm_transport,map_location='cpu',weights_only=False)
        warm_prefixes=('backbone.','view_embedding','view_attention.') if a.warm_visual_only else ('backbone.','view_embedding','view_attention.','context.')
        for key,value in prior['model'].items():
            if key.startswith(warm_prefixes):
                if key not in model.state_dict() or model.state_dict()[key].shape!=value.shape:
                    raise ValueError(f'Incompatible warm representation {key}')
                model.state_dict()[key].copy_(value);warm_fields.append(key)
        if not warm_fields or not all(any(k.startswith(prefix) for k in warm_fields) for prefix in warm_prefixes):
            raise ValueError('Warm checkpoint did not transfer the complete requested representation; refusing a silently fresh warm arm')
    if a.resume_local:
        prior=torch.load(a.resume_local,map_location='cpu',weights_only=False)
        if prior.get('architecture')!='shared_port_tcp_local_predicted_pose':
            raise ValueError('Only a predicted-port local actor can be resumed')
        fresh_output={k:v.clone() for k,v in model.state_dict().items() if k.startswith(('action_head.','sc_action_head.','sfp_action_head.')) or k.endswith(('_mean','_std'))}
        transfer=prior['model']
        decoder_prefixes=('action_head.','transformer.','sequence_head.','waypoint_tokens')
        if a.reset_decoder:transfer={k:v for k,v in transfer.items() if not k.startswith(decoder_prefixes)}
        missing,unexpected=model.load_state_dict(transfer,strict=False)
        if a.reset_action_head:
            for k,v in fresh_output.items():model.state_dict()[k].copy_(v)
        expected_missing=set()
        if a.reset_decoder:expected_missing|={k for k in model.state_dict() if k.startswith(decoder_prefixes)}
        if a.separate_family_orientation_head and not prior.get('separate_family_orientation_head'):
            expected_missing|={k for k in model.state_dict() if k.startswith(
                ('sc_orientation_head.','sfp_orientation_head.'))}
        if a.separate_family_action_head and not prior.get('separate_family_action_head'):
            expected_missing|={k for k in model.state_dict() if k.startswith(
                ('sc_action_head.','sfp_action_head.'))}
        if a.local_history and not prior.get('use_local_history'):
            expected_missing|={k for k in model.state_dict() if k.startswith('history_context.')}
        if include_tip and not prior.get('include_tip'):
            expected_missing|={'tip_mean','tip_std'}
            expected_missing|={k for k in model.state_dict() if k.startswith('tip_context.')}
        if set(missing)!=expected_missing or unexpected:
            raise ValueError(f'Incompatible orientation head transfer: {missing}, {unexpected}')
        if a.separate_family_orientation_head and not prior.get('separate_family_orientation_head'):
            model.initialize_family_orientation_heads_from_shared()
        if a.separate_family_action_head and not prior.get('separate_family_action_head'):
            model.initialize_family_action_heads_from_shared()
    if a.selection_orientation_tolerance_deg<=0:
        raise ValueError('Selection orientation tolerance must be positive')
    model.history_max_age_s=max(a.history_offsets_s) if a.history_offsets_s else 2.
    model.masked_state_indices=list(range(13,19)) if a.mask_command_feedback else []
    model.mask_history_command_feedback=bool(a.mask_command_feedback and a.local_history)
    model.masked_tip_indices=list(range(3,9)) if include_tip else []
    model.tip_input_mode=a.tip_input_mode
    model.constant_body_tip=constant_body_tip
    model.sc_action_reference=a.sc_action_reference
    model.tip_action_families=a.tip_action_families
    model.phase_reference=a.phase_reference
    model.contact_aware_phase=a.contact_aware_phase
    model.coordinate_heads_fp32=a.coordinate_heads_fp32
    model.control_path_fp32=a.control_path_fp32
    baseline_validation=evaluate(model,val_loader)
    opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-4)
    best=float('inf');best_step=0;stale=0;history=[];started=time.monotonic();reason='update_cap'
    for step,batch in enumerate(loader,start=1):
        image,state,task,pose,target,_,_,local_history=batch[:8]
        tip=tip_mask=None
        if include_tip:
            tip=batch[8].cuda(non_blocking=True);tip_mask=batch[9].cuda(non_blocking=True)
            if a.tip_input_mode=='disabled':tip_mask=torch.zeros_like(tip_mask)
        if step>a.updates:break
        if time.monotonic()-started>a.minutes*60:reason='wall_budget';break
        image=image.cuda(non_blocking=True);state=state.cuda(non_blocking=True)
        task=task.cuda(non_blocking=True);pose=pose.cuda(non_blocking=True)
        target=target.cuda(non_blocking=True)
        local_history=local_history.cuda(non_blocking=True)
        if a.tip_input_mode=='constant_translation':
            mean_body=pose.new_tensor(constant_body_tip)
            tip=tip.clone()
            tip[:,:3]=(rotation_from_pose9(pose.float())@mean_body[None,:,None]).squeeze(-1)+pose[:,:3]
        model.train();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output,internal=model(image,state,task,external_pose=pose,
                                  external_mask=torch.ones(len(pose),device='cuda',dtype=torch.bool),
                                  local_history=local_history if a.local_history else None,
                                  physical_tip=tip,tip_mask=tip_mask)
        residual=(output.float()-target)/model.target_std
        pose_residual=(internal.float()-pose)/model.pose_std
        decoded=decoded_tcp_targets(output,pose,tip,task,model.sc_action_reference,model.tip_action_families)
        recorded=(batch[10].cuda(non_blocking=True) if model.use_tip else target)
        command=body_translation_from_poses(pose.float(),decoded)
        teacher=body_translation_from_poses(pose.float(),recorded.float())
        loss=(F.smooth_l1_loss(residual[...,:3],torch.zeros_like(residual[...,:3]),beta=.1)
              +.1*F.smooth_l1_loss(residual[...,3:],torch.zeros_like(residual[...,3:]),beta=.1)
              +.3*F.smooth_l1_loss(pose_residual,torch.zeros_like(pose_residual),beta=.1)
              +.5*F.smooth_l1_loss((command-teacher)/.02,torch.zeros_like(command),beta=.1))
        if a.chunk_motion_loss_weight:
            predicted_change=output[:,1:,:3].float()-output[:,:-1,:3].float()
            recorded_change=target[:,1:,:3].float()-target[:,:-1,:3].float()
            loss+=a.chunk_motion_loss_weight*F.smooth_l1_loss(
                (predicted_change-recorded_change)/.0005,torch.zeros_like(predicted_change),beta=.2)
        phase_pose=(torch.where(tip_mask[:,None],tip,pose) if a.phase_reference=='predicted_tip' else pose)
        near=near_port_mask(phase_pose,task,a.lateral_buffer_m,a.contact_aware_phase)
        if near.any():
            delta=((output[near,0,:2]-pose[near,:2])-
                   (target[near,0,:2]-pose[near,:2]))/.002
            loss+=a.near_lateral_loss_weight*F.smooth_l1_loss(delta,torch.zeros_like(delta),beta=.2)
        if a.orientation_loss_weight:
            pred_rot=rotation_from_pose9(output.float())
            true_rot=rotation_from_pose9(target.float())
            cosine=((pred_rot.transpose(-1,-2)@true_rot).diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
            angular=(1-cosine.clamp(-1,1))
            loss+=a.orientation_loss_weight*(angular.mean()+angular[near].mean() if near.any() else angular.mean())
        if a.aligned_approach_axial_loss_weight:
            approach=((pose[:,2]>-.105)&(pose[:,2]<-.08)&
                      (torch.linalg.vector_norm(pose[:,:2],dim=1)<.005)&
                      (task[:,0]>.5))
            if approach.any():
                axial=(output[approach,0,2]-target[approach,0,2])/.002
                loss+=a.aligned_approach_axial_loss_weight*F.smooth_l1_loss(
                    axial,torch.zeros_like(axial),beta=.2)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%100==0:
            report=evaluate(model,val_loader)
            if a.selection_metric=='two_family_insertion':
                terms=[]
                for family in families:
                    terms.extend([report[family]['near_decoded_tcp_lateral_mm']['p95']/.5,
                                  report[family]['near_first_orientation_deg']['p95']/3.,
                                  report[family+'_far_approach']['first_body_command_mm']['p95']/3.,
                                  report[family]['axial_chunk_span_error_mm']['p95']/1.])
                metric=max(terms)
            elif a.selection_metric=='sc_near':
                metric=report['sc']['near_port_lateral_target_mm']['p95']
            elif a.selection_metric=='sc_near_orientation':
                metric=max(report['sc']['near_decoded_tcp_lateral_mm']['p95'],
                           report['sc']['near_first_orientation_deg']['p95']/a.selection_orientation_tolerance_deg)
            elif a.selection_metric in ('sc_correction','sc_motion'):
                correction=report['sc_continuation'];original=report['sc_original']
                metric=max(original['near_decoded_tcp_lateral_mm']['p95'],
                           correction['near_decoded_tcp_lateral_mm']['p95'],
                           correction['first_body_command_mm']['p95']/3.,
                           max(original['near_first_orientation_deg']['p95'],
                               correction['near_first_orientation_deg']['p95'])/a.selection_orientation_tolerance_deg)
                if a.selection_metric=='sc_motion':
                    metric=max(metric,original['axial_chunk_span_error_mm']['p95']/3.,
                               correction['axial_chunk_span_error_mm']['p95']/1.,
                               correction['false_axial_chunk_advance']['mean']/.2,
                               correction['missed_axial_chunk_advance']['mean']/.2)
            elif a.selection_metric=='sc_orientation':
                metric=(report['sc']['near_first_orientation_deg']['p95']+
                        .5*max(0.,report['sc']['near_port_lateral_target_mm']['p95']-1.5))
            elif a.selection_metric=='sc_advancing':
                metric=report['sc_advancing']['first_target_mm']['p95']
            elif a.selection_metric=='sfp_native':
                metric=report['sfp_native']['first_body_command_mm']['p95']
            elif a.selection_metric=='sfp_native_near':
                metric=max(report['sfp_native']['near_port_lateral_target_mm']['p95'],
                           report['sfp_native']['buffered_lateral_target_mm']['p95'])
            elif a.selection_metric=='sfp_aligned_approach':
                approach=report['sfp_native']['aligned_approach_sign_correct']
                metric=(1-approach['mean'])*20+report['sfp_native'][
                    'aligned_approach_axial_target_mm']['p95']
            else:
                metric=max(report[f]['first_body_command_mm']['p95'] for f in families)
            history.append({'step':step,'train_loss':float(loss.detach()),'selection_mm':metric,'validation':report})
            print(json.dumps({'step':step,'loss':float(loss.detach()),'sc_command_p95':report.get('sc',{}).get('first_body_command_mm',{}).get('p95'),
                              'sfp_command_p95':report['sfp']['first_body_command_mm']['p95']}),flush=True)
            if metric<best-.05:
                best=metric;best_step=step;stale=0
                torch.save({'model':model.state_dict(),'architecture':'shared_port_tcp_local_predicted_pose',
                            'trained_families':list(families),
                            'recorded_controller_contract':controller_contract,
                            'separate_family_orientation_head':a.separate_family_orientation_head,
                            'separate_family_action_head':a.separate_family_action_head,
                            'use_local_history':a.local_history,
                            'history_offsets_s':a.history_offsets_s,'history_masked':a.history_masked,
                            'masked_state_indices':model.masked_state_indices,
                            'mask_history_command_feedback':model.mask_history_command_feedback,
                            'include_tip':include_tip,'tip_input_mode':a.tip_input_mode,
                            'decoder_kind':a.decoder_kind,'coordinate_heads_fp32':a.coordinate_heads_fp32,
                            'control_path_fp32':a.control_path_fp32,
                            'sc_action_reference':a.sc_action_reference,
                            'tip_action_families':a.tip_action_families,
                            'physical_tip_families':a.tip_action_families if include_tip else [],
                            'training_phase_reference':a.phase_reference,
                            'contact_aware_phase':a.contact_aware_phase,
                            'actor_image_preprocess_contract':actor_image_contract,
                            'masked_tip_indices':model.masked_tip_indices,
                            'constant_body_tip_mean':constant_body_tip,
                            'pose_input':'frozen_RGB_estimated_port','target_frame':'estimated_port',
                            'warm_transport_representation_only':str(a.warm_transport) if a.warm_transport else None,
                            'warm_visual_only':a.warm_visual_only,
                            'resumed_local_checkpoint':str(a.resume_local) if a.resume_local else None,
                            'true_port_geometry_runtime_input':False,'update':step},a.output/'best.pt')
            else:stale+=1
            if step>=a.min_updates and stale>=a.stale_evaluations and not a.fixed_update_budget:
                reason='validation_plateau';break
    torch.save({'model':model.state_dict(),'architecture':'shared_port_tcp_local_predicted_pose',
                'optimizer_state':opt.state_dict(),
                'best_selection_metric':best,'best_update':best_step,
                'validation_manifest_sha256':validation_manifest_sha256,
                'dataset_rows_sha256':hashlib.sha256((a.dataset/'rows.jsonl').read_bytes()).hexdigest(),
                'warm_transport_representation_only':str(a.warm_transport) if a.warm_transport else None,
                'trained_families':list(families),
                'recorded_controller_contract':controller_contract,
                'separate_family_orientation_head':a.separate_family_orientation_head,
                'separate_family_action_head':a.separate_family_action_head,
                'use_local_history':a.local_history,
                            'history_offsets_s':a.history_offsets_s,'history_masked':a.history_masked,
                'masked_state_indices':model.masked_state_indices,
                'mask_history_command_feedback':model.mask_history_command_feedback,
                'include_tip':include_tip,'tip_input_mode':a.tip_input_mode,
                            'decoder_kind':a.decoder_kind,'coordinate_heads_fp32':a.coordinate_heads_fp32,
                            'control_path_fp32':a.control_path_fp32,
                            'sc_action_reference':a.sc_action_reference,
                            'tip_action_families':a.tip_action_families,
                            'physical_tip_families':a.tip_action_families if include_tip else [],
                            'training_phase_reference':a.phase_reference,
                            'contact_aware_phase':a.contact_aware_phase,
                            'actor_image_preprocess_contract':actor_image_contract,
                'masked_tip_indices':model.masked_tip_indices,
                'constant_body_tip_mean':constant_body_tip,
                'pose_input':'frozen_RGB_estimated_port','target_frame':'estimated_port',
                'warm_visual_only':a.warm_visual_only,
                'resumed_local_checkpoint':str(a.resume_local) if a.resume_local else None,
                'true_port_geometry_runtime_input':False,'update':step},a.output/'last.pt')
    selected=next((r['validation'] for r in history if r['step']==best_step),None)
    result={'schema':'aic_predicted_port_local_actor_bc/v1','best_step':best_step,'updates_completed':step,
            'stop_reason':reason,'selected_validation':selected,
            'initialization_validation':baseline_validation,
            'train_episodes':len({(r['family'],r['episode']) for r in train}),
            'validation_episodes':len({(r['family'],r['episode']) for r in val}),
            'train_rows':len(train),'validation_sample_rows':len(chosen),
            'full_validation':a.full_validation,'validation_manifest_sha256':validation_manifest_sha256,
            'extra_rows':str(a.extra_rows) if a.extra_rows else None,
            'extra_validation_only':a.extra_validation_only,
            'onpolicy_train_rows':sum(bool(r.get('onpolicy_teacher_match')) for r in train),
             'onpolicy_boost':a.onpolicy_boost,'workers':a.workers,
             'prefer_near_axial_progress':a.prefer_near_axial_progress,
            'warm_representation_tensors':len(warm_fields),
            'warm_visual_only':a.warm_visual_only,
            'mask_history_command_feedback':model.mask_history_command_feedback,
            'training_phase_reference':a.phase_reference,
            'contact_aware_phase':a.contact_aware_phase,
            'actor_image_preprocess_contract':actor_image_contract,
            'resumed_local_checkpoint':str(a.resume_local) if a.resume_local else None,
            'selection_metric':a.selection_metric,
            'masked_state_indices':model.masked_state_indices,
            'selection_orientation_tolerance_deg':a.selection_orientation_tolerance_deg,
            'tip_input_mode':a.tip_input_mode,'tip_orientation_input':False,
            'sc_action_reference':a.sc_action_reference,'reset_action_head':a.reset_action_head,
            'decoder_kind':a.decoder_kind,'reset_decoder':a.reset_decoder,
            'coordinate_heads_fp32':a.coordinate_heads_fp32,'chunk_motion_loss_weight':a.chunk_motion_loss_weight,
            'control_path_fp32':a.control_path_fp32,
            'fixed_update_budget':a.fixed_update_budget,'stale_evaluations_at_end':stale,
            'orientation_loss_weight':a.orientation_loss_weight,
            'near_lateral_loss_weight':a.near_lateral_loss_weight,
            'near_phase_sample_weight':a.near_phase_sample_weight,
            'near_phase_threshold_m':a.near_phase_threshold_m,
            'lateral_buffer_m':a.lateral_buffer_m,
            'separate_family_orientation_head':a.separate_family_orientation_head,
            'separate_family_action_head':a.separate_family_action_head,
            'use_local_history':a.local_history,
                            'history_offsets_s':a.history_offsets_s,'history_masked':a.history_masked,
            'pose_input':'frozen_RGB_estimated_port','teacher_targets_reexpressed_in_estimated_port':True,
            'sfp_native_only':a.sfp_native_only,
            'aligned_approach_sample_weight':a.aligned_approach_sample_weight,
            'aligned_approach_axial_loss_weight':a.aligned_approach_axial_loss_weight,
            'autonomous_insertions_measured':False,'reserved_final_opened':False}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    from project_gpu_guard import selected_gpu_reservation
    with selected_gpu_reservation():main()
