#!/usr/bin/env python3
"""Cache frozen observation-only transport features and offline route labels.

The SC MoveIt plan supplies targets and stage labels only. No plan value is
included in the feature tensor passed to the temporal actor.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18

from train_clean_transport_bc import TransportSamples
from train_shared_port_tcp_bc import SharedPortTCPActor, pose9, read_rows
from audit_temporal_transport_labels import STAGES


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sfp-labels',type=Path,required=True)
    p.add_argument('--sc-labels',type=Path,required=True)
    p.add_argument('--sc-spec',type=Path,required=True)
    p.add_argument('--sfp-cache',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=48)
    p.add_argument('--stride',type=int,default=4,
                   help='Number of 50 ms source rows between cached policy decisions')
    p.add_argument('--feature-source',choices=('policy_context','imagenet_views'),
                   default='policy_context')
    p.add_argument('--imagenet-resnet',type=Path,
                   help='Required for image-only, per-view ImageNet feature caching')
    a=p.parse_args()
    if a.stride<1:raise ValueError('Stride must be positive')
    if a.feature_source=='imagenet_views' and a.imagenet_resnet is None:
        p.error('--imagenet-resnet is required for imagenet_views')
    if a.output.exists():p.error('Output exists; this cache is immutable')
    a.output.mkdir(parents=True)
    torch.set_num_threads(8)
    corrective_sfp_keys=set()
    official_sfp_keys=set()
    actor_prefix_context_keys=set()
    pause_head_only_keys=set()
    pause_tail_keys=set()
    with a.sfp_labels.open() as f:
        for line in f:
            row=json.loads(line)
            key=(int(row['episode_index']),int(row['frame_index']))
            if row.get('pause_head_only_source'):
                pause_head_only_keys.add(key)
                if row.get('past_transport_cutoff'):
                    pause_tail_keys.add(key)
            if row.get('privileged_teacher_after_takeover'):
                if row.get('actor_prefix_used_as_supervision') is not False:
                    raise ValueError('Corrective label must exclude actor prefix')
                corrective_sfp_keys.add(key)
            if row.get('official_anchor_teacher'):
                official_sfp_keys.add(key)
            if row.get('actor_prefix_context_only'):
                if row.get('privileged_teacher_after_takeover') or row.get('actor_prefix_used_as_supervision'):
                    raise ValueError('Actor-prefix context cannot be a supervised teacher row')
                actor_prefix_context_keys.add(key)
    rows=read_rows([('sfp',a.sfp_labels),('sc',a.sc_labels)],target_frame='body')
    original_sc={}
    with a.sc_labels.open() as f:
        for line in f:
            x=json.loads(line);original_sc[(int(x['episode_index']),int(x['frame']))]=x
    spec=json.loads(a.sc_spec.read_text())
    episodes={e['case']:e for e in spec['episodes']}
    plans={}
    for case,e in episodes.items():
        folder=Path(e['insert_plan'])
        waypoints=json.loads((folder/'moveit_joint_trajectory.json').read_text())['waypoints']
        summary=json.loads((folder/'plan_summary.json').read_text())
        assert tuple(g['name'] for g in summary['candidate']['route_subgoals'])==STAGES
        plans[case]=(waypoints,[float(w['timestamp']) for w in waypoints],
                     {g['name']:g['pose'] for g in summary['candidate']['route_subgoals']})
    indices=[];last_key=None;position=0
    for i,r in enumerate(rows):
        key=(r['family'],r['episode'])
        if key!=last_key:position=0;last_key=key
        if position%a.stride==0:indices.append(i)
        position+=1
    source=TransportSamples(rows,a.sfp_cache)
    loader=DataLoader(Subset(source,indices),batch_size=a.batch_size,shuffle=False,
                      num_workers=6,pin_memory=True,persistent_workers=True)
    cp=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert cp['target_frame']=='body' and not cp.get('include_elapsed')
    stats={k:cp['model'][k].numpy() for k in
           ('state_mean','state_std','pose_mean','pose_std','target_mean','target_std')}
    if a.feature_source=='policy_context':
        model=SharedPortTCPActor(stats,pretrained=False,condition_on_pose=False).cuda().eval()
        model.load_state_dict(cp['model'],strict=True)
        model.masked_state_indices=cp.get('masked_state_indices',[])
        backbone=model.backbone
    else:
        backbone=resnet18(weights=None)
        backbone.fc=torch.nn.Identity()
        state=torch.load(a.imagenet_resnet,map_location='cpu',weights_only=True)
        backbone.load_state_dict({k:v for k,v in state.items() if not k.startswith('fc.')},strict=True)
        backbone=backbone.cuda().eval()
    features=[]
    with torch.inference_mode():
        for batch in loader:
            image,state,task,*_=batch
            n=image.shape[0]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                x=image.cuda(non_blocking=True).float().reshape(n*3,3,224,224)/255.
                x=((x-x.new_tensor((.485,.456,.406))[None,:,None,None]) /
                   x.new_tensor((.229,.224,.225))[None,:,None,None])
                views=backbone(x).reshape(n,3,512)
                if a.feature_source=='policy_context':
                    views=views+model.view_embedding[None]
                    views,_=model.view_attention(views,views,views,need_weights=False)
                    st=(state.cuda(non_blocking=True)-model.state_mean)/model.state_std
                    if model.masked_state_indices:
                        st[:,model.masked_state_indices]=0
                    feat=model.context(torch.cat((views.mean(1),st,task.cuda(non_blocking=True)),-1))
                else:
                    # Retain the three view identities and their global appearance
                    # separately; the temporal head learns the fusion.
                    feat=views.reshape(n,1536)
            features.append(feat.float().cpu().numpy().astype(np.float16))
    array=np.concatenate(features)
    assert len(array)==len(indices)
    np.save(a.output/'features.npy',array)
    counts=Counter();metadata=[]
    for index in indices:
        r=rows[index];family=r['family'];stage=6;handoff=-1
        subgoal=r['targets'][0]
        if family=='sc':
            src=original_sc[(r['episode'],r['frame'])]
            case=src['episode_id']
            if case in episodes:
                episode=episodes[case]
                plan,times,goals=plans[case]
                t=float(src['sim_time'])-float(src['episode_start_sim_time'])
                j=max(0,min(len(plan)-1,bisect_right(times,t)-1))
                name=plan[j]['diagnostics']['segment'];stage=STAGES.index(name)
                goal=goals[name]
                current=np.asarray(src['state'][:7]);cur_rot=Rotation.from_quat(current[3:])
                rel=cur_rot.inv()*Rotation.from_quat(goal['orientation_xyzw'])
                rel_pos=cur_rot.inv().apply(np.asarray(goal['position'])-current[:3])
                subgoal=pose9(np.r_[rel_pos,rel.as_quat()])
                # The local-start event is an offline handoff label; live gating
                # still requires observed progress and stability.
                handoff=int(stage==5 and 0<=episode['local_start_sim_s']-r['elapsed_s']-
                            float(src['episode_start_sim_time'])<=.5)
            elif src.get('body_target_only') and src.get('physical_tip_label_available') is False:
                # Scored qualification-family CheatCode transport has causal
                # command labels but no MoveIt stage plan or physical-tip label.
                # Neither is an input to the route diffusion actor.
                stage=6
            else:
                raise ValueError(f'Unrecognized SC episode without plan: {case}')
        counts[(r['split'],family,stage)]+=1
        metadata.append(dict(family=family,episode=r['episode'],scene=r['scene'],
                             split=r['split'],frame=r['frame'],time_s=r['elapsed_s'],
                             state=r['state'].tolist(),task=r['task'].tolist(),
                             target=r['targets'].tolist(),stage=stage,
                             corrective_teacher=(family=='sfp' and
                                 (r['episode'],r['frame']) in corrective_sfp_keys),
                             official_anchor_teacher=((r['episode'],r['frame']) in official_sfp_keys
                                 if family=='sfp' else bool(src.get('official_anchor_teacher'))),
                             actor_prefix_context_only=(family=='sfp' and
                                 (r['episode'],r['frame']) in actor_prefix_context_keys),
                             pause_head_only_source=(family=='sfp' and
                                 (r['episode'],r['frame']) in pause_head_only_keys),
                             past_transport_cutoff=(family=='sfp' and
                                 (r['episode'],r['frame']) in pause_tail_keys),
                             subgoal=np.asarray(subgoal).tolist(),handoff=handoff))
    with (a.output/'rows.jsonl').open('w') as f:
        for row in metadata:f.write(json.dumps(row,separators=(',',':'))+'\n')
    manifest=dict(schema='aic_temporal_transport_frozen_features/v1',rows=len(metadata),
                  counts={str(k):v for k,v in sorted(counts.items())},
                  feature_shape=list(array.shape),stage_names=list(STAGES)+['generic_no_plan_transport'],
                  stride=f'every {a.stride}th retained row per episode; timestamps preserved',
                  stride_source_rows=a.stride,
                  feature_source=a.feature_source,
                  corrective_sfp_cached_rows=sum(r['corrective_teacher'] for r in metadata),
                  official_anchor_cached_rows=sum(r['official_anchor_teacher'] for r in metadata),
                  pause_head_only_cached_rows=sum(r['pause_head_only_source'] for r in metadata),
                  past_transport_cutoff_cached_rows=sum(r['past_transport_cutoff'] for r in metadata),
                  feature_inputs=('three independent ImageNet ResNet18 view vectors; measured state/task are separate temporal inputs'
                                  if a.feature_source=='imagenet_views' else
                                  'three RGB cameras, measured 32D state, 10D task only'),
                  labels='teacher action; SC LLB plan stage/subgoal/local-start where available, generic stage for scored SC CheatCode transport; offline only',
                  sources_sha256={str(q):digest(q) for q in (a.sfp_labels,a.sc_labels,a.sc_spec,a.checkpoint,
                                                           *([a.imagenet_resnet] if a.imagenet_resnet else []))},
                  final_scenes_opened=False,rl_started=False)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(rows=len(metadata),feature_shape=array.shape,counts=manifest['counts'])))


if __name__=='__main__':main()
