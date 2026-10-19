#!/usr/bin/env python3
"""Versioned local BC labels using a causal, RGB-only static port lock.

Recorded teacher commands are full targets in the predicted port frame. True
port geometry is used only AFTER prediction freezing, for diagnostic scoring.
Original labels/images and complete-episode splits remain unchanged.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from sc_shadow_handoff_verifier import SCShadowHandoffVerifier
from shared_port_tcp_geometry import matrix_from_xyz_xyzw

ROOT=Path(__file__).resolve().parents[1]

def pose9(m):
    return np.r_[m[:3,3],m[:3,:2].T.reshape(-1)].tolist()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--transport',type=Path,required=True)
    p.add_argument('--local',type=Path,required=True)
    p.add_argument('--old-dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--reuse-predictions',type=Path,
                   help='Replay immutable RGB predictions; no inference or new images')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous artifacts')
    a.output.mkdir(parents=True)
    torch.set_num_threads(4);cv2.setNumThreads(2)
    modelroot=ROOT/'outputs/experiments/2026-10-02_two_family_perception'
    paths=[modelroot/'joint_locator_v1/best.pt',
           Path('/var/tmp/chmin_aic_20260918_act/shadow_handoff_20261003/sc_spatial_identity.pt'),
           Path('/var/tmp/chmin_aic_20260918_act/shadow_handoff_20261003/sc_orientation_oneport.pt'),
           modelroot/'sc_native_refiner_v1/best.pt',
           ROOT/'configs/hierarchical_recovery/sc_visual_standoff_20261001.json',
           ROOT/'configs/hierarchical_recovery/sc_board_to_port_training_fixed.json',
           modelroot/'local_sc_orientation_prior_v1.json']
    verifier=(None if a.reuse_predictions else SCShadowHandoffVerifier(*paths,device=a.device))
    reused=({r['episode']:r['predictions'] for r in json.loads(a.reuse_predictions.read_text())}
            if a.reuse_predictions else {})
    by_episode=defaultdict(list)
    for row in map(json.loads,a.transport.open()):by_episode[row['episode_id']].append(row)
    frozen=[];locks={};counts=Counter()
    for episode,source in sorted(by_episode.items()):
        source.sort(key=lambda r:r['sim_time']);end=source[-1]['sim_time'];chosen=[]
        for age in (.4,.2,0.):
            row=max((r for r in source if r['sim_time']<=end-age),key=lambda r:r['sim_time'])
            if not chosen or row['frame']!=chosen[-1]['frame']:chosen.append(row)
        predictions=[]
        for row in chosen:
            if a.reuse_predictions:
                predictions.append(next(r for r in reused[episode] if r['frame']==row['frame']))
                continue
            images={c:cv2.cvtColor(cv2.imread(row['images'][c]),cv2.COLOR_BGR2RGB)
                    for c in ('center','left','right')}
            report=verifier.inspect(images,np.asarray(row['state'],np.float32),
                                    int(np.argmax(row['task_vector'][2:4])))
            predictions.append({'time_s':row['sim_time'],'frame':row['frame'],
                                'images':row['images'],'report':report})
        # Pose observability is distinct from transport readiness. These two
        # rejection reasons occur AFTER identity/refinement/rail agreement and
        # provide a coherent pose. They are valid training pose measurements,
        # but NEVER authorize autonomous handoff or insertion.
        good=[r for r in predictions if r['report']['reason'] in (
            'clear_selected_port_and_safe_standoff','not_at_selected_standoff','wrong_tcp_posture')]
        lock=None
        if len(good)>=2:
            xyz=np.asarray([r['report']['predicted_opening_base_m'] for r in good])
            rotations=Rotation.from_quat([r['report']['predicted_port_quat_xyzw'] for r in good])
            center=rotations.mean();spread=np.max(np.linalg.norm(xyz-np.median(xyz,axis=0),axis=1))
            angular=np.max((center.inv()*rotations).magnitude())
            if spread<=.0005 and angular<=np.deg2rad(1.):
                m=np.eye(4);m[:3,3]=np.median(xyz,axis=0);m[:3,:3]=center.as_matrix()
                lock={'matrix':m.tolist(),'time_s':good[-1]['time_s'],
                      'translation_spread_mm':float(spread*1000),
                      'orientation_spread_deg':float(np.rad2deg(angular))}
                locks[episode]=lock
        frozen.append({'episode':episode,'split':source[0]['split'],
                       'predictions':predictions,'lock':lock,'geometry_input':False})
        counts['locked_episodes' if lock else 'rejected_episodes']+=1
        print(episode,bool(lock),flush=True)
    # Predictions are persisted before true geometry is accessed in scoring.
    (a.output/'frozen_predictions.json').write_text(json.dumps(frozen,indent=2)+'\n')
    rows=[];errors=defaultdict(list);episodes=defaultdict(set);scenes={}
    for source in map(json.loads,a.local.open()):
        episode=source['episode_id'];lock=locks.get(episode)
        if lock is None:counts['missing_lock_rows']+=1;continue
        if source['sim_time']<lock['time_s']:counts['pre_lock_rows']+=1;continue
        if source['official_tier3']<75 or source['action_owner']!='teacher':
            raise ValueError('Only verified teacher local rows may enter expert BC')
        predicted=np.asarray(lock['matrix']);tcp=matrix_from_xyz_xyzw(source['state'][:7])
        inv=np.linalg.inv(predicted)
        targets=[pose9(inv@matrix_from_xyz_xyzw(v))
                 for v in source['teacher_tcp_target_poses_base_frame']]
        if len(targets)!=4:raise ValueError('Four complete recorded command targets required')
        rows.append({'schema':'aic_predicted_port_local_bc/v1','family':'sc','episode':episode,
                     'episode_index':source['episode_index'],'scene':source['scene_sha256'],
                     'split':source['split'],'frame':source['frame'],'time_s':source['sim_time'],
                     'images':source['images'],'cache_index':None,'state':source['state'],
                     'task':source['task_vector'],'predicted_port_tcp_pose9':pose9(inv@tcp),
                     'teacher_tcp_targets_predicted_port_pose9':targets,
                     'perception_lock_time_s':lock['time_s'],
                     'perception_age_s':source['sim_time']-lock['time_s'],
                     'static_port_lock':True,'true_geometry_actor_input':False,
                     'recorded_complete_targets':True,'source_command_indices':source['command_indices'],
                     'source_command_offsets_s':source['source_command_time_offsets_s']})
        key=('sc',source['split']);episodes[key].add(episode)
        if episode not in scenes:scenes[episode]=source['scene_sha256']
        true=tcp@np.linalg.inv(matrix_from_xyz_xyzw(source['observed_tcp_pose_opening_frame']))
        errors[key].append([float(np.linalg.norm(predicted[:3,3]-true[:3,3])*1000),
                           float((Rotation.from_matrix(true[:3,:3]).inv()*Rotation.from_matrix(predicted[:3,:3])).magnitude()*180/np.pi)])
    # SFP data is held identical; this diagnostic repairs SC acquisition frames.
    for row in map(json.loads,(a.old_dataset/'rows.jsonl').open()):
        if row['family']=='sfp':rows.append(row);episodes[('sfp',row['split'])].add(row['episode'])
    with (a.output/'rows.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
    split_groups=defaultdict(set)
    for row in rows:split_groups[row['split']].add(row['scene'])
    assert not split_groups['train']&split_groups['validation'],'Reset split overlap'
    summary={'schema':'aic_locked_handoff_local_bc/v1','source_hashes':{
        str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.transport,a.local,a.old_dataset/'rows.jsonl']},
        'model_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'counts':dict(counts),'rows':dict(Counter(f"{r['family']}/{r['split']}" for r in rows)),
        'episodes':{str(k):len(v) for k,v in episodes.items()},'pose_errors_posthoc_only':{
        str(k):{'position_p50_p95_mm':np.percentile(np.asarray(v)[:,0],[50,95]).tolist(),
                'orientation_p50_p95_deg':np.percentile(np.asarray(v)[:,1],[50,95]).tolist()} for k,v in errors.items()},
        'original_sources_preserved':True,'frame_contract':'Full recorded TCP targets in causal static RGB-predicted opening frame',
        'runtime_requires_same_handoff_estimator_and_static_pose_lock':True,'final_scenes_opened':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
