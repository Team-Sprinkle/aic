#!/usr/bin/env python3
"""Build 8/12-slot recorded-action labels for the frozen transport feature cache.

The four original slots remain unchanged. Later slots use the first recorded
teacher command at the matching 50 ms row, converted into the anchor TCP frame.
Discontinuous clocks or command indices invalidate an anchor; no padding or
synthetic actions are used for training.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def pose_matrix(pose):
    result=np.eye(4)
    result[:3,:3]=Rotation.from_quat(pose[3:]).as_matrix()
    result[:3,3]=pose[:3]
    return result


def pose9(matrix):
    return np.r_[matrix[:3,3],matrix[:3,:2].T.reshape(-1)].astype(np.float32)


def source_time(family,row):
    return float(row['timestamp_s'] if family=='sfp' else row['sim_time'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--sfp-labels',type=Path,required=True)
    p.add_argument('--sc-labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--slot-stride',type=int,default=1,
                   help='Number of 50 ms source commands between action slots (1 or 2)')
    a=p.parse_args()
    if a.slot_stride not in (1,2):p.error('slot-stride must be 1 or 2')
    if a.output.exists():p.error('Output exists; preserve immutable labels')
    source={};episodes=defaultdict(list)
    for family,path in (('sfp',a.sfp_labels),('sc',a.sc_labels)):
        with path.open() as stream:
            for line in stream:
                row=json.loads(line)
                key=(family,int(row['episode_index']),int(row['frame_index'] if family=='sfp' else row['frame']))
                if key in source:raise ValueError(f'Duplicate source row {key}')
                source[key]=row
                episodes[key[:2]].append(key)
    positions={}
    for keys in episodes.values():
        keys.sort(key=lambda key:source_time(key[0],source[key]))
        for i,key in enumerate(keys):positions[key]=i
    cached=[json.loads(line) for line in (a.cache/'rows.jsonl').open()]
    target=np.zeros((len(cached),12,9),np.float32)
    valid=np.zeros((len(cached),12),np.bool_)
    reason=Counter();four_error=[]
    for n,row in enumerate(cached):
        family=row['family'];key=(family,int(row['episode']),int(row['frame']))
        if key not in source:raise ValueError(f'Cache row missing from source: {key}')
        anchor=source[key];keys=episodes[key[:2]];i=positions[key]
        anchor_pose=pose_matrix(anchor['observed_tcp_pose_port_frame'] if family=='sfp'
                                else anchor['observed_tcp_pose_opening_frame'])
        anchor_inv=np.linalg.inv(anchor_pose)
        targets=anchor['teacher_tcp_target_poses_port_frame'] if family=='sfp' else anchor['teacher_tcp_target_poses_opening_frame']
        original=np.asarray(row['target'],np.float32)
        if a.slot_stride==1:
            for j in range(4):
                reconstructed=pose9(anchor_inv@pose_matrix(targets[j]))
                four_error.append(float(np.linalg.norm((original[j]-reconstructed)[:3])*1000))
            # Preserve the trained four-step target exactly.
            target[n,:4]=original
            valid[n,:4]=True
        else:
            target[n,0]=pose9(anchor_inv@pose_matrix(targets[0]))
            valid[n,0]=True
            four_error.append(float(np.linalg.norm((original[0]-target[n,0])[:3])*1000))
        t0=source_time(family,anchor)
        anchor_command=(int(anchor['command_indices'][0]) if family=='sc' else None)
        if family=='sc' and any(int(c)!=anchor_command+j for j,c in enumerate(anchor['command_indices'])):
            reason['anchor_sc_command_gap']+=1;continue
        continuous=True
        for source_step in range(1,12*a.slot_stride):
            if i+source_step>=len(keys):reason['episode_end']+=1;break
            next_row=source[keys[i+source_step]]
            if family=='sfp' and keys[i+source_step][2]!=key[2]+source_step:
                reason['sfp_frame_gap']+=1;break
            offset=source_time(family,next_row)-t0
            if abs(offset-.05*source_step)>.025:
                reason['time_gap']+=1;break
            if family=='sc' and int(next_row['command_indices'][0])!=anchor_command+source_step:
                reason['sc_command_gap']+=1;break
            if next_row['task_vector']!=anchor['task_vector']:
                reason['task_changed']+=1;break
            if source_step%a.slot_stride:continue
            j=source_step//a.slot_stride
            if j>=4 or a.slot_stride!=1:
                later=next_row['teacher_tcp_target_poses_port_frame'][0] if family=='sfp' else next_row['teacher_tcp_target_poses_opening_frame'][0]
                target[n,j]=pose9(anchor_inv@pose_matrix(later))
                valid[n,j]=True
            elif family=='sc':
                later=next_row['teacher_tcp_target_poses_opening_frame'][0]
                error=np.linalg.norm((target[n,j]-pose9(anchor_inv@pose_matrix(later)))[:3])*1000
                if error>.25:
                    reason['sc_overlap_mismatch']+=1;continuous=False;break
        if not continuous:valid[n,4:]=False
    length=valid.sum(1)
    counts={str(k):dict(total=0,valid8=0,valid12=0) for k in ('train_sfp','train_sc','validation_sfp','validation_sc')}
    for row,l in zip(cached,length):
        key=f"{row['split']}_{row['family']}";c=counts[key];c['total']+=1;c['valid8']+=int(l>=8);c['valid12']+=int(l>=12)
    a.output.mkdir(parents=True)
    np.save(a.output/'targets12.npy',target)
    np.save(a.output/'valid12.npy',valid)
    manifest=dict(schema='aic_extended_route_chunks/v1',cache=str(a.cache),rows=len(cached),
                  slot_s=.05*a.slot_stride,source_sha256={str(path):digest(path) for path in (a.sfp_labels,a.sc_labels,a.cache/'rows.jsonl')},
                  counts=counts,invalid_reason=dict(reason),
                  original_four_reconstruction_translation_mm=dict(max=max(four_error),p95=float(np.quantile(four_error,.95))),
                  no_padding=True,episode_reset_split_preserved=True,final_scenes_opened=False)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
