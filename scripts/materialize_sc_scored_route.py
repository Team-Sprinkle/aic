#!/usr/bin/env python3
"""Causal four-command TCP-body transport labels from successful SC scored bags.

Port geometry defines offline labels only. The actor consumes images, measured
state, task identity, and previous executed commands, never this geometry.
"""
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from materialize_sfp_scored_transport import at, pose, posematrix, source_path
from lerobot_robot_aic.task_encoding import encode_task_vector


CAMERAS=('center','left','right')


def edge(edges,parent,child):
    value=edges[parent+'|'+child][0][1]
    result=np.eye(4)
    result[:3,3]=[value['translation'][k] for k in 'xyz']
    result[:3,:3]=Rotation.from_quat([value['rotation'][k] for k in 'xyzw']).as_matrix()
    return result


def port_opening(stream,port):
    edges=stream['edges'];parent=f'task_board/sc_port_{port}'
    world_base=edge(edges,'world','tabletop')@edge(edges,'tabletop','base_link')
    world_port=(edge(edges,'aic_world','task_board')@
                edge(edges,'task_board',parent)@
                edge(edges,parent,parent+'/sc_port_base_link_entrance'))
    return np.linalg.inv(world_base)@world_port


def materialize(audit,episode_index,split,out):
    stream=json.loads((audit/'scored_stream.json').read_text())
    scoring=yaml.safe_load((audit/'results/scoring.yaml').read_text())
    trial=next(value for key,value in scoring.items() if key.startswith('trial_'))
    if trial.get('tier_2',{}).get('categories',{}).get('contacts',{}).get('message')!='No contact detected.':
        raise ValueError('Scored contact in SC transport source')
    task=stream['task'];port=int(task['target_module_name'].split('_')[-1])
    opening=port_opening(stream,port)
    scene=hashlib.sha256(json.dumps({k:stream['reset_identity'][k] for k in ('scene','robot')},
                                   sort_keys=True).encode()).hexdigest()
    commands=[c for c in stream['commands'] if c['kind']=='pose']
    if len(commands)<100:raise ValueError('Too few pose commands')
    if {c['message']['header']['frame_id'] for c in commands}!={'gripper/tcp'}:
        raise ValueError('SC teacher commands are not TCP-frame pose targets')
    times=[c['sim_ns'] for c in commands]
    controllers=stream['streams']['/aic_controller/controller_state']
    joints=stream['streams']['/joint_states']
    wrenches=stream['streams']['/fts_broadcaster/wrench']
    rows=[];bad=Counter();reference_errors=[]
    for i,command in enumerate(commands):
        stamp=command['sim_ns']
        try:
            ct,ctrl=at(controllers,stamp)
            if stamp-ct>100_000_000:raise ValueError('stale command TCP')
            target=posematrix(ctrl['tcp_pose'])@posematrix(command['message']['pose'])
            camera_stamps={};images={}
            for camera in CAMERAS:
                it,path=at(stream['cameras'][camera],stamp)
                if stamp-it>=50_000_000:raise ValueError('stale image')
                camera_stamps[camera]=it;images[camera]=source_path(audit,path)
            observation=min(camera_stamps.values())
            if max(camera_stamps.values())-observation>1_000_000:
                raise ValueError('camera mismatch')
            ct,ctrl=at(controllers,observation)
            jt,joint=at(joints,observation)
            wt,wrench=at(wrenches,observation)
            if max(observation-ct,observation-jt,observation-wt)>100_000_000:
                raise ValueError('stale state')
            tcp=posematrix(ctrl['tcp_pose'])
            named=dict(zip(joint['name'],joint['position']))
            ordered=[named[k] for k in stream['joint_order']]+[named['gripper/left_finger_joint']/2]
            velocity=ctrl['tcp_velocity'];force=wrench['wrench']
            state=pose(tcp)+[velocity[k][a] for k in ('linear','angular') for a in 'xyz']+\
                list(ctrl['tcp_error'])+ordered+[force[k][a] for k in ('force','torque') for a in 'xyz']
            if len(state)!=32:raise ValueError('wrong state length')
            future=bisect_right(controllers,stamp+50_000_000,key=lambda r:r[0])-1
            if future>=0:
                ref=posematrix(controllers[future][1]['reference_tcp_pose'])
                reference_errors.append(float(np.linalg.norm(target[:3,3]-ref[:3,3])*1000))
            rows.append(dict(index=i,t=stamp,target=target,tcp=tcp,state=state,
                             relative=np.linalg.inv(opening)@tcp,images=images,
                             image_stamps=camera_stamps))
        except (ValueError,KeyError,FileNotFoundError) as exc:
            bad[type(exc).__name__+':'+str(exc)[:60]]+=1
    if len(rows)<100:raise ValueError(f'Insufficient causal SC rows: {dict(bad)}')
    # Retain only the transport prefix up to the first measured safe standoff.
    # This makes a later partial insertion usable as a *transport-only* label
    # without treating its failed insertion actions as expert behavior.
    buffered=next((r for r in rows if -100<=r['relative'][2,3]*1000<=-70 and
                   np.linalg.norm(r['relative'][:2,3])*1000<=10),None)
    if buffered is None:raise ValueError('No measured SC safe-buffer visit')
    cutoff=buffered['t']
    pre_cutoff_force_peak=max(float(np.linalg.norm(r['state'][26:29]))
                              for r in rows if r['t']<cutoff)
    if pre_cutoff_force_peak>40:
        raise ValueError(f'High pre-buffer measured wrist force {pre_cutoff_force_peak:.2f} N')
    by_index={r['index']:r for r in rows}
    task_vector=np.asarray(encode_task_vector(task_family='sc_to_sc',
        target_port_index=port,target_card_index=-1),float).tolist()
    written=0
    for anchor in rows:
        if anchor['t']>=cutoff:break
        indices=[bisect_right(times,anchor['t']+h*50_000_000)-1 for h in range(4)]
        if any(j not in by_index for j in indices):bad['missing_chunk_member']+=1;continue
        chunk=[by_index[j] for j in indices]
        if any(r['t']>cutoff or anchor['t']+h*50_000_000-r['t']>150_000_000
               for h,r in enumerate(chunk)):
            bad['stale_or_postcutoff_chunk']+=1;continue
        data=dict(schema='aic_sc_scored_route_chunk/v1',episode_index=episode_index,
                  episode_id=audit.name,scene_sha256=scene,split=split,
                  frame=anchor['index'],sim_time=anchor['t']/1e9,
                  episode_start_sim_time=rows[0]['t']/1e9,images=anchor['images'],
                  image_sim_ns=anchor['image_stamps'],
                  state=anchor['state'],task_vector=task_vector,
                  observed_tcp_pose_opening_frame=pose(anchor['relative']),
                  teacher_tcp_target_poses_opening_frame=[pose(np.linalg.inv(opening)@r['target'])
                                                          for r in chunk],
                  # The transport actor does not consume physical tip labels.
                  observed_sc_tip_pose_opening_frame=[0,0,0,0,0,0,1],
                  physical_tip_label_available=False,body_target_only=True,
                  source_scored_stream=str(audit/'scored_stream.json'),
                  target_source='recorded gripper/tcp command with causal measured TCP',
                  official_tier3=stream['official_tier3'],geometry_actor_input=False,
                  transport_only_from_nonfull=stream['official_tier3']<75)
        out.write(json.dumps(data,separators=(',',':'))+'\n');written+=1
    if written<30:raise ValueError(f'Too few SC chunks: {written}')
    return dict(audit=str(audit),episode_index=episode_index,split=split,
                selected_port=port,scene_group=scene,joined=len(rows),transport_chunks=written,
                cutoff_axial_mm=float(buffered['relative'][2,3]*1000),
                cutoff_lateral_mm=float(np.linalg.norm(buffered['relative'][:2,3])*1000),
                transport_only_from_nonfull=stream['official_tier3']<75,
                pre_buffer_measured_wrist_force_peak_N=pre_cutoff_force_peak,
                reference_target_p95_mm=(float(np.quantile(reference_errors,.95))
                    if reference_errors else None),rejected=dict(bad))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episode-index-start',type=int,default=2300000)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve immutable labels')
    a.output.mkdir(parents=True)
    plan=json.loads((a.collection/'plan.json').read_text())
    progress=json.loads((a.collection/'progress.json').read_text())
    summaries=[]
    with (a.output/'transport_chunks.jsonl').open('w') as out:
        for i,spec in enumerate(plan['scenes']):
            if spec['family']!='sc' or progress.get(spec['name'],{}).get('official_tier3')!=75:
                continue
            audit=Path(progress[spec['name']]['audit'])
            try:summaries.append(materialize(audit,a.episode_index_start+i,spec['split'],out))
            except (ValueError,KeyError,FileNotFoundError) as exc:
                summaries.append(dict(audit=str(audit),rejected_episode=str(exc)))
    (a.output/'manifest.json').write_text(json.dumps(dict(schema='aic_sc_scored_route_labels/v1',
        source=str(a.collection),episodes=summaries,final_scenes_opened=False),indent=2)+'\n')
    print(json.dumps(dict(admitted=sum('transport_chunks' in r for r in summaries),
                          rejected=sum('rejected_episode' in r for r in summaries))),flush=True)


if __name__=='__main__':main()
