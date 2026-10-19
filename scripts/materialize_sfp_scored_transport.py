#!/usr/bin/env python3
"""Build TCP-body transport chunks from freshly scored SFP CheatCode bags.

The controller's measured TCP at each command converts the recorded
`gripper/tcp` command into a complete base-frame target. The target is then
re-expressed in the TCP frame of the synchronized observation. The selected
port transform is used only for offline target-frame labels and for cutting
the transport prefix before insertion; it is never an actor input.
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

from build_two_family_port_projection_labels import sfp_ports
from lerobot_robot_aic.task_encoding import encode_task_vector


CAMERAS=('center','left','right')


def at(rows,stamp):
    i=bisect_right(rows,stamp,key=lambda r:r[0])-1
    if i<0:raise ValueError('no causal sample')
    return rows[i]


def posematrix(d):
    m=np.eye(4)
    m[:3,3]=[d['position'][k] for k in ('x','y','z')]
    m[:3,:3]=Rotation.from_quat([d['orientation'][k] for k in ('x','y','z','w')]).as_matrix()
    return m


def pose(m):
    return list(m[:3,3])+list(Rotation.from_matrix(m[:3,:3]).as_quat(canonical=True))


def source_path(audit,virtual):
    if virtual.startswith('/audit/'):
        p=audit/virtual[len('/audit/'):]
    else:
        # Host MCAP extraction preserves repo-relative camera paths. Accept
        # them only when they resolve inside this exact scored rollout.
        p=Path(virtual)
        if not p.is_absolute():
            p=Path(__file__).resolve().parents[1]/p
    p=p.resolve()
    if not p.is_relative_to(audit.resolve()):
        raise ValueError(f'Scored image escaped its rollout: {virtual}')
    if not p.exists():raise FileNotFoundError(p)
    return str(p)


def materialize(audit,calibration,episode_index,out,split,corrective=False,
                pause_tail_seconds=0.,full_insert_corrective=False,
                preserve_actor_prefix_context=False,allow_transport_only=False):
    if preserve_actor_prefix_context and not full_insert_corrective:
        raise ValueError('Actor-prefix context requires full-insertion corrective mode')
    if full_insert_corrective and not corrective:
        raise ValueError('Full-insertion corrective mode needs corrective=True')
    if pause_tail_seconds<0 or (corrective and pause_tail_seconds):
        raise ValueError('Pause tails require a noncorrective full-success episode')
    s=json.loads((audit/'scored_stream.json').read_text())
    takeover=None;buffer_event=None
    if corrective:
        takeover=json.loads((audit/'corrective_teacher_takeover.json').read_text())
        if not full_insert_corrective:
            buffer_event=json.loads((audit/'corrective_teacher_buffered_arrival.json').read_text())
        score=yaml.safe_load((audit/'results/scoring.yaml').read_text())
        trial=next(v for k,v in score.items() if k.startswith('trial_'))
        contact=float(trial.get('tier_2',{}).get('categories',{}).get('contacts',{}).get('score',0))
        expected_outcome=('teacher_returned_full_insert' if full_insert_corrective
                          else 'safe_buffered_arrival')
        if (not takeover.get('actor_prefix_completed') or
            takeover.get('teacher_outcome')!=expected_outcome or contact<0 or
            (full_insert_corrective and (trial.get('tier_3',{}).get('score',0)<75 or
                                         not takeover.get('require_full_insert')))):
            raise ValueError(f'Corrective teacher source did not pass safety gate: {audit}')
        if buffer_event is not None and not (-100<=buffer_event['axial_mm']<=-70 and
                                             buffer_event['lateral_mm']<=10):
            raise ValueError(f'Corrective buffer event outside label corridor: {audit}')
    elif s['official_tier3']!=75 and not allow_transport_only:
        raise ValueError(f'Unsuccessful source: {audit}')
    cfg=yaml.safe_load((audit/'eval_config.yaml').read_text())
    trial_id=next(iter(cfg['trials']))
    task=s['task']
    card=int(task['target_module_name'].rsplit('_',1)[-1])
    port=int(task['port_name'].rsplit('_',1)[-1])
    base_port=dict(((c,p),m) for c,p,m in sfp_ports(
        {'trial_yaml':str(audit/'eval_config.yaml'),'trial_id':trial_id},calibration))[(card,port)]
    scene_group=hashlib.sha256(json.dumps({k:s['reset_identity'][k] for k in ('scene','robot')},
        sort_keys=True).encode()).hexdigest()
    task_vector=np.asarray(encode_task_vector(task_family='sfp_to_nic',
        target_card_index=card,target_port_index=port),float).tolist()
    commands=[c for c in s['commands'] if c['kind']=='pose' and
              (not corrective or preserve_actor_prefix_context or
               c['sim_ns']/1e9>=takeover['takeover_sim_time_s']+.1)]
    if preserve_actor_prefix_context:
        # The actor may emit several base_link targets at the same simulator
        # timestamp. Only the last target can be the controller's effective
        # command for that moment; all rows in an episode must be strictly
        # ordered in time before building causal camera/action history.
        latest_by_sim_ns={c['sim_ns']:c for c in commands}
        commands=[latest_by_sim_ns[t] for t in sorted(latest_by_sim_ns)]
    if len(commands)<100:raise ValueError(f'Too few recorded commands: {audit}')
    teacher_commands=[c for c in commands if not corrective or
                      c['sim_ns']/1e9>=takeover['takeover_sim_time_s']+.1]
    if {c['message']['header']['frame_id'] for c in teacher_commands}!={'gripper/tcp'}:
        raise ValueError('Expected teacher gripper/TCP-frame commands')
    if {c['message']['header']['frame_id'] for c in commands}-{'gripper/tcp','base_link'}:
        raise ValueError('Unexpected actor-prefix command frame')
    times=[c['sim_ns'] for c in commands]
    ctrl=s['streams']['/aic_controller/controller_state']
    joints=s['streams']['/joint_states']
    wrenches=s['streams']['/fts_broadcaster/wrench']
    rows=[];bad=Counter();reference_error=[];reference_orientation_error=[]
    for i,c in enumerate(commands):
        t=c['sim_ns']
        try:
            ct,controller=at(ctrl,t)
            if t-ct>100000000:raise ValueError('stale command TCP')
            command_pose=posematrix(c['message']['pose'])
            target=(posematrix(controller['tcp_pose'])@command_pose
                    if c['message']['header']['frame_id']=='gripper/tcp' else command_pose)
            image_stamps={};images={}
            for camera in CAMERAS:
                it,f=at(s['cameras'][camera],t)
                if t-it>100000000:raise ValueError('stale image')
                image_stamps[camera]=it;images[camera]=source_path(audit,f)
            observation_t=min(image_stamps.values())
            if max(image_stamps.values())-observation_t>1000000:
                raise ValueError('unsynchronized RGB triplet')
            ct,controller=at(ctrl,observation_t)
            jt,joint=at(joints,observation_t)
            wt,wrench=at(wrenches,observation_t)
            if max(observation_t-ct,observation_t-jt,observation_t-wt)>100000000:
                raise ValueError('stale synchronized state')
            tcp=posematrix(controller['tcp_pose'])
            named=dict(zip(joint['name'],joint['position']))
            ordered=[named[k] for k in s['joint_order']]+[named['gripper/left_finger_joint']/2.]
            v=controller['tcp_velocity'];w=wrench['wrench']
            state=pose(tcp)+[v[k][a] for k in ('linear','angular') for a in ('x','y','z')]+\
                list(controller['tcp_error'])+ordered+\
                [w[k][a] for k in ('force','torque') for a in ('x','y','z')]
            if len(state)!=32:raise ValueError('wrong state length')
            rel=np.linalg.inv(base_port)@tcp
            rows.append(dict(index=i,t=t,target=target,tcp=tcp,state=state,
                             rel=rel,images=images,image_stamps=image_stamps))
            future_i=bisect_right(ctrl,t+50000000,key=lambda r:r[0])-1
            if future_i>=0:
                ref=posematrix(ctrl[future_i][1]['reference_tcp_pose'])
                reference_error.append(float(np.linalg.norm(target[:3,3]-ref[:3,3])*1000))
                reference_orientation_error.append(float((
                    Rotation.from_matrix(target[:3,:3]).inv()*
                    Rotation.from_matrix(ref[:3,:3])).magnitude()*180/np.pi))
        except (ValueError,KeyError,FileNotFoundError) as exc:
            bad[type(exc).__name__+':'+str(exc)[:45]]+=1
    if len(rows)<100:raise ValueError(f'Insufficient joined rows: {audit}, {dict(bad)}')
    cutoff_row=(next((r for r in rows if r['t']/1e9>=takeover['takeover_sim_time_s']+.1 and
                      -.100<=r['rel'][2,3]<=-.070 and
                      np.linalg.norm(r['rel'][:2,3])<=.010),None)
                if corrective else next((r for r in rows if r['rel'][2,3]>=-.070 and
                     np.linalg.norm(r['rel'][:2,3])<=.010),None))
    if cutoff_row is None:raise ValueError(f'No safe buffered approach before insertion: {audit}')
    cutoff=cutoff_row['t']
    stop_time=(takeover['teacher_end_sim_time_s'] if full_insert_corrective
               else buffer_event['sim_time_s']) if corrective else None
    if corrective and (cutoff/1e9<takeover['takeover_sim_time_s'] or
                       cutoff/1e9>stop_time):
        raise ValueError(f'Corrective controller-frame cutoff outside takeover/event order: {audit}')
    if not corrective and cutoff_row['rel'][2,3]>-.040:
        raise ValueError(f'First safe approach already beyond insertion standoff: {audit}')
    by_index={r['index']:r for r in rows}
    written=0;teacher_written=0;context_written=0
    for anchor in rows:
        if anchor['t']>=cutoff+int(pause_tail_seconds*1e9):break
        if anchor['t']>=cutoff and not pause_tail_seconds:break
        indices=[bisect_right(times,anchor['t']+h*50000000)-1 for h in range(4)]
        if any(j not in by_index for j in indices):bad['missing_chunk_member']+=1;continue
        chunk=[by_index[j] for j in indices]
        if any(r['t']>cutoff+int(pause_tail_seconds*1e9) for r in chunk):
            bad['crossed_transport_cutoff']+=1;continue
        if not pause_tail_seconds and any(r['t']>cutoff for r in chunk):
            bad['crossed_transport_cutoff']+=1;continue
        if any(anchor['t']+h*50000000-r['t']>150000000 for h,r in enumerate(chunk)):
            bad['command_hold_too_long']+=1;continue
        data=dict(schema='aic_sfp_scored_transport_chunk/v1',
            episode_index=episode_index,episode_id=audit.name,scene_sha256=scene_group,
            split=split,frame_index=anchor['index'],cache_frame_index=None,
            timestamp_s=anchor['t']/1e9,images=anchor['images'],
            image_kind='native_1152x1024_rgb_jpeg',state=anchor['state'],
            task_vector=task_vector,observed_tcp_pose_port_frame=pose(anchor['rel']),
            teacher_tcp_target_poses_port_frame=[pose(np.linalg.inv(base_port)@r['target']) for r in chunk],
            teacher_tcp_target_poses_body_frame=[pose(np.linalg.inv(anchor['tcp'])@r['target']) for r in chunk],
            target_source=('actor-prefix recorded base_link/TCP target converted to absolute base; context only, excluded from supervised loss'
                           if corrective and anchor['t']/1e9<takeover['takeover_sim_time_s']+.1 else
                           'post-actor-takeover CheatCode recorded gripper/tcp target converted with measured TCP'
                           if corrective else 'recorded gripper/tcp pose command converted with measured TCP at command time'),
            official_tier3=s['official_tier3'],geometry_label_only=True,selected_card=card,selected_port=port,
            full_insert_teacher=bool(s['official_tier3']==75),
            transport_only_prefix=bool(s['official_tier3']!=75),
            transport_cutoff_sim_s=cutoff/1e9,source_scored_stream=str(audit/'scored_stream.json'))
        if pause_tail_seconds:
            data['pause_head_only_source']=True
            data['past_transport_cutoff']=bool(anchor['t']>=cutoff)
        if corrective:
            context_only=bool(anchor['t']/1e9<takeover['takeover_sim_time_s']+.1)
            data['privileged_teacher_after_takeover']=not context_only
            data['actor_prefix_context_only']=context_only
            data['actor_prefix_used_as_supervision']=False
            data['takeover_sim_time_s']=takeover['takeover_sim_time_s']
            context_written+=int(context_only);teacher_written+=int(not context_only)
        out.write(json.dumps(data,separators=(',',':'))+'\n');written+=1
    if (teacher_written if corrective else written)<30:
        raise ValueError(f'Too few teacher transport chunks: {audit}, {teacher_written}')
    return dict(audit=str(audit),trial_id=trial_id,card=card,port=port,scene_group=scene_group,
        commands=len(commands),joined=len(rows),transport_chunks=written,
        cutoff_sim_s=cutoff/1e9,reference_target_error_mm=dict(
            median=float(np.median(reference_error)),p95=float(np.quantile(reference_error,.95))),
        reference_orientation_error_deg=dict(
            median=float(np.median(reference_orientation_error)),
            p95=float(np.quantile(reference_orientation_error,.95))),
        cutoff_axial_mm=float(cutoff_row['rel'][2,3]*1000),
        cutoff_lateral_mm=float(np.linalg.norm(cutoff_row['rel'][:2,3])*1000),
        cutoff_from_measured_controller_tcp=True,
        privileged_tf_event_sim_s=(buffer_event['sim_time_s'] if buffer_event is not None else None),
        controller_cutoff_to_tf_event_s=(buffer_event['sim_time_s']-cutoff/1e9
                                         if buffer_event is not None else None),
        rejected=dict(bad),corrective_teacher=corrective,
        full_insert_corrective=full_insert_corrective,
        actor_prefix_context_preserved=preserve_actor_prefix_context,
        actor_prefix_context_chunks=context_written,teacher_transport_chunks=teacher_written,
        pause_head_only_source=bool(pause_tail_seconds),pause_tail_seconds=pause_tail_seconds,
        actor_prefix_used_as_supervision=False if corrective else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,action='append',required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=('train','validation'),default='train')
    p.add_argument('--episode-index-start',type=int,default=1800000)
    p.add_argument('--corrective-teacher',action='store_true',
                   help='Use only safe recorded CheatCode commands after actor takeover')
    p.add_argument('--corrective-full-insert',action='store_true',
                   help='Require officially scored full insertion after actor takeover')
    p.add_argument('--preserve-actor-prefix-context',action='store_true',
                   help='Include actor-prefix observations/actions as history only, never loss targets')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous/partial dataset')
    a.output.mkdir(parents=True)
    calibration=json.loads(a.calibration.read_text())
    reports=[]
    with (a.output/'transport_chunks.jsonl').open('w') as out:
        for episode_index,audit in enumerate(a.audit,start=a.episode_index_start):
            reports.append(materialize(audit.resolve(),calibration,episode_index,out,a.split,
                                       corrective=a.corrective_teacher or a.corrective_full_insert,
                                       full_insert_corrective=a.corrective_full_insert,
                                       preserve_actor_prefix_context=a.preserve_actor_prefix_context))
            print(json.dumps(reports[-1]),flush=True)
    (a.output/'manifest.json').write_text(json.dumps(dict(
        schema='aic_sfp_scored_transport_admission/v1',episodes=reports,
        split=a.split,episode_index_start=a.episode_index_start,
        label_frame='complete target in observed TCP body and selected port frames',
        runtime_geometry_input=False,geometry_used_only_for_offline_cutoff_and_label=True,
        invalid_agent_action_arrays_used=False,
        corrective_teacher=a.corrective_teacher or a.corrective_full_insert,
        corrective_full_insert=a.corrective_full_insert,
        actor_prefix_context_preserved=a.preserve_actor_prefix_context,
        actor_prefix_used_as_supervision=False if (a.corrective_teacher or
                                                   a.corrective_full_insert) else None,
        final_scenes_opened=False),indent=2)+'\n')


if __name__=='__main__':main()
