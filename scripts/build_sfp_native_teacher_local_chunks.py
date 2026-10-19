#!/usr/bin/env python3
"""Materialize successful stock SFP pose commands beside native RGB anchors.

All actions are actual scored-bag Cartesian command targets. Simulator port
geometry only changes their coordinate frame for supervised training.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_sc_port_targets import pose_list, pose_matrix
from build_two_family_port_projection_labels import from_controller


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--tolerance-s',type=float,default=.035)
    p.add_argument('--post-perturb-release-only',action='store_true',
                   help='For corrective teacher runs, exclude deliberate perturbation and hold commands')
    p.add_argument('--post-teacher-takeover-only',action='store_true',
                   help='For DAgger runs, exclude actor commands and the teacher transition boundary')
    a=p.parse_args()
    rows=[json.loads(s) for s in (a.labels/'rows.jsonl').open()]
    cases=sorted({r['episode'] for r in rows});indices={case:2000000+i for i,case in enumerate(cases)}
    streams={};commands={};times={};controller_times={};release_times={};takeover_times={};admitted=Counter();rejected=Counter();split_groups=defaultdict(set)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w') as out:
        for row in rows:
            case=row['episode'];split_groups[row['split']].add(row['scene_group'])
            if case not in streams:
                stream=json.loads(Path(row['source_stream']).read_text())
                if stream['official_tier3']!=75:raise ValueError(f'Nonexpert SFP source: {case}')
                pose=[c for c in stream['commands'] if c['kind']=='pose' and
                      c['message']['header']['frame_id'] in ('base_link','gripper/tcp')]
                pose.sort(key=lambda x:x['sim_ns'])
                streams[case]=stream;commands[case]=pose;times[case]=[c['sim_ns'] for c in pose]
                controller_times[case]=[x[0] for x in stream['streams']['/aic_controller/controller_state']]
                if a.post_perturb_release_only:
                    event=Path(row['source_stream']).parent/'near_port_perturbation.json'
                    if not event.is_file():raise FileNotFoundError(event)
                    release_times[case]=int(json.loads(event.read_text())['sim_release_ns'])
                if a.post_teacher_takeover_only:
                    event=Path(row['source_stream']).parent/'teacher_takeover.json'
                    if not event.is_file():raise FileNotFoundError(event)
                    takeover=json.loads(event.read_text())
                    if not takeover['privileged_teacher_after_takeover']:
                        raise ValueError('Expected privileged teacher takeover')
                    takeover_times[case]=int(round(takeover['sim_time_s']*1e9))
            target=next(t for t in row['targets'] if t['identity']==row['selected_identity'])
            base_port=np.eye(4);base_port[:3,3]=target['position_base_m']
            base_port[:3,:3]=Rotation.from_quat(target['orientation_base_xyzw']).as_matrix()
            port_base=np.linalg.inv(base_port)
            tcp=np.eye(4);tcp[:3,3]=row['state'][:3]
            tcp[:3,:3]=Rotation.from_quat(row['state'][3:7]).as_matrix()
            observed=port_base@tcp
            if observed[2,3]<-.13:
                rejected['before_local_start']+=1;continue
            timestamp=int(round(row['time_s']*1e9));chosen=[];previous=-1
            if a.post_perturb_release_only and timestamp<release_times[case]+50_000_000:
                rejected['before_perturb_release']+=1;continue
            if a.post_teacher_takeover_only and timestamp<takeover_times[case]+50_000_000:
                rejected['before_teacher_takeover']+=1;continue
            for h in range(4):
                desired=timestamp+h*50_000_000
                j=bisect_left(times[case],desired)
                candidates=[k for k in (j-1,j,j+1) if previous<k<len(times[case])]
                if not candidates:break
                k=min(candidates,key=lambda n:abs(times[case][n]-desired))
                if abs(times[case][k]-desired)>a.tolerance_s*1e9:break
                chosen.append(k);previous=k
            if len(chosen)!=4:
                rejected['missing_four_synchronized_targets']+=1;continue
            targets=[]
            for k in chosen:
                recorded=commands[case][k];msg=recorded['message']['pose'];pos=msg['position'];quat=msg['orientation']
                command=np.eye(4);command[:3,3]=[pos[v] for v in 'xyz']
                command[:3,:3]=Rotation.from_quat([quat[v] for v in 'xyzw']).as_matrix()
                if recorded['message']['header']['frame_id']=='gripper/tcp':
                    controllers=streams[case]['streams']['/aic_controller/controller_state']
                    j=bisect_right(controller_times[case],recorded['sim_ns'])-1
                    if j<0 or recorded['sim_ns']-controller_times[case][j]>20_000_000:
                        raise ValueError(f'Missing causal TCP for body-frame target in {case}')
                    command=pose_matrix(from_controller(controllers[j][1]))@command
                targets.append(pose_list(port_base@command))
            card=int(row['selected_identity'].split('_')[2]);port=int(row['selected_identity'].split('_')[4])
            task=[1.,0.,float(port==0),float(port==1),
                  *[float(i==card) for i in range(5)],1.]
            out.write(json.dumps({'schema':'aic_sfp_native_local_chunks/v1',
                'family':'sfp','episode_id':case,'episode_index':indices[case],
                'scene_sha256':row['scene_group'],'split':row['split'],
                'frame_index':admitted[case],'timestamp_s':row['time_s'],
                'images':row['images'],'image_kind':row['image_kind'],
                'state':row['state'],'task_vector':task,
                'observed_tcp_pose_port_frame':pose_list(observed),
                'teacher_tcp_target_poses_port_frame':targets,
                'teacher_takeover_age_s':((timestamp-takeover_times[case])/1e9
                                          if a.post_teacher_takeover_only else None),
                'source_command_time_offsets_s':[(times[case][k]-timestamp)/1e9 for k in chosen],
                'source_command_frames':[commands[case][k]['message']['header']['frame_id'] for k in chosen],
                'target_source':'four recorded scored-bag Cartesian commands converted through causal measured TCP when body-relative',
                'official_tier3':75,'geometry_training_label_only':True,
                'runtime_true_geometry':False},separators=(',',':'))+'\n')
            admitted[case]+=1
    if split_groups['train']&split_groups['validation']:raise ValueError('Scene/reset leakage')
    report={'schema':'aic_sfp_native_local_chunks_summary/v1',
            'episodes':len([x for x in admitted.values() if x]),'rows':sum(admitted.values()),
            'rows_per_episode':dict(admitted),'rejected':dict(rejected),
            'train_scene_groups':len(split_groups['train']),
            'validation_scene_groups':len(split_groups['validation']),
            'source_actions_recorded':True,'geometry_training_label_only':True,
        'post_perturb_release_only':a.post_perturb_release_only,
        'post_teacher_takeover_only':a.post_teacher_takeover_only,
            'reserved_final_opened':False}
    a.summary.parent.mkdir(parents=True,exist_ok=True)
    a.summary.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
