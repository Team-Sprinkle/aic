#!/usr/bin/env python3
"""Recorded full targets in one causal RGB port/grasp frame, both families.

No scene or simulator geometry is read. A static camera-estimated body offset
is acquired from three consistent post-port-lock observations. Earlier rows
are excluded, never backfilled. SFP acquisition requires a public TCP gap of
at least120mm, as required by its free-approach-only approval.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from build_verified_predicted_port_actor_rows import convert_record
from shared_port_tcp_geometry import matrix_from_pose9,matrix_from_xyz_xyzw
from sc_native_grasp_observation import SCNativeGraspObservation
from sfp_native_grasp_observation import SFPNativeGraspObservation


def attach_grasp(row,body,lock_time):
    if lock_time>row['time_s']:raise ValueError('A future grasp cannot condition this action')
    result=dict(row);pose=matrix_from_pose9(row['predicted_port_tcp_pose9']);body=np.asarray(body)
    point=pose[:3,:3]@body+pose[:3,3]
    result['predicted_tip_port_pose9']=[*point.tolist(),1.,0.,0.,0.,1.,0.]
    result['predicted_tip_valid']=True
    result['recorded_tcp_targets_predicted_port_pose9']=row['teacher_tcp_targets_predicted_port_pose9']
    targets=[]
    for target in row['teacher_tcp_targets_predicted_port_pose9']:
        m=matrix_from_pose9(target);p=m[:3,3]+m[:3,:3]@body
        targets.append([*p.tolist(),*target[3:]])
    result.update(teacher_tcp_targets_predicted_port_pose9=targets,
        grasp_lock_time_s=lock_time,predicted_body_tip_position_m=body.tolist(),
        grasp_orientation_input=False,static_grasp_lock=True,
        action_reference='plug_tip_translation_tcp_orientation',
        target_encoding='complete recorded TCP command mapped through same RGB body estimate; not a residual action')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('commands','port-locks','sc-grasp','sfp-grasp','sfp-approval','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);cv2.setNumThreads(2)
    models=dict(sc=SCNativeGraspObservation(a.sc_grasp),sfp=SFPNativeGraspObservation(a.sfp_grasp,a.sfp_approval))
    locks={r['episode']:r for r in map(json.loads,a.port_locks.open())}
    source=json.loads((a.commands/'summary.json').read_text());counts=Counter();excluded=Counter();reports=[];groups=defaultdict(set)
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (a.port_locks,a.sc_grasp,a.sfp_grasp,a.sfp_approval)}
    (a.output/'plan.json').write_text(json.dumps(dict(command_source=str(a.commands),inputs=hashes,
        hypotheses='Lock visible body-to-plug translation before insertion; never update it from hidden inserted-tip images',
        gate='Three causal post-port-lock estimates, maximum0.5mm translation spread; SFP public TCP gap>=120mm',
        no_GT_read=True,backfill_prelock_rows=False,final_scenes_opened=False),indent=2)+'\n')
    with (a.output/'rows.jsonl').open('w') as writer,(a.output/'grasp_locks.jsonl').open('w') as lock_writer:
        for e in source['episodes']:
            lock=locks.get(e['episode']);family=e['family'];rows=list(map(json.loads,(a.commands/e['rows_file']).open()))
            if lock is None:excluded['no_port_lock']+=len(rows);continue
            history=[];grasp=None;inverse=np.linalg.inv(np.array(lock['base_port_matrix']))
            observations=list(map(json.loads,(a.commands/e['observation_prefix_file']).open())) if e.get('observation_prefix_file') else rows
            for row in observations:
                if row['time_s']<e.get('grasp_acquisition_start_s',-float('inf')):continue
                if row['time_s']<lock['time_s']:continue
                tcp=matrix_from_xyz_xyzw(row['state'][:7]);gap=-(inverse@tcp)[2,3]*1000
                if family=='sfp' and gap<120:history=[];continue
                images={c:cv2.cvtColor(cv2.imread(row['images'][c]),cv2.COLOR_BGR2RGB) for c in ('center','left','right')}
                kwargs=dict(already_training_jpeg=True)
                if family=='sfp':kwargs['estimated_TCP_axial_gap_mm']=float(gap)
                prediction=models[family].predict(images,**kwargs);body=np.array(prediction['predicted_tip_position_tcp_m'])
                if not np.isfinite(body).all() or np.linalg.norm(body)>.1:history=[];continue
                history=(history+[body])[-3:]
                if len(history)<3:continue
                mean=np.median(history,axis=0);spread=float(max(np.linalg.norm(x-mean) for x in history))
                if spread>.0005:continue
                grasp=dict(episode=row['episode'],family=family,scene=row['scene'],split=row['split'],
                    time_s=row['time_s'],body_tip_position_m=mean.tolist(),spread_mm=spread*1000,
                    source_images=row['images'],estimated_TCP_axial_gap_mm=float(gap),geometry_input=False)
                lock_writer.write(json.dumps(grasp)+'\n');break
            emitted=0
            if grasp is not None:
                for row in rows:
                    if row['time_s']<grasp['time_s']:excluded['pre_causal_grasp_lock']+=1;continue
                    converted=convert_record(row,lock)
                    if converted is None:raise ValueError('Grasp lock must follow port lock')
                    result=attach_grasp(converted,grasp['body_tip_position_m'],grasp['time_s'])
                    writer.write(json.dumps(result,separators=(',',':'))+'\n');counts[f'{family}/{row["split"]}']+=1;emitted+=1;groups[row['split']].add(row['scene'])
            else:excluded['no_approved_causal_grasp_lock']+=len(rows)
            report=dict(episode=e['episode'],family=family,split=e['split'],rows=emitted,
                        grasp_lock_time_s=grasp['time_s'] if grasp else None)
            reports.append(report)
            with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
            print(json.dumps(report),flush=True)
    if groups['train']&groups['validation']:raise ValueError('Physical reset leakage')
    summary=dict(rows=dict(counts),source_episodes={key:sum(r['rows']>0 and f"{r['family']}/{r['split']}"==key for r in reports) for key in counts},
        episodes=reports,exclusions=dict(excluded),all_inputs_observation_only=True,
        action_reference='plug_tip_translation_tcp_orientation',tip_action_families=['sc','sfp'],
        causal_port_and_grasp_locks=True,full_controller_contract_verified=True,
        original_data_modified=False,final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='episodes'}),flush=True)


if __name__=='__main__':
    from project_gpu_guard import selected_gpu_reservation
    with selected_gpu_reservation():main()
