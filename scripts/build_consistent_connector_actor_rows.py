#!/usr/bin/env python3
"""Copy matched BC rows into full connector SE(3), using RGB grasp estimates.

Uses the same causal acquisition images/translation as the original derivative.
Never reads simulator geometry, future observations or measured action labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from native_connector_pose_observation import NativeConnectorPoseObservation
from full_connector_geometry import recorded_command_to_connector_target,connector_target_to_tcp
from shared_port_tcp_geometry import matrix_from_pose9


def pose9(matrix):return [*matrix[:3,3],*matrix[:3,:2].T.reshape(-1)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','sc','sfp','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sc-position-mode',choices=['retained','joint'],default='retained')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    locks=list(map(json.loads,(a.source/'grasp_locks.jsonl').open()))
    models={f:NativeConnectorPoseObservation(getattr(a,f)) for f in ['sc','sfp']}
    estimated={};records=[]
    for lock in locks:
        images={c:cv2.cvtColor(cv2.imread(lock['source_images'][c]),cv2.COLOR_BGR2RGB) for c in ['center','left','right']}
        predicted=models[lock['family']].predict(images,already_training_jpeg=True,
            visual_tip_position_tcp_m=(None if lock['family']=='sc' and a.sc_position_mode=='joint' else lock['body_tip_position_m']))
        body=np.asarray(predicted['predicted_connector_in_tcp_matrix'])
        if lock['episode'] in estimated:raise ValueError('Ambiguous grasp/reset')
        estimated[lock['episode']]=body
        records.append(dict(lock,body_tip_position_m=body[:3,3].tolist(),estimated_connector_in_tcp=body.tolist(),
            rotation_checkpoint=str(getattr(a,lock['family'])),orientation_diagnostic_only=True,
            source_images_before_action=True,geometry_runtime_input=False))
        if len(records)%50==0:print('RGB grasp estimates',len(records),flush=True)
    (a.output/'full_grasp_locks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    counts={};maximum=0;total=0
    with (a.output/'rows.jsonl').open('w') as writer:
        for row in map(json.loads,(a.source/'rows.jsonl').open()):
            body=estimated[row['episode']]
            if row['time_s']<row['grasp_lock_time_s']:raise ValueError('Future grasp conditioning')
            current=matrix_from_pose9(row['predicted_port_tcp_pose9'])@body
            targets=[]
            for value in row['recorded_tcp_targets_predicted_port_pose9']:
                tcp=matrix_from_pose9(value)
                target=recorded_command_to_connector_target(tcp,body)
                maximum=max(maximum,float(np.max(abs(connector_target_to_tcp(target,body)-tcp))))
                targets.append(pose9(target))
            old_positions=np.asarray(row['teacher_tcp_targets_predicted_port_pose9'])[:,:3]
            if not (row['family']=='sc' and a.sc_position_mode=='joint') and not np.allclose(np.asarray(targets)[:,:3],old_positions,atol=2e-6):
                raise ValueError('Full orientation repair changed recorded point targets')
            row.update(predicted_tip_port_pose9=pose9(current),predicted_body_tip_position_m=body[:3,3].tolist(),
                teacher_tcp_targets_predicted_port_pose9=targets,
                estimated_connector_in_tcp=body.tolist(),grasp_orientation_input=True,
                action_reference='full_connector_pose',full_connector_pose_estimated_from_RGB=True,
                orientation_deployment_approved=False)
            writer.write(json.dumps(row)+'\n');total+=1
            key=row['family']+'/'+row['split'];counts[key]=counts.get(key,0)+1
    if maximum>1e-8:raise ValueError('Full-pose recorded command roundtrip failure')
    report=dict(rows=total,counts=counts,grasp_episodes=len(estimated),
        source=str(a.source),source_rows_sha256=hashlib.sha256((a.source/'rows.jsonl').read_bytes()).hexdigest(),
        orientation_checkpoints={f:dict(path=str(getattr(a,f)),sha256=hashlib.sha256(getattr(a,f).read_bytes()).hexdigest()) for f in ['sc','sfp']},
        retained_translation_estimator_fingerprints={f:dict(sha256=h) for f,h in {
            'sc':'d10db9f6e27acb0d18f4b8a58d659e0173f62f3f483f2077eb5268040d41b14f',
            'sfp':'e19cc1ff555b2c6f67b94aea84360ec7aa60e4904482e8c7d6c14c836ba3f095'}.items() if f!='sc' or a.sc_position_mode=='retained'},
        recorded_TCP_roundtrip_max_abs=maximum,original_point_targets_unchanged=a.sc_position_mode=='retained',original_recorded_TCP_targets_unchanged=True,position_modes={'sc':a.sc_position_mode,'sfp':'retained'},
        source_modified=False,ground_truth_input=False,final_scenes_opened=False,
        limitation='Training diagnostic uses the causal third acquisition image for orientation; live use additionally needs a causal stable orientation window.')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    from project_gpu_guard import selected_gpu_reservation
    with selected_gpu_reservation():main()
