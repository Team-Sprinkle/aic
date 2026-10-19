#!/usr/bin/env python3
"""Held-out geometry audit and prediction/GT comparison overlays.

Uses saved image-only predictions. Ground truth appears only in evaluation
metrics and explicitly labeled overlay arrows; it never chooses the crops.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from shared_port_tcp_geometry import matrix_from_pose9
from train_sfp_native_grasp_keypoints import metrics
from full_connector_geometry import (connector_in_port,recorded_command_to_connector_target,
    connector_target_to_tcp,connector_target_to_body_command)


def project(pose,optics,origin):
    # 8 mm coordinate arrows attached to the connector's mating frame.
    points=np.array([[0,0,0],[.008,0,0],[0,.008,0],[0,0,.008]])
    points=points@pose[:3,:3].T+pose[:3,3]
    camera=np.linalg.inv(optics);points=points@camera[:3,:3].T+camera[:3,3]
    from evaluate_sc_native_triangulation import FOCAL
    return points[:,:2]/points[:,2:]*FOCAL+np.array([576,512])-origin


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    summary=json.loads((a.run/'summary.json').read_text());plan=summary['plan'];source=Path(plan['source'])
    rows=json.loads((source/'manifest.json').read_text());ids=np.load(a.run/'heldout_indices.npy')
    predicted=np.load(a.run/'heldout_pose_predictions.npy');baseline=np.load(a.run/'baseline_heldout_poses.npy')
    truth=(np.load(source/'body_tip_matrices.npy') if plan['family']=='sfp' else
        np.stack([matrix_from_pose9(v) for v in np.load(source/'labels.npy')]))[ids]
    if plan.get('rotation_only'):
        predicted[:,:3,3]=baseline[:,:3,3]
    errors=(Rotation.from_matrix(truth[:,:3,:3]).inv()*Rotation.from_matrix(predicted[:,:3,:3])).magnitude()*180/np.pi
    groups=defaultdict(list)
    for j,index in enumerate(ids):groups[rows[index]['episode']].append(j)
    per_episode=[]
    for episode,index in groups.items():
        rotations=Rotation.from_matrix(truth[index,:3,:3]);mean=rotations.mean()
        estimate=Rotation.from_matrix(predicted[index,:3,:3]).mean()
        per_episode.append(dict(episode=episode,samples=len(index),
            angular_p50_p95_deg=np.percentile(errors[index],[50,95]).tolist(),
            all_frames_mean_orientation_error_deg=float((mean.inv()*estimate).magnitude()*180/np.pi),
            true_within_episode_p95_deg=float(np.percentile((mean.inv()*rotations).magnitude()*180/np.pi,95))))
    # Fixed, median and worst examples; declared selection avoids pretending
    # these illustrations alone establish generalization.
    selected=sorted(set([0,len(ids)//3,2*len(ids)//3,int(np.argsort(errors)[len(errors)//2]),int(np.argmax(errors))]))
    pixels=np.load(plan['pixels'],mmap_mode='r');origins=np.asarray(plan['crop_origins_native'])
    if plan['family']=='sfp':
        optics=json.loads((source/'summary.json').read_text())['tcp_to_optical']
    else:
        optics=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())['tcp_to_optical']
    panels=[]
    for j in selected:
        row=rows[ids[j]];views=[]
        for ci,c in enumerate(['center','left','right']):
            rgb=np.array(pixels[ids[j],ci],copy=True)
            if rgb.shape[0]==208:rgb=rgb[8:200,8:200]
            image=cv2.resize(rgb[...,::-1],(576,576),interpolation=cv2.INTER_NEAREST)
            gt=project(truth[j],np.asarray(optics[c]),origins[ci])*3
            pred=project(predicted[j],np.asarray(optics[c]),origins[ci])*3
            for k in range(1,4):
                # Pale dotted GT, solid colored predictions.
                for f in np.arange(0,1,.10):
                    u=gt[0]+f*(gt[k]-gt[0]);v=gt[0]+min(f+.05,1)*(gt[k]-gt[0])
                    cv2.line(image,tuple(np.rint(u).astype(int)),tuple(np.rint(v).astype(int)),(255,255,255),2)
                color=[(0,0,255),(0,255,0),(255,100,0)][k-1]
                cv2.arrowedLine(image,tuple(np.rint(pred[0]).astype(int)),tuple(np.rint(pred[k]).astype(int)),color,2,tipLength=.15)
                cv2.putText(image,'XYZ'[k-1],tuple(np.rint(pred[k]+[4,-4]).astype(int)),0,.6,color,2)
            cv2.putText(image,c+' | solid=prediction; dashed=GT',(8,24),0,.55,(0,255,255),1)
            views.append(image)
        strip=np.concatenate(views,axis=1)
        header=np.zeros((52,strip.shape[1],3),np.uint8)
        label=f'{plan["family"]} held-out frame {int(ids[j])}: orientation error {errors[j]:.3f} deg | {row["episode"]}'
        cv2.putText(header,label[:170],(8,30),0,.6,(255,255,255),1)
        filename=a.output/f'frame_{int(ids[j]):05d}.jpg';cv2.imwrite(str(filename),np.concatenate([header,strip]));panels.append(str(filename))
    # Full transform interface: arbitrary grasp rotations and changed bases
    # must preserve the physical desired connector pose.
    rng=np.random.default_rng(20261017);max_error=0
    def sample():
        m=np.eye(4);m[:3,:3]=Rotation.random(random_state=rng).as_matrix();m[:3,3]=rng.normal(0,.1,3);return m
    for _ in range(100):
        port,tcp,grasp,command,world=sample(),sample(),sample(),sample(),sample()
        current=connector_in_port(port,tcp,grasp)
        changed=connector_in_port(world@port,world@tcp,grasp)
        target=recorded_command_to_connector_target(command,grasp)
        decoded=connector_target_to_tcp(target,grasp)
        body=connector_target_to_body_command(tcp,port,target,grasp)
        max_error=max(max_error,float(np.max(abs(current-changed))),float(np.max(abs(decoded-command))),
                      float(np.max(abs(tcp@body@grasp-port@target))))
    if max_error>1e-10:raise ValueError('Full connector frame parity failed')
    report=dict(family=plan['family'],metrics=metrics(predicted,truth),baseline=metrics(baseline,truth),
        complete_episodes=len(groups),physical_resets=len({rows[i]['scene'] for i in ids}),
        prediction_source=str(a.run/'heldout_pose_predictions.npy'),
        position_source='existing landmark prediction' if plan.get('rotation_only') else 'direct trained head',
        per_episode=per_episode,overlays=panels,
        overlay_contract='RGB-only predictions; GT used only for metrics and dashed comparison arrows, never crop selection.',
        all_frame_mean_is_deployable=False,
        mean_note='All-frame episode averages are descriptive only; include future frames. Deployment requires a causal acquisition-window test.',
        full_transform_max_abs_error=max_error,automatic_runtime_approval=False,final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['family','metrics','baseline','complete_episodes','physical_resets','full_transform_max_abs_error']}))


if __name__=='__main__':main()
