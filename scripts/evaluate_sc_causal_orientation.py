#!/usr/bin/env python3
"""Score an observation-only stationary-port orientation history.

Runtime inputs are low-resolution RGB, task/state, predicted native keypoints,
measured TCP, and fixed camera/tool calibration. Scored TF is used only after
predictions to compute error, never to update history or select an image.
"""

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader

from audit_sc_port_targets import matrix, pose_matrix
from evaluate_sc_native_triangulation import chain, reprojection_residual_px, triangulate
from train_sc_pose_probe import Probe, SCFrames


def stats(values):
    return {'count':len(values),'median':float(np.median(values)) if values else None,
            'p95':float(np.percentile(values,95)) if values else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--keypoint-predictions',type=Path,required=True)
    p.add_argument('--edges',type=Path,required=True)
    p.add_argument('--orientation-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--window',type=int,default=40)
    p.add_argument('--reprojection-gate-px',type=float,default=1.5)
    args=p.parse_args()
    torch.set_num_threads(8)
    rows=[json.loads(x) for x in args.labels.open()]
    keypoints=[json.loads(x) for x in args.keypoint_predictions.open()]
    if [(r['trial'],r['frame']) for r in rows]!=[(r['trial'],r['frame']) for r in keypoints]:
        raise ValueError('Pose/keypoint rows do not match')
    saved=torch.load(args.orientation_checkpoint,map_location='cpu',weights_only=False)
    if saved['state_only'] or saved['label_frame']!='opening':raise ValueError('Wrong pose checkpoint')
    runtime_rows=[{**r,'images':r['lowres_images']} for r in rows]
    dataset=SCFrames(runtime_rows,'/data1/chmin/yj/ws_aic/src/aic','/workspace/isaaclab/aic',
                     saved['image_size'],True,saved['state_mean'],saved['state_std'],'opening')
    loader=DataLoader(dataset,batch_size=32,shuffle=False,num_workers=4,pin_memory=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=Probe(images=True,pretrained_path=None).to(device).eval()
    model.load_state_dict(saved['model'],strict=True)
    quats=[]
    with torch.inference_mode():
        for image,state,*_ in loader:
            _,q=model(image.to(device),state.to(device))
            quats.extend(q.cpu().numpy())
    edges={r['trial']:r for r in json.loads(args.edges.read_text())['rows']}
    first=next(iter(edges.values()))['edges']
    tool_tcp=chain(first,['tool0','cam_mount/cam_mount_link','ati/base_link','ati/tool_link',
                          'gripper/hande_base_link','gripper/tcp'])
    tcp_optical={}
    for cam in ('center','left','right'):
        tool_optical=chain(first,['tool0','cam_mount/cam_mount_link',f'{cam}_camera/camera_link',
                                  f'{cam}_camera/sensor_link',f'{cam}_camera/optical'])
        tcp_optical[cam]=np.linalg.inv(tool_tcp)@tool_optical
    tcp_tip=matrix(json.loads(Path('configs/hierarchical_recovery/sc_tcp_tip_gazebo_tf_251.json').read_text())['tcp_to_sc_tip'])
    histories={};last_update={};results=[]
    for row,keypoint,quat in zip(rows,keypoints,quats):
        base_tip=pose_matrix(row['state'][:7])@tcp_tip
        base_cameras={cam:pose_matrix(row['state'][:7])@offset for cam,offset in tcp_optical.items()}
        pixels={cam:keypoint['cameras'][cam]['opening_xy'] for cam in base_cameras}
        base_opening_xyz=triangulate(pixels,base_cameras)
        residual=reprojection_residual_px(base_opening_xyz,pixels,base_cameras)
        relative_rotation=Rotation.from_quat(quat)
        base_opening_rotation=Rotation.from_matrix(base_tip[:3,:3])*relative_rotation.inv()
        episode=row['episode_id']
        history=histories.setdefault(episode,deque(maxlen=args.window))
        elapsed=float(row['sim_time'])-last_update.get(episode,-float('inf'))
        accept=(np.linalg.norm(base_opening_xyz-base_tip[:3,3])<.1
                and residual<=args.reprojection_gate_px and elapsed>=.5)
        if accept:
            history.append(base_opening_rotation)
            last_update[episode]=float(row['sim_time'])
        if len(history)>=2:
            filtered=Rotation.from_matrix(np.stack([r.as_matrix() for r in history])).mean()
            relative_filtered=filtered.inv()*Rotation.from_matrix(base_tip[:3,:3])
        else:
            relative_filtered=relative_rotation
        truth=Rotation.from_quat(row['observed_sc_tip_pose_opening_frame'][3:7])
        near=abs(row['observed_sc_tip_pose_opening_frame'][2])<.03
        results.append({'trial':row['trial'],'split':row['split'],'near':near,
                        'raw_deg':float((truth.inv()*relative_rotation).magnitude()*180/np.pi),
                        'filtered_deg':float((truth.inv()*relative_filtered).magnitude()*180/np.pi),
                        'history_length':len(history),'accepted':accept,'residual_px':residual})
    report={'schema':'sc_causal_orientation/v1','checkpoint':str(args.orientation_checkpoint),
            'runtime_inputs':['RGB pose estimate','RGB keypoints','task/state','measured TCP','fixed camera/tool calibration'],
            'true_port_tf_runtime':False,'window':args.window,
            'reprojection_gate_px':args.reprojection_gate_px,'by_split':{}}
    for split in ('train','validation'):
        selected=[r for r in results if r['split']==split and r['near']]
        by_episode=defaultdict(list)
        for r in selected:by_episode[r['trial']].append(r)
        report['by_split'][split]={'episodes':len(by_episode),'near_count':len(selected),
            'raw_deg':stats([r['raw_deg'] for r in selected]),
            'filtered_deg':stats([r['filtered_deg'] for r in selected]),
            'history_initialized_rate':sum(r['history_length']>=2 for r in selected)/len(selected) if selected else None,
            'per_episode':{key:{'count':len(group),'raw_deg':stats([r['raw_deg'] for r in group]),
                                'filtered_deg':stats([r['filtered_deg'] for r in group])}
                           for key,group in by_episode.items()}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['by_split']['validation']))


if __name__=='__main__':main()
