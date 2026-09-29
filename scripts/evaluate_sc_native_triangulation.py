#!/usr/bin/env python3
"""Triangulate RGB-predicted SC opening pixels and score against bag TF.

Prediction uses fixed camera calibration and observed TCP only. Scored port TF
and projected ground-truth pixels are read solely in the scoring branch.
"""

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from audit_sc_port_targets import matrix, pose_matrix, select
from build_sc_native_pose_labels import WIDTH, HEIGHT, HFOV, chain


FOCAL=WIDTH/(2*math.tan(HFOV/2))


def triangulate(camera_pixels, base_optical):
    lhs=np.zeros((3,3),np.float64);rhs=np.zeros(3,np.float64)
    for cam,(u,v) in camera_pixels.items():
        transform=base_optical[cam]
        ray=transform[:3,:3]@np.array([(u-WIDTH/2)/FOCAL,(v-HEIGHT/2)/FOCAL,1.0])
        ray/=np.linalg.norm(ray)
        projection=np.eye(3)-np.outer(ray,ray)
        lhs+=projection;rhs+=projection@transform[:3,3]
    return np.linalg.solve(lhs+1e-9*np.eye(3),rhs)


def reprojection_residual_px(position_base, camera_pixels, base_optical):
    residual=[]
    for cam,(u,v) in camera_pixels.items():
        point=np.linalg.inv(base_optical[cam])@np.r_[position_base,1.0]
        if point[2]<=0:return float('inf')
        uv=np.array([WIDTH/2+FOCAL*point[0]/point[2],
                     HEIGHT/2+FOCAL*point[1]/point[2]])
        residual.append(float(np.linalg.norm(uv-np.array([u,v]))))
    return max(residual)


def stats(values):
    values=[v for v in values if v is not None and math.isfinite(v)]
    return {'count':len(values),'median':float(np.median(values)) if values else None,
            'p95':float(np.percentile(values,95)) if values else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--edges',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--causal-window',type=int,default=0,
                   help='Observation-only fixed-port median; update no faster than 0.5 s')
    p.add_argument('--reprojection-gate-px',type=float,default=0,
                   help='With causal window, only update history when all three views reproject within this pixel error')
    p.add_argument('--initial-reprojection-gate-px',type=float,
                   help='Optional looser reprojection gate until two visual port observations are accepted')
    p.add_argument('--cameras',nargs='+',choices=('center','left','right'),
                   default=['center','left','right'],
                   help='Fixed observation-only camera subset for this ablation')
    p.add_argument('--allow-far-history',action='store_true',
                   help='Permit fixed-port causal updates before the plug is within 100 mm')
    args=p.parse_args()
    labels=[json.loads(line) for line in args.labels.open()]
    predictions=[json.loads(line) for line in args.predictions.open()]
    if [(r['trial'],r['frame']) for r in labels]!=[(r['trial'],r['frame']) for r in predictions]:
        raise ValueError('Prediction/label identity or ordering mismatch')
    edge_rows={r['trial']:r for r in json.loads(args.edges.read_text())['rows']}
    first=next(iter(edge_rows.values()))['edges']
    tool_tcp=chain(first,['tool0','cam_mount/cam_mount_link','ati/base_link','ati/tool_link',
                          'gripper/hande_base_link','gripper/tcp'])
    tcp_tip=matrix(json.loads(Path('configs/hierarchical_recovery/sc_tcp_tip_gazebo_tf_251.json').read_text())['tcp_to_sc_tip'])
    tcp_optical={}
    for cam in ('center','left','right'):
        tool_optical=chain(first,['tool0','cam_mount/cam_mount_link',f'{cam}_camera/camera_link',
                                  f'{cam}_camera/sensor_link',f'{cam}_camera/optical'])
        tcp_optical[cam]=np.linalg.inv(tool_tcp)@tool_optical
    opening_offset=matrix(json.loads(Path('configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json').read_text())['base_to_opening'])
    true_openings={}
    for trial,item in edge_rows.items():
        edges=item['edges'];port=item['target_module_name'].rsplit('_',1)[-1]
        parent=f'task_board/sc_port_{port}';name=parent+'/sc_port_base_link'
        world_base=chain(edges,['world','tabletop','base_link'])
        world_port=chain(edges,['aic_world','task_board',parent,name])
        true_openings[trial]=np.linalg.inv(world_base)@world_port@opening_offset
    records=[]
    histories={}
    last_update={}
    for row,pred in zip(labels,predictions):
        base_tcp=pose_matrix(row['state'][:7])
        cameras={cam:base_tcp@tcp_optical[cam] for cam in args.cameras}
        rgb_pixels={cam:pred['cameras'][cam]['opening_xy'] for cam in cameras}
        predicted_base=triangulate(rgb_pixels,cameras)
        residual_px=reprojection_residual_px(predicted_base,rgb_pixels,cameras)
        accepted=False
        history_length=0
        if args.causal_window:
            tip_base=(base_tcp@tcp_tip)[:3,3]
            episode=row['episode_id']
            history=histories.setdefault(episode,deque(maxlen=args.causal_window))
            elapsed=float(row['sim_time'])-last_update.get(episode,-float('inf'))
            gate=(args.initial_reprojection_gate_px if len(history)<2 and args.initial_reprojection_gate_px is not None
                  else args.reprojection_gate_px)
            consistent=not gate or residual_px<=gate
            anchor_consistent=(not history or np.linalg.norm(predicted_base-np.median(np.stack(history),axis=0))<.005)
            if ((args.allow_far_history or np.linalg.norm(predicted_base-tip_base)<.1) and elapsed>=.5
                    and consistent and anchor_consistent):
                history.append(predicted_base)
                last_update[episode]=float(row['sim_time'])
                accepted=True
            history_length=len(history)
            if len(history)>=2:
                predicted_base=np.median(np.stack(history),axis=0)
        # Oracle triangulation verifies the camera/FOV/TF convention only.
        oracle_pixels={cam:row['projected_training_pixels'][cam]['opening'][:2]
                       for cam in cameras
                       if all(v is not None for v in row['projected_training_pixels'][cam]['opening'][:2])}
        oracle_base=triangulate(oracle_pixels,cameras) if len(oracle_pixels)>=2 else None
        true_frame=true_openings[row['trial']]
        true_position=true_frame[:3,3]
        local_error=true_frame[:3,:3].T@(predicted_base-true_position)*1000
        oracle_error=(true_frame[:3,:3].T@(oracle_base-true_position)*1000
                      if oracle_base is not None else None)
        near=abs(row['observed_sc_tip_pose_opening_frame'][2])<.03
        records.append({'trial':row['trial'],'frame':row['frame'],'split':row['split'],
                        'tier3':row['official_tier3'],'near':near,
                        'lateral_mm':float(np.linalg.norm(local_error[:2])),
                        'axial_mm':float(abs(local_error[2])),
                        'xyz_mm':local_error.tolist(),
                        'oracle_lateral_mm':float(np.linalg.norm(oracle_error[:2])) if oracle_error is not None else None,
                        'reprojection_residual_px':residual_px,
                        'accepted_into_history':accepted,
                        'history_length':history_length,
                        'images':row['images']})
    out={'schema':'sc_native_opening_triangulation/v1',
         'runtime_prediction_inputs':['RGB-only predicted opening pixels','observed TCP','fixed camera calibration'],
         'true_port_tf_runtime':False,'oracle_is_diagnostic_only':True,
         'causal_window':args.causal_window,
         'reprojection_gate_px':args.reprojection_gate_px,
         'initial_reprojection_gate_px':args.initial_reprojection_gate_px,
         'causal_filter_inputs':'predicted port, observed TCP, fixed TCP-to-tip; no true port TF',
         'camera_calibration':{'horizontal_fov_rad':HFOV,'image_width':WIDTH,'image_height':HEIGHT},
         'cameras':args.cameras,
         'allow_far_history':args.allow_far_history,
         'per_frame':records,
         'by_split':{}}
    out['by_phase']={}
    for name,lo,hi in [('initial_0_40',0,40),('transport_40_120',40,120),
                       ('mid_120_240',120,240),('near_240_400',240,400)]:
        phase=[r for r in records if lo<=r['frame']<hi and r['split']=='validation']
        out['by_phase'][name]={'frames':len(phase),
                              'lateral_mm':stats([r['lateral_mm'] for r in phase]),
                              'axial_mm':stats([r['axial_mm'] for r in phase]),
                              'reprojection_residual_px':stats([r['reprojection_residual_px'] for r in phase])}
    for split in ('train','validation'):
        near=[r for r in records if r['split']==split and r['near']]
        by_episode=defaultdict(list)
        for row in near:by_episode[row['trial']].append(row)
        out['by_split'][split]={'near_count':len(near),'episodes':len(by_episode),
                                'lateral_mm':stats([r['lateral_mm'] for r in near]),
                                'axial_mm':stats([r['axial_mm'] for r in near]),
                                'reprojection_residual_px':stats([r['reprojection_residual_px'] for r in near]),
                                'history_acceptance_rate':sum(r['accepted_into_history'] for r in near)/len(near) if near else None,
                                'history_initialized_rate':sum(r['history_length']>=2 for r in near)/len(near) if near else None,
                                'first_near_history_lengths':{trial:items[0]['history_length'] for trial,items in by_episode.items()},
                                'oracle_lateral_mm':stats([r['oracle_lateral_mm'] for r in near]),
                                'per_episode':{trial:{'count':len(items),'tier3':items[0]['tier3'],
                                    'lateral_mm':stats([r['lateral_mm'] for r in items]),
                                    'axial_mm':stats([r['axial_mm'] for r in items]),
                                    'history_initialized_rate':sum(r['history_length']>=2 for r in items)/len(items),
                                    'near_acceptance_rate':sum(r['accepted_into_history'] for r in items)/len(items),
                                    'reprojection_residual_px':stats([r['reprojection_residual_px'] for r in items])}
                                    for trial,items in by_episode.items()}}
    out['worst_validation']=[r for r in sorted(records,key=lambda r:r['lateral_mm'],reverse=True)
                             if r['split']=='validation' and r['near']][:20]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'val_lateral_p95_mm':out['by_split']['validation']['lateral_mm']['p95'],
                      'oracle_p95_mm':out['by_split']['validation']['oracle_lateral_mm']['p95']}))


if __name__=='__main__':main()
