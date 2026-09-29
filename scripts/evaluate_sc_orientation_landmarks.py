#!/usr/bin/env python3
"""Triangulate RGB-predicted SC port-axis landmarks and score orientation.

True port TF and virtual landmark projections are read only in the scoring
branch. The optional history uses RGB predictions and measured TCP causally.
"""

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_sc_port_targets import matrix, pose_matrix
from evaluate_sc_native_triangulation import chain, reprojection_residual_px, triangulate


POINTS=('opening','x_plus','x_minus','y_plus','y_minus')


def stats(values):
    return {'count':len(values),'median':float(np.median(values)) if values else None,
            'p95':float(np.percentile(values,95)) if values else None}


def orientation(points):
    x=points['x_plus']-points['x_minus']
    y=points['y_plus']-points['y_minus']
    matrix=np.stack((x,y,np.cross(x,y)),axis=1)
    u,_,vt=np.linalg.svd(matrix)
    result=u@vt
    if np.linalg.det(result)<0:
        u[:,-1]*=-1;result=u@vt
    return Rotation.from_matrix(result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--edges',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--window',type=int,default=40)
    p.add_argument('--reprojection-gate-px',type=float,default=2.0)
    p.add_argument('--initial-reprojection-gate-px',type=float,
                   help='Optional looser gate for the first two causal orientation observations')
    p.add_argument('--board-port-calibration-json',type=Path,
                   help='Fixed board-to-port quaternion fit on training scenes only; constrain orientation to board yaw')
    p.add_argument('--position-predictions',type=Path,
                   help='Optional RGB-only two-landmark predictions for a causal fixed-port position anchor')
    args=p.parse_args()
    rows=[json.loads(x) for x in args.labels.open()]
    predictions=[json.loads(x) for x in args.predictions.open()]
    if [(r['trial'],r['frame']) for r in rows]!=[(r['trial'],r['frame']) for r in predictions]:
        raise ValueError('Mismatched label/prediction rows')
    position_predictions=([json.loads(x) for x in args.position_predictions.open()]
                          if args.position_predictions else None)
    if position_predictions is not None and [(r['trial'],r['frame']) for r in rows]!=[(r['trial'],r['frame']) for r in position_predictions]:
        raise ValueError('Mismatched RGB position prediction rows')
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
    fixed_board_port=(Rotation.from_quat(json.loads(args.board_port_calibration_json.read_text())['board_to_port_xyzw'])
                      if args.board_port_calibration_json else None)
    history={};last_update={};position_history={};position_last_update={};records=[]
    for index,(row,pred) in enumerate(zip(rows,predictions)):
        base_tcp=pose_matrix(row['state'][:7]);base_tip=base_tcp@tcp_tip
        cameras={cam:base_tcp@offset for cam,offset in tcp_optical.items()}
        predicted_points={name:triangulate({cam:pred['cameras'][cam][name] for cam in cameras},cameras)
                          for name in POINTS}
        predicted_orientation=orientation(predicted_points)
        if fixed_board_port is not None:
            board_estimate=predicted_orientation*fixed_board_port.inv()
            board_yaw=board_estimate.as_euler('xyz')[2]
            predicted_orientation=Rotation.from_euler('z',board_yaw)*fixed_board_port
        residual=max(reprojection_residual_px(predicted_points[name],
                     {cam:pred['cameras'][cam][name] for cam in cameras},cameras) for name in POINTS)
        episode=row['episode_id'];stack=history.setdefault(episode,deque(maxlen=args.window))
        position_ready=False
        position_consistent=False
        if position_predictions is not None:
            pp=position_predictions[index]
            pixels={cam:pp['cameras'][cam]['opening_xy'] for cam in cameras}
            position=triangulate(pixels,cameras)
            position_residual=reprojection_residual_px(position,pixels,cameras)
            pos_stack=position_history.setdefault(episode,deque(maxlen=40))
            pos_elapsed=float(row['sim_time'])-position_last_update.get(episode,-float('inf'))
            pos_anchor=(np.median(np.stack(pos_stack),axis=0) if pos_stack else None)
            if (np.linalg.norm(position-base_tip[:3,3])<.1 and position_residual<=1.5
                    and pos_elapsed>=.5 and (pos_anchor is None or np.linalg.norm(position-pos_anchor)<.005)):
                pos_stack.append(position)
                position_last_update[episode]=float(row['sim_time'])
            position_ready=len(pos_stack)>=2
            if position_ready:
                anchor=np.median(np.stack(pos_stack),axis=0)
                position_consistent=np.linalg.norm(predicted_points['opening']-anchor)<.0005
        elapsed=float(row['sim_time'])-last_update.get(episode,-float('inf'))
        gate=(args.initial_reprojection_gate_px if len(stack)<2 and args.initial_reprojection_gate_px is not None
              else args.reprojection_gate_px)
        opening_tip_distance=float(np.linalg.norm(predicted_points['opening']-base_tip[:3,3]))
        board_yaw_in_range=(fixed_board_port is not None and
                            abs((predicted_orientation*fixed_board_port.inv()).as_euler('xyz')[2])<=.2)
        fallback=(len(stack)<2 and position_ready and position_consistent and board_yaw_in_range)
        accept=(opening_tip_distance<.1 and (residual<=gate or fallback) and elapsed>=.5)
        if accept:
            stack.append(predicted_orientation);last_update[episode]=float(row['sim_time'])
        filtered=(Rotation.from_matrix(np.stack([r.as_matrix() for r in stack])).mean()
                  if len(stack)>=2 else predicted_orientation)
        tip_orientation=Rotation.from_matrix(base_tip[:3,:3])
        truth=Rotation.from_quat(row['observed_sc_tip_pose_opening_frame'][3:7])
        raw_relative=predicted_orientation.inv()*tip_orientation
        filtered_relative=filtered.inv()*tip_orientation
        oracle_pixels={name:{cam:row['projected_orientation_landmarks'][cam][name][:2]
                             for cam in cameras} for name in POINTS}
        oracle_points={name:triangulate(oracle_pixels[name],cameras) for name in POINTS}
        oracle_relative=orientation(oracle_points).inv()*tip_orientation
        records.append({'trial':row['trial'],'frame':row['frame'],'split':row['split'],
            'near':abs(row['observed_sc_tip_pose_opening_frame'][2])<.03,
            'raw_deg':float((truth.inv()*raw_relative).magnitude()*180/np.pi),
            'filtered_deg':float((truth.inv()*filtered_relative).magnitude()*180/np.pi),
            'oracle_deg':float((truth.inv()*oracle_relative).magnitude()*180/np.pi),
            'residual_px':residual,'opening_tip_distance_m':opening_tip_distance,
            'accepted':bool(accept),'history_length':len(stack),
            'position_ready':bool(position_ready),'position_consistent':bool(position_consistent),
            'accepted_by_position_fallback':bool(accept and residual>gate)})
    report={'schema':'sc_orientation_landmark_triangulation/v1',
            'runtime_inputs':['RGB-selected crops','task one-hot','measured TCP','fixed camera/tool calibration'],
            'true_port_tf_runtime':False,'oracle_is_scoring_diagnostic_only':True,
            'window':args.window,'reprojection_gate_px':args.reprojection_gate_px,
            'initial_reprojection_gate_px':args.initial_reprojection_gate_px,'by_split':{}}
    report['fixed_board_port_calibration']=str(args.board_port_calibration_json) if args.board_port_calibration_json else None
    report['position_predictions']=str(args.position_predictions) if args.position_predictions else None
    report['position_fallback_inputs']='Only first two orientation updates: RGB-predicted opening, fixed-port causal history, 0.5mm consistency, board yaw within 0.2rad'
    for split in ('train','validation'):
        selected=[r for r in records if r['split']==split and r['near']]
        episodes=defaultdict(list)
        for r in selected:episodes[r['trial']].append(r)
        report['by_split'][split]={'episodes':len(episodes),'near_count':len(selected),
            'raw_deg':stats([r['raw_deg'] for r in selected]),
            'filtered_deg':stats([r['filtered_deg'] for r in selected]),
            'oracle_deg':stats([r['oracle_deg'] for r in selected]),
            'reprojection_residual_px':stats([r['residual_px'] for r in selected]),
            'history_initialized_rate':sum(r['history_length']>=2 for r in selected)/len(selected) if selected else None,
            'position_fallback_accepts':sum(r['accepted_by_position_fallback'] for r in selected),
            'per_episode':{trial:{'count':len(group),'raw_deg':stats([r['raw_deg'] for r in group]),
                                  'filtered_deg':stats([r['filtered_deg'] for r in group]),
                                  'first_near_history_length':group[0]['history_length'],
                                  'history_initialized_rate':sum(r['history_length']>=2 for r in group)/len(group),
                                  'near_acceptance_rate':sum(r['accepted'] for r in group)/len(group),
                                  'position_fallback_accepts':sum(r['accepted_by_position_fallback'] for r in group),
                                  'near_reprojection_residual_px':stats([r['residual_px'] for r in group]),
                                  'near_opening_tip_distance_m':stats([r['opening_tip_distance_m'] for r in group])}
                           for trial,group in episodes.items()}}
    report['worst_validation']=[r for r in sorted(records,key=lambda r:r['filtered_deg'],reverse=True)
                                if r['split']=='validation' and r['near']][:20]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['by_split']['validation']))


if __name__=='__main__':main()
