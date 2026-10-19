#!/usr/bin/env python3
"""Score observation-only SFP buffer goals on frozen actor takeover frames.

True port geometry is opened only after visual/robot-state goals are fixed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plans',required=True,type=Path)
    p.add_argument('--anchor',required=True,type=Path)
    p.add_argument('--pose-fusion',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous score')
    anchors=json.loads(a.anchor.read_text())['runs']
    fused=json.loads(a.pose_fusion.read_text())['runs']
    rows=[]
    for i in range(6):
        plan=json.loads((a.plans/f'case{i}.json').read_text())
        t=plan['takeover_sim_s']
        if anchors[i]['run']!=plan['audit'] or fused[i]['run']!=plan['audit']:
            raise ValueError(f'Case {i} source mismatch')
        observed=[r for r in anchors[i]['rows'] if r['sim_time_s']<=t]
        if not observed:raise ValueError(f'Case {i} has no actor observation')
        latest=observed[-1]
        anchor=(np.asarray(latest['anchor_base_m'],float) if latest['anchor_valid'] else None)
        measured_tcp=np.asarray(plan['start_tcp_base_m'],float)
        predictions={}
        if anchor is not None:
            away=measured_tcp-anchor
            away/=np.linalg.norm(away)
            predictions['measured_approach_ray']=(anchor+away*.085).tolist()
            # The vertical board/port approach direction is a fixed task prior,
            # not a pose read from this scene. Keep it separate from learned axes.
            predictions['nominal_vertical_axis']=(anchor+np.array([0.,0.,.085])).tolist()
        pose_rows=[r for r in fused[i]['fused'] if r['time_s']<=t and r['use_for_anchor']]
        predicted_pose=np.asarray(pose_rows[-1]['pose_base'],float) if pose_rows else None
        axis_rows=[r for r in fused[i]['fused'] if r['time_s']<=t and r['pose_base'] is not None]
        axis_pose=np.asarray(axis_rows[-1]['pose_base'],float) if axis_rows else None
        if anchor is not None and axis_pose is not None:
            predictions['selected_anchor_plus_visual_axis']=(anchor-axis_pose[:3,2]*.085).tolist()
        if predicted_pose is not None:
            predictions['full_pose_cascade']=(predicted_pose[:3,3]-
                                              predicted_pose[:3,2]*.085).tolist()
        # No scoring information is read above this point.
        true_goal=np.asarray(plan['buffer_goal_tcp_base_m'],float)
        true_port=np.asarray(plan['selected_port_base_m'],float)
        true_rot=Rotation.from_quat(plan['selected_port_quat_xyzw']).as_matrix()
        scored={}
        for name,goal in predictions.items():
            delta=true_rot.T@(np.asarray(goal)-true_goal)
            scored[name]=dict(goal_base_m=goal,error_3d_mm=float(np.linalg.norm(delta)*1000),
                              axial_error_mm=float(delta[2]*1000),
                              lateral_error_mm=float(np.linalg.norm(delta[:2])*1000))
        orientation_error=None
        if predicted_pose is not None:
            orientation_error=float((Rotation.from_matrix(true_rot).inv()*
                                     Rotation.from_matrix(predicted_pose[:3,:3])).magnitude()*
                                    180/np.pi)
        rows.append(dict(case=i,split='development' if i<3 else 'train',
                         scene_group=plan['scene_group'],request=[plan['request_card'],plan['request_port']],
                         takeover_sim_s=t,last_observation_sim_s=latest['sim_time_s'],
                         selected_anchor_available=anchor is not None,
                         selected_anchor_error_mm_scoring_only=latest['anchor_error_mm_scoring_only'],
                         full_pose_available=predicted_pose is not None,
                         visual_axis_available=axis_pose is not None,
                         visual_axis_time_s=axis_rows[-1]['time_s'] if axis_rows else None,
                         visual_axis_position_agreement_mm=axis_rows[-1]['position_agreement_mm'] if axis_rows else None,
                         full_pose_orientation_error_deg_scoring_only=orientation_error,
                         predicted_buffer_goals=scored,
                         true_buffer_goal_scoring_only=true_goal.tolist(),
                         true_port_scoring_only=true_port.tolist()))
    result=dict(schema='aic_sfp_observation_buffer_goal_audit/v1',
                source_anchor=str(a.anchor),source_pose_fusion=str(a.pose_fusion),
                true_geometry_used_for_scoring_only=True,final_scenes_opened=False,
                rows=rows)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    for row in rows:
        print(row['case'],row['selected_anchor_available'],row['full_pose_available'],
              {k:round(v['lateral_error_mm'],1) for k,v in row['predicted_buffer_goals'].items()})


if __name__=='__main__':main()
