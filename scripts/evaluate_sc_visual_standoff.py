#!/usr/bin/env python3
"""Score RGB-triangulated SC opening plus fixed transport standoff offset.

Six complete source groups fit a constant opening-to-TCP offset. Two held-out
groups only score it. Source-bag TF and recorded plans never enter inference.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_sc_port_targets import matrix,pose_matrix
from build_sc_native_pose_labels import chain
from evaluate_sc_native_triangulation import triangulate,reprojection_residual_px


CAMERAS=('center','left','right')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--edges',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    labels=[json.loads(s) for s in a.labels.open()]
    preds=[json.loads(s) for s in a.predictions.open()]
    if [(r['trial'],r['frame']) for r in labels]!=[(r['trial'],r['frame']) for r in preds]:
        raise ValueError('Prediction/label mismatch')
    edges={r['trial']:r for r in json.loads(a.edges.read_text())['rows']}
    source={r['case']:r for r in json.loads(a.dataset.read_text())['episodes']}
    opening_offset=matrix(json.loads(Path('configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json').read_text())['base_to_opening'])
    first=next(iter(edges.values()))['edges']
    tool_tcp=chain(first,['tool0','cam_mount/cam_mount_link','ati/base_link','ati/tool_link',
                          'gripper/hande_base_link','gripper/tcp'])
    tcp_optical={}
    for cam in CAMERAS:
        camera=chain(first,['tool0','cam_mount/cam_mount_link',f'{cam}_camera/camera_link',
                            f'{cam}_camera/sensor_link',f'{cam}_camera/optical'])
        tcp_optical[cam]=np.linalg.inv(tool_tcp)@camera
    calibration=[];goal_by_case={};opening_by_case={}
    for case,e in source.items():
        if case not in edges:continue
        scene_edges=edges[case]['edges'];port=e['selected_port']
        parent=f'task_board/sc_port_{port}'
        base=np.linalg.inv(chain(scene_edges,['world','tabletop','base_link']))
        opening=(base@chain(scene_edges,['aic_world','task_board',parent,parent+'/sc_port_base_link'])@
                 opening_offset)[:3,3]
        goal=np.asarray(json.loads((Path(e['insert_plan'])/'moveit_joint_trajectory.json').read_text())
                        ['waypoints'][-1]['tcp_pose']['position'])
        opening_by_case[case]=opening;goal_by_case[case]=goal
    # All eight BC-source cases have the same source split. The two cases in
    # the prediction file are sealed from this transport-offset fit.
    validation_cases={r['trial'] for r in labels}
    train_cases=set(edges)-validation_cases
    if len(train_cases)!=6 or len(validation_cases)!=2:
        raise ValueError('Expected six calibration and two held-out scenes')
    calibration=np.asarray([goal_by_case[c]-opening_by_case[c] for c in sorted(train_cases)])
    fixed_offset=np.median(calibration,axis=0)
    results=[]
    for row,pred in zip(labels,preds):
        case=row['trial'];tcp=pose_matrix(row['state'][:7]);cameras={cam:tcp@tcp_optical[cam] for cam in CAMERAS}
        pix={cam:pred['cameras'][cam]['opening_xy'] for cam in CAMERAS}
        estimated=triangulate(pix,cameras)+fixed_offset
        target=goal_by_case[case]
        true_distance=np.linalg.norm(target-np.asarray(row['state'][:3]))*1000
        residual=reprojection_residual_px(estimated-fixed_offset,pix,cameras)
        results.append({'case':case,'split':row['split'],'frame':row['frame'],
            'sim_time':row['sim_time'],'true_tcp_to_standoff_mm':float(true_distance),
            'predicted_standoff_base_m':estimated.tolist(),
            'position_error_mm':float(np.linalg.norm(estimated-target)*1000),
            'reprojection_residual_px':float(residual)})
    groups={}
    for split in ('train','validation'):
        by_distance={}
        for name,lo,hi in [('near20_50',20,50),('near50_80',50,80),('far80plus',80,float('inf'))]:
            chosen=[r for r in results if r['split']==split and lo<=r['true_tcp_to_standoff_mm']<hi]
            error=[r['position_error_mm'] for r in chosen]
            by_distance[name]={'frames':len(chosen),
                'episodes':len({r['case'] for r in chosen}),
                'median_mm':float(np.median(error)) if error else None,
                'p95_mm':float(np.quantile(error,.95)) if error else None,
                'reprojection_p95_px':float(np.quantile([r['reprojection_residual_px'] for r in chosen],.95)) if error else None}
        groups[split]=by_distance
    report={'schema':'aic_sc_visual_standoff_diagnostic/v1',
        'calibration_train_cases':sorted(train_cases),
        'fixed_opening_to_transport_tcp_base_offset_m':fixed_offset.tolist(),
        'calibration_residual_mm_p95':float(np.quantile(np.linalg.norm(calibration-fixed_offset,axis=1)*1000,.95)),
        'by_split_distance':groups,'per_frame':results,
        'runtime_inputs':'RGB, ordinary measured TCP, public selected-port code, fixed camera calibration and fitted transport offset',
        'true_scene_geometry_runtime_input':False,'final_scenes_opened':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='per_frame'},indent=2))


if __name__=='__main__':main()
