#!/usr/bin/env python3
"""Offline SFP requested-port corridor planning from a measured actor state."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from scipy.spatial.transform import Rotation
import yaml

from aic_teacher_official.expert_generator.collision_scene import object_geometries_from_gazebo_assets
from aic_teacher_official.expert_generator.moveit_py_backend import (
    MoveItPyBackendConfig,MoveItPyPlanningBackend,_apply_verified_scene,
    _pose_stamped_from_serializable)
from aic_teacher_official.expert_generator.scene_snapshot import SerializablePose
from build_two_family_port_projection_labels import sfp_ports
from probe_sfp_actor_state_moveit_path import NAMES,sample_at

CAL=Path('configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--buffer-mm',type=float,default=85.)
    p.add_argument('--enable-cable-side-variants',action='store_true',
                   help='Bounded five-card cable-clearance hypotheses; privileged offline labels only')
    p.add_argument('--start-json',type=Path,help='Live measured takeover joints/TCP from the same Gazebo run')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous plan')
    if not 70<=a.buffer_mm<=100:p.error('Buffer outside safe transport band')
    root=Path(__file__).resolve().parents[1]
    config=a.audit/'eval_config.yaml'
    data=json.loads((a.audit/'scored_stream.json').read_text())
    takeover=float(json.loads((a.audit/'corrective_teacher_takeover.json').read_text())['takeover_sim_time_s'])
    task=data['task'];card=int(task['target_module_name'].rsplit('_',1)[1]);port=int(task['port_name'].rsplit('_',1)[1])
    cfg_yaml=yaml.safe_load(config.read_text());tid=next(iter(cfg_yaml['trials']))
    calibration=json.loads((root/CAL).read_text())
    port_matrix=dict(((c,n),matrix) for c,n,matrix in sfp_ports(
        dict(trial_yaml=str(config),trial_id=tid),calibration))[(card,port)]
    ctrl=data['streams']['/aic_controller/controller_state']
    controller,ctrl_time=sample_at(ctrl,takeover)
    current=controller[1]['tcp_pose'];start_xyz=np.array([current['position'][k] for k in 'xyz'])
    start_quat=np.array([current['orientation'][k] for k in 'xyzw'])
    joint_row,joint_time=sample_at(data['streams']['/joint_states'],takeover)
    joint_by=dict(zip(joint_row[1]['name'],joint_row[1]['position']))
    start_q=np.array([joint_by[n] for n in NAMES],float)
    if a.start_json:
        live=json.loads(a.start_json.read_text())
        if live.get('schema')!='aic_sfp_moveit_live_start/v1':p.error('Unexpected live-start schema')
        if live['task']['target_module_name']!=task['target_module_name'] or live['task']['port_name']!=task['port_name']:
            p.error('Live task differs from saved requested port')
        start_xyz=np.asarray(live['tcp_base_m'],float)
        start_quat=np.asarray(live['tcp_quat_xyzw'],float)
        start_q=np.asarray(live['joints'],float)
        takeover=float(live['sim_s'])
        if start_xyz.shape!=(3,) or start_quat.shape!=(4,) or start_q.shape!=(6,):
            p.error('Malformed live start')
    rel=port_matrix[:3,:3].T@(start_xyz-port_matrix[:3,3])
    buffer=-a.buffer_mm/1000
    if rel[2]>=buffer:
        p.error(f'Actor already past buffered axial target: {rel[2]:.4f}')
    def base(local):return (port_matrix[:3,3]+port_matrix[:3,:3]@np.asarray(local,float)).tolist()
    final=base([0,0,buffer])
    # Alternatives expose a genuine path choice. The first aligns before
    # axial approach; the second retreats farther before alignment.
    align_z=min(float(rel[2]),-.15)
    variants={
        'direct':[('buffered_port',final)],
        'align_then_approach':[('align_far',base([0,0,align_z])),('buffered_port',final)],
        'retreat_align_approach':[
            ('retreat',base([rel[0],rel[1],min(float(rel[2])-.04,-.22)])),
            ('align_far',base([0,0,-.22])),('buffered_port',final)],
    }
    if a.enable_cable_side_variants:
        high=-.22
        variants['side_down_then_center']=[
            ('retreat',base([rel[0],rel[1],min(float(rel[2])-.04,high)])),
            ('side_high',base([0,-.06,high])),
            ('side_lower',base([0,-.06,-.12])),
            ('center_120mm',base([0,0,-.12])),
            ('buffered_port',final)]
        variants['side_sweep_high']=[
            ('retreat',base([rel[0],rel[1],min(float(rel[2])-.04,high)])),
            ('side_high',base([0,-.08,high])),
            ('align_far',base([0,0,high])),
            ('buffered_port',final)]
    objects=object_geometries_from_gazebo_assets(config,repo_root=root)
    cfg=MoveItPyBackendConfig(workspace_dir=str(root),use_sim_time=False,planning_time=10.)
    backend=MoveItPyPlanningBackend(cfg)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    start_file=a.output.with_suffix('.start_joint.json')
    start_file.write_text(json.dumps(dict(name=NAMES,position=start_q.tolist()),indent=2)+'\n')
    with a.output.with_suffix('.joint_publisher.log').open('w') as pub_log:
        publisher=subprocess.Popen([sys.executable,str(root/'scripts/publish_moveit_joint_state.py'),
                                    str(start_file)],stdout=pub_log,stderr=subprocess.STDOUT)
        try:
            time.sleep(.5)
            moveit=MoveItPy(node_name='aic_sfp_actor_corridor',config_dict=backend._moveit_config_dict())
            scene_count=_apply_verified_scene(moveit,objects,required=True)
            model=moveit.get_robot_model();group=moveit.get_planning_component(cfg.planning_group)
            start_state=RobotState(model)
            start_state.set_joint_group_active_positions(cfg.planning_group,start_q);start_state.update()
            fk=start_state.get_pose(cfg.end_effector_link)
            fk_error=float(np.linalg.norm(np.array([fk.position.x,fk.position.y,fk.position.z])-start_xyz)*1000)
            results={}
            for variant,goals in variants.items():
                last=start_state;segments=[];whole=[];success=True
                for name,xyz in goals:
                    group.set_start_state(robot_state=last)
                    goal=SerializablePose(list(xyz),start_quat.tolist(),'base_link')
                    group.set_goal_state(pose_stamped_msg=_pose_stamped_from_serializable(goal),
                                         pose_link=cfg.end_effector_link)
                    planned=group.plan();trajectory=getattr(planned,'trajectory',None)
                    msg=trajectory.get_robot_trajectory_msg() if trajectory is not None else None
                    points=list(msg.joint_trajectory.points) if msg is not None else []
                    if not getattr(planned,'success',bool(points)) or not points:
                        segments.append(dict(name=name,success=False,goal_tcp_base_m=xyz,
                                             reason='moveit_plan_failed'))
                        success=False;break
                    names=list(msg.joint_trajectory.joint_names)
                    samples=[]
                    for point in points:
                        by=dict(zip(names,point.positions));q=np.array([by[n] for n in NAMES],float)
                        state=RobotState(model)
                        state.set_joint_group_active_positions(cfg.planning_group,q);state.update()
                        pose=state.get_pose(cfg.end_effector_link)
                        samples.append(dict(joints=q.tolist(),tcp_base_m=[pose.position.x,pose.position.y,pose.position.z],
                                            tcp_quat_xyzw=[pose.orientation.x,pose.orientation.y,
                                                           pose.orientation.z,pose.orientation.w]))
                    whole.extend(samples[1:] if whole else samples)
                    last=RobotState(model)
                    last.set_joint_group_active_positions(cfg.planning_group,
                        np.asarray(samples[-1]['joints'],float));last.update()
                    segments.append(dict(name=name,success=True,goal_tcp_base_m=xyz,
                                         points=len(samples),end_tcp_error_mm=float(np.linalg.norm(
                                             np.asarray(samples[-1]['tcp_base_m'])-xyz)*1000)))
                path=np.array([x['tcp_base_m'] for x in whole],float)
                results[variant]=dict(success=success,segments=segments,
                    tcp_path_mm=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()*1000) if len(path)>1 else 0.,
                    first_step_base_m=(path[1]-path[0]).tolist() if len(path)>1 else None,
                    samples=whole)
        finally:
            publisher.terminate();publisher.wait(timeout=10)
    result=dict(schema='aic_sfp_actor_state_requested_port_corridor/v1',audit=str(a.audit),
                config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                scene_group=data['scene_group_id'],request_card=card,request_port=port,
                task=data['task'],takeover_sim_s=takeover,controller_sim_s=ctrl_time,joint_sim_s=joint_time,
                start_tcp_base_m=start_xyz.tolist(),start_tcp_quat_xyzw=start_quat.tolist(),
                start_port_frame_m=rel.tolist(),selected_port_base_m=port_matrix[:3,3].tolist(),
                selected_port_quat_xyzw=Rotation.from_matrix(port_matrix[:3,:3]).as_quat().tolist(),
                buffer_goal_tcp_base_m=final,buffer_mm=a.buffer_mm,
                collision_objects_applied=scene_count,start_fk_error_mm=fk_error,
                goal_from_requested_port_geometry=True,teacher_future_joint_goal_used=False,
                cable_side_variants_enabled=a.enable_cable_side_variants,
                start_state_source='live_gazebo_takeover' if a.start_json else 'saved_scored_bag',
                route_variants=results,cable_collision_checked=False,
                gazebo_executed=False,final_scenes_opened=False)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(scene_group=result['scene_group'],request=(card,port),
                          variants={k:v['success'] for k,v in results.items()},
                          objects=scene_count,fk_mm=fk_error)))

if __name__=='__main__':main()
