#!/usr/bin/env python3
"""Offline rigid MoveIt path from measured actor takeover to teacher recovery joint state."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import sys
import time

import numpy as np
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy

from aic_teacher_official.expert_generator.collision_scene import object_geometries_from_gazebo_assets
from aic_teacher_official.expert_generator.moveit_py_backend import (
    MoveItPyBackendConfig,MoveItPyPlanningBackend,_apply_verified_scene)

NAMES=('shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
       'wrist_1_joint','wrist_2_joint','wrist_3_joint')


def sample_at(stream,second):
    times=np.asarray([x[0]*1e-9 for x in stream])
    index=int(np.argmin(np.abs(times-second)))
    return stream[index],float(times[index])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--net-mm',type=float,default=20.)
    p.add_argument('--max-lookahead-s',type=float,default=8.)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous probe')
    if not 10<=a.net_mm<=40 or not 2<=a.max_lookahead_s<=10:p.error('Invalid target bounds')
    event=json.loads((a.audit/'corrective_teacher_takeover.json').read_text())
    takeover=float(event['takeover_sim_time_s'])
    data=json.loads((a.audit/'scored_stream.json').read_text())
    ctrl=data['streams']['/aic_controller/controller_state']
    times=np.asarray([x[0]*1e-9 for x in ctrl]);xyz=np.asarray([
        [x[1]['tcp_pose']['position'][k] for k in 'xyz'] for x in ctrl])
    k=int(np.searchsorted(times,takeover))
    if k>=len(times):p.error('No TCP after takeover')
    target=next((j for j in range(k+1,len(times)) if
        times[j]-times[k]<=a.max_lookahead_s and
        np.linalg.norm(xyz[j]-xyz[k])*1000>=a.net_mm),None)
    if target is None:
        summary=dict(schema='aic_sfp_actor_state_moveit_probe/v1',audit=str(a.audit),
                     scene_group=data['scene_group_id'],request=data['task'],
                     takeover_sim_s=takeover,target_available=False,
                     reason='no_teacher_net_target_within_cap',net_mm=a.net_mm,
                     max_lookahead_s=a.max_lookahead_s,moveit_success=False,
                     collision_objects_applied=0,trajectory_points=0,
                     start_fk_error_mm=None,rigid_collision_only=True,
                     cable_collision_checked=False,gazebo_executed=False,
                     final_scenes_opened=False)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary))
        return
    joints=data['streams']['/joint_states']
    start_row,start_t=sample_at(joints,float(times[k]));goal_row,goal_t=sample_at(joints,float(times[target]))
    def positions(row):
        payload=row[1];by=dict(zip(payload['name'],payload['position']))
        return np.asarray([by[n] for n in NAMES],float)
    start_q=positions(start_row);goal_q=positions(goal_row)
    root=Path(__file__).resolve().parents[1]
    config=a.audit/'eval_config.yaml'
    objects=object_geometries_from_gazebo_assets(config,repo_root=root)
    cfg=MoveItPyBackendConfig(workspace_dir=str(root),use_sim_time=False,planning_time=10.)
    backend=MoveItPyPlanningBackend(cfg)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    joint_source=a.output.with_suffix('.start_joint.json')
    joint_source.write_text(json.dumps(dict(name=NAMES,position=start_q.tolist()),indent=2)+'\n')
    with (a.output.with_suffix('.joint_publisher.log')).open('w') as publisher_log:
        publisher=subprocess.Popen([sys.executable,str(root/'scripts/publish_moveit_joint_state.py'),
                                    str(joint_source)],stdout=publisher_log,stderr=subprocess.STDOUT)
        try:
            time.sleep(.5)
            moveit=MoveItPy(node_name='aic_sfp_actor_state_probe',config_dict=backend._moveit_config_dict())
            count=_apply_verified_scene(moveit,objects,required=True)
            model=moveit.get_robot_model();group=moveit.get_planning_component(cfg.planning_group)
            states=[]
            for q in (start_q,goal_q):
                state=RobotState(model);state.set_joint_group_active_positions(cfg.planning_group,q);state.update()
                states.append(state)
            group.set_start_state(robot_state=states[0]);group.set_goal_state(robot_state=states[1])
            result=group.plan();trajectory=getattr(result,'trajectory',None)
            message=trajectory.get_robot_trajectory_msg() if trajectory is not None else None
            points=list(message.joint_trajectory.points) if message is not None else []
            planned_xyz=[];planned_q=[]
            if points:
                joint_names=list(message.joint_trajectory.joint_names)
                for point in points:
                    by=dict(zip(joint_names,point.positions))
                    q=np.asarray([by[n] for n in NAMES],float)
                    state=RobotState(model)
                    state.set_joint_group_active_positions(cfg.planning_group,q)
                    state.update();pose=state.get_pose(cfg.end_effector_link)
                    planned_xyz.append([pose.position.x,pose.position.y,pose.position.z])
                    planned_q.append(q.tolist())
            fk=states[0].get_pose(cfg.end_effector_link)
            fk_xyz=np.array([fk.position.x,fk.position.y,fk.position.z])
            fk_error=float(np.linalg.norm(fk_xyz-xyz[k])*1000)
        finally:
            publisher.terminate()
            publisher.wait(timeout=10)
    summary=dict(schema='aic_sfp_actor_state_moveit_probe/v1',
                 audit=str(a.audit),config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                 scene_group=data['scene_group_id'],request=data['task'],
                 takeover_sim_s=takeover,goal_sim_s=float(times[target]),
                 source_tcp_base_m=xyz[k].tolist(),target_tcp_base_m=xyz[target].tolist(),
                 net_target_mm=float(np.linalg.norm(xyz[target]-xyz[k])*1000),
                 start_joint_sample_sim_s=start_t,goal_joint_sample_sim_s=goal_t,
                 start_joint_positions=start_q.tolist(),goal_joint_positions=goal_q.tolist(),
                 start_fk_error_mm=fk_error,collision_objects_applied=count,
                 target_available=True,
                 moveit_success=bool(getattr(result,'success',bool(points))) and bool(points),
                 trajectory_points=len(points),
                 trajectory_tcp_base_m=planned_xyz,
                 trajectory_joint_positions=planned_q,
                 trajectory_tcp_path_mm=float(np.linalg.norm(np.diff(np.asarray(planned_xyz),axis=0),axis=1).sum()*1000) if len(planned_xyz)>1 else 0.,
                 rigid_collision_only=True,cable_collision_checked=False,
                 gazebo_executed=False,final_scenes_opened=False)
    a.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))

if __name__=='__main__':main()
