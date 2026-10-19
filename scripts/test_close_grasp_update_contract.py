#!/usr/bin/env python3
"""Exercise the actual new hold/update method without starting ROS or Gazebo.

Extract the method AST to avoid importing the entire ROS controller ancestry;
this is a control-contract test, not a replacement for a live rollout.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from close_grasp_consensus import CloseGraspConsensus,compare_grasp_update
from shared_port_tcp_geometry import matrix_from_xyz_xyzw

source=Path('aic_example_policies/aic_example_policies/ros/RunVisualConnectorServoRecoveryV21.py')
node=ast.parse(source.read_text())
cls=next(x for x in node.body if isinstance(x,ast.ClassDef))
method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_reestimate_close_grasp')
namespace=dict(np=np,CloseGraspConsensus=CloseGraspConsensus,compare_grasp_update=compare_grasp_update,
               matrix_from_xyz_xyzw=matrix_from_xyz_xyzw,
               RunRuleTransport=SimpleNamespace(native=lambda x:x),
               time=SimpleNamespace(monotonic=lambda:0.,perf_counter=lambda:0.,sleep=lambda _:None))
exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
update=namespace['_reestimate_close_grasp']


def run(angle,force=0.,rotation_only=False):
    body=np.array([0.,0.,.015]);state=np.zeros(32);state[:7]=[0,0,-.04,0,0,0,1]
    reference=np.eye(4);reference[:3,3]=[.002,-.001,-.037]
    model=SimpleNamespace(rotation_only=rotation_only,
        predict=lambda _,visual_tip_position_tcp_m=None:dict(
        predicted_tip_position_tcp_m=(body+[.0001,0,0]).tolist() if visual_tip_position_tcp_m is None else visual_tip_position_tcp_m.tolist(),
        orientation_for_diagnostic_only=Rotation.from_euler('x',angle,degrees=True).as_matrix().tolist()))
    local=SimpleNamespace(governor_seed=(reference.copy(),0.),visual_body_tip=body.copy(),
        visual_body_connector_rotation=np.eye(3),visual_port=np.eye(4),
        assembler=SimpleNamespace(assemble_ros=lambda _:state.copy()),
        _control_sim_time=lambda x:x.sim,clear_pending_actions=lambda:None,governor='old')
    events=[];commands=[]
    controller=SimpleNamespace(local=local,close_grasp_model=model,episode_deadline=100.,
        _command=lambda _,value,**kw:commands.append(value.copy()),_force=lambda _:force,
        _trace=lambda **kw:events.append(kw),servo_stall_history=['old'])
    count=0
    def observation():
        nonlocal count
        count+=1
        if count>20:raise AssertionError('Update failed to reach a bounded decision')
        sim=count*.2;s=SimpleNamespace(sec=int(sim),nanosec=round((sim-int(sim))*1e9))
        image=SimpleNamespace(header=SimpleNamespace(stamp=s))
        return SimpleNamespace(sim=sim,center_image=image,left_image=image,right_image=image)
    accepted=update(controller,observation,None)
    assert all(np.array_equal(command,reference) for command in commands),'Observation hold changed API target'
    assert np.array_equal(local.visual_port,np.eye(4)),'Port frame changed'
    return accepted,controller,events,count,body,reference

accepted,c,events,count,body,reference=run(2.)
assert accepted and count==5
assert np.array_equal(c.local.governor_seed[0],reference)
assert np.allclose(c.local.visual_body_tip,body+[.0001,0,0])
assert c.local.governor is None and c.servo_stall_history==[]
accepted,c,events,count,body,_=run(5.)
assert not accepted and count==10
assert np.array_equal(c.local.visual_body_tip,body)
assert any(x['event']=='close_grasp_large_disagreement_second_window' for x in events)
assert events[-1]['reason']=='persistent_large_disagreement_reacquisition_required'
accepted,c,events,count,body,_=run(2.,force=8.)
assert not accepted and count==1 and np.array_equal(c.local.visual_body_tip,body)
accepted,c,events,count,body,_=run(2.,rotation_only=True)
assert accepted and count==5 and np.array_equal(c.local.visual_body_tip,body)
print(json.dumps(dict(small_update_five_distinct_frames=True,large_disagreement_two_independent_windows_rejected=True,
    loaded_hold_rejected=True,API_reference_unchanged=True,port_frame_unchanged=True,
    rotation_only_keeps_visual_translation=True,
    live_Gazebo_test=False)))
