#!/usr/bin/env python3
"""Exercise actual V31 seeding in two port orientations; no ROS required."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from shared_port_tcp_geometry import matrix_from_xyz_xyzw
from insertion_reference_governor_v2 import InsertionReferenceGovernor,ReferenceLimits

path=Path('aic_example_policies/aic_example_policies/ros/RunVisualConnectorServoBiasSeedV31.py')
tree=ast.parse(path.read_text())
method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_close_pose_servo')
namespace=dict(np=np,Rotation=Rotation,matrix_from_xyz_xyzw=matrix_from_xyz_xyzw,
               InsertionReferenceGovernor=InsertionReferenceGovernor,ReferenceLimits=ReferenceLimits,
               time=SimpleNamespace(monotonic=lambda:0.,sleep=lambda _:None,perf_counter=lambda:0.))
exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
for angle in [0,90]:
    port=np.eye(4);port[:3,:3]=Rotation.from_euler('z',angle,degrees=True).as_matrix()
    position=port[:3,:3]@np.array([.0004,0,.0156]);bias=np.array([.002,0,.003])
    seed=port.copy();seed[:3,3]=position+bias
    state=np.zeros(32);state[:3]=position;state[3:7]=Rotation.from_matrix(port[:3,:3]).as_quat()
    state[13:16]=bias;state[26]=20
    count=[0];events=[];commands=[]
    def observation():
        count[0]+=1
        assert count[0]<100
        return SimpleNamespace(sim=count[0]*.05)
    local=SimpleNamespace(task_vector=lambda _:[0,1],visual_port=port,visual_body_tip=np.zeros(3),
        visual_body_connector_rotation=np.eye(3),governor_seed=(seed.copy(),0.),
        assembler=SimpleNamespace(assemble_ros=lambda _:state.copy()),_control_sim_time=lambda x:x.sim)
    controller=SimpleNamespace(local=local,servo_goal_translation=np.zeros(3),servo_goal_rotation=np.eye(3),
        episode_deadline=100.,mode='none',_force=lambda _:0.,
        _command=lambda _,value,**kwargs:commands.append(value.copy()),_trace=lambda **kwargs:events.append(kwargs))
    namespace['_close_pose_servo'](controller,None,observation,None,None)
    event=next(e for e in events if e['event']=='lateral_integral_seed_from_public_tracking_error')
    expected=port[:3,:3].T@bias
    assert np.allclose(event['new_integral_port_m'][:2],expected[:2],atol=1e-9)
    assert event['new_integral_port_m'][2]==event['previous_integral_port_m'][2]
    assert np.allclose(np.array(event['previous_integral_port_m'])[:2]-expected[:2],[.0004,0],atol=1e-9)
    assert np.linalg.norm(commands[0][:3,3]-seed[:3,3])<=.0013,'Governor must bound the first reference change'
print('PASS: base-to-port bias projection, misalignment excluded, axial seed preserved, initial command bounded')
