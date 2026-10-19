#!/usr/bin/env python3
"""Exercise the actual V26 SFP loop: pause a spike, abort sustained load."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from shared_port_tcp_geometry import matrix_from_xyz_xyzw
from insertion_reference_governor_v2 import InsertionReferenceGovernor, ReferenceLimits

path=Path('aic_example_policies/aic_example_policies/ros/RunVisualConnectorServoRecoveryV26.py')
cls=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.ClassDef))
method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_close_pose_servo')
namespace=dict(np=np,Rotation=Rotation,matrix_from_xyz_xyzw=matrix_from_xyz_xyzw,
               InsertionReferenceGovernor=InsertionReferenceGovernor,ReferenceLimits=ReferenceLimits,
               time=SimpleNamespace(monotonic=lambda:0.,sleep=lambda _:None,perf_counter=lambda:0.))
exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)

def run(forces, *, depth=.0465, raw=20):
    events=[];commands=[];index=0
    seed=np.eye(4);seed[2,3]=depth
    def observation():
        nonlocal index
        index+=1
        if index>100:raise AssertionError('Controller failed to finish a bounded test')
        state=np.zeros(32);state[:7]=[0,0,depth,0,0,0,1]
        state[26]=raw;state[27]=forces[min(index-1,len(forces)-1)]
        return SimpleNamespace(sim=index*.05,state=state)
    local=SimpleNamespace(task_vector=lambda _:[1],visual_port=np.eye(4),visual_body_tip=np.zeros(3),
        visual_body_connector_rotation=np.eye(3),governor_seed=(seed.copy(),0.),
        assembler=SimpleNamespace(assemble_ros=lambda x:x.state),_control_sim_time=lambda x:x.sim)
    def command(_,matrix,*,owner):
        if owner=='SFP_seating_force_pause':
            assert np.array_equal(matrix,commands[-1][1] if commands else seed)
        commands.append((owner,matrix.copy()))
    controller=SimpleNamespace(local=local,servo_goal_translation=np.zeros(3),servo_goal_rotation=np.eye(3),
        episode_deadline=100.,mode='none',_force=lambda x:x[27],_command=command,
        _trace=lambda **kw:events.append(kw))
    success=namespace['_close_pose_servo'](controller,None,observation,None,None)
    return success,events,commands

success,events,commands=run([2,8.5,2])
assert success and any(x['event']=='SFP_seating_force_pause' for x in events)
assert any(x['event']=='SFP_seating_force_spike_cleared' for x in events)
success,events,_=run([8.5])
assert not success and events[-1]['reason']=='visual_servo_persistent_seat_load_guard'
success,events,_=run([8.5],depth=.040)
assert not success and events[-1]['reason']=='visual_servo_force_guard_outside_seat'
success,events,_=run([2],raw=41)
assert not success and events[-1]['reason']=='visual_servo_raw_force_guard'
print('PASS: transient-seat pause, persistent-load abort, outside-seat abort, raw40N immediate abort; no forward command during pause')
