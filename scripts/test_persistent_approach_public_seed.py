#!/usr/bin/env python3
"""Exercise actual approach/seed methods through commands and JSON traces, without ROS."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from shared_port_tcp_geometry import matrix_from_xyz_xyzw

directory = Path('aic_example_policies/aic_example_policies/ros')
target_rotation = Rotation.from_euler('x',21.,degrees=True).as_matrix()


class LockBase:
    def _lock(self,*args,**kwargs):
        return True

    def _approach_tcp_rotation(self,port):
        return target_rotation


source = directory/'RunSample25mmThenInsertion.py'
original = next(node for node in ast.parse(source.read_text()).body if isinstance(node,ast.ClassDef))
method = next(node for node in original.body if isinstance(node,ast.FunctionDef) and node.name=='_lock')
cls = ast.ClassDef(name='Approach',bases=[ast.Name(id='LockBase',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[])
namespace = dict(LockBase=LockBase,np=np,Rotation=Rotation,matrix_from_xyz_xyzw=matrix_from_xyz_xyzw,
                 time=SimpleNamespace(monotonic=lambda:0.,sleep=lambda _:None,perf_counter=lambda:0.))
exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(source),'exec'),namespace)
source = directory/'RunVisualConnectorServoPersistentApproachV48.py'
cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node,ast.ClassDef))
cls.bases = [ast.Name(id='Approach',ctx=ast.Load())]
exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(source),'exec'),namespace)
policy = namespace[cls.name]()
state = np.zeros(32)
state[2],state[6] = -.080,1.
reference = np.eye(4)
reference[2,3] = -.080
commands,traces = [],[]
tick = 0


class Done(Exception):
    pass


def observation():
    global tick
    tick += 1
    if tick>5:
        raise Done()
    return SimpleNamespace(stamp=tick*.05)


def sim_time(obs):
    policy.local.last_observation_stamp = obs.stamp
    return obs.stamp


def command(move_robot,matrix,*,owner):
    commands.append(matrix.copy())
    policy.local.governor_seed = (matrix.copy(),policy.local.last_observation_stamp)


def trace(**fields):
    traces.append(json.loads(json.dumps(fields)))


policy.local = SimpleNamespace(visual_port=np.eye(4),visual_body_tip=np.zeros(3),governor_seed=None,
    last_observation_stamp=0.,assembler=SimpleNamespace(assemble_ros=lambda obs:state.copy()),
    _control_sim_time=sim_time)
policy.fresh_public_reference = reference.copy()
policy.persistent_approach_orientation = True
policy.approach_antiwindup = True
policy._command,policy._trace = command,trace
policy._force = lambda state:0.
policy._approach_speed = lambda axial:.04
policy._approach_orientation_tolerance = lambda:np.deg2rad(2.)
try:
    policy._lock([1,0],observation,None,fresh_context=True)
except Done:
    pass
assert len(commands)==5
assert len([row for row in traces if row['event']=='near_visual_approach'])==5
assert all(row['persistent_orientation_reference'] is not None
           for row in traces if row['event']=='near_visual_approach')
assert any(row['event']=='persistent_approach_seed_from_public_hold' for row in traces)
print(json.dumps(dict(actual_first_approach_commands=len(commands),JSON_trace_roundtrip=True,
                      initial_governor_seed_None_handled=True,public_API_seed=True)),flush=True)
