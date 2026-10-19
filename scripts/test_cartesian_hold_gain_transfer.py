#!/usr/bin/env python3
"""Check public holding-effort conservation and actual free-approach gain seams."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as S
import numpy as np
from scipy.spatial.transform import Rotation
from shared_port_tcp_geometry import matrix_from_xyz_xyzw
from cartesian_hold_gain_transfer import transfer_hold

measured = np.eye(4)
reference = np.eye(4)
reference[:3,3] = [.001,-.002,.009]
reference[:3,:3] = Rotation.from_euler('xyz',[.2,-.3,.1],degrees=True).as_matrix()
old = np.array([500.,500.,90.,50.,50.,50.])
new = np.array([500.,500.,240.,100.,100.,100.])
adjusted = transfer_hold(measured,reference,old,new)
assert np.allclose(old[:3]*reference[:3,3],new[:3]*adjusted[:3,3],atol=1e-10)
assert np.allclose(old[3:]*Rotation.from_matrix(reference[:3,:3]).as_rotvec(),
                   new[3:]*Rotation.from_matrix(adjusted[:3,:3]).as_rotvec(),atol=1e-10)
assert np.allclose(transfer_hold(measured,adjusted,new,old),reference,atol=1e-10)


class Base:
    def _lock(self,*args,**kwargs):
        self._approach_tcp_rotation(np.eye(4))
        if self.fail_after_entry:
            raise RuntimeError('expected sentinel')
        return True

    def _approach_tcp_rotation(self,port):
        return port[:3,:3]

    def _command(self,move_robot,matrix,*,owner):
        self.commands.append((matrix.copy(),list(self.local.cartesian_stiffness)))
        q = Rotation.from_matrix(matrix[:3,:3]).as_quat()
        self.observation.controller_state.reference_tcp_pose = S(
            position=S(**dict(zip('xyz',matrix[:3,3]))),orientation=S(**dict(zip('xyzw',q))))


path = Path('aic_example_policies/aic_example_policies/ros/RunVisualConnectorServoFreeApproachV50.py')
cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node,ast.ClassDef))
cls.bases = [ast.Name(id='Base',ctx=ast.Load())]
namespace = dict(Base=Base,np=np,transfer_hold=transfer_hold,matrix_from_xyz_xyzw=matrix_from_xyz_xyzw)
exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(path),'exec'),namespace)
for failure in (False,True):
    policy = namespace[cls.name]()
    policy.observation = S(controller_state=S(reference_tcp_pose=S(
        position=S(x=0.,y=0.,z=.009),orientation=S(x=0.,y=0.,z=0.,w=1.))))
    state = np.zeros(32)
    state[6] = 1.
    policy.local = S(cartesian_stiffness=old.tolist(),assembler=S(assemble_ros=lambda obs:state),
                     _control_sim_time=lambda obs:.1)
    policy.rule = S(direct_empty_selected=False)
    policy._trace = lambda **fields:json.dumps(fields)
    policy.commands = []
    policy.fail_after_entry = failure
    try:
        policy._lock([1,0],lambda:policy.observation,None,fresh_context=True)
    except RuntimeError:
        assert failure
    assert len(policy.commands)==2
    assert policy.commands[0][1][2]==240. and policy.commands[1][1][2]==90.
    assert abs(policy.commands[1][0][2,3]-.009)<1e-10
    assert policy.gain_get_observation is None and policy.gain_move_robot is None
print(json.dumps(dict(static_translation_effort_preserved=True,rotation_effort_preserved=True,
                      actual_entry_and_exit_methods=True,exception_cleanup=True)),flush=True)
