#!/usr/bin/env python3
"""Use actual acquisition methods to compare joint versus serial pose windows."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as S
import numpy as np
from scipy.spatial.transform import Rotation
from shared_port_tcp_geometry import matrix_from_xyz_xyzw

directory = Path('aic_example_policies/aic_example_policies/ros')


class ReportBase:
    def _fresh_report(self,observation,task):
        self.calls += 1
        if getattr(self,'required_SC_far_context_frames',0) and self.calls>1:
            self.far_grasp_frames.append(self.calls)
        return dict(valid=True,reason='native_fit',predicted_opening_base_m=[0.,0.,0.],
                    predicted_port_quat_xyzw=[0.,0.,0.,1.]),{},self.state.copy(),self.calls*.2


path = directory/'RunFullConnectorInsertion.py'
original = next(node for node in ast.parse(path.read_text()).body if isinstance(node,ast.ClassDef))
method = next(node for node in original.body if isinstance(node,ast.FunctionDef) and node.name=='_fresh_report')
cls = ast.ClassDef(name='Report',bases=[ast.Name(id='ReportBase',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[])
namespace = dict(ReportBase=ReportBase,np=np,Rotation=Rotation,
                 matrix_from_xyz_xyzw=matrix_from_xyz_xyzw,
                 time=S(monotonic=lambda:0.,perf_counter=lambda:0.,sleep=lambda _:None))
exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(path),'exec'),namespace)
path = directory/'RunRuleThenSharedRecovery.py'
original = next(node for node in ast.parse(path.read_text()).body if isinstance(node,ast.ClassDef) and node.name=='RunRuleThenSharedRecovery')
method = next(node for node in original.body if isinstance(node,ast.FunctionDef) and node.name=='_lock')
exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)


def run(joint,unstable=False,required_far=0):
    policy = namespace['Report']()
    policy.calls = 0
    policy.joint_grasp_acquisition = joint
    policy.required_SC_far_context_frames = required_far
    policy.far_grasp_frames = []
    policy.state = np.zeros(32)
    policy.state[2],policy.state[6] = -.2,1.
    policy.connector_rotation_window = []
    policy.locked_connector_rotation = None
    def rotation(images,**kwargs):
        angle = policy.calls*5. if unstable else .02
        return dict(orientation_for_diagnostic_only=Rotation.from_euler('x',angle,degrees=True).as_matrix().tolist())
    policy.connector_models = dict(sfp=S(predict=rotation),sc=S(predict=rotation))
    policy.local = S(visual_port=None,visual_body_tip=None,robot_history=[],clear_pending_actions=lambda:None,
        _control_sim_time=lambda obs:(policy.calls+1)*.2,
        assembler=S(assemble_ros=lambda obs:policy.state),
        bind_visual_handoff=lambda *args:None,bind_predicted_body_tip=lambda *args:None)
    policy._command = lambda *args,**kwargs:None
    policy._trace = lambda **fields:json.dumps(fields)
    policy.grasp = S(predict=lambda *args,**kwargs:dict(predicted_tip_position_tcp_m=[0.,0.,.05]))
    policy.acquisition_state = policy.state.copy()
    policy.perception_latencies_ms = []
    def observation():
        if policy.calls>=8:
            raise RuntimeError('bounded inconsistent-window stop')
        return object()
    accepted = namespace['_lock'](policy,np.array([0,1]) if required_far else np.array([1,0]),observation,None,fresh_context=False)
    return accepted,policy.calls


assert run(False)==(True,5)
assert run(True)==(True,3)
assert run(True,required_far=3)==(True,4)
try:
    run(True,unstable=True)
except RuntimeError as error:
    assert str(error)=='bounded inconsistent-window stop'
else:
    raise AssertionError('Unstable rotation accepted despite stable translation')
print(json.dumps(dict(actual_methods=True,legacy_frames=5,joint_frames=3,
                      SC_quiet_context_frames=3,SC_total_with_first_moving_frame=4,
                      unstable_rotation_rejected=True,translation_threshold_unchanged=True,
                      JSON_trace=True)),flush=True)
