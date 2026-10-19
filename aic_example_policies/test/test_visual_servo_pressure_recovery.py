"""Distinguish contact pressure from ordinary public gravity-tracking bias."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'aic_example_policies/ros/RunVisualConnectorServoRecoveryV8.py'
module=ast.parse(path.read_text());cls=next(x for x in module.body if isinstance(x,ast.ClassDef));method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_supervise_servo_stall')
class Transfer(Exception):
    pass
ns={'np':np,'measured_pose':lambda state:state[:7].copy(),'RecoveryTransfer':Transfer}
exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),ns)


def run_stall(*,pressure=.002,force=1.6,speed=0.,depth=.002):
    calls=[];events=[];clears=[]
    owner=SimpleNamespace(servo_stall_history=[],route='fixed',last_command_m=.008,
        unloaded_reference_bias_base_m=np.array([0.,0.,.006]),
        local=SimpleNamespace(visual_port=np.eye(4),clear_pending_actions=lambda:clears.append(True)),
        supervisor=SimpleNamespace(_trigger=lambda *args:calls.append(args),step=lambda **kwargs:SimpleNamespace(level='local')),
        _trace=lambda **kwargs:events.append(kwargs))
    state=np.zeros(32);issued=np.eye(4);issued[2,3]=.006+pressure
    def execute():
        for i in range(10):
            stamp=i*.05;point=np.eye(4);point[2,3]=depth+stamp*speed
            ns['_supervise_servo_stall'](owner,state,issued,point,stamp,force,.0156)
    return execute,calls,events,clears


def test_loaded_excess_pressure_triggers_before_one_second():
    execute,calls,events,clears=run_stall()
    with pytest.raises(Transfer):execute()
    assert calls[0][1]==1.6 and clears==[True]
    assert .19<=events[0]['progress_window_s']<1
    assert events[0]['excess_axial_tracking_pressure_mm']==pytest.approx(2)
    assert not events[0]['synthetic_force_used']


@pytest.mark.parametrize('options',[{'pressure':0.},{'force':1.49},{'speed':.006},{'depth':.0155}])
def test_bias_only_normal_motion_or_seating_does_not_trigger(options):
    execute,calls,events,clears=run_stall(**options)
    execute()
    assert not calls and not events and not clears
