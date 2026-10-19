"""Safety regression for the loaded-stall trigger, independent of ROS/Gazebo."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

source=Path(__file__).resolve().parents[1]/'aic_example_policies/ros/RunVisualConnectorServoRecoveryV5.py'
module=ast.parse(source.read_text())
method=next(n for n in next(n for n in module.body if isinstance(n,ast.ClassDef)).body
            if isinstance(n,ast.FunctionDef) and n.name=='_supervise_servo_stall')
class Transfer(Exception):
    pass
namespace={'np':np,'measured_pose':lambda state:state[:7].copy(),'RecoveryTransfer':Transfer}
exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
trigger=namespace['_supervise_servo_stall']


def scenario(*,force=3.5,speed=0.,depth=.002,tracking=.005):
    events=[];calls=[];clears=[]
    owner=SimpleNamespace(servo_stall_history=[],route='same_corridor',last_command_m=.005,
        supervisor=SimpleNamespace(_trigger=lambda *args:calls.append(args),step=lambda **kwargs:SimpleNamespace(level='local')),
        local=SimpleNamespace(clear_pending_actions=lambda:clears.append(True)),
        _trace=lambda **kwargs:events.append(kwargs))
    state=np.zeros(32);issued=np.eye(4);issued[0,3]=tracking
    def run():
        for tick in range(34):
            stamp=tick*.05;point=np.eye(4);point[2,3]=depth+speed*stamp
            trigger(owner,state,issued,point,stamp,force,.0156)
    return run,events,calls,clears


def test_sustained_loaded_stall_transfers_with_real_force():
    run,events,calls,clears=scenario()
    with pytest.raises(Transfer):run()
    assert len(calls)==1 and clears==[True]
    assert calls[0][1]==3.5
    assert events[-1]['synthetic_force_used'] is False
    assert events[-1]['progress_window_s']>=1.5


@pytest.mark.parametrize('options',[{'force':2.9},{'speed':.006},{'depth':.0155},{'tracking':.0029}])
def test_normal_progress_seating_or_unloaded_hold_does_not_recover(options):
    run,events,calls,clears=scenario(**options)
    run()
    assert not events and not calls and not clears
