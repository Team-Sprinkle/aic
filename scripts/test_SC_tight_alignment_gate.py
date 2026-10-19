#!/usr/bin/env python3
"""Check the actual V29 lateral-advance latch, without ROS or a simulator."""
import ast
from pathlib import Path
import numpy as np

path=Path('aic_example_policies/aic_example_policies/ros/RunVisualConnectorServoAlignedV29.py')
tree=ast.parse(path.read_text())
method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_close_pose_servo')
loop=next(n for n in method.body if isinstance(n,ast.While))
start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and
           any(isinstance(x,ast.Name) and x.id=='alignment_streak' for x in n.targets))
end=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and
         any(isinstance(x,ast.Name) and x.id=='aligned' for x in n.targets))
nodes=loop.body[start:end+1]
namespace={'np':np}
def step(lateral,angle,streak,latched):
    values=dict(lateral=lateral,angle=angle,alignment_streak=streak,alignment_latched=latched)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace,values)
    return values['aligned'],values['alignment_streak'],values['alignment_latched']

streak=0;latched=False
for _ in range(3):
    advance,streak,latched=step(.0004,0,streak,latched)
    assert not advance,'The previously accepted0.4mm must not permit advance'
for i in range(3):
    advance,streak,latched=step(.00009,0,streak,latched)
    assert advance==(i==2),'Three quiet control ticks are required'
advance,streak,latched=step(.00013,0,streak,latched)
assert advance,'Small jitter inside the hysteresis band should not chatter'
advance,streak,latched=step(.00016,0,streak,latched)
assert not advance,'Advance must pause above0.15mm'
advance,streak,latched=step(.00001,np.deg2rad(.6),streak,latched)
assert not advance,'Existing0.5degree angular guard remains active'
print('PASS:0.4mmcannotadvance;0.1mmentryrequires3ticks;0.15mmexit;angular guard preserved')
