"""Regression checks for causal labels and reset grouping, independent of model weights."""
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT=Path(__file__).resolve().parents[2]


def run(script,*args):
    result=subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==0,result.stderr


def board():
    return {'pose':dict(x=.16,y=-.21,z=1.14,roll=0,pitch=0,yaw=3.1),
            **{f'nic_rail_{i}':{'entity_present':i==0,'entity_pose':dict(translation=0,roll=0,pitch=0,yaw=0)} for i in range(5)}}


def test_projection_uses_image_time_past_pose_not_row_pose_or_future(tmp_path):
    case=tmp_path/'case';case.mkdir()
    (case/'eval_config.yaml').write_text(yaml.safe_dump({'trials':{'trial':{'scene':{'task_board':board()}}}}))
    def state(x):return {'tcp_pose':{'position':dict(x=x,y=.19,z=.32),'orientation':dict(x=1,y=0,z=0,w=0)}}
    (case/'scored_stream.json').write_text(json.dumps({'streams':{'/aic_controller/controller_state':[[995000000,state(-.37)],[1005000000,state(.99)]]}}))
    source=tmp_path/'source.jsonl'
    source.write_text(json.dumps({'scene_sha256':'scene','source_scored_stream':str(case/'scored_stream.json'),
        'timestamp_s':1.04,'episode_index':1,'split':'train','state':[999]*7,
        'images':{c:str(case/f'0_1000000000_{c}.jpg') for c in ('center','left','right')},
        'task_vector':[1,0,1,0,1,0,0,0,0,1]})+'\n')
    output=tmp_path/'output'
    run('project_native_transport_images.py','--source',source,'--output',output)
    rows=[json.loads(x) for x in (output/'rows.jsonl').read_text().splitlines()]
    assert len(rows)==1
    assert rows[0]['state'][0]==-.37
    assert rows[0]['image_pose_age_ms']==dict(center=5,left=5,right=5)


def test_different_requested_ports_do_not_create_independent_reset_splits(tmp_path):
    rows=[]
    for i,split in enumerate(('train','validation')):
        case=tmp_path/str(i);case.mkdir()
        config={'robot':{'home':[0]*6},'trials':{f'trial{i}':{'scene':{'task_board':board()},'tasks':{'task':{'port_name':f'port_{i}'}}}}}
        (case/'eval_config.yaml').write_text(yaml.safe_dump(config))
        (case/'scored_stream.json').write_text(json.dumps({'commands':[{'kind':'pose','sim_ns':2000000000}]}))
        rows.append(dict(family='sfp',episode=f'episode{i}',scene_group=f'oldgroup{i}',split=split,time_s=1,
                         images={c:str(case/'frames'/f'0_1000000000_{c}.jpg') for c in ('center','left','right')}))
    source=tmp_path/'source.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in rows));output=tmp_path/'output'
    run('canonicalize_native_refiner_splits.py','--source',source,'--output',output)
    summary=json.loads((output/'summary.json').read_text())
    assert summary['rows']==0
    assert len(summary['quarantined_groups'])==1
    assert summary['rejected']['rows_quarantined_cross_source_reset_split_overlap']==2
