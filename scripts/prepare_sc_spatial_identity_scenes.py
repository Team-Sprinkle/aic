#!/usr/bin/env python3
"""Make bounded, distinct one-port SC reset groups for a spatial identity test."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import yaml


# Fixed before fitting. Development scenes are evaluated once after the model is frozen.
CASES = [
    ('train_p0_a',0,.126,-.211,3.025,-.041,(0,1,4)),
    ('train_p0_b',0,.176,-.174,3.144,.031,(1,2,3,4)),
    ('train_p0_c',0,.154,-.237,2.975,.011,(0,2)),
    ('train_p0_d',0,.183,-.153,3.183,-.012,(1,3,4)),
    ('train_p1_a',1,.134,-.238,3.115,.052,(0,2,4)),
    ('train_p1_b',1,.183,-.172,2.973,-.043,(1,2)),
    ('train_p1_c',1,.152,-.146,3.175,.009,(0,1,2,3,4)),
    ('train_p1_d',1,.171,-.210,3.045,-.018,(0,3)),
    ('development_p0_a',0,.139,-.182,3.169,.048,(0,1,2,3,4)),
    ('development_p0_b',0,.169,-.228,2.986,-.027,(1,3)),
    ('development_p1_a',1,.125,-.195,3.077,-.051,(0,1,4)),
    ('development_p1_b',1,.177,-.143,3.192,.027,(0,2,3,4)),
]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-port0',type=Path,required=True)
    p.add_argument('--source-port1',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output exists; preserve prior scenes')
    a.output.mkdir(parents=True)
    records=[];seen=set()
    for name,port,x,y,yaw,rail,cards in CASES:
        source=a.source_port0 if port==0 else a.source_port1
        dest=a.output/name;dest.mkdir()
        for file in ('world_audit.sdf','bridge_audit.yaml'):
            shutil.copy2(source/file,dest/file)
        cfg=yaml.safe_load((source/'eval_config.yaml').read_text())
        trial=next(iter(cfg['trials'].values()))
        task=next(iter(trial['tasks'].values()))
        if task['target_module_name']!=f'sc_port_{port}':raise ValueError((name,task))
        board=trial['scene']['task_board']
        board['pose'].update(x=x,y=y,yaw=yaw)
        board[f'sc_rail_{1-port}']={'entity_present':False}
        board[f'sc_rail_{port}']['entity_pose']['translation']=rail
        for card in range(5):
            if card not in cards:board[f'nic_rail_{card}']={'entity_present':False}
        identity=json.dumps({'scene':trial['scene'],'robot':cfg.get('robot'),'task':task},sort_keys=True)
        digest=hashlib.sha256(identity.encode()).hexdigest()
        if digest in seen:raise ValueError(f'Duplicate reset: {name}')
        seen.add(digest)
        (dest/'eval_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        records.append({'name':name,'scene_dir':str(dest),'split':name.split('_')[0],
            'selected_port':port,'other_port_present':False,'cards':list(cards),
            'board_xy_yaw':[x,y,yaw],'rail_translation_m':rail,
            'complete_reset_sha256':digest})
    (a.output/'manifest.json').write_text(json.dumps({'schema':'sc_spatial_identity_scenes/v1',
        'purpose':'one-port selected-identity supervision; no policy expert labels',
        'sources':[str(a.source_port0),str(a.source_port1)],
        'scenes':records,'final_scenes_opened':False},indent=2)+'\n')
    print(json.dumps({'scenes':len(records),'output':str(a.output)}))


if __name__=='__main__':main()
