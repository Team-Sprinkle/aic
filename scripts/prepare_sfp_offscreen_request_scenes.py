#!/usr/bin/env python3
"""Fixed bounded SFP card-4 starts for an offscreen-request correspondence test."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import yaml


# Distinct from the earlier card-4 development scene. Development entries stay
# unused until a training-source coverage check passes.
CASES=(
    ('train_a',.157,-.222,3.100,-.014,.012,.017),
    ('train_b',.188,-.204,3.145,.012,-.006,-.009),
    ('train_c',.149,-.245,3.185,.001,.021,.013),
    ('train_d',.170,-.183,3.035,-.009,-.012,.019),
    ('development_a',.182,-.218,3.085,.019,.007,-.014),
    ('development_b',.143,-.195,3.165,-.016,-.015,.004),
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing scenes')
    a.output.mkdir(parents=True)
    seen=set();records=[]
    for name,x,y,yaw,rail0,rail2,rail4 in CASES:
        dest=a.output/name;dest.mkdir()
        for file in ('world_audit.sdf','bridge_audit.yaml'):
            shutil.copy2(a.source/file,dest/file)
        cfg=yaml.safe_load((a.source/'eval_config.yaml').read_text())
        trial=next(iter(cfg['trials'].values()))
        task=next(iter(trial['tasks'].values()))
        if task['target_module_name']!='nic_card_mount_4':raise ValueError('Wrong template')
        board=trial['scene']['task_board']
        board['pose'].update(x=x,y=y,yaw=yaw)
        for card,value in ((0,rail0),(2,rail2),(4,rail4)):
            board[f'nic_rail_{card}']['entity_pose']['translation']=value
        identity=json.dumps({'scene':trial['scene'],'robot':cfg.get('robot'),
                             'task':task},sort_keys=True)
        digest=hashlib.sha256(identity.encode()).hexdigest()
        if digest in seen:raise ValueError('Duplicate reset')
        seen.add(digest)
        (dest/'eval_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        records.append({'name':name,'scene_dir':str(dest),
            'split':name.split('_')[0],'target':'sfp_card_4_port_0',
            'board_xy_yaw':[x,y,yaw],'rails_0_2_4':[rail0,rail2,rail4],
            'complete_reset_sha256':digest})
    (a.output/'manifest.json').write_text(json.dumps({'schema':'sfp_offscreen_request_scenes/v1',
        'source':str(a.source),'purpose':'targeted offscreen-card perception only',
        'scenes':records,'final_scenes_opened':False},indent=2)+'\n')
    (a.output/'collection_manifest.json').write_text(json.dumps({
        'rows':[{'case':r['name'],
                 'split':'validation' if r['split']=='development' else r['split'],
                 'card_count':3}
                for r in records]},indent=2)+'\n')
    print(json.dumps({'output':str(a.output),'scenes':len(records)}))


if __name__=='__main__':main()
