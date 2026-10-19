#!/usr/bin/env python3
"""Choose grouped, diverse non-final SFP resets for native RGB supervision.

Source agent trajectories are not used as expert actions. Only their public
reset configs are replayed with stock CheatCode for camera/pose labels.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil

import yaml


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path(
        'outputs/trajectory_datasets/clean_including_no_insert_trajs/sfp_to_nic/agent'))
    p.add_argument('--template',type=Path,default=Path(
        'outputs/experiments/2026-10-01_sfp_transport_decisive/scene'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output already exists; preserve prior selection')
    groups=defaultdict(list)
    for path in sorted(a.source.rglob('trials/trial_*.yaml')):
        cfg=yaml.safe_load(path.read_text());trial=next(iter(cfg['trials'].values()))
        task=next(iter(trial['tasks'].values()));board=trial['scene']['task_board']
        card=int(task['target_module_name'].rsplit('_',1)[-1])
        port=int(task['port_name'].rsplit('_',1)[-1])
        count=sum(bool(board[f'nic_rail_{i}']['entity_present']) for i in range(5))
        if not board[f'nic_rail_{card}']['entity_present']:continue
        reset=json.dumps({'scene':trial['scene'],'robot':cfg.get('robot'),
                          'task':task},sort_keys=True)
        digest=hashlib.sha256(reset.encode()).hexdigest()
        groups[(card,port,count)].append((digest,path))
    rows=[];used=set();a.output.mkdir(parents=True)
    for card in range(5):
        for port in range(2):
            for count,split in ((5,'train'),(1,'train'),(3,'validation')):
                pool=sorted(groups[(card,port,count)])
                selected=next(((digest,path) for digest,path in pool if digest not in used),None)
                if selected is None:raise ValueError(f'Missing distinct reset {(card,port,count)}')
                digest,path=selected;used.add(digest)
                name=f'card{card}_port{port}_cards{count}_{split}'
                directory=a.output/name;directory.mkdir()
                shutil.copy2(path,directory/'eval_config.yaml')
                for filename in ('world_audit.sdf','bridge_audit.yaml'):
                    shutil.copy2(a.template/filename,directory/filename)
                record={'case':name,'source_trial_yaml':str(path.resolve()),
                        'scene_reset_sha256':digest,'split':split,
                        'selected_card':card,'selected_port':port,'card_count':count,
                        'expert_action_source':False,'reserved_final':False}
                (directory/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
                rows.append(record)
    (a.output/'collection_manifest.json').write_text(json.dumps({
        'schema':'aic_sfp_native_perception_selection/v1','rows':rows,
        'train_scene_groups':20,'validation_scene_groups':10,
        'selection':'all ten selected identities, each with five, one, and three cards',
        'source_actions_used':False,'reserved_final_opened':False},indent=2)+'\n')
    print(json.dumps({'scenes':len(rows),'train':20,'validation':10,'output':str(a.output)}))


if __name__=='__main__':main()
