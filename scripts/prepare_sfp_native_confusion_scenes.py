#!/usr/bin/env python3
"""New SFP starts for the specific wrong-card/neighbor-port RGB hypothesis."""
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
    p.add_argument('--exclude-manifest',type=Path,action='append',required=True)
    p.add_argument('--three-card-train-only',action='store_true',
                   help='Create one independent three-card training start for every selected port')
    p.add_argument('--template',type=Path,default=Path(
        'outputs/experiments/2026-10-01_sfp_transport_decisive/scene'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing scene selection')
    excluded={r['scene_reset_sha256'] for manifest in a.exclude_manifest
              for r in json.loads(manifest.read_text())['rows']}
    groups=defaultdict(list)
    for path in sorted(a.source.rglob('trials/trial_*.yaml')):
        cfg=yaml.safe_load(path.read_text());trial=next(iter(cfg['trials'].values()))
        task=next(iter(trial['tasks'].values()));board=trial['scene']['task_board']
        card=int(task['target_module_name'].rsplit('_',1)[-1]);port=int(task['port_name'].rsplit('_',1)[-1])
        count=sum(bool(board[f'nic_rail_{i}']['entity_present']) for i in range(5))
        if not board[f'nic_rail_{card}']['entity_present']:continue
        reset=json.dumps({'scene':trial['scene'],'robot':cfg.get('robot'),'task':task},sort_keys=True)
        digest=hashlib.sha256(reset.encode()).hexdigest()
        if digest not in excluded:groups[card,port,count].append((digest,path))
    plan=([(3,'train',0)] if a.three_card_train_only else
          [(5,'train',0),(5,'train',1),(3,'development',0)])
    rows=[];used=set();a.output.mkdir(parents=True)
    for card in range(5):
        for port in range(2):
            for count,split,number in plan:
                choices=sorted(groups[card,port,count])
                pick=next(((digest,path) for digest,path in choices if digest not in used),None)
                if pick is None:raise ValueError((card,port,count,split))
                digest,path=pick;used.add(digest)
                name=f'card{card}_port{port}_cards{count}_extra{number}_{split}'
                directory=a.output/name;directory.mkdir()
                shutil.copy2(path,directory/'eval_config.yaml')
                for filename in ('world_audit.sdf','bridge_audit.yaml'):
                    shutil.copy2(a.template/filename,directory/filename)
                item={'case':name,'source_trial_yaml':str(path.resolve()),
                      'scene_reset_sha256':digest,'split':split,'selected_card':card,
                      'selected_port':port,'card_count':count,'expert_action_source':False,
                      'reserved_final':False}
                (directory/'manifest.json').write_text(json.dumps(item,indent=2)+'\n')
                rows.append(item)
    manifest={'schema':'aic_sfp_native_confusion_selection/v1','rows':rows,
              'hypothesis':'Only two prior native training starts per selected card/port identity; wrong-card/neighbor-port predictions need new grouped starts',
              'old_reset_hashes_excluded':len(excluded),
              'new_train_starts':sum(split=='train' for _,split,_ in plan)*10,
              'new_development_starts':sum(split=='development' for _,split,_ in plan)*10,
              'reserved_final_opened':False}
    (a.output/'collection_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'scenes':len(rows),'train':manifest['new_train_starts'],
                      'development':manifest['new_development_starts']}))


if __name__=='__main__':main()
