#!/usr/bin/env python3
"""Freeze fresh nonfinal 2/3/5-card starts for corrected-vs-control live gate."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from collect_sfp_diverse_teacher import SOURCE_ROOT, PATTERNS, config, digest, board_count
from run_sfp_cheatcode_target_pilot import TEMPLATE

PORT_BY_COUNT={2:1,3:0,5:1}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,action='append',required=True,
                   help='Pass every actor train/validation label source; scenes from all are excluded')
    p.add_argument('--previous-live-plan',type=Path,required=True)
    p.add_argument('--exclude-plan',type=Path,action='append',default=[],
                   help='Additional nonfinal scene plans to keep disjoint')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous live plan')
    blocked={json.loads(line)['scene_sha256'] for source in a.labels
             for line in source.open()}
    previous=json.loads(a.previous_live_plan.read_text())
    if previous['final_scenes_opened']:
        p.error('Previous live plan opened final scenes')
    blocked.update(x['scene_digest'] for x in previous['candidate_scenes'])
    for extra in a.exclude_plan:
        item=json.loads(extra.read_text())
        blocked.update(x['scene_digest'] for x in item.get('candidate_scenes',item.get('pairs',[])))
    pool={count:[] for count in PORT_BY_COUNT}
    for pattern in PATTERNS:
        for path in SOURCE_ROOT.glob(pattern):
            cfg=config(str(path));trial=next(iter(cfg['trials'].values()))
            board=trial['scene']['task_board'];task=next(iter(trial['tasks'].values()))
            count=board_count(board)
            if count not in pool:continue
            card=int(task['target_module_name'].rsplit('_',1)[-1])
            port=int(task['port_name'].rsplit('_',1)[-1])
            if (port!=PORT_BY_COUNT[count] or card==0 or
                not board[f'nic_rail_{card}']['entity_present']):continue
            scene=digest({'scene':trial['scene'],'robot':cfg.get('robot')})
            if scene not in blocked:pool[count].append((path,scene,card,port))
    chosen=[];used=set()
    for count in PORT_BY_COUNT:
        candidates=sorted(pool[count],key=lambda x:hashlib.sha256(str(x[0]).encode()).hexdigest())
        for path,scene,card,port in candidates:
            if scene in used:continue
            chosen.append((count,path,scene,card,port));used.add(scene)
            if sum(x[0]==count for x in chosen)==3:break
        if sum(x[0]==count for x in chosen)<3:
            p.error(f'Only {len(pool[count])} unused {count}-card starts for port {PORT_BY_COUNT[count]}')
    a.output.mkdir(parents=True)
    specs=[]
    for count,path,scene,card,port in chosen:
        name=f'fresh_cards{count}_card{card}p{port}_{path.parent.parent.name}_{path.stem}'
        folder=a.output/'scenes'/name;folder.mkdir(parents=True)
        shutil.copy2(path,folder/'eval_config.yaml')
        for filename in ('world_audit.sdf','bridge_audit.yaml'):
            shutil.copy2(TEMPLATE/filename,folder/filename)
        specs.append(dict(name=name,card_count=count,request=[card,port],
                          source_trial=str(path),scene_digest=scene,scene_dir=str(folder),
                          config_sha256=hashlib.sha256((folder/'eval_config.yaml').read_bytes()).hexdigest()))
    (a.output/'plan.json').write_text(json.dumps(dict(
        schema='aic_sfp_paired_actor_live_starts/v1',candidate_scenes=specs,
        maximum_teacher_candidates=9,maximum_actor_starts=3,
        selected_port_by_card_count=PORT_BY_COUNT,
        previous_live_groups_excluded=True,
        label_sources=[str(path) for path in a.labels],
        complete_reset_group_excluded_from_train_and_development=True,
        final_scenes_opened=False,teacher_actions_actor_input=False),indent=2)+'\n')
    print(json.dumps(dict(candidate_scenes=len(specs),port_by_count=PORT_BY_COUNT)))


if __name__=='__main__':main()
