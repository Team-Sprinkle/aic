#!/usr/bin/env python3
"""Prepare six training-only same-board alternate SFP request controls.

Every source board/reset has a previously scored successful request. Only the
public target module and port fields change in the paired configuration.
No simulator is launched by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import yaml


ALTERNATES = {
    'card1_port0_trial_000305': (4, 1),
    'card3_port0_trial_000492': (0, 0),
    'card3_port1_trial_000464': (1, 1),
    'card2_port1_trial_000402': (4, 0),
    'card2_port1_trial_000453': (0, 1),
    'card3_port0_trial_000514': (2, 0),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-plan',type=Path,required=True)
    p.add_argument('--source-rollouts',type=Path,required=True)
    p.add_argument('--template-scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error(f'Output exists: {a.output}')
    plan=json.loads(a.source_plan.read_text())['scenes']
    if {x['name'] for x in plan}!=set(ALTERNATES):
        raise ValueError('Source scene set changed')
    rows=[]
    for source in plan:
        if source.get('source_split')!='train':
            raise ValueError(f'Nontraining source scene: {source["name"]}')
        name=source['name'];card,port=ALTERNATES[name]
        config=yaml.safe_load(Path(source['source_trial']).read_text())
        trial=next(iter(config['trials'].values()))
        board=trial['scene']['task_board']
        if not all(board[f'nic_rail_{j}']['entity_present'] for j in range(5)):
            raise ValueError(f'Expected five installed cards: {name}')
        task=next(iter(trial['tasks'].values()))
        original=(int(task['target_module_name'].rsplit('_',1)[-1]),
                  int(task['port_name'].rsplit('_',1)[-1]))
        if original==(card,port):raise ValueError(f'No alternate target: {name}')
        scored=a.source_rollouts/f'teacher_{name}'/'results/scoring.yaml'
        if not scored.exists():raise FileNotFoundError(scored)
        scored_data=yaml.safe_load(scored.read_text())
        scored_trial=next((value for key,value in scored_data.items()
                           if key.startswith('trial_')),None)
        if scored_trial is None or float(scored_trial.get('tier_3',{}).get('score',0))<75:
            raise ValueError(f'Original request did not fully insert: {name}')
        original_scene_digest=digest({'scene':trial['scene'],
                                      'robot':config.get('robot')})
        task['target_module_name']=f'nic_card_mount_{card}'
        task['port_name']=f'sfp_port_{port}'
        if digest({'scene':trial['scene'],'robot':config.get('robot')})!=original_scene_digest:
            raise AssertionError('Task edit changed scene or robot')
        folder=a.output/name
        folder.mkdir(parents=True)
        (folder/'eval_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        for item in ('world_audit.sdf','bridge_audit.yaml'):
            shutil.copy2(a.template_scene/item,folder/item)
        rows.append({'name':name,'scene_dir':str(folder.resolve()),
                     'source_trial':source['source_trial'],
                     'source_successful_rollout':str(scored.parent.parent),
                     'original_request':list(original),
                     'alternate_request':[card,port],
                     'same_board_robot_scene_sha256':original_scene_digest,
                     'trial_yaml_sha256':hashlib.sha256((folder/'eval_config.yaml').read_bytes()).hexdigest(),
                     'split':'train','task_change_only':True})
    result={'schema':'sfp_same_board_paired_request_plan/v1',
            'purpose':'test whether same-board alternate requests supply the missing task-code discrimination',
            'planned_new_rootless_teacher_rollouts':len(rows),
            'existing_original_teacher_rollouts':len(rows),
            'train_only':True,'source_actions_from_agent_dataset_used':False,
            'prelaunch_gate':'rootless Docker user key quota has a free slot; active Isaac job untouched',
            'teacher_gate':'at least 4 of 6 alternate requests fully insert and reach measured safe buffered region without scored contact',
            'actor_gate':'same-image task-code sensitivity must reflect paired teacher motion; held-out first-action p95 improves at least 20%; live safe arrivals at least 2 of 3 new development starts with no greater contact',
            'final_scenes_opened':False,'pairs':rows}
    (a.output/'plan.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'pairs':len(rows),'original_requests':sorted({tuple(r['original_request']) for r in rows}),
                      'alternate_requests':sorted({tuple(r['alternate_request']) for r in rows})}))


if __name__=='__main__':main()
