#!/usr/bin/env python3
"""Plan a bounded second paired collection on unused grouped train/dev resets."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil

import yaml

from collect_sfp_divergent_pairs import TEMPLATE


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--prior-plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-per-count', type=int, default=5)
    p.add_argument('--development-count', type=int, default=9)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve existing collection')
    used = {int(x['original_episode']) for x in json.loads(a.prior_plan.read_text())['scenes']}
    episodes = {}
    for line in a.labels.open():
        row = json.loads(line)
        episode = int(row['episode_index'])
        if 2400000 <= episode < 3200000 and episode not in used:
            episodes.setdefault(episode, row)
    by_split_count = defaultdict(list)
    for row in episodes.values():
        source = Path(row['source_scored_stream']).parent / 'eval_config.yaml'
        cfg = yaml.safe_load(source.read_text())
        trial = next(iter(cfg['trials'].values()))
        board = trial['scene']['task_board']
        present = [i for i in range(5) if board[f'nic_rail_{i}']['entity_present']]
        if len(present) < 2:
            continue
        original = int(row['selected_card'])
        alternate = max(present, key=lambda i: abs(i-original))
        if abs(alternate-original) < (2 if row['split']=='train' else 1):
            continue
        by_split_count[(row['split'],len(present))].append((row,source,cfg,alternate))
    chosen = []
    for count in range(2,6):
        items = sorted(by_split_count[('train',count)],
                       key=lambda x:(-abs(x[3]-int(x[0]['selected_card'])),
                                     int(x[0]['episode_index'])))
        if len(items) < a.train_per_count:
            p.error(f'Only {len(items)} unused train scenes with {count} cards')
        chosen.extend(items[:a.train_per_count])
    dev = sorted(by_split_count[('validation',2)],
                 key=lambda x:(-abs(x[3]-int(x[0]['selected_card'])),
                               int(x[0]['episode_index'])))
    if len(dev) < a.development_count:
        p.error(f'Only {len(dev)} unused development scenes with two cards')
    chosen.extend(dev[:a.development_count])
    a.output.mkdir(parents=True)
    scenes = []
    for row,source,cfg,alternate in chosen:
        original = [int(row['selected_card']),int(row['selected_port'])]
        desired = [alternate,1-original[1]]
        trial = next(iter(cfg['trials'].values()))
        task = next(iter(trial['tasks'].values()))
        task['target_module_name'] = f'nic_card_mount_{desired[0]}'
        task['port_name'] = f'sfp_port_{desired[1]}'
        name = (f"episode{row['episode_index']}_card{original[0]}p{original[1]}"
                f'_to_card{desired[0]}p{desired[1]}')
        folder = a.output/'scenes'/name
        folder.mkdir(parents=True)
        (folder/'eval_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        for filename in ('world_audit.sdf','bridge_audit.yaml'):
            shutil.copy2(TEMPLATE/filename,folder/filename)
        board=trial['scene']['task_board']
        scenes.append(dict(name=name,source_split=row['split'],
                           original_episode=int(row['episode_index']),
                           source_scene_hash=row['scene_sha256'],
                           source_config=str(source),original_request=original,
                           alternate_request=desired,
                           card_count=sum(board[f'nic_rail_{i}']['entity_present'] for i in range(5)),
                           scene_dir=str(folder),config_sha256=digest(folder/'eval_config.yaml')))
    plan=dict(schema='aic_sfp_divergent_pairs/v2',train_only=False,
              source_agent_actions_used=False,final_scenes_opened=False,
              first_five_min_full=4,planned_count=len(scenes),scenes=scenes,
              source_episode_grouped=True,previously_used_source_resets_excluded=True)
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(planned=len(scenes),train=sum(x['source_split']=='train' for x in scenes),
                          development=sum(x['source_split']=='validation' for x in scenes))))


if __name__=='__main__':
    main()
