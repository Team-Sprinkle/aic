#!/usr/bin/env python3
"""Prepare disjoint one/two-card alternate-request SFP controls."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil

import yaml

from run_sfp_cheatcode_target_pilot import ROOT, TEMPLATE


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-split',choices=('train','validation'),default='train')
    p.add_argument('--per-count',type=int,default=5)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior plan/rollouts')
    by_episode={}
    for line in a.labels.open():
        r=json.loads(line)
        if r['split']==a.source_split and r['episode_index'] not in by_episode:
            by_episode[r['episode_index']]=r
    by_count=defaultdict(list)
    for r in by_episode.values():
        source=r.get('source_scene_yaml')
        if not source or r.get('official_tier3',75)<75:continue
        cfg=yaml.safe_load(Path(source).read_text());trial=next(iter(cfg['trials'].values()))
        board=trial['scene']['task_board']
        present=[i for i in range(5) if board[f'nic_rail_{i}']['entity_present']]
        if len(present) not in (1,2):continue
        by_count[len(present)].append((r,source,cfg,present))
    if not 1<=a.per_count<=5:p.error('Per-count cap must be 1–5')
    if any(len(by_count[n])<a.per_count for n in (1,2)):
        raise ValueError('Insufficient sparse source scenes')
    opposite='validation' if a.source_split=='train' else 'train'
    opposite_scenes={r['scene_sha256'] for line in a.labels.open()
                     if (r:=json.loads(line))['split']==opposite}
    selected=[]
    for n in (1,2):
        items=sorted(by_count[n],key=lambda x:hashlib.sha256(x[1].encode()).hexdigest())
        unique={}
        for item in items:
            scene=item[0]['scene_sha256']
            if scene not in opposite_scenes:unique.setdefault(scene,item)
        if len(unique)<a.per_count:raise ValueError('Insufficient disjoint sparse scenes')
        selected.extend(list(unique.values())[:a.per_count])
    a.output.mkdir(parents=True)
    rows=[]
    for r,source,cfg,present in selected:
        original_task=next(iter(next(iter(cfg['trials'].values()))['tasks'].values()))
        original=(int(original_task['target_module_name'].rsplit('_',1)[-1]),
                  int(original_task['port_name'].rsplit('_',1)[-1]))
        alt_card=present[-1] if len(present)==2 and present[-1]!=original[0] else present[0]
        alt_port=1-original[1]
        name=f'cards{len(present)}_episode{r["episode_index"]}_to_card{alt_card}_port{alt_port}'
        folder=a.output/'scenes'/name;folder.mkdir(parents=True)
        original_digest=digest({'scene':next(iter(cfg['trials'].values()))['scene'],
                                'robot':cfg.get('robot')})
        original_task['target_module_name']=f'nic_card_mount_{alt_card}'
        original_task['port_name']=f'sfp_port_{alt_port}'
        if digest({'scene':next(iter(cfg['trials'].values()))['scene'],
                   'robot':cfg.get('robot')})!=original_digest:raise AssertionError('Scene changed')
        (folder/'eval_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        for file in ('world_audit.sdf','bridge_audit.yaml'):
            shutil.copy2(TEMPLATE/file,folder/file)
        rows.append(dict(name=name,source_trial=source,original_request=list(original),
                         alternate_request=[alt_card,alt_port],card_count=len(present),
                         scene_digest=original_digest,scene_dir=str(folder),
                         config_sha256=hashlib.sha256((folder/'eval_config.yaml').read_bytes()).hexdigest(),
                         original_score={'tier3':75}))
    plan=dict(schema='aic_sfp_paired_coverage_v2',train_only=a.source_split=='train',
              source_split=('train' if a.source_split=='train' else 'development'),
              final_scenes_opened=False,
              source_agent_actions_used=False,
              gate='first five >=4 full insertions; do not admit unsuccessful action labels',
              scenes=rows)
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(planned=len(rows),card_counts=[r['card_count'] for r in rows],
                          requests=[r['alternate_request'] for r in rows])))


if __name__=='__main__':main()
