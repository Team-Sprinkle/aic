#!/usr/bin/env python3
"""Reuse verified new SFP scenes for a separate grouped coarse-model audit."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import yaml


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sfp-labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds-from-first',type=float,default=3.)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous output')
    episodes=defaultdict(list)
    with a.sfp_labels.open() as f:
        for line in f:
            row=json.loads(line)
            if int(row['episode_index'])>=2400000 and row['official_tier3']==75:
                episodes[int(row['episode_index'])].append(row)
    if len(episodes)<100:raise ValueError('Expected full-success grouped diverse data')
    result=[];counts=Counter();groups=defaultdict(set)
    for episode,rows in sorted(episodes.items()):
        rows.sort(key=lambda r:r['timestamp_s'])
        first=rows[0]['timestamp_s']
        trial_path=Path(rows[0]['source_scored_stream']).parent/'eval_config.yaml'
        board=next(iter(yaml.safe_load(trial_path.read_text())['trials'].values()))['scene']['task_board']
        card_count=sum(bool(board[f'nic_rail_{i}']['entity_present']) for i in range(5))
        for row in rows:
            if row['timestamp_s']-first>a.seconds_from_first:break
            card=int(row.get('selected_card',max(range(5),key=lambda i:row['task_vector'][4+i])))
            port=int(row.get('selected_port',max(range(2),key=lambda i:row['task_vector'][2+i])))
            rec=dict(family='sfp',episode=str(episode),
                     scene_group=row['scene_sha256'],split=row['split'],
                     time_s=row['timestamp_s'],state=row['state'],images=row['images'],
                     image_kind=row['image_kind'],source_scored_stream=row['source_scored_stream'],
                     targets=[],selected_identity=f'sfp_card_{card}_port_{port}',
                     card_count=card_count,autonomous_actor=False,
                     official_tier3=75,geometry_runtime_input=False)
            result.append(rec);counts[(row['split'],card_count,card,port)]+=1
            groups[row['split']].add(row['scene_sha256'])
    if groups['train'] & groups['validation']:
        raise ValueError('Complete reset-group leakage')
    a.output.mkdir(parents=True)
    with (a.output/'rows.jsonl').open('w') as f:
        for r in result:f.write(json.dumps(r,separators=(',',':'))+'\n')
    summary=dict(schema='aic_sfp_diverse_coarse_source/v1',
                 source_labels=str(a.sfp_labels),seconds_from_first=a.seconds_from_first,
                 rows=len(result),groups={k:len(v) for k,v in groups.items()},
                 counts={str(k):v for k,v in sorted(counts.items())},
                 action_labels_used=False,projected_visibility_claim=False,
                 final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='counts'}),flush=True)


if __name__=='__main__':main()
