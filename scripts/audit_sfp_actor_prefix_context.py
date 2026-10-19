#!/usr/bin/env python3
"""Check causal actor-prefix context and supervised teacher boundary."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from PIL import Image


def audit(path,expected_split):
    episodes=defaultdict(list)
    for line in path.open():
        row=json.loads(line)
        if row['split']!=expected_split:raise ValueError('Unexpected split')
        episodes[int(row['episode_index'])].append(row)
    results=[]
    for episode,rows in episodes.items():
        times=[float(x['timestamp_s']) for x in rows]
        if any(a>=b for a,b in zip(times,times[1:])):
            raise ValueError(f'Noncausal command order in {episode}')
        if len({x['scene_sha256'] for x in rows})!=1:
            raise ValueError(f'Mixed reset scene in {episode}')
        prefix=[x for x in rows if x.get('actor_prefix_context_only')]
        teacher=[x for x in rows if x.get('privileged_teacher_after_takeover')]
        if len(prefix)<5 or len(teacher)<30 or len(prefix)+len(teacher)!=len(rows):
            raise ValueError(f'Missing context or teacher actions in {episode}')
        if (prefix[-1]['timestamp_s']>=teacher[0]['timestamp_s'] or
                teacher[0]['timestamp_s']-prefix[-1]['timestamp_s']>1):
            raise ValueError(f'Broken actor/teacher boundary in {episode}')
        if any(x.get('actor_prefix_used_as_supervision') is not False for x in rows):
            raise ValueError(f'Actor prefix admitted as a label in {episode}')
        if any(x.get('privileged_teacher_after_takeover') for x in prefix):
            raise ValueError(f'Privileged teacher flag on actor context in {episode}')
        if any(x.get('actor_prefix_context_only') for x in teacher):
            raise ValueError(f'Context flag on teacher label in {episode}')
        samples=(prefix[0],prefix[-1],teacher[0],teacher[-1])
        dimensions={}
        for camera in ('center','left','right'):
            for row in samples:
                with Image.open(row['images'][camera]) as image:
                    image.verify();dimensions[camera]=image.size
        results.append(dict(episode=episode,scene=rows[0]['scene_sha256'],
                            requested_port=[rows[0]['selected_card'],rows[0]['selected_port']],
                            prefix_rows=len(prefix),teacher_rows=len(teacher),
                            boundary_gap_s=teacher[0]['timestamp_s']-prefix[-1]['timestamp_s'],
                            images_verified=12,camera_dimensions=dimensions))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous audit')
    train=audit(a.train,'train');val=audit(a.validation,'validation')
    overlap={x['scene'] for x in train}&{x['scene'] for x in val}
    if overlap:raise ValueError(f'Train/validation reset overlap: {sorted(overlap)}')
    result=dict(schema='aic_sfp_actor_prefix_context_audit/v1',
                train_episodes=len(train),validation_episodes=len(val),
                train_context_rows=sum(x['prefix_rows'] for x in train),
                train_teacher_rows=sum(x['teacher_rows'] for x in train),
                validation_context_rows=sum(x['prefix_rows'] for x in val),
                validation_teacher_rows=sum(x['teacher_rows'] for x in val),
                train_validation_scene_overlap=0,
                actor_prefix_used_as_supervision=False,
                final_scenes_opened=False,episodes=train+val)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'}))


if __name__=='__main__':main()
