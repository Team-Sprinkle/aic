#!/usr/bin/env python3
"""Merge audited corrective teachers while retaining actor prefixes as context only."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--train-context',type=Path,action='append',required=True)
    p.add_argument('--validation-context',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous merge')
    sources=[('base',a.base),*[('train_context',x) for x in a.train_context],
             ('validation_context',a.validation_context)]
    for role,path in sources[1:]:
        manifest=json.loads((path.parent/'manifest.json').read_text())
        if not (manifest['corrective_teacher'] and manifest['corrective_full_insert'] and
                manifest['actor_prefix_context_preserved'] and
                manifest['split']==('train' if role=='train_context' else 'validation')):
            p.error(f'Unadmitted context source {path}')
    counts=Counter();scenes=defaultdict(set);episodes=defaultdict(set);keys=set()
    a.output.mkdir(parents=True)
    with (a.output/'sfp_labels.jsonl').open('w') as out:
        for role,path in sources:
            for line in path.open():
                row=json.loads(line)
                split=row['split'];context=bool(row.get('actor_prefix_context_only',False))
                teacher=bool(row.get('privileged_teacher_after_takeover',False))
                if role!='base':
                    if (split!=('train' if role=='train_context' else 'validation') or
                            context==teacher or
                            row.get('actor_prefix_used_as_supervision') is not False or
                            row.get('geometry_label_only') is not True):
                        raise ValueError(f'Invalid actor-prefix/teacher boundary: {path}')
                    task=row['task_vector']
                    card=max(range(5),key=lambda c:task[4+c]);port=max(range(2),key=lambda q:task[2+q])
                    if (card,port)!=(row['selected_card'],row['selected_port']):
                        raise ValueError('Task code disagrees with selected port')
                key=(int(row['episode_index']),int(row['frame_index']))
                if key in keys:raise ValueError(f'Duplicate episode/frame {key}')
                keys.add(key);scenes[split].add(row['scene_sha256'])
                counts[(role,split,'context' if context else 'teacher' if teacher else 'base')]+=1
                episodes[(role,split)].add(int(row['episode_index']))
                out.write(line)
    overlap=scenes['train']&scenes['validation']
    if overlap:raise ValueError(f'Train/validation reset overlap: {sorted(overlap)[:4]}')
    result=dict(schema='aic_sfp_actor_prefix_context_merge/v1',
                source_sha256={str(path):sha(path) for _,path in sources},
                counts={str(k):v for k,v in sorted(counts.items())},
                episode_counts={str(k):len(v) for k,v in sorted(episodes.items())},
                scene_counts={k:len(v) for k,v in scenes.items()},
                train_validation_scene_overlap=0,
                actor_prefix_used_as_supervision=False,
                final_scenes_opened=False)
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
