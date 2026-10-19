#!/usr/bin/env python3
"""Merge native and historical perception-only SFP rows with reset isolation."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--agent-easy',type=Path,required=True)
    p.add_argument('--agent-hard',type=Path,required=True)
    p.add_argument('--old-labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing output')
    native=[json.loads(s) for s in (a.native/'rows.jsonl').open()]
    easy=[json.loads(s) for s in (a.agent_easy/'rows.jsonl').open()]
    hard=[json.loads(s) for s in (a.agent_hard/'rows.jsonl').open()]
    for family,rows in [('easy',easy),('hard',hard)]:
        for r in rows:r['episode']=family+'_'+r['episode']
    old_val={json.loads(s)['scene_group'] for s in (a.old_labels/'rows.jsonl').open()
             if '"split":"validation"' in s or '"split": "validation"' in s}
    train={r['scene_group'] for r in native+easy+hard if r['split']=='train'}
    validation={r['scene_group'] for r in native if r['split']=='validation'}|old_val
    if train&validation:raise ValueError(f'Scene/reset leakage: {len(train&validation)} groups')
    seen={};duplicate=[]
    for source,rows in [('native',native),('easy',easy),('hard',hard)]:
        for r in rows:
            prior=seen.setdefault(r['scene_group'],source)
            if prior!=source:duplicate.append((r['scene_group'],prior,source))
    if duplicate:raise ValueError(f'Cross-source repeated reset groups: {duplicate[:3]}')
    a.output.mkdir(parents=True)
    with (a.output/'rows.jsonl').open('w') as out:
        for row in native+easy+hard:out.write(json.dumps(row,separators=(',',':'))+'\n')
    counts=Counter((r['image_kind'],r['split']) for r in native+easy+hard)
    summary={'schema':'aic_sfp_perception_merged/v1','rows':len(native+easy+hard),
             'episodes':len({r['episode'] for r in native+easy+hard}),
             'source_rows':{str(k):v for k,v in counts.items()},
             'train_scene_groups':len(train),'validation_scene_groups':len(validation),
             'cross_source_scene_overlap':False,'final_scenes_opened':False,
             'agent_actions_used':False,'literal_visibility_labels':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
