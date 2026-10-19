#!/usr/bin/env python3
"""Admit only scored successful, causal transport labels from paired controls."""
import argparse
from collections import Counter, defaultdict
import io
import json
from pathlib import Path

from materialize_sfp_scored_transport import materialize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',type=Path,required=True)
    p.add_argument('--collection',type=Path,action='append',required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episode-base',type=int,default=2100000,
                   help='First ID for the new collection; must not overlap old IDs')
    p.add_argument('--development-validation-first',type=int,default=0,
                   help='Hold out the first N planned new scene groups as development validation')
    p.add_argument('--min-admitted-per-collection',type=int,default=4,
                   help='Require this many successful causal admissions per collection')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous admission')
    old_rows=[json.loads(line) for line in a.old.open()]
    old_ids={r['episode_index'] for r in old_rows}
    val_scenes={r['scene_sha256'] for r in old_rows if r['split']=='validation'}
    train_scenes={r['scene_sha256'] for r in old_rows if r['split']=='train'}
    cal=json.loads(a.calibration.read_text())
    admitted=[];excluded=[];new_rows=[];seen=set()
    for batch,folder in enumerate(a.collection):
        plan=json.loads((folder/'plan.json').read_text())
        progress=json.loads((folder/'progress.json').read_text())['results']
        if plan['final_scenes_opened'] or plan['schema'] not in (
                'aic_sfp_divergent_pairs/v1', 'aic_sfp_divergent_pairs/v2'):
            raise ValueError('Expected nonfinal paired collection')
        if {x['name'] for x in plan['scenes']}!=progress.keys():
            raise ValueError(f'Incomplete collection: {folder}')
        for number,spec in enumerate(plan['scenes']):
            entry=progress[spec['name']]
            if entry['official_score'].get('tier3',0)<75:
                excluded.append(dict(name=spec['name'],reason='no_full_insertion',
                                     score=entry['official_score']))
                continue
            audit=Path(entry['audit'])
            if not (audit/'scored_stream.json').exists():
                raise FileNotFoundError(audit/'scored_stream.json')
            episode=a.episode_base+batch*100000+number
            if episode in old_ids or episode in seen:raise ValueError('Episode collision')
            buffer=io.StringIO()
            split=(spec['source_split'] if 'source_split' in spec else
                   'validation' if number<a.development_validation_first else 'train')
            if split not in ('train','validation'):
                raise ValueError('Invalid source split')
            try:report=materialize(audit.resolve(),cal,episode,buffer,split)
            except Exception as exc:
                excluded.append(dict(name=spec['name'],reason=type(exc).__name__+': '+str(exc)))
                continue
            rows=[json.loads(line) for line in buffer.getvalue().splitlines()]
            if not rows or {r['scene_sha256'] for r in rows}!={report['scene_group']}:
                raise ValueError('Inconsistent new scene group')
            if report['scene_group'] in val_scenes and split != 'validation':
                raise ValueError('Validation scene leakage')
            if report['scene_group'] in train_scenes and split != 'train':
                raise ValueError('Training scene leakage')
            if report['scene_group'] not in (val_scenes if split=='validation' else train_scenes):
                raise ValueError('Alternate route does not match a prior source scene')
            request=spec.get('alternate_request',spec.get('request'))
            if [report['card'],report['port']]!=request:
                raise ValueError('Request label mismatch')
            seen.add(episode);new_rows.extend(rows)
            admitted.append(dict(collection=str(folder),name=spec['name'],
                                 split=split,
                                 card_count=spec['card_count'],paired_with_prior_train_scene=(report['scene_group'] in train_scenes),
                                 **report))
    if len(admitted)<len(a.collection)*a.min_admitted_per_collection:
        raise ValueError('Too few admitted controls')
    a.output.mkdir(parents=True)
    with (a.output/'sfp_labels.jsonl').open('w') as f:
        for row in old_rows+new_rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
    counts=Counter((r['card_count'],r['card'],r['port']) for r in admitted)
    manifest=dict(schema='aic_sfp_paired_coverage_admission/v1',
        source_labels=str(a.old),collections=[str(x) for x in a.collection],
        old_rows=len(old_rows),new_rows=len(new_rows),old_episodes=len(old_ids),
        new_episodes=len(admitted),excluded=excluded,admitted=admitted,
        new_development_validation_episodes=sum(r['split']=='validation' for r in admitted),
        admitted_count_request={str(k):v for k,v in sorted(counts.items())},
        new_validation_scene_overlap=0,source_agent_actions_used=False,
        geometry_runtime_input=False,final_scenes_opened=False)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(new_episodes=len(admitted),new_rows=len(new_rows),
                          excluded=len(excluded),paired_with_prior_train=sum(r['paired_with_prior_train_scene'] for r in admitted),
                          counts=manifest['admitted_count_request'])),flush=True)


if __name__=='__main__':main()
