#!/usr/bin/env python3
"""Admit post-takeover full-insertion labels and make a grouped BC comparison.

Four training corrections cover 2/3/5 cards; one two-card and one five-card
correction are withheld. Ordinary base rows on the withheld reset scenes are
removed from optimization to avoid a new train/validation scene overlap.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from materialize_sfp_scored_transport import materialize


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gate',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    gate=json.loads((a.gate/'summary_reconciled.json').read_text())
    if (not gate['gate_passed'] or gate['completed']!=6 or gate['safe_full_insertions']!=6 or
        gate['final_scenes_opened']):
        p.error('Six safe, officially scored training-reset continuations required')
    runs=gate['runs']
    if [r['card_count'] for r in runs]!=[2,2,3,3,5,5]:
        p.error('Unexpected frozen card-count ordering')
    holdout={1,5}
    holdout_scenes={runs[i]['scene_digest'] for i in holdout}
    if a.output.exists():p.error('Preserve prior labels')
    a.output.mkdir(parents=True)
    calibration=json.loads(a.calibration.read_text())
    reports=[]
    corrections_path=a.output/'correction_labels.jsonl'
    with corrections_path.open('w') as stream:
        for i,r in enumerate(runs):
            audit=Path(r['audit'])
            if not (audit/'scored_stream.json').exists():
                raise FileNotFoundError(f'Extract causal scored stream first: {audit}')
            report=materialize(audit,calibration,2600000+i,stream,
                               'validation' if i in holdout else 'train',
                               corrective=True,full_insert_corrective=True)
            if report['scene_group']!=r['scene_digest']:
                raise ValueError(f'Reset digest mismatch: {audit}')
            reports.append(report)
    removed=Counter();groups=defaultdict(set);keys=set();counts=Counter()
    merged=a.output/'sfp_labels.jsonl'
    with merged.open('w') as out:
        for source,path in (('ordinary',a.base),('later_correction',corrections_path)):
            for line in path.open():
                row=json.loads(line)
                split=row['split'];scene=row['scene_sha256']
                if source=='ordinary' and split=='train' and scene in holdout_scenes:
                    removed[scene]+=1
                    continue
                if source=='later_correction' and (not row.get('privileged_teacher_after_takeover') or
                    row.get('actor_prefix_used_as_supervision') is not False or
                    row.get('official_tier3')!=75):
                    raise ValueError('Invalid later-state correction provenance')
                key=(row['episode_index'],row['frame_index'])
                if key in keys:raise ValueError(f'Duplicate row {key}')
                keys.add(key)
                groups[split].add(scene);counts[(source,split)]+=1
                out.write(line)
    if groups['train'] & groups['validation']:
        raise ValueError('Merged train/validation reset-group overlap')
    manifest=dict(schema='aic_sfp_later_correction_labels/v1',
                  gate=str(a.gate/'summary_reconciled.json'),
                  gate_sha256=digest(a.gate/'summary_reconciled.json'),
                  base=str(a.base),base_sha256=digest(a.base),
                  calibration=str(a.calibration),calibration_sha256=digest(a.calibration),
                  correction_reports=reports,
                  correction_train_episode_indices=[2600000+i for i in range(6) if i not in holdout],
                  correction_validation_episode_indices=[2600000+i for i in sorted(holdout)],
                  withheld_scene_digests=sorted(holdout_scenes),
                  ordinary_train_rows_removed_by_scene=dict(removed),
                  rows_by_source_split={str(k):v for k,v in sorted(counts.items())},
                  train_validation_scene_overlap=0,
                  prior_checkpoint_may_have_seen_original_teacher_episodes_on_withheld_scenes=True,
                  actor_prefix_used_as_action_supervision=False,
                  privileged_geometry_actor_input=False,final_scenes_opened=False)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(correction_train_episodes=4,correction_validation_episodes=2,
                          correction_chunks=sum(r['transport_chunks'] for r in reports),
                          removed_base_train_rows=sum(removed.values()),
                          train_validation_scene_overlap=0)),flush=True)


if __name__=='__main__':main()
