#!/usr/bin/env python3
"""Append new grouped native RGB rows to the existing perception-only corpus."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--existing',type=Path,required=True)
    p.add_argument('--new',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior dataset')
    old=[json.loads(s) for s in (a.existing/'rows.jsonl').open()]
    fresh=[json.loads(s) for s in (a.new/'rows.jsonl').open()]
    groups=Counter((r['scene_group'],r['split']) for r in old+fresh)
    split={}
    for scene,part in groups:
        if scene in split and split[scene]!=part:raise ValueError('Scene/reset split leakage')
        split[scene]=part
    old_scenes={r['scene_group'] for r in old}
    existing_train_overlap=old_scenes & {r['scene_group'] for r in fresh}
    if any(r['split']!='train' for r in fresh):
        raise ValueError('First-ten derivative must contain training scenes only')
    a.output.mkdir(parents=True)
    with (a.output/'rows.jsonl').open('w') as out:
        for r in old+fresh:out.write(json.dumps(r,separators=(',',':'))+'\n')
    summary={'schema':'sfp_native_confusion_merged_labels/v1','existing_rows':len(old),
             'new_rows':len(fresh),'new_episodes':len({r['episode'] for r in fresh}),
             'new_scene_groups':len({r['scene_group'] for r in fresh}),
             'existing_train_scene_overlap':len(existing_train_overlap),
             'total_rows':len(old)+len(fresh),'split_scene_overlap':False,
             'source_actions_used':False,'reserved_final_opened':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
