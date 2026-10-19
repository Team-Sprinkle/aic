#!/usr/bin/env python3
"""Remove reserved stress groups from board predictions before crop fitting."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split-rail',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing filtered predictions')
    allowed={r['scene_group'] for line in (a.split_rail/'rows.jsonl').open()
             if (r:=json.loads(line))}
    kept=0;seen=set()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w') as out:
        for line in a.predictions.open():
            row=json.loads(line)
            if row['scene_group'] in allowed:
                out.write(line);kept+=1;seen.add(row['scene_group'])
    if seen!=allowed:
        a.output.unlink()
        raise ValueError(f'Missing {len(allowed-seen)} group predictions')
    print(json.dumps(dict(rows=kept,groups=len(seen),reserved_groups_excluded=True)))


if __name__=='__main__':main()
