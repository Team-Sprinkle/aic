#!/usr/bin/env python3
"""Freeze disjoint new SFP crop training, development and stress scene groups."""
import argparse
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in (path/'rows.jsonl').open()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rail',type=Path,required=True)
    p.add_argument('--ports',type=Path,required=True)
    p.add_argument('--reserved',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing group split')
    rail=read(a.rail);ports=read(a.ports)
    reserved={r['scene_group'] for r in read(a.reserved)}
    groups={r['scene_group']:(r['split'],int(r['card_count'])) for r in rail}
    if len(groups)!=136 or len(reserved)!=18 or any(groups[g][0]!='train' for g in reserved):
        raise ValueError('Unexpected diverse or reserved scene groups')
    dense=sorted(g for g,(split,n) in groups.items() if split=='train' and n>=3 and g not in reserved)
    chosen=[]
    for count in (3,4,5):
        candidates=[g for g in dense if groups[g][1]==count]
        chosen.extend(candidates[:(4 if count==3 else 3)])
    if len(chosen)!=10:raise ValueError('Insufficient dense crop-development groups')
    selected=set(chosen)
    split={g:('reserved_stress' if g in reserved else
              'validation' if original=='validation' or g in selected else 'train')
           for g,(original,_) in groups.items()}
    if any(sum(split[g]==s for g in groups)!=n for s,n in
           [('train',91),('validation',27),('reserved_stress',18)]):
        raise ValueError('Unexpected group count')
    a.output.mkdir(parents=True)
    for source,name in ((rail,'rail'),(ports,'ports')):
        folder=a.output/name;folder.mkdir()
        with (folder/'rows.jsonl').open('w') as f:
            for row in source:
                role=split[row['scene_group']]
                if role=='reserved_stress':continue
                row=dict(row,split=role)
                f.write(json.dumps(row,separators=(',',':'))+'\n')
    (a.output/'split.json').write_text(json.dumps(dict(
        schema='aic_sfp_diverse_crop_split/v1',train_groups=sorted(g for g in groups if split[g]=='train'),
        development_groups=sorted(g for g in groups if split[g]=='validation'),
        reserved_stress_groups=sorted(reserved),original_sparse_validation_groups=17,
        new_dense_development_groups=10,final_scenes_opened=False),indent=2)+'\n')
    print(json.dumps(dict(train_groups=91,development_groups=27,reserved_stress_groups=18)))


if __name__=='__main__':main()
