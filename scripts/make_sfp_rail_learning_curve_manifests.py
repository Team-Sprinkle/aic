#!/usr/bin/env python3
"""Deterministic nested complete-group 20/40/70 SFP rail split."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior output')
    rows=[json.loads(line) for line in a.rows.open()]
    groups={}
    for r in rows:
        if r['split']!='train':continue
        key=r['scene_group'];value=(r['selected_identity'],r['card_count'])
        if key in groups and groups[key]!=value:raise ValueError('Inconsistent group')
        groups[key]=value
    if len(groups)!=70:raise ValueError('Expected 70 train groups')
    by_request=defaultdict(list)
    for group,(request,count) in groups.items():
        by_request[request].append((hashlib.sha256(group.encode()).hexdigest(),group))
    if len(by_request)!=10 or min(map(len,by_request.values()))<4:
        raise ValueError('Cannot balance 20/40 groups over ten requests')
    for request in by_request:by_request[request].sort()
    selected={20:sorted(g for values in by_request.values() for _,g in values[:2]),
              40:sorted(g for values in by_request.values() for _,g in values[:4]),
              70:sorted(groups)}
    if not set(selected[20]).issubset(selected[40]) or not set(selected[40]).issubset(selected[70]):
        raise ValueError('Subsets not nested')
    a.output.mkdir(parents=True)
    for n,ids in selected.items():
        payload={'schema':'sfp_rail_learning_curve_groups/v1','groups':n,
                 'selection':'sha256-sorted first 2/4 per requested identity; 70 all',
                 'scene_groups':ids,
                 'request_counts':{name:sum(groups[g][0]==name for g in ids)
                                   for name in sorted(by_request)}}
        (a.output/f'train_groups_{n}.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({n:len(v) for n,v in selected.items()}))


if __name__=='__main__':main()
