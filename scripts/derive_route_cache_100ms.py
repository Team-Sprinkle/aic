#!/usr/bin/env python3
"""Make an immutable 100 ms action cache at a chosen decision cadence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--extended-labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--decision-stride',type=int,default=2,
                   help='Rows of the 300 ms source cache per decision (2=600 ms, 3=900 ms)')
    a=p.parse_args()
    if a.decision_stride not in (2,3):p.error('Expected decision stride 2 or 3')
    if a.output.exists():p.error('Output exists; source and derivative are immutable')
    source_manifest=json.loads((a.cache/'manifest.json').read_text())
    label_manifest=json.loads((a.extended_labels/'manifest.json').read_text())
    if label_manifest['slot_s']!=.1 or label_manifest['rows']!=source_manifest['rows']:
        p.error('Expected aligned 100 ms extended labels')
    rows=[json.loads(x) for x in (a.cache/'rows.jsonl').open()]
    targets=np.load(a.extended_labels/'targets12.npy',mmap_mode='r')
    valid=np.load(a.extended_labels/'valid12.npy',mmap_mode='r')
    feature=np.load(a.cache/'features.npy',mmap_mode='r')
    camera=np.load(a.cache/'camera_offsets.npy',mmap_mode='r')
    selected=[]; last=None;position=0
    for i,row in enumerate(rows):
        key=(row['family'],row['episode'])
        if key!=last:position=0;last=key
        if position%a.decision_stride==0:selected.append(i)
        position+=1
    a.output.mkdir(parents=True)
    counts=Counter();kept=[]
    with (a.output/'rows.jsonl').open('w') as out:
        for i in selected:
            row=rows[i].copy();row['target']=np.asarray(targets[i,:4]).tolist()
            row['action_slot_s']=.1
            out.write(json.dumps(row,separators=(',',':'))+'\n')
            kept.append(row)
            counts[(row['split'],row['family'])]+=1
    np.save(a.output/'features.npy',np.asarray(feature[selected]))
    np.save(a.output/'camera_offsets.npy',np.asarray(camera[selected]))
    np.save(a.output/'targets12.npy',np.asarray(targets[selected]))
    np.save(a.output/'valid12.npy',np.asarray(valid[selected]))
    manifest=dict(schema='aic_route_100ms_cache_derivative/v1',rows=len(selected),
                  source_rows=len(rows),slot_s=.1,action_slot_s=.1,
                  nominal_decision_s=.3*a.decision_stride,decision_stride=a.decision_stride,
                  source_cache=str(a.cache.resolve()),extended_labels=str(a.extended_labels.resolve()),
                  source_sha256={str(path):digest(path) for path in
                                 (a.cache/'manifest.json',a.extended_labels/'manifest.json')},
                  counts={str(k):v for k,v in sorted(counts.items())},
                  valid12_counts={str(k):sum(bool(valid[i,11]) for i in selected
                                               if (rows[i]['split'],rows[i]['family'])==k)
                                  for k in counts},
                  input_images_unchanged=True,source_episodes_mutated=False,
                  final_scenes_opened=False)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
