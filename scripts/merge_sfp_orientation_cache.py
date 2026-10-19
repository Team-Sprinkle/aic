#!/usr/bin/env python3
"""Concatenate frozen RGB features and candidate predictions for appended labels."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-features',type=Path,required=True)
    p.add_argument('--new-features',type=Path,required=True)
    p.add_argument('--old-predictions',type=Path,required=True)
    p.add_argument('--new-predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing output')
    old=np.load(a.old_features,mmap_mode='r');new=np.load(a.new_features,mmap_mode='r')
    if old.ndim!=3 or old.shape[1:]!=(3,768) or new.ndim!=3 or new.shape[1:]!=(3,768):
        raise ValueError('Expected three frozen 768D camera features per row')
    a.output.mkdir(parents=True)
    merged=np.lib.format.open_memmap(a.output/'features.npy',mode='w+',dtype=np.float16,
                                      shape=(len(old)+len(new),3,768))
    merged[:len(old)]=old;merged[len(old):]=new;merged.flush()
    counts=[]
    with (a.output/'predictions.jsonl').open('w') as out:
        for path,offset,expected in ((a.old_predictions,0,len(old)),
                                     (a.new_predictions,len(old),len(new))):
            count=0
            for line in path.open():
                item=json.loads(line)
                if not 0<=item['row']<expected:raise ValueError('Prediction row outside source labels')
                item['row']+=offset
                out.write(json.dumps(item,separators=(',',':'))+'\n');count+=1
            if count!=expected*3:raise ValueError('Expected one prediction for each camera and row')
            counts.append(count)
    summary={'schema':'aic_sfp_orientation_cache_merge/v1','old_rows':len(old),
             'new_rows':len(new),'prediction_views':counts,'frozen_features':True,
             'reserved_final_opened':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
