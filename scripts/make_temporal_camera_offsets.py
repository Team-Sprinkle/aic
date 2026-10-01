#!/usr/bin/env python3
"""Attach SC per-camera capture ages to the frozen feature cache.

SFP cache rows lack original individual camera timestamps; their offset-known
bit is zero. This metadata contains observation timing only, no geometry.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--sc-rows',type=Path,required=True)
    a=p.parse_args()
    output=a.cache/'camera_offsets.npy'
    if output.exists():p.error('Camera timing derivative already exists')
    rows=[json.loads(line) for line in (a.cache/'rows.jsonl').open()]
    selected={(r['episode'],r['frame']) for r in rows if r['family']=='sc'}
    clocks={}
    with a.sc_rows.open() as f:
        for line in f:
            r=json.loads(line);key=(int(r['episode_index']),int(r['frame']))
            if key in selected:
                clocks[key]=[float(r['sim_time'])-float(r['image_sim_ns'][c])*1e-9
                             for c in ('center','left','right')]
    assert len(clocks)==len(selected)
    offsets=np.zeros((len(rows),4),dtype=np.float32)
    for i,r in enumerate(rows):
        if r['family']=='sc':
            offsets[i,:3]=np.maximum(0.,clocks[(r['episode'],r['frame'])])
            offsets[i,3]=1
    assert np.isfinite(offsets).all() and (offsets[:,:3]>=0).all()
    assert offsets[:,:3].max()<.05
    np.save(output,offsets)
    print(json.dumps(dict(rows=len(rows),sc_known=int(offsets[:,3].sum()),
                          min_signed_camera_age_ms=float(offsets[:,:3].min()*1000),
                          max_signed_camera_age_ms=float(offsets[:,:3].max()*1000),
                          output=str(output))))


if __name__=='__main__':main()
