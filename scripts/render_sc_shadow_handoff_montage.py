#!/usr/bin/env python3
"""Mark the RGB-predicted requested opening on saved shadow-inspection views."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous montage')
    checks=[json.loads(s) for s in (a.audit/'runtime_trace.jsonl').open()
            if '"event": "shadow_handoff_inspection"' in s]
    frames=list(csv.DictReader((a.audit/'frame_manifest.csv').open()))
    strips=[]
    for check in checks:
        cells=[];t=check['sim_time']
        for camera in ('center','left','right'):
            row=min((r for r in frames if r['camera']==camera),
                    key=lambda r:abs(int(r['sim_ns'])/1e9-t))
            bgr=cv2.imread(str(a.audit/'frames'/row['file']))
            if bgr is None:raise FileNotFoundError(row['file'])
            bgr=cv2.resize(bgr,(576,512),interpolation=cv2.INTER_AREA)
            xy=check.get('refined_pixels',{}).get(camera)
            if xy:
                x,y=(int(round(v/2)) for v in xy)
                cv2.drawMarker(bgr,(x,y),(0,255,255),cv2.MARKER_CROSS,24,2)
                cv2.circle(bgr,(x,y),20,(0,255,255),2)
            label=f"{camera} | port {check['requested_port']} | {check['reason']}"
            cv2.rectangle(bgr,(0,0),(576,32),(0,0,0),-1)
            cv2.putText(bgr,label,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
            cells.append(bgr)
        strips.append(np.concatenate(cells,axis=1))
    if not strips:raise ValueError('No shadow inspections')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(a.output),np.concatenate(strips,axis=0),
                [cv2.IMWRITE_JPEG_QUALITY,91])
    print(json.dumps({'output':str(a.output),'inspections':len(checks),
                      'geometry_labels_used':False}))


if __name__=='__main__':main()
