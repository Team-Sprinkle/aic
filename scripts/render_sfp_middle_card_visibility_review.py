#!/usr/bin/env python3
"""Label-only review sheets for middle-card identifiability, never model input."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

CAMERAS=('center','left','right')


def review_crop(row,cam,size=256):
    image=cv2.imread(row['images'][cam])
    if image is None:raise FileNotFoundError(row['images'][cam])
    selected=next(t for t in row['targets'] if t['identity']==row['selected_identity'])
    uv=selected['cameras'][cam]
    if not uv[3]:
        crop=cv2.resize(image,(size,size),interpolation=cv2.INTER_AREA)
        cv2.putText(crop,f'{cam} t={row["time_s"]:.1f} OFFSCREEN',(6,size-12),
                    cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,0),2)
        return crop
    cx,cy=round(uv[0]),round(uv[1]);pad=size//2
    padded=cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=(0,0,0))
    crop=padded[cy:cy+size,cx:cx+size].copy()
    for t in row['targets']:
        point=t['cameras'][cam]
        if not point[3]:continue
        x=int(round(point[0]-cx+pad));y=int(round(point[1]-cy+pad))
        if not 0<=x<size or not 0<=y<size:continue
        if t['identity']==row['selected_identity']:
            cv2.drawMarker(crop,(x,y),(0,255,0),cv2.MARKER_CROSS,20,2)
        else:
            cv2.circle(crop,(x,y),5,(0,0,255),1)
    cv2.putText(crop,f'{cam} t={row["time_s"]:.1f}',(6,size-12),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,0),2)
    return crop


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior review')
    by=defaultdict(list)
    for line in (a.labels/'rows.jsonl').open():
        row=json.loads(line)
        if row['split']=='validation' and 1<=int(row['selected_identity'].split('_')[2])<=3:
            by[row['episode']].append(row)
    a.output.mkdir(parents=True)
    manifest=[]
    for episode,rows in sorted(by.items()):
        rows.sort(key=lambda x:x['time_s']);samples=[rows[0],rows[len(rows)//2],rows[-1]]
        sheet=np.concatenate([np.concatenate([review_crop(r,c) for c in CAMERAS],axis=1) for r in samples],axis=0)
        path=a.output/(episode+'.jpg');cv2.imwrite(str(path),sheet,[cv2.IMWRITE_JPEG_QUALITY,95])
        manifest.append({'episode':episode,'requested':rows[0]['selected_identity'],
                         'times_s':[r['time_s'] for r in samples],'review_sheet':str(path),
                         'green_cross':'true projected requested opening for human review only',
                         'red_circles':'other physically present projected openings for human review only',
                         'true_geometry_model_input':False})
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'sheets':len(manifest),'output':str(a.output)}))


if __name__=='__main__':main()
