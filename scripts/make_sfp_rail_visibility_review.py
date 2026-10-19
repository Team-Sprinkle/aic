#!/usr/bin/env python3
"""Fixed single-reviewer SFP endpoint packet without model predictions."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


CAMS=('center','left','right')
SCALE=np.asarray([1151.,1023.])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate-kind',choices=('rail','outline'),default='rail')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior output')
    a.output.mkdir(parents=True)
    rows=[json.loads(line) for line in a.rows.open()]
    preds=[json.loads(line) for line in a.predictions.open()]
    by_key=defaultdict(lambda:defaultdict(list))
    for pred in preds:
        row=rows[pred['row']]
        if row['split']!='validation' or row['scene_group']!=pred['scene_group']:
            raise ValueError('Development join mismatch')
        for ci,cam in enumerate(CAMS):
            for end,slot in enumerate((0,4)):
                key=('rail_projections_training_label_only' if a.candidate_kind=='rail'
                     else 'corner_projections_training_label_only')
                label=row[key][cam][slot if a.candidate_kind=='rail' else end]
                if not label[3]:continue
                if a.candidate_kind=='rail' and pred['endpoint_inframe_prob'][ci][end]<.5:continue
                center=np.asarray(pred['endpoint_xy_normalized'][ci][end])*SCALE
                if a.candidate_kind=='rail' and np.any(np.abs(center-label[:2])>79.5):continue
                if a.candidate_kind=='outline':center=np.asarray(label[:2])
                by_key[(cam,end)][row['scene_group']].append((row,pred,label,center))
    selected=[]
    for cam in CAMS:
        for end in (0,1):
            groups=sorted(by_key[(cam,end)],key=lambda s:hashlib.sha256(s.encode()).hexdigest())
            if len(groups)<5:raise ValueError(f'Under five groups: {cam}/{end}')
            for j,group in enumerate(groups[:5]):
                options=sorted(by_key[(cam,end)][group],key=lambda x:x[0]['time_s'])
                row,pred,label,center=options[0 if j%2==0 else -1]
                selected.append((cam,end,row,pred,label,center))
    manifest=[]
    for sheet in range(2):
        canvas=np.zeros((3*400,5*580,3),np.uint8)
        for local,item in enumerate(selected[sheet*15:(sheet+1)*15]):
            cam,end,row,pred,label,center=item
            source=cv2.imread(row['images'][cam]);truth=np.asarray(label[:2])
            full=cv2.resize(source,(384,341),interpolation=cv2.INTER_AREA)
            cv2.circle(full,tuple(np.rint(truth/np.array([3.,3.])).astype(int)),5,(0,255,0),2)
            crop=cv2.getRectSubPix(source,(160,160),tuple(float(x) for x in center))
            local_truth=np.rint(truth-center+79.5).astype(int)
            cv2.circle(crop,tuple(local_truth),5,(0,255,0),2)
            tile=np.zeros((400,580,3),np.uint8)
            tile[35:376,5:389]=full
            tile[35:195,400:560]=crop
            cv2.putText(tile,f'#{sheet*15+local:02d} {cam} {a.candidate_kind}{end} {row["scene_group"][:7]}',
                        (7,23),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
            cv2.putText(tile,f't={row["time_s"]:.2f}s',(402,220),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
            canvas[(local//5)*400:(local//5+1)*400,(local%5)*580:(local%5+1)*580]=tile
            manifest.append({'panel':sheet*15+local,'camera':cam,'endpoint':end,
                             'candidate_kind':a.candidate_kind,
                             'scene_group':row['scene_group'],'episode':row['episode'],
                             'time_s':row['time_s'],'image':row['images'][cam],
                             'projected_pixel_label_only':label[:2],
                             'review_label':None})
        cv2.imwrite(str(a.output/f'contact_sheet_{sheet+1}.jpg'),canvas,
                    [cv2.IMWRITE_JPEG_QUALITY,93])
    (a.output/'review_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (a.output/'rubric.md').write_text(
        f'# Single-reviewer {a.candidate_kind} visibility rubric\n\n'
        'For each numbered panel, classify the green projected point as '
        '`unique_visible`, `ambiguous_or_occluded`, or `uncertain`. '
        'Unique visible requires a distinct physical corner/mark at that exact point '
        'and enough full-image context to link it to the requested rail end. '
        'A repeated card/rail edge or blank projected surface is ambiguous. '
        'This review cannot certify absolute numbered-port identity.\n')
    print(json.dumps({'panels':len(manifest),'groups':len(set(x['scene_group'] for x in manifest))}))


if __name__=='__main__':main()
