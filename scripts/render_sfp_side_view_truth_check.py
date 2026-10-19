#!/usr/bin/env python3
"""Posthoc marker copies for checking interpretation; never model input."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np


CAMERAS=('center','left','right')
STAGES=('baseline_view','positive_view','negative_view')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--labels',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('Preserve previous marker images')
    rows=defaultdict(dict)
    for s in (args.labels/'rows.jsonl').open():
        r=json.loads(s);rows[r['episode']][r['stage']]=r
    args.output.mkdir(parents=True)
    for episode,stages in sorted(rows.items()):
        strips=[]
        for stage in STAGES:
            r=stages[stage];tiles=[]
            selected=next(t for t in r['targets'] if t['identity']==r['selected_identity'])
            card=int(r['selected_identity'].split('_')[2]);port=int(r['selected_identity'].split('_')[-1])
            other=next(t for t in r['targets'] if t['identity']==f'sfp_card_{card}_port_{1-port}')
            for camera in CAMERAS:
                img=cv2.imread(r['images'][camera],cv2.IMREAD_COLOR)
                if img is None:raise FileNotFoundError(r['images'][camera])
                for target,color,text in ((other,(255,255,0),'other'),
                                          (selected,(255,0,255),'requested')):
                    u,v,_,inframe=target['cameras'][camera]
                    if not inframe:continue
                    point=(round(u),round(v))
                    cv2.circle(img,point,12,color,2,cv2.LINE_AA)
                    cv2.drawMarker(img,point,color,cv2.MARKER_CROSS,24,2,cv2.LINE_AA)
                    cv2.putText(img,text,(point[0]+14,point[1]-6),
                                cv2.FONT_HERSHEY_SIMPLEX,.8,color,2,cv2.LINE_AA)
                tile=cv2.resize(img,(384,341),interpolation=cv2.INTER_AREA)
                cv2.putText(tile,f'POSTHOC LABEL ONLY / {stage} / {camera}',
                            (4,20),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,255),1,cv2.LINE_AA)
                tiles.append(tile)
            strips.append(np.concatenate(tiles,axis=1))
        path=args.output/f'{episode}_posthoc_truth_check.jpg'
        cv2.imwrite(str(path),np.concatenate(strips,axis=0),[cv2.IMWRITE_JPEG_QUALITY,95])
    print(json.dumps(dict(episodes=len(rows),output=str(args.output),
                          posthoc_only=True,model_input=False)))


if __name__=='__main__':main()
