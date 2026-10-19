#!/usr/bin/env python3
"""Recorded RGB views at close update, first entry and terminal; no geometry overlays."""
import cv2
cv2.setNumThreads(2)
import argparse
import csv
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('audit',type=Path)
a=p.parse_args()
rows=list(csv.DictReader((a.audit/'frame_manifest.csv').open()))
events=[json.loads(line) for line in (a.audit/'runtime_trace.jsonl').open()]
accepted=[r for r in events if r['event']=='close_grasp_update_accepted']
terminal=[r for r in events if r['event']=='terminal_before_return'][-1]
if not accepted:raise ValueError('No accepted close-up estimate')
start=accepted[0]['sim_time']
servo=[r for r in events if r['event']=='visual_servo_step' and r.get('sim_time',-1)>=start]
# Public estimated axial coordinate may be represented differently across versions.
entry=next((r for r in servo if float(r.get('estimated_tip_port_m',[0,0,-1])[2])>=0),None)
times=[('accepted close update',start),('insertion midway',(start+terminal['sim_time'])/2),('terminal',terminal['sim_time'])]
sheets=[]
manifest=[]
for name,stamp in times:
    panels=[]
    for camera in ('center','left','right'):
        choices=[r for r in rows if r['camera']==camera and float(r['sim_ns'])/1e9<=terminal['sim_time']]
        row=min(choices,key=lambda r:abs(float(r['sim_ns'])/1e9-stamp))
        image=cv2.imread(str(a.audit/'frames'/row['file']))
        scale=min(480/image.shape[1],360/image.shape[0])
        image=cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)))
        panel=np.zeros((400,480,3),dtype=np.uint8)
        y=40+(360-image.shape[0])//2;x=(480-image.shape[1])//2
        panel[y:y+image.shape[0],x:x+image.shape[1]]=image
        cv2.putText(panel,f'{name}: {camera} {float(row["sim_ns"])/1e9:.2f}s',(8,24),0,.45,(0,255,0),1)
        panels.append(panel)
        manifest.append(dict(stage=name,camera=camera,requested_s=stamp,recorded_ns=int(row['sim_ns']),source=row['file']))
    sheets.append(np.hstack(panels))
path=a.audit/'close_update_rgb_review.jpg'
cv2.imwrite(str(path),np.vstack(sheets),[cv2.IMWRITE_JPEG_QUALITY,94])
(a.audit/'close_update_rgb_review.json').write_text(json.dumps(dict(recorded_RGB_only=True,ground_truth_overlays=False,frames=manifest),indent=2)+'\n')
print(path)
