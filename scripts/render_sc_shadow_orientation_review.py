#!/usr/bin/env python3
"""Save original-RGB locator-centered inspection crops for human review."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


CAMERAS=('center','left','right')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous review')
    trace=[json.loads(s) for s in (a.audit/'runtime_trace.jsonl').open()]
    inspections=[r for r in trace if r.get('event')=='shadow_handoff_inspection']
    if not inspections:raise ValueError('No inspections')
    frames={}
    for row in csv.DictReader((a.audit/'frame_manifest.csv').open()):
        if row['camera'] in CAMERAS:
            frames.setdefault(int(row['sim_ns']),{})[row['camera']]=a.audit/'frames'/row['file']
    a.output.mkdir(parents=True)
    panels=[]
    manifest=[]
    for j,inspection in enumerate(inspections):
        stamp=min((s for s,v in frames.items() if len(v)==3),
                  key=lambda s:abs(s-inspection['sim_time']*1e9))
        cams=[]
        for i,cam in enumerate(CAMERAS):
            image=cv2.imread(str(frames[stamp][cam]))
            cx,cy=np.rint(inspection['coarse_pixels'][i]).astype(int)
            padded=cv2.copyMakeBorder(image,80,80,80,80,cv2.BORDER_CONSTANT,
                                      value=(0,0,0))
            crop=padded[cy:cy+160,cx:cx+160]
            cv2.putText(crop,cam,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.5,
                        (255,255,255),1)
            cams.append(crop)
        panel=np.hstack(cams)
        cv2.putText(panel,f"{j+1} {inspection['sim_time']:.2f}s", (5,155),
                    cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        cv2.imwrite(str(a.output/f'inspection_{j+1:02d}.jpg'),panel)
        panels.append(panel)
        manifest.append({'inspection':j+1,'sim_time_s':inspection['sim_time'],
                         'image_sim_ns':stamp,'crop_centers_from_rgb_locator':True,
                         'true_geometry_used':False,
                         'reason':inspection['reason'],
                         'posture_error_deg':inspection.get('measured_tcp_posture_error_deg')})
    cv2.imwrite(str(a.output/'all_inspections.jpg'),np.vstack(panels))
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'inspections':len(panels),'output':str(a.output)}))
if __name__=='__main__':main()
