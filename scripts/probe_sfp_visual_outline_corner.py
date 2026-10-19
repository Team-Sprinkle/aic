#!/usr/bin/env python3
"""Privileged-window diagnostic of dark-board silhouette corners in native RGB.

The CAD projection centers the inspection window, so results are never a
deployable crop selector. This tests whether a visually distinct replacement
landmark exists near the inaccurate collision-envelope corner label.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous output')
    a.output.mkdir(parents=True)
    panels=json.loads(a.manifest.read_text())
    report=[]
    canvas=np.zeros((6*260,5*260,3),np.uint8)
    for item in panels:
        frame=cv2.imread(item['image'])
        x,y=item['projected_pixel_label_only']
        crop=cv2.getRectSubPix(frame,(220,220),(float(x),float(y)))
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        mask=np.uint8(gray<130)*255
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        vertices=[]
        for contour in contours:
            if cv2.contourArea(contour)<1000:continue
            poly=cv2.approxPolyDP(contour,5,True)
            for v in poly[:,0,:]:
                if np.linalg.norm(v-np.array([110,110]))<=75:
                    vertices.append(v)
        nearest=min(vertices,key=lambda v:np.linalg.norm(v-np.array([110,110]))) if vertices else None
        draw=crop.copy()
        for vertex in vertices:cv2.circle(draw,tuple(vertex),3,(0,255,255),1)
        cv2.drawMarker(draw,(110,110),(0,255,0),cv2.MARKER_CROSS,18,2)
        if nearest is not None:cv2.drawMarker(draw,tuple(nearest),(0,0,255),cv2.MARKER_CROSS,18,2)
        panel=item['panel'];tile=np.zeros((260,260,3),np.uint8)
        tile[35:255,20:240]=draw
        cv2.putText(tile,f'#{panel} {item["camera"]} corner{item["endpoint"]}',
                    (8,24),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
        canvas[panel//5*260:(panel//5+1)*260,panel%5*260:(panel%5+1)*260]=tile
        report.append({'panel':panel,'candidate_pixel':
                       (np.asarray(nearest,dtype=float)+[x-110,y-110]).tolist()
                       if nearest is not None else None,
                       'cad_to_candidate_px':float(np.linalg.norm(nearest-[110,110]))
                       if nearest is not None else None,'vertices':len(vertices)})
    cv2.imwrite(str(a.output/'contour_candidates.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,94])
    (a.output/'candidate_pixels.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'panels':len(report),'with_candidate':sum(r['candidate_pixel'] is not None for r in report)}))


if __name__=='__main__':main()
