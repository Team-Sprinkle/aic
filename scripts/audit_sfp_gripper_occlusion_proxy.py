#!/usr/bin/env python3
"""Audit RGB-only bottom-connected dark gripper occlusion of projected SFP ports.

This is a conservative image heuristic, not a physical visibility label.
Ground-truth projections and checkpoint errors are used only for posthoc audit.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

import cv2
import numpy as np


def bottom_dark_component(image,point):
    small=cv2.resize(image,(288,256),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    mask=(gray<40).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    px=int(round(point[0]*287/1151));py=int(round(point[1]*255/1023))
    if not 0<=px<288 or not 0<=py<256:return False
    label=int(labels[py,px])
    if label==0:return False
    x,y,w,h,area=stats[label]
    return bool(y+h>=255 and area>=288*256*.005)


def main():
    cv2.setNumThreads(1)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[json.loads(s) for s in (a.labels/'rows.jsonl').open()]
    pred={(q['row'],q['camera']):q for q in map(json.loads,a.predictions.open())}
    errors=defaultdict(list);by_row=defaultdict(Counter);examples=[]
    for i,r in enumerate(rows):
        selected=next(t for t in r['targets'] if t['identity']==r['selected_identity'])
        for cam in ('center','left','right'):
            truth=selected['cameras'][cam]
            if not truth[3]:continue
            image=cv2.imread(r['images'][cam],cv2.IMREAD_COLOR)
            if image is None:raise FileNotFoundError(r['images'][cam])
            hidden=bottom_dark_component(image,truth[:2])
            group='likely_gripper_occluded' if hidden else 'not_gripper_occluded'
            by_row[(i,r['split'])][group]+=1
            if (i,cam) in pred:
                xy=pred[(i,cam)]['predicted'][r['selected_identity']]['xy']
                error=float(np.linalg.norm(np.asarray(xy)-truth[:2]))
                errors[(r['split'],group)].append(error)
                errors[(r['split'],r['selected_identity'],group)].append(error)
            if len(examples)<20 and hidden:
                examples.append({'episode':r['episode'],'camera':cam,'image':r['images'][cam],
                                 'projected_xy':truth[:2]})
    summary={'schema':'aic_sfp_gripper_occlusion_proxy_audit/v1',
             'classification':'RGB dark connected component touching image bottom; not certified visibility',
             'groups':{str(k):{'n':len(v),'median_px':float(np.median(v)),
                                'p95_px':float(np.quantile(v,.95))} for k,v in errors.items() if v},
             'rows_with_two_non_gripper_views':sum(v['not_gripper_occluded']>=2 for v in by_row.values()),
             'rows_with_one_non_gripper_view':sum(v['not_gripper_occluded']==1 for v in by_row.values()),
             'rows_with_zero_non_gripper_views':sum(v['not_gripper_occluded']==0 for v in by_row.values()),
             'rows_total':len(by_row),'examples':examples,'geometry_runtime':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'rows':len(by_row),'views':{str(k):len(v) for k,v in errors.items() if len(k)==2}}))


if __name__=='__main__':main()
