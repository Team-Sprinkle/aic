#!/usr/bin/env python3
"""Conservative RGB mask for projected SFP points hidden by gripper housing.

This is a specific visual obstruction test, not a general visibility label.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from train_requested_history_crop import CAMERAS, rows_from


def black_housing_component(image):
    # Original capture is 1152x1024; half-scale preserves the large housing.
    small=cv2.resize(image,(576,512),interpolation=cv2.INTER_AREA)
    dark=(np.max(small,axis=2)<50).astype('uint8')
    _,component,stats,_=cv2.connectedComponentsWithStats(dark)
    bottom=set(int(v) for v in np.unique(component[505:,:]) if v)
    return component,{v for v in bottom if stats[v,cv2.CC_STAT_AREA]>5000}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--labels',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():ap.error('Preserve prior output')
    rows=rows_from(a.labels)
    a.output.mkdir(parents=True)
    selected=Counter();all_count=Counter();review=[]
    with (a.output/'rows.jsonl').open('w') as f:
        for r in rows:
            components={}
            for cam in CAMERAS:
                image=cv2.imread(r['images'][cam],cv2.IMREAD_COLOR)
                if image is None:raise FileNotFoundError(r['images'][cam])
                components[cam]=black_housing_component(image)
            for t in r['targets']:
                flags={}
                for cam in CAMERAS:
                    u,v,_,inside=t['cameras'][cam]
                    hit=False
                    if inside:
                        component,bottom=components[cam]
                        label=int(component[min(511,max(0,int(v/2))),min(575,max(0,int(u/2)))])
                        hit=label in bottom
                    flags[cam]=hit
                    all_count[(r['split'],cam)]+=hit
                    if t['identity']==r['selected_identity']:
                        selected[(r['split'],cam)]+=hit
                        if hit and len(review)<30:
                            review.append({'episode':r['episode'],'time_s':r['time_s'],
                                           'camera':cam,'image':r['images'][cam],
                                           'point_px':t['cameras'][cam][:2]})
                t['housing_occluded_rgb_heuristic']=flags
            r['housing_occlusion_is_not_full_visibility']=True
            f.write(json.dumps(r,separators=(',',':'))+'\n')
    summary={'schema':'sfp_rgb_housing_occlusion_audit/v1','source_labels':str(a.labels),
             'rows':len(rows),'selected_flags':{f'{s}/{c}':n for (s,c),n in selected.items()},
             'all_target_flags':{f'{s}/{c}':n for (s,c),n in all_count.items()},
             'review_examples':review,'not_general_visibility':True,
             'simulator_geometry_only_for_offline_label_position':True,'final_scenes_opened':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='review_examples'}))

if __name__=='__main__':main()
