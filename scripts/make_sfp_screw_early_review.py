#!/usr/bin/env python3
"""Fixed early native-RGB review for channel-specific SFP mesh screws."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np


CHANNELS=(('center',0),('left',0),('center',1),('right',1))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screw-projections',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous output')
    a.output.mkdir(parents=True)
    rows=[json.loads(line) for line in a.screw_projections.open()]
    by_scene=defaultdict(list)
    for r in rows:by_scene[r['scene_group']].append(r)
    if len(by_scene)!=10:raise ValueError('Expected ten complete development groups')
    entries=[]
    for scene in sorted(by_scene):
        rr=sorted(by_scene[scene],key=lambda r:r['time_s'])
        early=[r for r in rr if r['time_s']-rr[0]['time_s']<=3]
        for cam,end in CHANNELS:
            candidates=[r for r in early if r['screw_projections_training_label_only'][cam][end][3]]
            r=candidates[0] if candidates else None
            entries.append({'scene_group':scene,'episode':rr[0]['episode'],
                            'camera':cam,'screw':end,'time_s':r['time_s'] if r else None,
                            'image':r['images'][cam] if r else None,
                            'projection':r['screw_projections_training_label_only'][cam][end] if r else None,
                            'single_reviewer_label':None})
    for sheet in range(2):
        canvas=np.zeros((4*400,5*580,3),np.uint8)
        for local,item in enumerate(entries[sheet*20:(sheet+1)*20]):
            tile=np.zeros((400,580,3),np.uint8)
            if item['image']:
                source=cv2.imread(item['image']);xy=np.asarray(item['projection'][:2])
                full=cv2.resize(source,(384,341),interpolation=cv2.INTER_AREA)
                crop=cv2.getRectSubPix(source,(160,160),tuple(float(x) for x in xy))
                cv2.circle(full,tuple(np.rint(xy/3).astype(int)),5,(0,255,0),2)
                cv2.circle(crop,(80,80),5,(0,255,0),2)
                tile[35:376,5:389]=full;tile[35:195,400:560]=crop
            else:
                cv2.putText(tile,'NOT PROJECTED IN FRAME',(20,170),
                            cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,255),2)
            cv2.putText(tile,f'#{sheet*20+local:02d} {item["camera"]} s{item["screw"]} {item["scene_group"][:7]}',
                        (7,23),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            canvas[local//5*400:(local//5+1)*400,local%5*580:(local%5+1)*580]=tile
        cv2.imwrite(str(a.output/f'early_sheet_{sheet+1}.jpg'),canvas,
                    [cv2.IMWRITE_JPEG_QUALITY,94])
    (a.output/'review_manifest.json').write_text(json.dumps(entries,indent=2)+'\n')
    print(json.dumps({'panels':len(entries),'projected_inframe':sum(x['image'] is not None for x in entries)}))


if __name__=='__main__':main()
