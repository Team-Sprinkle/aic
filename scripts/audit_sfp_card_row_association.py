#!/usr/bin/env python3
"""Frozen multiport RGB candidate audit; no teacher geometry chooses candidates."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate,reprojection_residual_px
from train_two_family_port_locator import CAMERAS,LOOKUP,NAMES,Locator


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=('train','validation'),default='validation')
    p.add_argument('--input-width',type=int,default=288)
    p.add_argument('--input-height',type=int,default=256)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior output')
    rows=[json.loads(s) for s in (a.labels/'rows.jsonl').open() if f'"split":"{a.split}"' in s]
    model=Locator(None).cuda().eval()
    model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model'])
    cal=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics={c:np.asarray(cal['tcp_to_optical'][c],float) for c in CAMERAS}
    output=[]
    for r in rows:
        imgs=[];meta=[]
        for ci,cam in enumerate(CAMERAS):
            image=cv2.imread(r['images'][cam])
            if image is None:raise FileNotFoundError(r['images'][cam])
            imgs.append(cv2.cvtColor(cv2.resize(image,(a.input_width,a.input_height),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB))
            m=np.zeros(5,np.float32);m[1]=1;m[2+ci]=1;meta.append(m)
        with torch.inference_mode():
            xy,flag=model(torch.from_numpy(np.stack(imgs).copy()).cuda().permute(0,3,1,2).float()/255.,
                          torch.from_numpy(np.stack(meta)).cuda())
        xy=xy.float().cpu().numpy()*[1151,1023];prob=flag.float().sigmoid().cpu().numpy()
        poses={c:pose_matrix(r['state'])@optics[c] for c in CAMERAS}
        candidates={}
        # Runtime does not know which optional NIC cards are present. Enumerate
        # every SFP output channel, including physically absent slots; the
        # labels below are retained only for posthoc scoring.
        for name in NAMES[2:]:
            j=LOOKUP[name]
            views={c:xy[i,j].tolist() for i,c in enumerate(CAMERAS) if prob[i,j]>=.5}
            candidate={'views':views,'probability':{c:float(prob[i,j]) for i,c in enumerate(CAMERAS)},
                       'predicted_base_m':None,'reprojection_px':None,'three_view_consistent':False}
            if len(views)>=2:
                try:
                    point=triangulate(views,poses);reproj=reprojection_residual_px(point,views,poses)
                    candidate.update(predicted_base_m=point.tolist(),reprojection_px=float(reproj),
                                     three_view_consistent=bool(len(views)==3 and reproj<=5))
                except np.linalg.LinAlgError:pass
            candidates[name]=candidate
        output.append({'episode':r['episode'],'scene_group':r['scene_group'],'time_s':r['time_s'],
                       'requested':r['selected_identity'],'candidates':candidates,
                       'true_positions_label_only':{t['identity']:t['position_base_m'] for t in r['targets']}})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w') as f:
        for r in output:f.write(json.dumps(r,separators=(',',':'))+'\n')
    counts=defaultdict(int)
    for r in output:
        for name,c in r['candidates'].items():
            if c['three_view_consistent']:counts[name]+=1
    print(json.dumps({'rows':len(output),'consistent_candidates':counts},default=dict))


if __name__=='__main__':main()
