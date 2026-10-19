#!/usr/bin/env python3
"""Audit a task-selected rail crop built from RGB predictions and CAD spacing.

The requested true port is read only after crop placement for audit. Runtime
placement uses predicted port pairs, public card index, measured TCP, fixed
camera calibration, and fixed 40 mm rail spacing from task-board URDF.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_sc_port_targets import pose_matrix
from build_sc_native_pose_labels import WIDTH,HEIGHT,FOCAL
from train_two_family_port_locator import CAMERAS


def project(point, optical):
    q=np.linalg.inv(optical)@np.r_[point,1.]
    if q[2]<=0:return None
    return np.array([WIDTH/2+FOCAL*q[0]/q[2],HEIGHT/2+FOCAL*q[1]/q[2]])


def propose(row,optics,halfsize,source_reprojection_px):
    requested=row['requested'];parts=requested.split('_');target_card=int(parts[2]);target_port=int(parts[-1])
    candidates=[]
    for source in range(5):
        left=row['candidates'].get(f'sfp_card_{source}_port_0')
        right=row['candidates'].get(f'sfp_card_{source}_port_1')
        if not left or not right:continue
        if (len(left['views'])!=3 or len(right['views'])!=3 or
            left['reprojection_px'] is None or right['reprojection_px'] is None or
            left['reprojection_px']>source_reprojection_px or
            right['reprojection_px']>source_reprojection_px):continue
        p0=np.asarray(left['predicted_base_m']);p1=np.asarray(right['predicted_base_m'])
        delta=p1-p0;gap=np.linalg.norm(delta)
        if not .015<=gap<=.035:continue
        horizontal=delta.copy();horizontal[2]=0
        if np.linalg.norm(horizontal)<.015:continue
        e1=horizontal/np.linalg.norm(horizontal)
        row_axis=-np.cross([0,0,1.],e1)
        center=p0+(target_card-source)*.04*row_axis+target_port*delta
        candidates.append((abs(target_card-source),source,center,gap))
    if not candidates:return None
    _,source,center,gap=min(candidates,key=lambda x:(x[0],x[1]))
    pixels={cam:project(center,pose_matrix(row['state'])@optics[cam]) for cam in CAMERAS}
    if any(x is None for x in pixels.values()):return None
    return {'source_card':source,'source_pair_gap_mm':gap*1000,
            'estimated_center_base_m':center.tolist(),
            'center_px':{c:p.tolist() for c,p in pixels.items()},'crop_halfsize_px':halfsize}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--crop-size',type=int,default=480)
    p.add_argument('--split',choices=('train','validation'),default='validation')
    p.add_argument('--source-reprojection-px',type=float,default=5.)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing audit')
    if a.crop_size!=480:p.error('First bounded diagnostic fixes native crop at 480 pixels')
    cal=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics={c:np.asarray(cal['tcp_to_optical'][c],float) for c in CAMERAS}
    labels=[json.loads(s) for s in (a.labels/'rows.jsonl').open() if f'"split":"{a.split}"' in s]
    by_key={(r['episode'],r['time_s']):r for r in labels}
    results=[];by_episode=defaultdict(list)
    for line in a.candidates.open():
        row=json.loads(line);label=by_key[(row['episode'],row['time_s'])]
        row['state']=label['state']
        crop=propose(row,optics,a.crop_size//2,a.source_reprojection_px)
        target=next(t for t in label['targets'] if t['identity']==row['requested'])
        scored={}
        if crop:
            for cam in CAMERAS:
                uv=target['cameras'][cam]
                delta=np.asarray(uv[:2])-crop['center_px'][cam] if uv[3] else None
                scored[cam]={'projected_inframe':bool(uv[3]),
                             'target_inside_crop':bool(delta is not None and np.max(np.abs(delta))<=a.crop_size/2),
                             'center_error_px':float(np.linalg.norm(delta)) if delta is not None else None,
                             'target_offset_px':delta.tolist() if delta is not None else None}
        item={'episode':row['episode'],'time_s':row['time_s'],'requested':row['requested'],
              'crop':crop,'audit_only':scored}
        results.append(item);by_episode[row['episode']].append(item)
    early=0;any_episode=0;fully_inside=0;available=0
    episode_rows=[]
    for episode,rows in sorted(by_episode.items()):
        rows.sort(key=lambda r:r['time_s']);t0=rows[0]['time_s']
        usable=[r for r in rows if r['crop'] and all(v['target_inside_crop'] for v in r['audit_only'].values())]
        proposed=[r for r in rows if r['crop']]
        any_episode+=bool(usable)
        early+=bool(usable and usable[0]['time_s']-t0<=3)
        episode_rows.append({'episode':episode,'requested':rows[0]['requested'],'sampled_rows':len(rows),
                             'crop_proposed':len(proposed),'all_three_target_inside':len(usable),
                             'first_usable_s_from_start':usable[0]['time_s']-t0 if usable else None})
        for r in rows:
            available+=bool(r['crop'])
            fully_inside+=bool(r['crop'] and all(v['target_inside_crop'] for v in r['audit_only'].values()))
    summary={'schema':'sfp_observed_rail_crop_audit/v1','rows':len(results),
             'crop_proposed':available,'target_inside_all_three_crops':fully_inside,
             'episodes_any_usable_crop':any_episode,'episodes_usable_crop_within_3s':early,
             'episode_rows':episode_rows,'crop_size_px':a.crop_size,
             'source_reprojection_px':a.source_reprojection_px,
             'runtime_true_geometry_input':False,'true_geometry_used_only_for_audit':True,
             'final_scenes_opened':False}
    a.output.mkdir(parents=True)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (a.output/'rows.jsonl').open('w') as f:
        for r in results:f.write(json.dumps(r,separators=(',',':'))+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
