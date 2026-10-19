#!/usr/bin/env python3
"""Jointly match frozen requested-port heatmap peaks from three cameras.

Runtime selection uses RGB heatmaps, measured TCP, and fixed camera calibration.
Projected simulator port labels are accessed only for posthoc scoring.
"""
import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path

import cv2
cv2.setNumThreads(1)
import numpy as np
import torch
from torch.utils.data import DataLoader

from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate,reprojection_residual_px
from train_sfp_observed_rail_crop import Crops
from train_target_conditioned_port_locator import ConditionalLocator
from train_two_family_port_locator import CAMERAS


def read(path):return [json.loads(s) for s in (path/'rows.jsonl').open()]


def peaks(heat,center,k=5,radius=9):
    score=np.asarray(heat,np.float32)*4
    kernel=np.ones((radius,radius),np.uint8)
    maxima=cv2.dilate(score,kernel)
    yy,xx=np.where(score>=maxima-1e-5)
    inds=np.argsort(score[yy,xx])[::-1][:k]
    h,w=score.shape
    top=float(score.max())
    return [{'xy':[float(center[0]+(x/(w-1)-.5)*480),
                    float(center[1]+(y/(h-1)-.5)*480)],
             'gap':float(top-score[y,x]),'cell':[int(x),int(y)]}
            for y,x in zip(yy[inds],xx[inds])]


def select(views,cameras,k=5,weight=2.):
    if set(views)!=set(CAMERAS) or any(views[c]['prob']<.5 for c in CAMERAS):return None
    best=None
    for triple in product(*(views[c]['peaks'][:k] for c in CAMERAS)):
        points={c:triple[i]['xy'] for i,c in enumerate(CAMERAS)}
        try:
            xyz=triangulate(points,cameras)
            residual=reprojection_residual_px(xyz,points,cameras)
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(residual):continue
        cost=residual+weight*sum(q['gap'] for q in triple)
        if best is None or cost<best['cost']:
            best={'position_base_m':xyz.tolist(),'reprojection_px':float(residual),
                  'cost':float(cost),'points':points,
                  'peak_cells':{c:triple[i]['cell'] for i,c in enumerate(CAMERAS)}}
    return best


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--proposals',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--top-k',type=int,default=5)
    p.add_argument('--logit-gap-weight',type=float,default=2.)
    p.add_argument('--reprojection-gate-px',type=float,default=5.)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous evaluation')
    labels={(r['episode'],r['time_s']):r for r in read(a.labels)}
    rows=[]
    for q in read(a.proposals):
        lab=labels[(q['episode'],q['time_s'])]
        assert q['scene_group']==lab['scene_group']
        rows.append({'episode':q['episode'],'time_s':q['time_s'],
                     'scene_group':q['scene_group'],'images':lab['images'],
                     'state':lab['state'],'targets':lab['targets'],
                     'requested':q['requested'],'proposal':q['crop']})
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if ckpt.get('history_context') or ckpt.get('source_offset_meta'):
        p.error('Expected no-history/no-offset crop checkpoint')
    torch.set_num_threads(8)
    model=ConditionalLocator(None).cuda().eval()
    model.load_state_dict(ckpt['model'])
    loader=DataLoader(Crops(rows,[(i,c) for i in range(len(rows)) for c in range(3)],False),
                      batch_size=24,num_workers=4,pin_memory=True,multiprocessing_context='spawn')
    cal=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics={c:np.asarray(cal['tcp_to_optical'][c],float) for c in CAMERAS}
    predictions=defaultdict(dict)
    for batch in loader:
        image,meta,target,_,_,ri,ci=batch
        _,flag,heat=model(image.cuda().float()/255.,meta.cuda(),target.cuda())
        flag=flag.sigmoid().float().cpu().numpy();heat=heat.float().cpu().numpy()
        for i in range(len(image)):
            j=int(ri[i]);cam=CAMERAS[int(ci[i])]
            center=rows[j]['proposal']['center_px'][cam]
            predictions[j][cam]={'prob':float(flag[i]),'peaks':peaks(heat[i],center,a.top_k)}
    outcomes=[];errors=[];wrong=0;byepisode=defaultdict(list)
    for j,row in enumerate(rows):
        cameras={c:pose_matrix(row['state'])@optics[c] for c in CAMERAS}
        chosen=select(predictions[j],cameras,a.top_k,a.logit_gap_weight)
        accepted=chosen is not None and chosen['reprojection_px']<=a.reprojection_gate_px
        rec={'episode':row['episode'],'time_s':row['time_s'],'requested':row['requested'],
             'accepted':accepted,'selected':chosen,'view_prob':{c:predictions[j][c]['prob'] for c in CAMERAS},
             'error_mm_label_only':None,'nearest_identity_label_only':None}
        if accepted:
            pos=np.asarray(chosen['position_base_m'])
            target=next(t for t in row['targets'] if t['identity']==row['requested'])
            err=float(np.linalg.norm(pos-np.asarray(target['position_base_m']))*1000)
            nearest=min(row['targets'],key=lambda t:np.linalg.norm(pos-np.asarray(t['position_base_m'])))
            rec['error_mm_label_only']=err;rec['nearest_identity_label_only']=nearest['identity']
            errors.append(err);wrong+=nearest['identity']!=row['requested']
        outcomes.append(rec);byepisode[row['episode']].append(rec)
    early=0;anygroup=0
    for items in byepisode.values():
        items.sort(key=lambda r:r['time_s'])
        first=next((r for r in items if r['accepted']),None)
        anygroup+=first is not None
        early+=first is not None and first['time_s']-items[0]['time_s']<=3
    summary={'schema':'sfp_joint_crop_candidates/v1','groups':len(byepisode),'rows':len(rows),
             'accepted':len(errors),'wrong':wrong,'accepted_3d_p95_mm':float(np.quantile(errors,.95)) if errors else None,
             'groups_with_any':anygroup,'groups_early':early,'top_k':a.top_k,
             'logit_gap_weight':a.logit_gap_weight,'reprojection_gate_px':a.reprojection_gate_px,
             'checkpoint':str(a.checkpoint),'labels':str(a.labels),'proposals':str(a.proposals),
             'runtime_true_geometry_input':False,'final_scenes_opened':False}
    a.output.mkdir(parents=True)
    with (a.output/'predictions.jsonl').open('w') as f:
        for r in outcomes:f.write(json.dumps(r,separators=(',',':'))+'\n')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
