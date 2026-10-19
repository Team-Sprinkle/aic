#!/usr/bin/env python3
"""Inspect rejected held-out SFP corrective frames with truth for diagnosis only."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from sfp_port_cascade import SfpPortCascade


def report(values):
    return {'n': len(values), 'median': float(np.median(values)),
            'p95': float(np.quantile(values, .95))} if values else {'n': 0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--coarse',type=Path,required=True)
    p.add_argument('--broad',type=Path,required=True)
    p.add_argument('--fine',type=Path,required=True)
    p.add_argument('--orientation',type=Path,required=True)
    p.add_argument('--orientation-secondary',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous diagnosis')
    cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(4)
    cascade=SfpPortCascade(a.coarse,a.broad,a.fine,a.orientation,a.calibration,
                           orientation_path_secondary=a.orientation_secondary,
                           max_reprojection_px=1e6)
    rows=[]
    for row in map(json.loads,(a.labels/'rows.jsonl').open()):
        if row['time_s']<20:continue
        native={cam:cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
                for cam,path in row['images'].items()}
        small={cam:cv2.cvtColor(cv2.imdecode(cv2.imencode('.jpg',
            cv2.resize(image,(288,256),interpolation=cv2.INTER_AREA)[...,::-1],
            [cv2.IMWRITE_JPEG_QUALITY,95])[1],cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
            for cam,image in native.items()}
        target=next(t for t in row['targets'] if t['identity']==row['selected_identity'])
        card=int(row['selected_identity'].split('_')[2]);port=int(row['selected_identity'].split('_')[4])
        task=[1.,0.,float(port==0),float(port==1),
              *[float(j==card) for j in range(5)],1.]
        pred,diag=cascade.predict(native,small,row['state'],task)
        truth=np.asarray(target['position_base_m'])
        rows.append({'episode':row['episode'],'time_s':row['time_s'],
                     'reprojection_px':diag.get('reprojection_px'),
                     'position_error_mm':float(np.linalg.norm(pred[:3,3]-truth)*1000) if pred is not None else None,
                     'fine_pixels_px':diag.get('fine_pixels_px'),
                     'broad_pixels_px':diag.get('broad_pixels_px'),
                     'coarse_pixels_px':diag.get('coarse_pixels_px'),
                     'true_pixels_px':{cam:target['cameras'][cam][:2] for cam in ('center','left','right')},
                     'reason':diag.get('reason')})
    summary={}
    for episode in sorted({r['episode'] for r in rows}):
        subset=[r for r in rows if r['episode']==episode]
        summary[episode]={'frames':len(subset),'passes_2px':sum(r['reprojection_px'] is not None and r['reprojection_px']<2 for r in subset),
                          'reprojection_px':report([r['reprojection_px'] for r in subset if r['reprojection_px'] is not None]),
                          'position_error_mm':report([r['position_error_mm'] for r in subset if r['position_error_mm'] is not None]),
                          'pixel_error_by_camera':{cam:report([float(np.linalg.norm(np.array(r['fine_pixels_px'][i])-np.array(r['true_pixels_px'][cam])))
                                  for r in subset if r['fine_pixels_px'] is not None])
                                  for i,cam in enumerate(('center','left','right'))}}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'schema':'aic_sfp_recovery_view_diagnosis/v1',
        'truth_used_only_in_posthoc_diagnosis':True,'summary':summary,'rows':rows,
        'reserved_final_opened':False},indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
