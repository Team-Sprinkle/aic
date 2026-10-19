#!/usr/bin/env python3
"""Cache selected 6D port pose from RGB predictions and frozen visual features.

Only observed TCP and fixed camera calibration enter this cache. Ground-truth
port transforms are absent. Rows with fewer than two accepted cameras have
no 3D position and must not satisfy a local-actor handoff.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate,reprojection_residual_px
from train_two_family_port_locator import NAMES,CAMERAS
from train_two_family_port_orientation import Model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--sc-predictions',type=Path)
    p.add_argument('--sfp-predictions',type=Path)
    p.add_argument('--refined-predictions',type=Path,
                   help='Optional native crop result; coarse RGB chooses every crop')
    p.add_argument('--visual-features',type=Path,required=True)
    p.add_argument('--orientation-checkpoint',type=Path,required=True)
    p.add_argument('--camera-calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(8)
    if not (a.sc_predictions or a.sfp_predictions):p.error('At least one prediction file required')
    rows=[json.loads(l) for l in (a.labels/'rows.jsonl').open()]
    pred=defaultdict(dict)
    for path in (a.sc_predictions,a.sfp_predictions):
        if path is None:continue
        for item in map(json.loads,path.open()):pred[item['row']][item['camera']]=item
    refined=({(item['row'],item['camera']):item
              for item in map(json.loads,a.refined_predictions.open())}
             if a.refined_predictions else {})
    visual=np.load(a.visual_features,mmap_mode='r')
    if visual.shape!=(len(rows),3,768):raise ValueError(visual.shape)
    calibration=json.loads(a.camera_calibration.read_text())
    optics={c:np.asarray(calibration['tcp_to_optical'][c],float) for c in CAMERAS}
    features=[];selected=[];records=[]
    for i,r in enumerate(rows):
        camera_pose={c:pose_matrix(r['state'][:7])@optics[c] for c in CAMERAS}
        candidates=[];present=[];selected_pixels={};selected_position=None;selected_residual=None;selected_refined=False
        for name in NAMES:
            pixels={c:pred[i][c]['predicted'][name]['xy'] for c in CAMERAS
                    if c in pred[i] and name in pred[i][c]['predicted'] and
                    pred[i][c]['predicted'][name]['projected_inframe_probability']>=.5}
            if len(pixels)>=2:
                try:point=triangulate(pixels,camera_pose)
                except np.linalg.LinAlgError:point=np.zeros(3)
                valid=float(np.isfinite(point).all())
            else:point=np.zeros(3);valid=0.
            if name==r['selected_identity'] and valid and np.linalg.norm(point-r['state'][:3])<.2:
                native={c:refined[(i,c)]['predicted_xy'] for c in pixels
                        if (i,c) in refined and
                        np.linalg.norm(np.asarray(refined[(i,c)]['predicted_xy'])-pixels[c])<=15}
                if len(native)>=2:
                    try:
                        refined_point=triangulate(native,camera_pose)
                        residual=reprojection_residual_px(refined_point,native,camera_pose)
                        if residual<=4 and np.linalg.norm(refined_point-point)<.008:
                            point=refined_point;pixels=native;selected_refined=True
                    except np.linalg.LinAlgError:pass
            candidates.extend(point.tolist() if valid else [0.,0.,0.]);present.append(valid)
            if name==r['selected_identity'] and valid:
                selected_position=point
                selected_pixels=pixels
                selected_residual=reprojection_residual_px(point,pixels,camera_pose)
        code=np.zeros(len(NAMES));code[NAMES.index(r['selected_identity'])]=1.
        feature=np.r_[candidates,present,code,r['state'][:7],
                      np.asarray(visual[i],dtype=np.float32).reshape(-1)].astype(np.float32)
        features.append(feature);selected.append(selected_position)
        records.append({'label_row':i,'episode':r['episode'],'scene_group':r['scene_group'],
                        'split':r['split'],'family':r['family'],'time_s':r['time_s'],
                        'selected_identity':r['selected_identity'],'position_base_m':(
                            selected_position.tolist() if selected_position is not None else None),
                        'accepted_views':list(selected_pixels),'reprojection_residual_px':selected_residual,
                        'native_refined':selected_refined,
                        'runtime_geometry_input':False,'visibility_certified':False})
    x=torch.as_tensor(np.stack(features),device='cuda')
    checkpoint=torch.load(a.orientation_checkpoint,map_location='cpu',weights_only=False)
    mean=checkpoint['model']['mean'];std=checkpoint['model']['std']
    model=Model(mean,std).cuda().eval();model.load_state_dict(checkpoint['model'])
    with torch.inference_mode():
        orientation=torch.cat([model(part).cpu() for part in x.split(1024)]).numpy()
    quat=Rotation.from_matrix(orientation).as_quat()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w') as file:
        for i,r in enumerate(records):
            r['orientation_base_xyzw']=quat[i].tolist() if selected[i] is not None else None
            file.write(json.dumps(r,separators=(',',':'))+'\n')
    print(json.dumps({'rows':len(records),'3d_available':sum(r['position_base_m'] is not None for r in records),
                      'output':str(a.output),'runtime_geometry':False}))


if __name__=='__main__':main()
