#!/usr/bin/env python3
"""Evaluate RGB-only selected-port localization as 3D geometry on held-out scenes.

Only RGB, observed TCP, public target code, and fixed camera calibration enter
the estimate. Target geometry is read afterward for scoring. This is not an
occlusion or orientation validation.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader

from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate, reprojection_residual_px
from train_two_family_port_locator import Locator, Views, CAMERAS, LOOKUP, WIDTH, HEIGHT


def summary(values):
    a=np.asarray(values,dtype=float)
    return {'n':len(a),'median':float(np.median(a)) if len(a) else None,
            'p95':float(np.quantile(a,.95)) if len(a) else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--sfp-cache',type=Path)
    p.add_argument('--imagenet',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--camera-calibration',type=Path,required=True)
    p.add_argument('--refined-predictions',type=Path,
                   help='Observation-selected native SC crop predictions, validation only')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(8)
    rows=[json.loads(l) for l in (a.labels/'rows.jsonl').open()]
    indices=[(i,c) for i,r in enumerate(rows) if r['split']=='validation' for c in range(3)]
    dataset=Views(rows,a.sfp_cache,indices,False)
    loader=DataLoader(dataset,batch_size=96,shuffle=False,num_workers=4,pin_memory=True)
    model=Locator(a.imagenet).cuda().eval()
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(ckpt['model'])
    calibration=json.loads(a.camera_calibration.read_text())
    optics={c:np.asarray(calibration['tcp_to_optical'][c],float) for c in CAMERAS}
    estimates=defaultdict(dict)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for image,meta,_,_,_,selected,_,index,cam in loader:
            pred,flag=model(image.cuda(non_blocking=True).float()/255.,meta.cuda(non_blocking=True))
            pred=pred.float().cpu().numpy();flag=flag.float().sigmoid().cpu().numpy()
            for j in range(len(index)):
                i=int(index[j]);c=int(cam[j]);s=int(selected[j])
                estimates[i][CAMERAS[c]]={'xy':(pred[j,s]*[WIDTH-1,HEIGHT-1]).tolist(),
                                          'projected_inframe_probability':float(flag[j,s])}
    if a.refined_predictions:
        lookup={(r['episode'],r['time_s'],r['camera']):r
                for r in map(json.loads,a.refined_predictions.open())}
        for i,cams in estimates.items():
            r=rows[i]
            for cam in CAMERAS:
                key=(r['episode'],r['time_s'],cam)
                if key in lookup:
                    cams[cam]['xy']=lookup[key]['predicted_xy']
                    cams[cam]['native_refined']=True
    by=defaultdict(list);records=[]
    for i,cams in estimates.items():
        r=rows[i];name=r['selected_identity'];target=next(t for t in r['targets'] if t['identity']==name)
        camera_pose={c:pose_matrix(r['state'][:7])@optics[c] for c in CAMERAS}
        pixels={c:cams[c]['xy'] for c in CAMERAS if cams[c]['projected_inframe_probability']>=.5}
        if len(pixels)<2:continue
        try:
            estimated=triangulate(pixels,camera_pose)
            reproj=reprojection_residual_px(estimated,pixels,camera_pose)
        except np.linalg.LinAlgError:continue
        truth=np.asarray(target['position_base_m'],float)
        dist=float(np.linalg.norm(truth-r['state'][:3]))
        err=float(np.linalg.norm(estimated-truth)*1000)
        local_error=Rotation.from_quat(target['orientation_base_xyzw']).inv().apply(estimated-truth)*1000
        lateral=float(np.linalg.norm(local_error[:2]))
        axial=float(abs(local_error[2]))
        group='near' if dist<.1 else 'middle' if dist<.2 else 'far'
        by[(r['family'],group,'xyz_mm')].append(err)
        by[(r['family'],group,'lateral_mm')].append(lateral)
        by[(r['family'],group,'axial_mm')].append(axial)
        by[(r['family'],group,'reprojection_px')].append(reproj)
        records.append({'episode':r['episode'],'scene_group':r['scene_group'],'family':r['family'],
                        'selected_identity':name,'time_s':r['time_s'],'tcp_distance_to_port_mm':dist*1000,
                        'estimated_position_base_m':estimated.tolist(),'truth_position_base_m':truth.tolist(),
                        'translation_error_mm':err,'lateral_error_mm':lateral,
                        'axial_error_mm':axial,'reprojection_residual_px':reproj,
                        'views':len(pixels),'predicted_inframe':list(pixels)})
    result={'schema':'two_family_port_geometry_eval/v1','checkpoint':str(a.checkpoint),
            'runtime_inputs':['RGB','selected target from public task code','measured TCP','fixed TCP-to-camera calibration'],
            'target_geometry_used_for_estimation':False,'orientation_estimated':False,
            'visibility_certified':False,'native_refinement':str(a.refined_predictions) if a.refined_predictions else None,
            'reserved_final_opened':False,
            'validation_episodes':len({r['episode'] for r in records}),
            'by_family_distance':{str(k):summary(v) for k,v in by.items()},'records':records}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'episodes':result['validation_episodes'],'stats':result['by_family_distance']}))


if __name__=='__main__':main()
