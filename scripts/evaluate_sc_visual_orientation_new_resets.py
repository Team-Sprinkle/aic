#!/usr/bin/env python3
"""Score a frozen RGB SC port-orientation model on independent native reset groups."""
import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate
from train_two_family_port_locator import CAMERAS, NAMES, Locator
from train_two_family_port_orientation import Model


def stats(values):
    if not values:return {'n':0,'median_deg':None,'p95_deg':None}
    return {'n':len(values),'median_deg':float(np.median(values)),
            'p95_deg':float(np.quantile(values,.95))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels-glob')
    p.add_argument('--labels',type=Path,action='append',
                   help='Explicit fixed label paths, used when camera sampling schedules differ')
    p.add_argument('--locator',type=Path,required=True)
    p.add_argument('--orientation',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--max-triplets-per-group',type=int,default=80)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior result')
    paths=a.labels if a.labels else [Path(x) for x in sorted(glob.glob(a.labels_glob or ''))]
    if len(paths)!=4:raise ValueError(f'Expected four fixed development groups, got {len(paths)}')
    samples=[]
    for path in paths:
        by_time=defaultdict(dict)
        for line in path.open():
            r=json.loads(line);by_time[r['sim_ns']][r['camera']]=r
        eligible=[]
        for t,views in sorted(by_time.items()):
            if set(views)!=set(CAMERAS):continue
            if not all(views[c]['projected_inframe'] for c in CAMERAS):continue
            eligible.append({'path':path,'sim_ns':t,'views':views})
        if len(eligible)>a.max_triplets_per_group:
            eligible=[eligible[int(i)] for i in np.linspace(0,len(eligible)-1,
                         a.max_triplets_per_group,dtype=int)]
        samples.extend(eligible)
    locator=Locator(None).to(a.device).eval()
    locator.load_state_dict(torch.load(a.locator,map_location='cpu',weights_only=False)['model'])
    saved=torch.load(a.orientation,map_location='cpu',weights_only=False)
    orient=Model(saved['model']['mean'],saved['model']['std']).to(a.device).eval()
    orient.load_state_dict(saved['model'])
    optics={c:np.asarray(x,float) for c,x in
            json.loads(a.calibration.read_text())['tcp_to_optical'].items()}
    rows=[]
    for sample in samples:
        views=sample['views'];images=[]
        for c in CAMERAS:
            bgr=cv2.imread(views[c]['image'])
            if bgr is None:raise FileNotFoundError(views[c]['image'])
            images.append(cv2.cvtColor(cv2.resize(bgr,(288,256),interpolation=cv2.INTER_AREA),
                                       cv2.COLOR_BGR2RGB))
        rgb=torch.from_numpy(np.stack(images)).permute(0,3,1,2).to(a.device).float()/255
        meta=np.zeros((3,5),np.float32);meta[:,0]=1
        for j in range(3):meta[j,2+j]=1
        with torch.inference_mode():
            points,flags,features=locator(rgb,torch.from_numpy(meta).to(a.device),
                                          return_features=True)
        points=points.float().cpu().numpy()*[1151,1023]
        flags=flags.float().sigmoid().cpu().numpy()
        state=np.asarray(views['center']['measured_tcp_base_pose'],float)
        tcp=pose_matrix(state);cameras={c:tcp@optics[c] for c in CAMERAS}
        candidates=[];mask=[]
        for k in range(len(NAMES)):
            pix={c:points[j,k] for j,c in enumerate(CAMERAS) if flags[j,k]>=.5}
            if len(pix)>=2:
                try:xyz=triangulate(pix,cameras);valid=np.isfinite(xyz).all()
                except np.linalg.LinAlgError:valid=False
            else:valid=False
            candidates.extend(xyz.tolist() if valid else [0.,0.,0.]);mask.append(float(valid))
        port=int(views['center']['target_port']);code=np.zeros(len(NAMES))
        code[port]=1
        vector=np.r_[candidates,mask,code,state,features.float().cpu().numpy().reshape(-1)].astype(np.float32)
        with torch.inference_mode():pred=orient(torch.from_numpy(vector).to(a.device)[None]).float().cpu().numpy()[0]
        rotation=Rotation.from_matrix(pred)
        truth=Rotation.from_quat(state[3:7])*Rotation.from_matrix(
            views['center']['relative_opening_rotation_tcp'])
        error=float(np.rad2deg((rotation.inv()*truth).magnitude()))
        rows.append({'episode':views['center']['episode_id'],
            'scene_group':views['center']['scene_group_id'],
            'sim_ns':sample['sim_ns'],'target_port':port,
            'tcp_to_opening_mm':float(np.linalg.norm(views['center']['relative_opening_tcp_m'])*1000),
            'selected_valid_triangulation':bool(mask[port]),
            'selected_flags_by_camera':flags[:,port].tolist(),
            'orientation_error_deg_posthoc':error,
            'images':[views[c]['image'] for c in CAMERAS]})
    result={'schema':'aic_sc_frozen_visual_orientation_independent_resets/v1',
        'checkpoint':str(a.orientation),'locator':str(a.locator),
        'source_paths':[str(x) for x in paths],
        'max_triplets_per_group':a.max_triplets_per_group,
        'truth_used_after_prediction_only':True,'runtime_true_geometry_input':False,
        'complete_reset_groups':len(paths),'complete_inframe_triplets':len(rows),
        'all':stats([r['orientation_error_deg_posthoc'] for r in rows]),
        'near_100mm':stats([r['orientation_error_deg_posthoc'] for r in rows
                            if r['tcp_to_opening_mm']<100]),
        'by_episode':{name:stats([r['orientation_error_deg_posthoc'] for r in rows
                                  if r['episode']==name]) for name in sorted({r['episode'] for r in rows})},
        'rows':rows,'final_scenes_opened':False}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('complete_reset_groups','complete_inframe_triplets',
                    'all','near_100mm','by_episode')},indent=2))


if __name__=='__main__':main()
