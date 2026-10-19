#!/usr/bin/env python3
"""Matched old/coverage model actions on held-out public-anchor teacher states."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from train_route_aware_diffusion_transport import RouteDiffusion
from train_temporal_transport_probe import LONG_AGES,build_data


def label_map(paths):
    result={}
    for path in paths:
        with path.open() as stream:
            for line in stream:
                row=json.loads(line)
                pose=(row.get('observed_tcp_pose_port_frame') or
                      row.get('observed_tcp_pose_opening_frame'))
                result[(row['episode_index'],row.get('frame',row.get('frame_index')))] = pose
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--base-checkpoint',type=Path,required=True)
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--anchor-labels',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous diagnosis')
    torch.set_num_threads(8)
    data=build_data(a.cache,a.base_checkpoint,'cuda',LONG_AGES)
    poses=label_map(a.anchor_labels)
    rows=data['rows']
    selected=[i for i,r in enumerate(rows) if r['split']=='validation' and
              r.get('official_anchor_teacher') and (r['episode'],r['frame']) in poses]
    if len(selected)<50:raise ValueError(f'Too few grouped anchor rows: {len(selected)}')
    results={}
    for arm,path in (('control',a.control),('candidate',a.candidate)):
        model=RouteDiffusion(data['features'].shape[1],data['mean'],data['std']).cuda().eval()
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
        group=defaultdict(lambda:defaultdict(list))
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for offset in range(0,len(selected),48):
                ids=selected[offset:offset+48]
                idx=torch.as_tensor(ids,device='cuda')
                action,_=model.sample(model.context(data,idx),seed=20261014+offset,sample_steps=12)
                action=action.float()*model.target_std+model.target_mean
                pred=action[:,0,:3].cpu().numpy()
                truth=data['target'][idx,0,:3].cpu().numpy()
                for j,i in enumerate(ids):
                    r=rows[i];pose=np.asarray(poses[(r['episode'],r['frame'])],float)
                    relative=pose[:3]*1000
                    axial=relative[2];lateral=np.linalg.norm(relative[:2])
                    rot=Rotation.from_quat(pose[3:7])
                    pred_port=rot.apply(pred[j])*1000
                    true_port=rot.apply(truth[j])*1000
                    toward=-relative[:2]/max(lateral,1e-6)
                    phase=('near' if (-200<=axial<=-100 if r['family']=='sc' else -150<=axial<=-70)
                           else 'far')
                    key=f"{r['family']}_{phase}"
                    group[key]['error_mm'].append(float(np.linalg.norm(pred[j]-truth[j])*1000))
                    group[key]['pred_toward_mm'].append(float(pred_port[:2]@toward))
                    group[key]['teacher_toward_mm'].append(float(true_port[:2]@toward))
                    group[key]['same_lateral_sign'].append(bool((pred_port[:2]@toward)*
                                                                (true_port[:2]@toward)>0))
                    group[key]['episode'].append(r['episode'])
        results[arm]={}
        for key,metrics in group.items():
            results[arm][key]=dict(rows=len(metrics['error_mm']),
                episodes=len(set(metrics['episode'])),
                first_action_error_mm_p50=float(np.median(metrics['error_mm'])),
                first_action_error_mm_p95=float(np.quantile(metrics['error_mm'],.95)),
                predicted_toward_port_mm_p50=float(np.median(metrics['pred_toward_mm'])),
                teacher_toward_port_mm_p50=float(np.median(metrics['teacher_toward_mm'])),
                lateral_correction_sign_accuracy=float(np.mean(metrics['same_lateral_sign'])))
    out=dict(schema='aic_anchor_teacher_action_diagnosis/v1',results=results,
             heldout_complete_reset=True,privileged_geometry_posthoc_only=True,
             final_scenes_opened=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
