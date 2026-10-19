#!/usr/bin/env python3
"""Check whether a full-action actor changes its action under its frozen pose input."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_temporal_transport_probe import LONG_AGES,TemporalTransport,build_data,predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','base-checkpoint','values','candidate','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous diagnostic')
    cp=torch.load(a.candidate,map_location='cpu',weights_only=False)
    data=build_data(a.cache,a.base_checkpoint,'cuda',history_ages=LONG_AGES,
                    route_goal_values=a.values)
    m=TemporalTransport(data['mean'],data['std'],raw_state_skip=True,
                        feature_dim=data['features'].shape[1],state_task_dim=42,
                        route_conditioning=False,route_goal_conditioning=True).cuda().eval()
    m.load_state_dict(cp['model'])
    idx=[i for i,r in enumerate(data['rows']) if r['split']=='validation' and
         int(r['episode'])>=8100000 and r.get('corrective_teacher')]
    q=data['route_goal'][idx].clone();truth=data['target'][idx,0].float()
    with torch.inference_mode():
        indices=torch.tensor(idx,device='cuda');actual,*_=predict(m,data,indices,False)
        data['route_goal'][idx]=0;zero,*_=predict(m,data,indices,False)
        actual=(actual[:,0]*m.target_std+m.target_mean).float()
        zero=(zero[:,0]*m.target_std+m.target_mean).float()
        delta=(actual[:,:3]-zero[:,:3]).norm(dim=-1).cpu().numpy()*1000
        error_actual=(actual[:,:3]-truth[:,:3]).norm(dim=-1).cpu().numpy()*1000
        error_zero=(zero[:,:3]-truth[:,:3]).norm(dim=-1).cpu().numpy()*1000
        norms=q.norm(dim=-1).cpu().numpy();valid=norms>1e-8
    result=dict(schema='aic_sfp_route_pose_input_use/v1',rows=len(idx),
                accepted_pose_rows=int(valid.sum()),
                pose_projection_weight_norm=float(m.route_goal_proj.weight.norm().item()),
                input_norm_p50=float(np.median(norms[valid])) if valid.any() else None,
                action_change_mm_p50=float(np.median(delta)),
                action_change_mm_p95=float(np.percentile(delta,95)),
                accepted_action_change_mm_p95=float(np.percentile(delta[valid],95)) if valid.any() else None,
                accepted_teacher_error_p95_with_pose=float(np.percentile(error_actual[valid],95)) if valid.any() else None,
                accepted_teacher_error_p95_zero_pose=float(np.percentile(error_zero[valid],95)) if valid.any() else None,
                final_scenes_opened=False)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
