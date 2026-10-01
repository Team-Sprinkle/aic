#!/usr/bin/env python3
"""Evaluate the frozen single-observation BC action head on temporal cache rows."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from train_shared_port_tcp_bc import SharedPortTCPActor, rotation_from_pose9


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[json.loads(line) for line in (a.cache/'rows.jsonl').open()]
    features=np.load(a.cache/'features.npy',mmap_mode='r')
    cp=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    stats={k:cp['model'][k].numpy() for k in
           ('state_mean','state_std','pose_mean','pose_std','target_mean','target_std')}
    model=SharedPortTCPActor(stats,pretrained=False,condition_on_pose=False).cuda().eval()
    model.load_state_dict(cp['model'],strict=True)
    groups=defaultdict(lambda:defaultdict(list))
    with torch.inference_mode():
        for begin in range(0,len(rows),512):
            end=min(len(rows),begin+512)
            selected=[i for i in range(begin,end) if rows[i]['split']=='validation']
            if not selected:continue
            context=torch.as_tensor(np.asarray(features[selected]).copy(),device='cuda').float()
            condition=model.pose_context(torch.cat((context,torch.zeros(len(selected),9,device='cuda')),-1))
            tokens=model.waypoint_tokens+condition[:,None]
            pred=model.action_head(model.transformer(tokens))*model.target_std+model.target_mean
            truth=torch.as_tensor(np.asarray([rows[i]['target'] for i in selected],dtype=np.float32),device='cuda')
            mm=(pred[:,0,:3]-truth[:,0,:3]).norm(dim=-1)*1e3
            relative=rotation_from_pose9(pred[:,0]).transpose(-1,-2)@rotation_from_pose9(truth[:,0])
            deg=(((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1).acos()*180/np.pi)
            for j,i in enumerate(selected):
                family=rows[i]['family'];groups[family]['translation_mm'].append(float(mm[j]));
                groups[family]['orientation_deg'].append(float(deg[j]))
    result=dict(schema='aic_single_observation_cached_validation/v1',checkpoint=str(a.checkpoint),
                by_family={f:{k:dict(count=len(v),median=float(np.median(v)),p95=float(np.quantile(v,.95)))
                              for k,v in values.items()} for f,values in groups.items()},
                scope='same selected 200ms decision rows as temporal cache; teacher-frame error only',
                final_scenes_opened=False,rl_started=False)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['by_family']))


if __name__=='__main__':main()
