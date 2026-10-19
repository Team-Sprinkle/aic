#!/usr/bin/env python3
"""Evaluate two frozen route diffusion checkpoints on identical grouped rows."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from train_temporal_transport_probe import LONG_AGES, build_data
from train_route_aware_diffusion_transport import RouteDiffusion, evaluate, route_targets


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--base-checkpoint',type=Path,required=True)
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--selected',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prior-dropout',type=float,default=0.,
                   help='Evaluation-time chance of hiding previous teacher chunks; 1.0 is zero-prior')
    p.add_argument('--sample-steps',type=int,default=12)
    p.add_argument('--prior-lag-eval',type=int,choices=(0,1,2),default=0)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing comparison')
    torch.set_num_threads(8)
    data=build_data(a.cache,a.base_checkpoint,'cuda',LONG_AGES)
    if a.prior_lag_eval:
        groups=defaultdict(list)
        for i,row in enumerate(data['rows']):groups[(row['family'],row['episode'])].append(i)
        delayed=torch.zeros_like(data['prev'])
        for ids in groups.values():
            lag=a.prior_lag_eval
            if len(ids)>lag:delayed[ids[lag:]]=data['prev'][ids[:-lag]]
        data[f'prev_lag{a.prior_lag_eval}']=delayed
    selected=json.loads(a.selected.read_text())
    assert selected and all(data['rows'][i]['split']=='validation' for i in selected)
    target,valid=route_targets(data['rows'])
    route=torch.as_tensor(target/.05,device='cuda')
    result=dict(schema='aic_route_diffusion_same_rows_offline/v1',
                cache=str(a.cache),cache_manifest_sha256=sha(a.cache/'manifest.json'),
                selected_rows=str(a.selected),selected_sha256=sha(a.selected),
                prior_dropout=a.prior_dropout,sample_steps=a.sample_steps,
                prior_lag_eval=a.prior_lag_eval,
                final_scenes_opened=False,checkpoints={})
    for name,path in (('control',a.control),('candidate',a.candidate)):
        cp=torch.load(path,map_location='cpu',weights_only=False)
        model=RouteDiffusion(data['features'].shape[1],data['mean'],data['std']).cuda()
        model.load_state_dict(cp['model'])
        metrics=evaluate(model,data,selected,route,valid,a.sample_steps,20261014,
                         prior_dropout=a.prior_dropout,eval_prior_lag=a.prior_lag_eval)
        result['checkpoints'][name]=dict(path=str(path),sha256=sha(path),metrics=metrics)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v['metrics'] for k,v in result['checkpoints'].items()},indent=2))


if __name__=='__main__':main()
