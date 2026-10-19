#!/usr/bin/env python3
"""Distinguish diffusion sampling spread from systematic action error."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from compare_transport_five_way import select_rows
from train_diffusion_transport_probe import DiffusionTransport
from train_temporal_transport_probe import build_data,LONG_AGES


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--diffusion',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(8)
    data=build_data(a.cache,a.base,'cuda',LONG_AGES)
    ck=torch.load(a.diffusion,map_location='cpu',weights_only=False)
    model=DiffusionTransport(data['features'].shape[1],data['mean'],data['std']).cuda().eval()
    model.load_state_dict(ck['model'])
    selected=select_rows(data['rows']);groups=defaultdict(lambda:defaultdict(list))
    with torch.inference_mode():
        for offset in range(0,len(selected),96):
            part=selected[offset:offset+96]
            context=model.context(data,torch.as_tensor(part,device='cuda'))
            draws=torch.stack([model.sample(context,20261001+offset+100000*seed)
                               for seed in range(4)])
            xyz=(draws*model.target_std+model.target_mean)[:,:,0,:3]
            truth=data['target'][part,0,:3]
            error=(xyz-truth[None]).norm(dim=-1)*1000
            mean_error=(xyz.mean(0)-truth).norm(dim=-1)*1000
            spread=(xyz-xyz.mean(0)[None]).norm(dim=-1).mean(0)*1000
            for j,i in enumerate(part):
                family=data['rows'][i]['family'];value=groups[family]
                value['sample_error_mm'].append(float(error[0,j]))
                value['four_sample_mean_error_mm'].append(float(mean_error[j]))
                value['oracle_best_of_four_mm'].append(float(error[:,j].min()))
                value['sample_spread_mm'].append(float(spread[j]))
    result={family:{name:dict(median=float(np.quantile(vals,.5)),
                              p95=float(np.quantile(vals,.95)))
                    for name,vals in metrics.items()} for family,metrics in groups.items()}
    summary=dict(schema='aic_diffusion_sampling_diagnostic/v1',
                 draws=4,oracle_best_of_four_is_not_deployable=True,
                 validation_episodes=dict(sfp=32,sc=21),results=result)
    a.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
