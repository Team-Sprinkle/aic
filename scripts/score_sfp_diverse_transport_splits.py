#!/usr/bin/env python3
"""Compare frozen transport checkpoints on old and new grouped development scenes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from train_temporal_transport_probe import LONG_AGES,TemporalTransport,build_data,evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--model',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--new-episode-base',type=int,default=2400000)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing score')
    torch.set_num_threads(8)
    data=build_data(a.cache,a.base,a.device,LONG_AGES)
    groups=defaultdict(lambda:defaultdict(list))
    for i,row in enumerate(data['rows']):
        if row['split']!='validation':continue
        section=('new_diverse' if int(row['episode'])>=a.new_episode_base else
                 'old_sfp' if row['family']=='sfp' else 'sc')
        groups[section][int(row['episode'])].append(i)
    indices={}
    for section,episodes in groups.items():
        selected=[]
        for _,rows in sorted(episodes.items()):
            cap=8 if section=='sc' else 12
            selected.extend(np.asarray(rows)[np.linspace(0,len(rows)-1,min(cap,len(rows))).astype(int)].tolist())
        indices[section]=np.asarray(sorted(set(selected)),dtype=np.int64)
    results={}
    for path in a.model:
        saved=torch.load(path,map_location='cpu',weights_only=False)
        manifest=saved['source_manifest']
        if not manifest['full_state_skip'] or not manifest['long_history'] or manifest['route_conditioning']:
            raise ValueError(f'Unsupported comparison model: {path}')
        model=TemporalTransport(data['mean'],data['std'],raw_state_skip=True,
                                feature_dim=data['features'].shape[1],state_task_dim=42,
                                route_conditioning=False).to(a.device).eval()
        model.load_state_dict(saved['model'])
        results[str(path)]={section:dict(episodes=len(groups[section]),
            rows=len(values),metrics=evaluate(model,data,values,False)['overall'])
            for section,values in indices.items()}
        print(json.dumps({k:{'episodes':v['episodes'],'metrics':v['metrics']}
                          for k,v in results[str(path)].items()}),flush=True)
    a.output.write_text(json.dumps(dict(schema='aic_sfp_diverse_transport_split_score/v1',
        cache=str(a.cache),new_episode_base=a.new_episode_base,
        episode_grouped=True,final_scenes_opened=False,results=results),indent=2)+'\n')


if __name__=='__main__':main()
