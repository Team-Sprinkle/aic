#!/usr/bin/env python3
"""Group the frozen SFP proposal-head event test by old/new development scenes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from handoff_region_prior import load_prior
from train_sfp_frozen_handoff import event_metrics
from train_temporal_transport_probe import LONG_AGES,TemporalTransport,build_data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--model',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--new-episode-base',type=int,default=2400000)
    p.add_argument('--head-only-episode-base',type=int,default=3100000)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior score')
    torch.set_num_threads(8)
    data=build_data(a.cache,a.base,'cuda',LONG_AGES)
    rows=data['rows'];prior=load_prior(a.prior)
    indices={
        'old_sfp':np.asarray([i for i,r in enumerate(rows) if r['family']=='sfp' and
            r['split']=='validation' and int(r['episode'])<a.new_episode_base],dtype=np.int64),
        'new_diverse':np.asarray([i for i,r in enumerate(rows) if r['family']=='sfp' and
            r['split']=='validation' and a.new_episode_base<=int(r['episode'])<a.head_only_episode_base],dtype=np.int64),
        'head_only_tail':np.asarray([i for i,r in enumerate(rows) if r['family']=='sfp' and
            r['split']=='validation' and int(r['episode'])>=a.head_only_episode_base],dtype=np.int64)}
    indices['original_sfp']=np.concatenate([indices['old_sfp'],indices['new_diverse']])
    indices['all_sfp']=np.concatenate([indices['original_sfp'],indices['head_only_tail']])
    results={}
    for path in a.model:
        saved=torch.load(path,map_location='cpu',weights_only=False)
        manifest=saved['source_manifest']
        if not manifest['full_state_skip'] or not manifest['long_history'] or manifest['route_conditioning']:
            raise ValueError(f'Unsupported proposal checkpoint: {path}')
        model=TemporalTransport(data['mean'],data['std'],raw_state_skip=True,
                                feature_dim=data['features'].shape[1],state_task_dim=42,
                                route_conditioning=False).cuda().eval()
        model.load_state_dict(saved['model'])
        results[str(path)]={}
        for section,idx in indices.items():
            summary,episodes=event_metrics(model,data,idx,prior)
            results[str(path)][section]=dict(summary=summary,episodes=episodes)
        print(json.dumps({k:v['summary'] for k,v in results[str(path)].items()}),flush=True)
    a.output.write_text(json.dumps(dict(schema='aic_sfp_frozen_handoff_split_score/v1',
        source_cache=str(a.cache),prior=str(a.prior),threshold=.95,streak=3,
        new_episode_base=a.new_episode_base,
        head_only_episode_base=a.head_only_episode_base,final_scenes_opened=False,
        action_weights_frozen=True,results=results),indent=2)+'\n')


if __name__=='__main__':main()
