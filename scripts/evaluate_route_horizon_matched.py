#!/usr/bin/env python3
"""Matched grouped-row comparison of 4, 8, and 12 target transport chunks."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train_temporal_transport_probe import LONG_AGES, build_data
from train_route_aware_diffusion_transport import RouteDiffusion


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def summarize(values):
    x=np.asarray(values,float)
    return dict(n=len(x),median_mm=float(np.median(x)),p95_mm=float(np.quantile(x,.95))) if len(x) else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--base-checkpoint',type=Path,required=True)
    p.add_argument('--extended-labels',type=Path,required=True)
    p.add_argument('--checkpoint4',type=Path,required=True)
    p.add_argument('--checkpoint8',type=Path)
    p.add_argument('--checkpoint12',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sample-steps',type=int,default=12)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing output')
    if a.checkpoint8 is None and a.checkpoint12 is None:p.error('Provide a longer-horizon checkpoint')
    torch.set_num_threads(8)
    data=build_data(a.cache,a.base_checkpoint,'cuda',LONG_AGES)
    labels=np.load(a.extended_labels/'targets12.npy',mmap_mode='r')
    valid=np.load(a.extended_labels/'valid12.npy',mmap_mode='r')
    if len(labels)!=len(data['rows']):raise ValueError('Label/cache row mismatch')
    episodes=defaultdict(list)
    for i,r in enumerate(data['rows']):
        if r['split']=='validation' and valid[i,11] and not r.get('actor_prefix_context_only'):
            episodes[(r['family'],r['episode'])].append(i)
    selected=[]
    for ids in episodes.values():
        ids.sort(key=lambda i:data['rows'][i]['time_s'])
        selected.extend(np.asarray(ids)[np.linspace(0,len(ids)-1,min(12,len(ids))).astype(int)].tolist())
    selected=sorted(set(selected))
    if not selected:raise ValueError('No matched validation rows')
    report=dict(schema='aic_matched_route_horizon/v1',slot_s=.05,
                common_validation_rows=len(selected),common_episode_groups=len(episodes),
                labels=str(a.extended_labels),label_manifest_sha256=sha(a.extended_labels/'manifest.json'),
                sample_steps=a.sample_steps,final_scenes_opened=False,models={})
    entries=[(4,4,a.checkpoint4)]
    if a.checkpoint8:entries.append((8,4,a.checkpoint8))
    if a.checkpoint12:entries.append((12,6,a.checkpoint12))
    for horizon,execute,checkpoint in entries:
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        manifest=cp['source_manifest']
        if int(manifest.get('action_horizon',4))!=horizon:
            raise ValueError(f'Horizon mismatch: {checkpoint}')
        model=RouteDiffusion(data['features'].shape[1],data['mean'],data['std'],
                             action_horizon=horizon).cuda().eval()
        model.load_state_dict(cp['model'])
        groups=defaultdict(lambda:defaultdict(list))
        for offset in range(0,len(selected),48):
            ids=selected[offset:offset+48]
            idx=torch.as_tensor(ids,device='cuda')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                context=model.context(data,idx)
                actions,_=model.sample(context,seed=20261014+offset,sample_steps=a.sample_steps)
            pred=(actions.float()*model.target_std+model.target_mean).cpu().numpy()
            truth=np.asarray(labels[ids,:horizon],np.float32)
            error=np.linalg.norm(pred[:,:,:3]-truth[:,:,:3],axis=-1)*1000
            for j,i in enumerate(ids):
                r=data['rows'][i]
                phase='early' if r['family']=='sfp' and 4<=r['time_s']<=12 else 'other'
                key=f"{r['family']}_{phase}"
                for name in (key,r['family'],'all'):
                    group=groups[name]
                    group['first'].append(float(error[j,0]))
                    group['first4_mean'].append(float(error[j,:4].mean()))
                    group['executed_mean'].append(float(error[j,:execute].mean()))
                    group['last_executed'].append(float(error[j,execute-1]))
                    group['full_mean'].append(float(error[j].mean()))
                    group['last_predicted'].append(float(error[j,-1]))
        report['models'][str(horizon)]=dict(checkpoint=str(checkpoint),sha256=sha(checkpoint),
            selected_update=cp.get('update'),predicted_slots=horizon,executed_slots=execute,
            metrics={name:{metric:summarize(values) for metric,values in v.items()}
                     for name,v in sorted(groups.items())})
        del model
        torch.cuda.empty_cache()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({h:{g:m['first4_mean'] for g,m in row['metrics'].items() if g in ('sfp','sc')}
                      for h,row in report['models'].items()},indent=2))


if __name__=='__main__':main()
