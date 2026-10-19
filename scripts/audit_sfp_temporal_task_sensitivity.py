#!/usr/bin/env python3
"""Same-observation card/port code swaps for a frozen temporal transport actor."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from train_temporal_transport_probe import LONG_AGES, TemporalTransport, build_data


@torch.inference_mode()
def actions(model, data, index):
    h = data['history'][index:index+1]
    base = (data['features'][h].float(), data['prev'][h].float(),
            data['motion'][h].float(), data['age'][index:index+1],
            data['gap'][index:index+1], data['camera_clock'][h].float(),
            data['mask'][index:index+1], data['state_task_full'][h].float())
    out = []
    for card in range(5):
        for port in range(2):
            features, prior, motion, age, gap, clock, mask, state_task = base
            task = state_task.clone()
            task[...,32:] = 0
            task[...,32] = 1
            task[...,32+2+port] = 1
            task[...,32+4+card] = 1
            scaled_clock = clock.clone()
            scaled_clock[...,:3] /= .05
            pred, _, _, _ = model(features,prior,motion,age,gap,
                                  scaled_clock,mask,task,
                                  spatial=(data['spatial'][index:index+1].float()
                                           if model.spatial_request_attention else None))
            first = (pred[0,0].float().cpu().numpy()*data['std'][0] +
                     data['mean'][0])
            out.append({'card':card,'port':port,
                        'first_target_body_translation_mm':(first[:3]*1000).tolist()})
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--source-checkpoint',type=Path,required=True)
    p.add_argument('--actor-checkpoint',type=Path,required=True)
    p.add_argument('--spatial-cache',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--ages',type=float,nargs='+',default=(0,1,2,4,8),
                   help='Elapsed seconds from first saved frame for same-image task-code swaps')
    a=p.parse_args()
    if a.output.exists():p.error(f'Output exists: {a.output}')
    torch.set_num_threads(4)
    cp=torch.load(a.actor_checkpoint,map_location='cpu',weights_only=False)
    manifest=cp['source_manifest']
    data=build_data(a.cache,a.source_checkpoint,a.device,LONG_AGES,
                    spatial_cache=a.spatial_cache)
    if bool(a.spatial_cache)!=bool(manifest.get('spatial_request_attention',False)):
        raise ValueError('Spatial cache and checkpoint mode mismatch')
    if not (manifest['full_state_skip'] and manifest['long_history']
            and not manifest['route_conditioning']):
        raise ValueError('Unexpected actor architecture')
    model=TemporalTransport(data['mean'],data['std'],raw_state_skip=True,
                            feature_dim=data['features'].shape[1],
                            state_task_dim=42,route_conditioning=False,
                            task_gated_sfp=bool(cp.get('task_gated_sfp',False)),
                            task_route_film=manifest.get('task_route_film',False),
                            spatial_request_attention=manifest.get('spatial_request_attention',False)).to(a.device).eval()
    model.load_state_dict(cp['model'])
    grouped=defaultdict(list)
    for i,row in enumerate(data['rows']):
        if row['family']=='sfp' and row['split']=='validation':
            grouped[row['episode']].append(i)
    records=[]
    for episode,indices in grouped.items():
        indices.sort(key=lambda i:data['rows'][i]['time_s'])
        initial=data['rows'][indices[0]]['time_s']
        for target_age in a.ages:
            i=min(indices,key=lambda j:abs(data['rows'][j]['time_s']-initial-target_age))
            row=data['rows'][i]
            preds=actions(model,data,i)
            true=(int(np.argmax(row['task'][4:9])),int(np.argmax(row['task'][2:4])))
            original=next(x for x in preds if (x['card'],x['port'])==true)
            xyz=np.asarray([x['first_target_body_translation_mm'] for x in preds])
            record={'episode':episode,'time_s':row['time_s'],
                    'age_from_first_s':row['time_s']-initial,
                    'requested_card_port':list(true),
                    'observed_request_first_action_mm':original['first_target_body_translation_mm'],
                    'same_image_code_swaps':preds,
                    'ten_code_action_diameter_mm':float(max(np.linalg.norm(x-y)
                        for x in xyz for y in xyz)),
                    'nearest_other_code_action_delta_mm':float(min(np.linalg.norm(
                        xyz[true[0]*2+true[1]]-x) for j,x in enumerate(xyz)
                        if j!=true[0]*2+true[1]))}
            records.append(record)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result={'schema':'aic_sfp_temporal_task_sensitivity/v1',
            'checkpoint':str(a.actor_checkpoint),'cache':str(a.cache),
            'episodes':len(grouped),'records':records,
            'sample_ages_s':list(a.ages),
            'median_ten_code_action_diameter_mm':float(np.median(
                [x['ten_code_action_diameter_mm'] for x in records])),
            'median_nearest_code_delta_mm':float(np.median(
                [x['nearest_other_code_action_delta_mm'] for x in records])),
            'same_observation_only_task_code_changes':True,
            'teacher_geometry_actor_input':False,'final_scenes_opened':False}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
