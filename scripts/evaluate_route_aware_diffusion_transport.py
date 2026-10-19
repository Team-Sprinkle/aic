#!/usr/bin/env python3
"""Held-out request-swap, route-goal and local latency checks for diffusion."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torchvision.models import resnet18

from train_route_aware_diffusion_transport import RouteDiffusion, route_targets, evaluate
from train_temporal_transport_probe import LONG_AGES, build_data


def alternate_context(model, data, index, requested_task):
    h = data['history'][index]
    mask = data['mask'][index]
    clock = data['camera_clock'][h].float().clone()
    clock[..., :3] /= .05
    state_task = data['state_task_full'][h].float().clone()
    state_task[..., -10:] = requested_task
    x = torch.cat((data['features'][h].float(), data['prev'][h].float(),
                   data['motion'][h].float(), data['age'][index][...,None],
                   data['gap'][index][...,None], clock, state_task), -1)
    x = model.history_net(model.history_input(x), src_key_padding_mask=~mask)
    return x[:, -1] + model.task_token(requested_task)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--paired-audit', type=Path, required=True)
    p.add_argument('--imagenet', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output already exists')
    saved = torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    manifest = saved['source_manifest']
    data = build_data(Path(manifest['cache']),Path(manifest['checkpoint']),'cuda',LONG_AGES)
    rows = data['rows']
    target, valid = route_targets(rows)
    route = torch.as_tensor(target/.05,device='cuda')
    model = RouteDiffusion(data['features'].shape[1],data['mean'],data['std']).cuda().eval()
    model.load_state_dict(saved['model'])
    paired = json.loads(a.paired_audit.read_text())
    assert len(rows) > 50000
    records = [r for r in paired['records'] if r['split']=='validation'
               and r['same_observation'] and r['future_path_delta_mm']>=10]
    results = []
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for record in records:
            left,right=record['index_a'],record['index_b']
            if not (valid[left] and valid[right]):
                continue
            if rows[left]['scene']!=rows[right]['scene'] or rows[left]['scene']!=record['scene']:
                raise RuntimeError('Paired scene or row-order mismatch')
            li=torch.tensor([left],device='cuda')
            other_task=data['state_task_full'][right,-10:][None].float()
            a_context=model.context(data,li)
            b_context=alternate_context(model,data,li,other_task)
            pred_a=model.goal(a_context)[0].float().cpu().numpy()*.05
            pred_b=model.goal(b_context)[0].float().cpu().numpy()*.05
            action_a,_=model.sample(a_context,seed=20261011,sample_steps=manifest['inference_steps'])
            action_b,_=model.sample(b_context,seed=20261011,sample_steps=manifest['inference_steps'])
            action_delta=((action_a[0,0,:3]-action_b[0,0,:3]).float()
                          *model.target_std[:3]).cpu().numpy()
            teacher_action_delta=(data['target'][left,0,:3]-data['target'][right,0,:3]).cpu().numpy()
            truth=target[left]-target[right]
            delta=pred_a-pred_b
            cosine=float(delta@truth/max(1e-9,np.linalg.norm(delta)*np.linalg.norm(truth)))
            results.append(dict(scene=record['scene'],left=left,right=right,
                                teacher_future_difference_mm=float(np.linalg.norm(truth)*1000),
                                predicted_future_difference_mm=float(np.linalg.norm(delta)*1000),
                                difference_error_mm=float(np.linalg.norm(delta-truth)*1000),
                                direction_cosine=cosine,
                                teacher_first_action_difference_mm=float(np.linalg.norm(teacher_action_delta)*1000),
                                predicted_first_action_difference_mm=float(np.linalg.norm(action_delta)*1000),
                                first_action_difference_error_mm=float(np.linalg.norm(action_delta-teacher_action_delta)*1000)))
    if not results:
        raise RuntimeError('No valid held-out separable route pairs')
    # Representative grouped held-out teacher rows, independent of fit's exact sample list.
    selected=json.loads((a.checkpoint.parent/'selected_validation_rows.json').read_text())
    metrics=evaluate(model,data,selected,route,valid,manifest['inference_steps'],manifest['seed'])
    # Local complete component benchmark: fresh three-view CNN, context, twelve denoising steps.
    encoder=resnet18(weights=None);encoder.fc=torch.nn.Identity()
    weights=torch.load(a.imagenet,map_location='cpu',weights_only=True)
    encoder.load_state_dict({k:v for k,v in weights.items() if not k.startswith('fc.')},strict=True)
    encoder=encoder.cuda().eval()
    idx=torch.tensor([selected[0]],device='cuda')
    images=torch.randn((3,3,224,224),device='cuda')
    timings=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for step in range(40):
            torch.cuda.synchronize();started=time.perf_counter()
            encoded=encoder(images).reshape(1,1536)
            # Replace only latest cached visual feature in a compact one-row proxy.
            # Context still uses saved causal history; this measures fresh CNN cost.
            context=model.context(data,idx)+encoded.mean()*0
            model.sample(context,seed=step,sample_steps=manifest['inference_steps'])
            torch.cuda.synchronize()
            if step>=5:timings.append((time.perf_counter()-started)*1000)
    summary=dict(schema='aic_route_aware_diffusion_heldout/v1',
        checkpoint=str(a.checkpoint),paired_audit=str(a.paired_audit),
        heldout=metrics,paired_scenes=len({r['scene'] for r in results}),
        separable_pairs=len(results),
        branch_direction_accuracy_cosine_ge_point7=float(np.mean([r['direction_cosine']>=.7 for r in results])),
        branch_difference_error_mm_p50=float(np.median([r['difference_error_mm'] for r in results])),
        branch_magnitude_ratio_p50=float(np.median([
            r['predicted_future_difference_mm']/r['teacher_future_difference_mm'] for r in results])),
        branch_action_pairs_with_teacher_difference_ge_5mm=sum(r['teacher_first_action_difference_mm']>=5 for r in results),
        branch_action_difference_error_mm_p50=float(np.median([r['first_action_difference_error_mm'] for r in results])),
        branch_records=results,
        local_encoder_plus_model_p50_ms=float(np.median(timings)),
        local_encoder_plus_model_p95_ms=float(np.quantile(timings,.95)),
        latency_scope='local GPU CNN + cached-history Transformer + diffusion; excludes ROS image decode and controller',
        autonomous_test_run=False,final_scenes_opened=False)
    a.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='branch_records'}))


if __name__=='__main__':
    main()
