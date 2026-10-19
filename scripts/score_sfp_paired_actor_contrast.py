#!/usr/bin/env python3
"""Score requested-route changes on the identical cached observation history."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_temporal_transport_probe import (LONG_AGES, TemporalTransport,
                                           build_data, paired_counterfactual_first)
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--source-checkpoint', type=Path, required=True)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--route-goal-model', type=Path, help='Frozen route-goal probe used by conditioned actors')
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve existing result')
    torch.set_num_threads(8)
    manifest = json.loads(a.pairs.read_text())
    pairs = manifest['validation_pairs']
    if not pairs:
        p.error('No episode-grouped validation pairs')
    results = {}
    for path in a.checkpoint:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        sm = saved['source_manifest']
        if not sm['long_history'] or not sm['full_state_skip'] or sm['route_conditioning']:
            raise ValueError(f'Unsupported actor: {path}')
        route_goal_enabled=bool(sm.get('route_goal_conditioning',False))
        goal_path=a.route_goal_model
        if route_goal_enabled and not goal_path:
            raise ValueError('Route-goal checkpoint has no frozen probe path')
        data = build_data(a.cache, a.source_checkpoint, a.device, LONG_AGES,
                          route_goal_model=goal_path if route_goal_enabled else None,
                          route_goal_zero_control=sm.get('route_goal_zero_control',False),
                          route_goal_progress_gate_mm=sm.get('route_goal_progress_gate_mm'))
        left = torch.as_tensor([x['left_index'] for x in pairs], device=a.device)
        right = torch.as_tensor([x['right_index'] for x in pairs], device=a.device)
        true = (data['target'][left, 0, :3] - data['target'][right, 0, :3]).cpu().numpy() * 1000
        model = TemporalTransport(data['mean'], data['std'], raw_state_skip=True,
                                  feature_dim=data['features'].shape[1],
                                  state_task_dim=42, route_conditioning=False,
                                  route_goal_conditioning=route_goal_enabled).to(a.device).eval()
        model.load_state_dict(saved['model'])
        pred = []
        with torch.inference_mode():
            for l, r in zip(left.split(64), right.split(64)):
                if not route_goal_enabled:
                    x, y = paired_counterfactual_first(model, data, l, r)
                else:
                    h=data['history'][l];mask=data['mask'][l]
                    feature=data['features'][h].float()
                    prior=data['prev'][h].float();motion=data['motion'][h].float()
                    clock=data['camera_clock'][h].float().clone();clock[...,:3]/=.05
                    state=data['state_task_full'][h].float()
                    alternate=state.clone()
                    alternate[...,-10:]=data['state_task_full'][r,-10:].float()[:,None,:]
                    original_goal=data['route_goal'][l]
                    alternate_goal=torch.zeros_like(original_goal)
                    if not sm.get('route_goal_zero_control',False):
                        ridge=dict(np.load(goal_path))
                        public=np.concatenate((data['features'][l].cpu().numpy(),
                                               np.asarray([data['rows'][int(i)]['state'][:7] for i in l],dtype=np.float32),
                                               np.asarray([data['rows'][int(i)]['task'] for i in r],dtype=np.float32)),axis=1)
                        base=((public-ridge['center'])/ridge['scale'])@ridge['coefficients']+ridge['target_mean']
                        raw=np.asarray([data['rows'][int(i)]['state'] for i in l],dtype=np.float32)
                        body=Rotation.from_quat(raw[:,3:7]).inv().apply(base)/.05
                        alternate_goal=torch.as_tensor(body,device=a.device,dtype=torch.float32)
                        active=original_goal.norm(dim=-1)>0
                        alternate_goal=alternate_goal*active[:,None]
                    action,*_=model(torch.cat((feature,feature)),torch.cat((prior,prior)),
                                    torch.cat((motion,motion)),torch.cat((data['age'][l],data['age'][l])),
                                    torch.cat((data['gap'][l],data['gap'][l])),torch.cat((clock,clock)),
                                    torch.cat((mask,mask)),torch.cat((state,alternate)),
                                    route_goal=torch.cat((original_goal,alternate_goal)))
                    x,y=action[:len(l),0],action[len(l):,0]
                pred.append(((x[:, :3] - y[:, :3]) * model.target_std[:3]).cpu().numpy() * 1000)
        pred = np.concatenate(pred)
        err = np.linalg.norm(pred - true, axis=1)
        true_norm = np.linalg.norm(true, axis=1)
        pred_norm = np.linalg.norm(pred, axis=1)
        cosine = np.sum(pred * true, axis=1) / np.maximum(1e-6, pred_norm * true_norm)
        results[str(path)] = dict(pairs=len(pairs), scenes=len(set(x['scene'] for x in pairs)),
                                  delta_error_mm_p50=float(np.median(err)),
                                  delta_error_mm_p95=float(np.percentile(err, 95)),
                                  predicted_delta_mm_p50=float(np.median(pred_norm)),
                                  teacher_delta_mm_p50=float(np.median(true_norm)),
                                  correct_direction_fraction=float(np.mean(cosine > 0)),
                                  rows=[dict(scene=x['scene'], predicted_delta_mm=pred[i].tolist(),
                                             target_delta_mm=true[i].tolist(), error_mm=float(err[i]))
                                        for i, x in enumerate(pairs)])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(schema='aic_sfp_paired_actor_contrast/v1',
                                        same_observation_only_request_code_changes=True,
                                        final_scenes_opened=False, results=results), indent=2) + '\n')
    print(json.dumps({k: {j: v for j, v in value.items() if j != 'rows'}
                      for k, value in results.items()}))


if __name__ == '__main__':
    main()
