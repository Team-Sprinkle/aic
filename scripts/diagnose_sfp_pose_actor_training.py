#!/usr/bin/env python3
"""Bounded capacity check on grouped, actor-visited SFP correction labels."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_temporal_transport_probe import LONG_AGES, TemporalTransport, build_data, predict


def score(model, data, indices):
    model.eval()
    index = torch.as_tensor(indices, device='cuda')
    pose = data['route_goal'][index].clone()
    truth = data['target'][index, 0, :3].float()
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        real, *_ = predict(model, data, index, False)
        data['route_goal'][index] = 0
        zero, *_ = predict(model, data, index, False)
        data['route_goal'][index] = pose[torch.randperm(len(index), device='cuda')]
        shuffled, *_ = predict(model, data, index, False)
        data['route_goal'][index] = pose
    def xyz(value):
        return (value[:, 0] * model.target_std + model.target_mean)[:, :3].float()
    real, zero, shuffled = map(xyz, (real, zero, shuffled))
    def p95(error): return float(torch.quantile(error * 1000, .95).item())
    return dict(rows=len(indices), episodes=len({data['rows'][i]['episode'] for i in indices}),
                error_mm_p95=p95((real - truth).norm(dim=-1)),
                zero_pose_error_mm_p95=p95((zero - truth).norm(dim=-1)),
                shuffled_pose_error_mm_p95=p95((shuffled - truth).norm(dim=-1)),
                zero_pose_action_change_mm_p95=p95((real - zero).norm(dim=-1)),
                shuffled_pose_action_change_mm_p95=p95((real - shuffled).norm(dim=-1)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'base-checkpoint', 'values', 'prior', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--seed', type=int, default=20261008)
    p.add_argument('--reset-action-head', action='store_true')
    args = p.parse_args()
    if args.output.exists(): p.error('Preserve earlier diagnostic')
    if not 1 <= args.steps <= 1000: p.error('Bound steps to 1..1000')
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    data = build_data(args.cache, args.base_checkpoint, 'cuda', history_ages=LONG_AGES,
                      route_goal_values=args.values)
    rows = data['rows']; values = data['route_goal']
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if (row['family'] == 'sfp' and row.get('corrective_teacher') and
                8100000 <= int(row['episode']) < 8400000 and
                float(values[i].norm()) > 1e-8):
            groups[row['split']].append(i)
    train, validation = groups['train'], groups['validation']
    if (len(train) < 100 or len(validation) < 40 or
            len({rows[i]['episode'] for i in train}) != 5 or
            len({rows[i]['episode'] for i in validation}) != 3):
        raise ValueError('Unexpected accepted-pose episode split')
    train_scenes = {rows[i]['scene'] for i in train}
    if train_scenes & {rows[i]['scene'] for i in validation}:
        raise ValueError('Reset configuration leaked across splits')
    # Verify the nearest history before each first teacher label is an actor row.
    joins = []
    for episode in sorted({rows[i]['episode'] for i in train + validation}):
        teacher = next(i for i, row in enumerate(rows) if row['episode'] == episode and
                       row.get('corrective_teacher'))
        previous = rows[teacher - 1]
        if (previous['episode'] != episode or not previous.get('actor_prefix_context_only') or
                not 0 < rows[teacher]['time_s'] - previous['time_s'] < 1):
            raise ValueError(f'Invalid first-action causal join: {episode}')
        joins.append(dict(episode=int(episode), split=rows[teacher]['split'],
                          gap_s=rows[teacher]['time_s'] - previous['time_s'],
                          first_teacher_pose_accepted=bool(float(values[teacher].norm()) > 1e-8)))
    prior = torch.load(args.prior, map_location='cpu', weights_only=False)
    model = TemporalTransport(data['mean'], data['std'], raw_state_skip=True,
                              feature_dim=data['features'].shape[1], state_task_dim=42,
                              route_conditioning=False, route_goal_conditioning=True).cuda()
    missing = model.load_state_dict(prior['model'], strict=False)
    if missing.missing_keys != ['route_goal_proj.weight'] or missing.unexpected_keys:
        raise ValueError(f'Unexpected prior structure: {missing}')
    if args.reset_action_head:
        # Retain the learned visual/history context and reset only the full-action decoder.
        for module in model.action.modules():
            if module is not model.action and hasattr(module, 'reset_parameters'):
                module.reset_parameters()
    args.output.mkdir(parents=True)
    manifest = dict(schema='aic_sfp_pose_actor_capacity/v1', train_rows=len(train),
                    validation_rows=len(validation), train_episodes=5, validation_episodes=3,
                    train_pose_accepted_by_episode={str(e):sum(rows[i]['episode'] == e for i in train)
                                                    for e in sorted({rows[i]['episode'] for i in train})},
                    validation_pose_accepted_by_episode={str(e):sum(rows[i]['episode'] == e for i in validation)
                                                         for e in sorted({rows[i]['episode'] for i in validation})},
                    steps=args.steps, seed=args.seed,
                    reset_action_head=args.reset_action_head,
                    first_teacher_joins=joins,
                    final_scenes_opened=False, autonomous_rollouts=0)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    history = []
    for step in range(1, args.steps + 1):
        model.train(); opt.zero_grad(set_to_none=True)
        selected = np.random.choice(train, size=32, replace=True)
        index = torch.as_tensor(selected, device='cuda')
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            action, *_ = predict(model, data, index, False)
            truth = (data['target'][index] - model.target_mean) / model.target_std
            loss = nn.functional.smooth_l1_loss(action, truth, beta=.2)
        loss.backward()
        gradient = float(model.route_goal_proj.weight.grad.norm().item())
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        if step % 100 == 0 or step == 1:
            entry = dict(step=step, loss=float(loss.item()), pose_projection_grad_norm=gradient,
                         train=score(model, data, train), validation=score(model, data, validation))
            history.append(entry)
            print(json.dumps(entry), flush=True)
    torch.save(dict(model=model.state_dict(), manifest=manifest), args.output / 'last.pt')
    (args.output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(history[-1], indent=2) + '\n')


if __name__ == '__main__': main()
