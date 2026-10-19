#!/usr/bin/env python3
"""Matched offline oracle-target-cue diagnostic; never a deployable actor."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.nn import functional as F

from train_temporal_transport_probe import LONG_AGES, TemporalTransport, build_data, predict


class Head(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.model = nn.Sequential(nn.LayerNorm(275), nn.Linear(275, 256),
                                   nn.GELU(), nn.Linear(256, 36))
        with torch.no_grad():
            self.model[0].weight[:272].copy_(base.action[0].weight)
            self.model[0].bias[:272].copy_(base.action[0].bias)
            self.model[1].weight[:, :272].copy_(base.action[1].weight)
            self.model[1].weight[:, 272:].zero_()
            self.model[1].bias.copy_(base.action[1].bias)
            self.model[3].load_state_dict(base.action[3].state_dict())

    def forward(self, hidden, cue):
        padding = torch.zeros((len(hidden), 16), device=hidden.device)
        return self.model(torch.cat((hidden, padding, cue), -1)).reshape(-1, 4, 9)


def label_map(path):
    result = {}
    for line in (path / 'rows.jsonl').open():
        row = json.loads(line)
        for target in row['targets']:
            key = (row['scene_group'], target['identity'])
            xyz = np.asarray(target['position_base_m'], dtype=float)
            if key in result:
                old_split, old = result[key]
                if old_split != row['split'] or np.linalg.norm(old - xyz) > .001:
                    raise ValueError(f'Conflicting scene geometry/split: {key}')
            result[key] = row['split'], xyz
    return result


def request(row):
    code = np.asarray(row['task'])
    return f'sfp_card_{code[4:9].argmax()}_port_{code[2:4].argmax()}'


@torch.inference_mode()
def score(head, hidden, cue, truth, rows, indices, mean, std):
    head.eval()
    errors = []
    early = []
    by_task = defaultdict(list)
    for chunk in np.array_split(indices, max(1, (len(indices) + 255) // 256)):
        idx = torch.as_tensor(chunk, device=hidden.device)
        pred = head(hidden[idx], cue[idx]).float() * std + mean
        mm = torch.linalg.vector_norm((pred[:, 0, :3] - truth[idx, 0, :3]) * 1000, dim=-1)
        for j, value in zip(chunk, mm.cpu().numpy()):
            value = float(value)
            errors.append(value)
            row = rows[j]
            by_task[request(row)].append(value)
            if .3 <= row['time_s'] <= 1.2:
                early.append(value)
    return {'rows': len(errors), 'early_rows': len(early),
            'first_target_p95_mm': float(np.percentile(errors, 95)),
            'early_first_target_p95_mm': float(np.percentile(early, 95)) if early else None,
            'by_task_p95_mm': {k: float(np.percentile(v, 95))
                               for k, v in sorted(by_task.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--actor-checkpoint', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--minutes', type=float, default=20)
    parser.add_argument('--cue-scale-m', type=float, default=.1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve previous output')
    if args.updates > 1000 or args.minutes > 20:
        parser.error('Bounded diagnostic cap 1000 updates/20 minutes')
    if args.cue_scale_m <= 0:
        parser.error('Positive cue scale required')
    random.seed(20261007)
    np.random.seed(20261007)
    torch.manual_seed(20261007)
    torch.set_num_threads(4)
    label = label_map(args.labels)
    data = build_data(args.cache, args.source_checkpoint, 'cuda', LONG_AGES)
    cp = torch.load(args.actor_checkpoint, map_location='cpu', weights_only=False)
    source = cp['source_manifest']
    if not (source['full_state_skip'] and source['long_history'] and
            not source['route_conditioning']):
        raise ValueError('Unexpected frozen actor architecture')
    base = TemporalTransport(data['mean'], data['std'], raw_state_skip=True,
                             feature_dim=data['features'].shape[1],
                             state_task_dim=42, route_conditioning=False).cuda().eval()
    base.load_state_dict(cp['model'])
    base.requires_grad_(False)
    selected = []
    cues = []
    groups = defaultdict(set)
    by_task = defaultdict(list)
    for i, row in enumerate(data['rows']):
        if row['family'] != 'sfp':
            continue
        key = (row['scene'], request(row))
        if key not in label:
            continue
        split, port = label[key]
        if split != row['split']:
            raise ValueError(f'Leak at row {i}')
        state = np.asarray(row['state'], dtype=float)
        relative = Rotation.from_quat(state[3:7]).inv().apply(port - state[:3])
        selected.append(i)
        cues.append(relative.astype(np.float32))
        groups[split].add(row['scene'])
        if split == 'train':
            by_task[request(row)].append(len(selected) - 1)
    if len(groups['train']) != 50 or len(groups['validation']) != 10 or len(by_task) != 10:
        raise ValueError(f'Unexpected joined groups: {dict((k,len(v)) for k,v in groups.items())}')
    if groups['train'] & groups['validation']:
        raise ValueError('Complete-reset overlap')
    args.output.mkdir(parents=True)
    (args.output / 'manifest.json').write_text(json.dumps({
        'schema': 'sfp_oracle_transport_cue/v1', 'train_groups': 50,
        'development_groups': 10, 'joined_rows': len(selected),
        'cache': str(args.cache), 'actor': str(args.actor_checkpoint),
        'labels': str(args.labels), 'geometry_is_privileged_input': True,
        'cue_scale_m': args.cue_scale_m,
        'use': 'offline upper bound only; never deploy',
        'final_scenes_opened': False}, indent=2) + '\n')
    hidden = torch.empty((len(selected), 256), device='cuda')
    for start in range(0, len(selected), 256):
        stop = min(start + 256, len(selected))
        idx = torch.as_tensor(selected[start:stop], device='cuda')
        with torch.inference_mode():
            *_, h = predict(base, data, idx, False, return_hidden=True)
        hidden[start:stop] = h
    cue_true = torch.as_tensor(np.asarray(cues) / args.cue_scale_m, device='cuda')
    cue_zero = torch.zeros_like(cue_true)
    truth = data['target'][torch.as_tensor(selected, device='cuda')]
    rows = [data['rows'][i] for i in selected]
    dev = np.asarray([j for j, row in enumerate(rows) if row['split'] == 'validation'])
    mean = base.target_mean
    std = base.target_std
    initial = Head(base).cuda()
    heads = {'zero': Head(base).cuda(), 'oracle': Head(base).cuda()}
    for head in heads.values():
        head.load_state_dict(initial.state_dict())
    optimizers = {k: torch.optim.AdamW(h.parameters(), lr=1e-4, weight_decay=1e-4)
                  for k, h in heads.items()}
    history = []
    best = {'zero': (float('inf'), None), 'oracle': (float('inf'), None)}
    start_time = time.monotonic()
    reason = 'update_cap'
    tasks = sorted(by_task)
    for step in range(1, args.updates + 1):
        if time.monotonic() - start_time > args.minutes * 60:
            reason = 'time_cap'
            break
        chosen = np.asarray([random.choice(by_task[random.choice(tasks)])
                             for _ in range(64)])
        idx = torch.as_tensor(chosen, device='cuda')
        desired = (truth[idx] - mean) / std
        for name, head in heads.items():
            head.train()
            optimizers[name].zero_grad(set_to_none=True)
            cue = cue_true if name == 'oracle' else cue_zero
            with torch.autocast('cuda', dtype=torch.bfloat16):
                predicted = head(hidden[idx], cue[idx])
                loss = F.smooth_l1_loss(predicted.float(), desired, beta=.2)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizers[name].step()
        if step % 200 == 0:
            record = {'step': step}
            for name, head in heads.items():
                cue = cue_true if name == 'oracle' else cue_zero
                metric = score(head, hidden, cue, truth, rows, dev, mean, std)
                record[name] = metric
                objective = metric['early_first_target_p95_mm']
                if objective < best[name][0]:
                    best[name] = objective, step
                    torch.save({'head': head.state_dict(), 'step': step,
                                'privileged_geometry_input': name == 'oracle'},
                               args.output / f'{name}_best.pt')
            history.append(record)
            print(json.dumps({'step': step,
                              'zero_early_p95': record['zero']['early_first_target_p95_mm'],
                              'oracle_early_p95': record['oracle']['early_first_target_p95_mm']}),
                  flush=True)
    selected_metrics = {}
    for name in heads:
        checkpoint = torch.load(args.output / f'{name}_best.pt', map_location='cpu',
                                weights_only=False)
        chosen = next(x[name] for x in history if x['step'] == checkpoint['step'])
        selected_metrics[name] = {'step': checkpoint['step'], **chosen}
    zero = selected_metrics['zero']
    oracle = selected_metrics['oracle']
    early_improvement = 1 - oracle['early_first_target_p95_mm'] / zero['early_first_target_p95_mm']
    gate = early_improvement >= .2 and oracle['first_target_p95_mm'] <= zero['first_target_p95_mm']
    summary = {'schema': 'sfp_oracle_transport_cue_result/v1',
               'steps_completed': step, 'stop_reason': reason,
               'selected': selected_metrics, 'early_p95_fractional_improvement': early_improvement,
               'diagnostic_gate_passed': bool(gate),
               'autonomous_or_deployable_result': False, 'final_scenes_opened': False}
    (args.output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
