#!/usr/bin/env python3
"""Train only the SC arrival head on a frozen, observation-only transport actor."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from train_temporal_transport_probe import LONG_AGES, TemporalTransport, build_data, predict


def event_metrics(model, data, indices, threshold=.95, streak=2):
    grouped = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for part in np.array_split(indices, max(1, (len(indices)+255)//256)):
            ind = torch.as_tensor(part, device='cuda')
            logits = predict(model, data, ind, False)[3].sigmoid().cpu().numpy()
            for i, value in zip(part, logits):
                row = data['rows'][int(i)]
                grouped[row['episode']].append((row['time_s'], bool(row['handoff']), float(value)))
    result = {'early': 0, 'correct': 0, 'late': 0, 'missed': 0}
    for rows in grouped.values():
        rows.sort()
        onset = min(t for t, truth, _ in rows if truth)
        end = max(t for t, truth, _ in rows if truth)
        run = 0
        trigger = None
        for t, _, prob in rows:
            run = run+1 if prob >= threshold else 0
            if run >= streak:
                trigger = t
                break
        if trigger is None: result['missed'] += 1
        elif trigger < onset: result['early'] += 1
        elif trigger <= end: result['correct'] += 1
        else: result['late'] += 1
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--actor', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=1200)
    a = p.parse_args()
    if a.output.exists(): p.error('Preserve existing output')
    a.output.mkdir(parents=True)
    random.seed(20261001); np.random.seed(20261001); torch.manual_seed(20261001)
    torch.set_num_threads(8)
    saved = torch.load(a.actor, map_location='cpu', weights_only=False)
    manifest = dict(saved['source_manifest'])
    if not manifest.get('zero_previous_command') or not manifest.get('long_history'):
        p.error('Expected zero-previous long-history actor')
    data = build_data(a.cache, a.base, 'cuda', LONG_AGES)
    data['prev'].zero_()
    model = TemporalTransport(data['mean'], data['std'], raw_state_skip=True,
                              feature_dim=data['features'].shape[1], state_task_dim=42,
                              route_conditioning=False).cuda()
    model.load_state_dict(saved['model'])
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith('handoff.'))
    opt = torch.optim.AdamW(model.handoff.parameters(), lr=1e-3, weight_decay=1e-3)
    rows = data['rows']
    positive = np.asarray([i for i, r in enumerate(rows) if r['split']=='train' and
                           r['family']=='sc' and r['handoff']], dtype=np.int64)
    late_negative = np.asarray([i for i, r in enumerate(rows) if r['split']=='train' and
                                r['family']=='sc' and r['stage']==5 and not r['handoff']], dtype=np.int64)
    other_negative = np.asarray([i for i, r in enumerate(rows) if r['split']=='train' and
                                 r['family']=='sc' and r['stage']!=5], dtype=np.int64)
    val = np.asarray([i for i, r in enumerate(rows) if r['split']=='validation' and
                      r['family']=='sc'], dtype=np.int64)
    manifest.update({'handoff_aux_loss': 1., 'handoff_training':
        'frozen action/temporal backbone; only linear handoff head trained on SC labels',
        'handoff_train_counts': {'positive': len(positive), 'stage5_negative': len(late_negative),
                                 'other_negative': len(other_negative)},
        'handoff_validation_rows': len(val), 'handoff_update_cap': a.updates,
        'handoff_stop_calibration': {'threshold': .95, 'streak': 2},
        'handoff_actor_checkpoint': str(a.actor)})
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    best = float('inf'); best_update = 0; history = []
    for update in range(1, a.updates+1):
        selection = np.r_[np.random.choice(positive, 24),
                          np.random.choice(late_negative, 24),
                          np.random.choice(other_negative, 16)]
        ind = torch.as_tensor(selection, device='cuda')
        model.eval(); opt.zero_grad(set_to_none=True)
        logits = predict(model, data, ind, False)[3]
        truth = data['handoff'][ind]
        loss = nn.functional.binary_cross_entropy_with_logits(logits, truth)
        loss.backward(); opt.step()
        if update%100==0 or update==a.updates:
            metrics = event_metrics(model, data, val)
            # A false early stop is worse than a missed handoff.
            score = metrics['early']*100+metrics['missed']+metrics['late']
            history.append({'update': update, 'train_loss': float(loss.detach()),
                            'event_metrics': metrics, 'score': score})
            print(json.dumps(history[-1]), flush=True)
            if score < best:
                best = score; best_update = update
                torch.save({'model': model.state_dict(), 'update': update,
                    'history_enabled': True, 'source_manifest': manifest,
                    'final_scenes_opened': False, 'rl_started': False}, a.output/'best.pt')
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps({'best_update': best_update,
        'best_event_metrics': next(r['event_metrics'] for r in history if r['update']==best_update),
        'train_episodes': manifest['train_episodes'],
        'validation_episodes': manifest['validation_episodes'],
        'action_weights_frozen': True, 'final_scenes_opened': False},indent=2)+'\n')


if __name__ == '__main__': main()
