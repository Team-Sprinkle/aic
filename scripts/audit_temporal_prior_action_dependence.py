#!/usr/bin/env python3
"""Measure held-out action sensitivity to the prior-command input."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from train_temporal_transport_probe import (LONG_AGES, AGES, TemporalTransport,
                                             build_data, predict)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(8)
    saved = torch.load(a.model, map_location='cpu', weights_only=False)
    manifest = saved['source_manifest']
    data = build_data(a.cache, a.base, 'cuda', LONG_AGES if manifest['long_history'] else AGES)
    model = TemporalTransport(data['mean'], data['std'], raw_state_skip=True,
                              feature_dim=data['features'].shape[1], state_task_dim=42,
                              route_conditioning=manifest['route_conditioning']).cuda().eval()
    model.load_state_dict(saved['model'])
    groups = defaultdict(list)
    for i, row in enumerate(data['rows']):
        if row['split'] == 'validation':
            groups[(row['family'], row['episode'], row['stage'])].append(i)
    chosen = []
    for (family, _, _), items in groups.items():
        chosen += np.asarray(items)[np.linspace(0, len(items)-1,
                          min(8 if family == 'sc' else 12, len(items))).astype(int)].tolist()
    chosen = np.asarray(sorted(set(chosen)), dtype=np.int64)
    original_prev = data['prev'].clone()
    results = {}
    with torch.inference_mode():
        for condition in ('teacher_previous', 'zero_previous'):
            if condition == 'zero_previous':
                data['prev'].zero_()
            records = defaultdict(list)
            for indices in np.array_split(chosen, max(1, (len(chosen)+255)//256)):
                ind = torch.as_tensor(indices, device='cuda')
                prediction = predict(model, data, ind, not saved['history_enabled'])[0]
                xyz = (prediction[:, 0] * model.target_std + model.target_mean)[:, :3]
                mm = ((xyz - data['target'][ind, 0, :3]).norm(dim=-1)*1000).cpu().numpy()
                for index, error in zip(indices, mm):
                    row = data['rows'][int(index)]
                    records[(row['family'], row['stage'])].append(float(error))
            results[condition] = {f'{family}_stage{stage}': {'n': len(values),
                     'median_mm': float(np.median(values)), 'p95_mm': float(np.quantile(values, .95))}
                   for (family, stage), values in records.items()}
            results[condition]['overall'] = {family: {'n': sum(len(v) for (f, _), v in records.items() if f == family),
                'p95_mm': float(np.quantile([e for (f, _), v in records.items() if f == family for e in v], .95))}
                for family in ('sc', 'sfp')}
    data['prev'] = original_prev
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({'schema': 'aic_temporal_prior_action_audit/v1',
        'checkpoint': str(a.model), 'validation_rows': len(chosen), 'results': results,
        'final_scenes_opened': False}, indent=2)+'\n')
    print(json.dumps({k: v['overall'] for k, v in results.items()}, indent=2))


if __name__ == '__main__':
    main()
