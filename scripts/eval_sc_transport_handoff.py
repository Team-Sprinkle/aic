#!/usr/bin/env python3
"""Episode-grouped offline calibration of an observation-only SC stop signal."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from train_temporal_transport_probe import LONG_AGES, AGES, TemporalTransport, build_data, predict


def first_streak(rows, threshold, streak, require_stage):
    run = 0
    for row in rows:
        valid = row['probability'] >= threshold and (not require_stage or row['stage_probability'] >= .7)
        run = run + 1 if valid else 0
        if run >= streak: return row
    return None


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
    if manifest.get('zero_previous_command'): data['prev'].zero_()
    model = TemporalTransport(data['mean'], data['std'], raw_state_skip=True,
                              feature_dim=data['features'].shape[1], state_task_dim=42,
                              route_conditioning=manifest['route_conditioning']).cuda().eval()
    model.load_state_dict(saved['model'])
    indices = np.asarray([i for i, r in enumerate(data['rows'])
                          if r['split'] == 'validation' and r['family'] == 'sc'])
    grouped = defaultdict(list)
    with torch.inference_mode():
        for part in np.array_split(indices, max(1, (len(indices)+255)//256)):
            ind = torch.as_tensor(part, device='cuda')
            _, stage, _, handoff = predict(model, data, ind, not saved['history_enabled'])
            probs = handoff.sigmoid().cpu().numpy()
            stage_probs = stage.softmax(-1)[:, 5].cpu().numpy()
            for i, prob, stage_prob in zip(part, probs, stage_probs):
                row = data['rows'][int(i)]
                grouped[row['episode']].append({'time_s': row['time_s'],
                    'label': bool(row['handoff']), 'probability': float(prob),
                    'stage_probability': float(stage_prob)})
    for rows in grouped.values(): rows.sort(key=lambda r: r['time_s'])
    variants = []
    for threshold in (.5, .7, .8, .9, .95, .98, .99):
        for streak in (1, 2, 3):
            for require_stage in (False, True):
                early = correct = missed = late = 0
                offsets = []
                for rows in grouped.values():
                    onset = min(r['time_s'] for r in rows if r['label'])
                    ending = max(r['time_s'] for r in rows if r['label'])
                    trigger = first_streak(rows, threshold, streak, require_stage)
                    if trigger is None: missed += 1
                    elif trigger['time_s'] < onset:
                        early += 1; offsets.append(float(trigger['time_s']-onset))
                    elif trigger['time_s'] <= ending:
                        correct += 1; offsets.append(float(trigger['time_s']-onset))
                    else: late += 1; offsets.append(float(trigger['time_s']-onset))
                variants.append({'threshold': threshold, 'streak': streak,
                    'require_stage': require_stage, 'early': early, 'correct': correct,
                    'late': late, 'missed': missed,
                    'median_trigger_offset_s': float(np.median(offsets)) if offsets else None})
    report = {'schema': 'aic_sc_transport_handoff_calibration/v1',
              'checkpoint': str(a.model), 'validation_episodes': len(grouped),
              'validation_rows': len(indices), 'variants': variants,
              'final_scenes_opened': False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    for row in sorted(variants, key=lambda x: (x['early'], -x['correct'], x['missed']))[:8]:
        print(row)


if __name__ == '__main__': main()
