#!/usr/bin/env python3
"""Freeze 30 mm image-history context for the native SFP crop ablation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_distance_coarse_crop import CoarseCrop, batch, coarse_cache, distance_histories
from train_requested_history_crop import rows_from


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--stress-labels', type=Path, required=True)
    p.add_argument('--feature-cache', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve old cache')
    rows = rows_from(a.labels) + rows_from(a.stress_labels)
    manifest = json.loads((a.feature_cache / 'features_manifest.json').read_text())
    expected = [{'episode': r['episode'], 'time_s': r['time_s'],
                 'scene_group': r['scene_group'], 'images': r['images']} for r in rows]
    if manifest != expected:
        raise ValueError('Feature image/row manifest mismatch')
    full = np.load(a.feature_cache / 'features.npy', mmap_mode='r')
    small = coarse_cache(full, a.feature_cache / 'distance_4x5_features.npy', 'cuda')
    histories = distance_histories(rows)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model = CoarseCrop(history=not state['history_off'],
                       state_conditioned=state['state_conditioned']).cuda().eval()
    model.load_state_dict(state['model'])
    a.output.mkdir(parents=True)
    context = np.lib.format.open_memmap(a.output / 'context.npy', mode='w+',
                                        dtype=np.float16, shape=(len(rows), 3, 128))
    with torch.inference_mode():
        for i in range(0, len(rows), 8):
            chosen = list(range(i, min(i+8, len(rows))))
            inp = batch(rows, full, small, histories, chosen, 'cuda')
            _, _, _, q, _ = model(*inp[:5], state=inp[7], return_context=True)
            context[i:i+len(chosen)] = q.cpu().numpy().astype(np.float16)
    context.flush()
    (a.output / 'manifest.json').write_text(json.dumps(expected, indent=2) + '\n')
    (a.output / 'summary.json').write_text(json.dumps({
        'schema': 'sfp_frozen_distance_history_context/v1',
        'rows': len(rows), 'shape': [len(rows), 3, 128],
        'source_checkpoint': str(a.checkpoint), 'distance_step_mm': 30,
        'history_slots': 12, 'runtime_true_geometry_input': False,
        'final_scenes_opened': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
