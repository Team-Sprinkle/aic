#!/usr/bin/env python3
"""Evaluate a frozen board-reference head on independent episode groups."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_sfp_board_reference import (BoardReference, BoardReferenceLocal,
                                       RailViews, evaluate)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-scenes', type=int, default=10)
    p.add_argument('--split', choices=('all','train','validation'), default='all')
    p.add_argument('--allow-in-sample-cache', action='store_true',
                   help='Only for predicted crop centers on train scenes, never evaluation')
    args = p.parse_args()
    if args.output.exists():
        p.error('Preserve existing output')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    old_rows = [json.loads(line) for line in (Path(state['labels']) / 'rows.jsonl').open()]
    rows = [json.loads(line) for line in (args.labels / 'rows.jsonl').open()]
    if args.split != 'all':
        rows = [row for row in rows if row['split'] == args.split]
    old_groups = {row['scene_group'] for row in old_rows}
    groups = {row['scene_group'] for row in rows}
    if args.allow_in_sample_cache and args.split != 'train':
        raise ValueError('In-sample cache permission applies to training rows only')
    if old_groups & groups and not args.allow_in_sample_cache:
        raise ValueError('Checkpoint fit/development scenes overlap this test')
    if len(groups) != args.expected_scenes:
        raise ValueError(f'Expected {args.expected_scenes} test scenes, got {len(groups)}')
    torch.set_num_threads(4)
    model = (BoardReferenceLocal(state['init']) if state.get('head') == 'slot_local'
             else BoardReference(state['init']))
    model.load_state_dict(state['model'])
    model.cuda().eval()
    loader = DataLoader(RailViews(rows, list(range(len(rows)))), batch_size=6,
                        num_workers=4, pin_memory=True,
                        multiprocessing_context='spawn')
    args.output.mkdir(parents=True)
    metrics = evaluate(model, loader, rows, 'cuda', args.output / 'predictions.jsonl')
    report = {'schema': 'sfp_board_reference_frozen_independent/v1',
              'checkpoint': str(args.checkpoint),
              'head': state.get('head', 'pooled'),
              'checkpoint_update': state['update'],
              'label_manifest': str(args.labels),
              'checkpoint_manifest': state['labels'],
              'scene_overlap': bool(old_groups & groups),
              'allow_in_sample_cache': bool(args.allow_in_sample_cache),
              'geometry_runtime_input': False,
              'final_scenes_opened': False,
              'metrics': metrics}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in metrics.items() if k != 'episode_details'}))


if __name__ == '__main__':
    main()
