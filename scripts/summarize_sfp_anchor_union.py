#!/usr/bin/env python3
"""Read-only matched audit of frozen SFP anchor predictions."""
import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


def read(path):
    rows = [json.loads(line) for line in path.open()]
    return {(r['episode'], r['time_s']): r for r in rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', action='append', nargs=2, metavar=('NAME', 'JSONL'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fresh-scenes', action='store_true',
                   help='Model weights were frozen before these independent reset groups')
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve an existing audit')
    models = {name: read(Path(path)) for name, path in a.model}
    if len(models) != len(a.model):
        p.error('Model names must be unique')
    keys = sorted(set().union(*(rows.keys() for rows in models.values())))
    episode_times = defaultdict(list)
    for episode, time_s in keys:
        episode_times[episode].append(time_s)
    rows = []
    conflicts = []
    for episode, times in sorted(episode_times.items()):
        start = min(times)
        accepted = []
        all_accepted = []
        for time_s in sorted(times):
            options = [(name, models[name].get((episode, time_s))) for name in models]
            options = [(name, row) for name, row in options if row is not None and row['accepted']]
            for name, row in options:
                all_accepted.append((time_s, name, row))
            for (name_a, row_a), (name_b, row_b) in combinations(options, 2):
                gap = float(np.linalg.norm(np.asarray(row_a['predicted_base_m']) - np.asarray(row_b['predicted_base_m'])) * 1000)
                if gap > 10:
                    conflicts.append({'episode': episode, 'time_s': time_s,
                                      'models': [name_a, name_b], 'gap_mm': gap})
            if options:
                accepted.append((time_s, options))
        wrong = [(time_s, name, row.get('nearest_identity_label_only'), row['requested'])
                 for time_s, name, row in all_accepted
                 if row.get('nearest_identity_label_only') != row['requested']]
        rows.append({'episode': episode, 'requested': next((r['requested'] for m in models.values()
                     for (ep, _), r in m.items() if ep == episode)),
                     'first_sample_s': start,
                     'first_any_accept_s_from_start': accepted[0][0] - start if accepted else None,
                     'early_within_3s': bool(accepted and accepted[0][0] - start <= 3),
                     'any_accept': bool(accepted),
                     'first_accept_models': [name for name, _ in accepted[0][1]] if accepted else [],
                     'accepted_wrong_count': len(wrong)})
    summary = {'schema': 'sfp_anchor_union_frozen_audit/v1',
               'models': {name: str(path) for name, path in a.model},
               'episode_count': len(rows),
               'episodes_early_within_3s': sum(r['early_within_3s'] for r in rows),
               'episodes_any_accept': sum(r['any_accept'] for r in rows),
               'accepted_wrong_count': sum(r['accepted_wrong_count'] for r in rows),
               'cross_model_pairs_more_than_10mm': len(conflicts),
               'cross_model_conflicts': conflicts,
               'episodes': rows,
               'selection_caveat': (
                   'Frozen checkpoints on independent non-final reset groups; complete live latency and autonomous handoff still require checks.'
                   if a.fresh_scenes else
                   'Some checkpoints were selected on these development episodes; union is exploratory, not independent validation.')}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('episodes', 'cross_model_conflicts')}))


if __name__ == '__main__':
    main()
