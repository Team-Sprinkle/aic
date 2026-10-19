#!/usr/bin/env python3
"""Score whether frozen RGB candidate sets contain the requested opening.

This is a diagnostic ceiling, never a runtime selector: true positions are read
only after the saved candidate predictions and are used only for scoring.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(rows, threshold_mm):
    groups = defaultdict(list)
    for row in rows:
        groups[row['scene_group']].append(row)
    episodes = []
    for scene, items in sorted(groups.items()):
        items.sort(key=lambda x: x['time_s'])
        start = items[0]['time_s']
        early = [x for x in items if x['time_s'] - start <= 3]
        episodes.append({
            'scene_group': scene,
            'request': items[0]['requested'],
            'early_rows': len(early),
            'early_any_named_correct': any(x['named_correct'] for x in early),
            'early_any_set_contains_requested': any(x['set_contains_requested'] for x in early),
            'early_any_two_distinct_card_slots': any(x['distinct_true_card_slots_covered'] >= 2 for x in early),
            'early_max_distinct_card_slots': max(x['distinct_true_card_slots_covered'] for x in early),
            'early_min_requested_set_error_mm': min(
                (x['requested_set_error_mm'] for x in early
                 if x['requested_set_error_mm'] is not None), default=None),
        })
    return {
        'threshold_mm': threshold_mm,
        'episodes': len(episodes),
        'early_named_correct': sum(x['early_any_named_correct'] for x in episodes),
        'early_set_contains_requested': sum(x['early_any_set_contains_requested'] for x in episodes),
        'early_two_distinct_card_slots': sum(x['early_any_two_distinct_card_slots'] for x in episodes),
        'by_episode': episodes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threshold-mm', type=float, default=20.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve prior output')
    if not 0 < args.threshold_mm <= 50:
        parser.error('threshold must be in (0, 50] mm')
    rows = []
    with args.candidates.open() as source:
        for line in source:
            raw = json.loads(line)
            truth = raw['true_positions_label_only']
            requested = raw['requested']
            if requested not in truth:
                raise ValueError(f'Missing request truth: {requested}')
            points = {}
            for name, candidate in raw['candidates'].items():
                point = candidate.get('predicted_base_m')
                if candidate.get('three_view_consistent') and point is not None:
                    points[name] = np.asarray(point, dtype=float)
            target = np.asarray(truth[requested], dtype=float)
            distance = {name: float(np.linalg.norm(point - target) * 1000)
                        for name, point in points.items()}
            requested_set_error = min(distance.values(), default=None)
            covered_cards = set()
            for name, position in truth.items():
                exact = np.asarray(position, dtype=float)
                if any(np.linalg.norm(point - exact) * 1000 <= args.threshold_mm
                       for point in points.values()):
                    covered_cards.add(int(name.split('_')[2]))
            rows.append({
                'scene_group': raw['scene_group'], 'episode': raw['episode'],
                'time_s': raw['time_s'], 'requested': requested,
                'candidate_count': len(points),
                'named_correct': (distance.get(requested) is not None and
                                  distance[requested] <= args.threshold_mm),
                'set_contains_requested': (requested_set_error is not None and
                                           requested_set_error <= args.threshold_mm),
                'requested_named_error_mm': distance.get(requested),
                'requested_set_error_mm': requested_set_error,
                'nearest_candidate_channel_label_only': min(distance, key=distance.get) if distance else None,
                'distinct_true_card_slots_covered': len(covered_cards),
            })
    result = summarize(rows, args.threshold_mm)
    result.update(schema='sfp_candidate_set_ceiling/v1',
                  source=str(args.candidates),
                  note='True positions and nearest-channel labels are posthoc scoring only; this is not a deployable selector.',
                  final_scenes_opened=False)
    args.output.mkdir(parents=True)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (args.output / 'rows.jsonl').open('w') as target:
        for row in rows:
            target.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('episodes', 'early_named_correct',
          'early_set_contains_requested', 'early_two_distinct_card_slots')}, indent=2))


if __name__ == '__main__':
    main()
