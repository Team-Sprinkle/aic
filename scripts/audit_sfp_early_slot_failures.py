#!/usr/bin/env python3
"""Summarize early SFP request identity on frozen saved predictions.

Projection is a geometric availability diagnostic, not visual visibility.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path


CAMERAS = ('center', 'left', 'right')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve previous audit')
    labels = defaultdict(list)
    for row in map(json.loads, (a.labels / 'rows.jsonl').open()):
        labels[row['episode']].append(row)
    predictions = {(r['episode'], r['time_s']): r for r in
                   map(json.loads, a.predictions.open())}
    report = []
    for episode, rows in sorted(labels.items()):
        rows.sort(key=lambda r: r['time_s'])
        t0 = rows[0]['time_s']
        early = [r for r in rows if r['time_s'] - t0 <= 3]
        selected = rows[0]['selected_identity']
        projected = 0
        accepted = 0
        for row in early:
            target = next(t for t in row['targets'] if t['identity'] == selected)
            projected += all(target['cameras'][c][3] for c in CAMERAS)
            pred = predictions[(row['episode'], row['time_s'])]
            accepted += bool(pred['accepted'] and
                             pred['nearest_identity_label_only'] == selected)
        report.append(dict(episode=episode, requested=selected,
                           early_rows=len(early), projected_all_three=projected,
                           accepted_correct=accepted))
    result = dict(schema='sfp_early_slot_failure_audit/v1', episodes=len(report),
                  rows=report, early_correct_episodes=sum(r['accepted_correct'] > 0 for r in report),
                  geometry_used_for_posthoc_projection_only=True,
                  projection_is_not_visual_visibility=True,
                  final_scenes_opened=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(episodes=result['episodes'],
                          early_correct=result['early_correct_episodes'])))


if __name__ == '__main__':
    main()
