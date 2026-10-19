#!/usr/bin/env python3
"""Check frozen one-second route-goal extrapolation beyond its early fit window."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--pose-model', type=Path, required=True)
    p.add_argument('--visual-model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve existing score')
    rows = [json.loads(s) for s in (a.cache / 'rows.jsonl').open()]
    visual = np.load(a.cache / 'features.npy', mmap_mode='r')
    pose_model, visual_model = (dict(np.load(path)) for path in (a.pose_model, a.visual_model))
    episodes = defaultdict(list)
    for i, row in enumerate(rows):
        if row['family'] == 'sfp' and row['split'] == 'validation':
            episodes[row['episode']].append(i)
    records = []
    for episode, ids in episodes.items():
        ids.sort(key=lambda i: rows[i]['time_s'])
        times = np.asarray([rows[i]['time_s'] for i in ids])
        for i in ids:
            r = rows[i]
            j = ids[int(np.argmin(np.abs(times - (r['time_s'] + 1))))]
            if abs(rows[j]['time_s'] - (r['time_s'] + 1)) > .2:
                continue
            state = np.asarray(r['state'])
            task = np.asarray(r['task'])
            truth = np.asarray(rows[j]['state'][:3]) - state[:3]
            item = {'episode': episode, 'scene': r['scene'], 'time_s': r['time_s']}
            for model, name, x in ((pose_model, 'pose', np.r_[state[:7], task]),
                                   (visual_model, 'visual', np.r_[visual[i], state[:7], task])):
                pred = ((x - model['center']) / model['scale']) @ model['coefficients'] + model['target_mean']
                item[name + '_error_mm'] = float(np.linalg.norm(pred - truth) * 1000)
            records.append(item)
    grouped = {}
    for lo, hi in ((0, 1.6), (1.6, 3), (3, 5), (5, 8), (8, 12), (12, 20)):
        part = [r for r in records if lo <= r['time_s'] < hi]
        if not part:
            continue
        grouped[f'{lo:g}-{hi:g}s'] = {'rows': len(part), 'episodes': len({r['episode'] for r in part}),
            **{name + '_error_mm_p50_p95': [float(np.median([r[name + '_error_mm'] for r in part])),
                                            float(np.percentile([r[name + '_error_mm'] for r in part], 95))]
               for name in ('pose', 'visual')}}
    result = {'schema': 'aic_sfp_route_goal_phase_score/v1', 'trained_on_time_s': [0, 1.6],
              'heldout_scene_grouped': True, 'by_phase': grouped,
              'final_scenes_opened': False, 'records': records}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(grouped, indent=2))


if __name__ == '__main__':
    main()
