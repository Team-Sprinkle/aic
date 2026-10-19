#!/usr/bin/env python3
"""Audit same-reset alternate-request route-branch labels without new training."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def req(row):
    task = np.asarray(row['task'])
    return int(task[4:9].argmax()), int(task[2:4].argmax())


def summarize(rows):
    if not rows:
        return {'rows': 0, 'scenes': 0}
    return {'rows': len(rows), 'scenes': len({x['scene'] for x in rows}),
            'first_action_delta_mm_median': float(np.median([x['action_delta_mm'] for x in rows])),
            'future_one_second_path_delta_mm_median': float(np.median(
                [x['future_path_delta_mm'] for x in rows if x['future_path_delta_mm'] is not None]))
            if any(x['future_path_delta_mm'] is not None for x in rows) else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-time-s', type=float, default=12)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve existing result')
    rows = [json.loads(s) for s in (a.cache / 'rows.jsonl').open()]
    features = np.load(a.cache / 'features.npy', mmap_mode='r')
    scenes = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        if row['family'] == 'sfp' and row['time_s'] <= a.max_time_s:
            scenes[(row['split'], row['scene'])][(row['episode'], req(row))].append(i)
    records = []
    for (split, scene), episodes in scenes.items():
        choices = list(episodes.items())
        if len({key[1] for key, _ in choices}) < 2:
            continue
        for u in range(len(choices)):
            (episode_a, req_a), left = choices[u]
            for v in range(u + 1, len(choices)):
                (episode_b, req_b), right = choices[v]
                if req_a == req_b or episode_a == episode_b:
                    continue
                right_time = np.asarray([rows[j]['time_s'] for j in right])
                left_time = np.asarray([rows[i]['time_s'] for i in left])
                for i in left:
                    t = rows[i]['time_s']
                    j = right[int(np.argmin(np.abs(right_time - t)))]
                    dt = abs(rows[j]['time_s'] - t)
                    if dt > .11:
                        continue
                    r, s = rows[i], rows[j]
                    xyz = np.linalg.norm(np.asarray(r['state'][:3]) - s['state'][:3]) * 1000
                    angle = (Rotation.from_quat(r['state'][3:7]).inv() *
                             Rotation.from_quat(s['state'][3:7])).magnitude() * 180 / np.pi
                    x = np.asarray(features[i], dtype=float)
                    y = np.asarray(features[j], dtype=float)
                    cosine = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
                    action = np.linalg.norm(np.asarray(r['target'][0][:3]) -
                                            s['target'][0][:3]) * 1000
                    ia = left[int(np.argmin(np.abs(left_time - (t + 1))))]
                    ib = right[int(np.argmin(np.abs(right_time - (t + 1))))]
                    future = None
                    if abs(rows[ia]['time_s'] - (t + 1)) <= .2 and abs(rows[ib]['time_s'] - (t + 1)) <= .2:
                        da = np.asarray(rows[ia]['state'][:3]) - r['state'][:3]
                        db = np.asarray(rows[ib]['state'][:3]) - s['state'][:3]
                        future = float(np.linalg.norm(da - db) * 1000)
                    same_observation = bool(xyz <= 2 and angle <= 2 and cosine >= .995)
                    records.append(dict(split=split, scene=scene,
                                        episode_a=episode_a, episode_b=episode_b,
                                        request_a=req_a, request_b=req_b,
                                        index_a=i, index_b=j, time_s=float(t),
                                        time_difference_s=float(dt), tcp_difference_mm=float(xyz),
                                        orientation_difference_deg=float(angle),
                                        image_feature_cosine=cosine,
                                        action_delta_mm=float(action),
                                        future_path_delta_mm=future,
                                        same_observation=same_observation))
    summary = {}
    for split in ('train', 'validation'):
        split_rows = [r for r in records if r['split'] == split]
        same = [r for r in split_rows if r['same_observation']]
        summary[split] = {'aligned': summarize(split_rows), 'same_observation': summarize(same),
                          'same_observation_action_ge_5mm': summarize([r for r in same if r['action_delta_mm'] >= 5]),
                          'same_observation_action_ge_10mm': summarize([r for r in same if r['action_delta_mm'] >= 10]),
                          'same_observation_future_path_ge_10mm': summarize(
                              [r for r in same if r['future_path_delta_mm'] is not None and r['future_path_delta_mm'] >= 10]),
                          'same_observation_future_path_ge_20mm': summarize(
                              [r for r in same if r['future_path_delta_mm'] is not None and r['future_path_delta_mm'] >= 20])}
        first = []
        for scene in {r['scene'] for r in split_rows}:
            match = sorted([r for r in split_rows if r['scene'] == scene and r['tcp_difference_mm'] >= 10],
                           key=lambda r: r['time_s'])
            if match:
                first.append({'scene': scene, 'time_s': match[0]['time_s'],
                              'index_a': match[0]['index_a'], 'index_b': match[0]['index_b'],
                              'request_a': match[0]['request_a'], 'request_b': match[0]['request_b']})
        summary[split]['first_measured_route_gap_ge_10mm'] = first
    result = {'schema': 'aic_sfp_route_branch_support/v1', 'cache': str(a.cache),
              'max_time_s': a.max_time_s,
              'same_observation_rule': {'tcp_mm': 2, 'orientation_deg': 2,
                                        'feature_cosine': .995, 'time_gap_s': .11},
              'summary': summary, 'records': records,
              'geometry_actor_input': False, 'final_scenes_opened': False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({split: {key: (len(value) if isinstance(value, list) else value)
                              for key, value in group.items()}
                      for split, group in summary.items()}, indent=2))


if __name__ == '__main__':
    main()
