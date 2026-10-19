#!/usr/bin/env python3
"""Measure distance-sampled visual-history spacing on saved SFP teacher labels."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


CAMERAS = ('center', 'left', 'right')
STEPS_MM = (20, 30, 40, 50)
HISTORY_SLOTS = (6, 8, 10, 12)
LAGS_S = (1.5, 3, 6)


def percentile(values, q):
    return round(float(np.percentile(values, q)), 2) if values else None


def selected_pixels(row):
    target = next(t for t in row['targets'] if t['identity'] == row['selected_identity'])
    return {cam: target['cameras'][cam] for cam in CAMERAS}


def image_vector(path, cache):
    if path not in cache:
        with Image.open(path) as im:
            cache[path] = np.asarray(im.convert('L').resize((48, 48)), np.float32) / 255.
    return cache[path]


def pair_metrics(first, second, image_cache):
    shifts = []
    a, b = selected_pixels(first), selected_pixels(second)
    for cam in CAMERAS:
        p, q = a[cam], b[cam]
        if p[3] and q[3] and 0 <= p[0] < 1152 and 0 <= q[0] < 1152 and 0 <= p[1] < 1024 and 0 <= q[1] < 1024:
            shifts.append(math.hypot(p[0]-q[0], p[1]-q[1]))
    visual = [float(np.mean(np.abs(image_vector(first['images'][cam], image_cache) -
                                    image_vector(second['images'][cam], image_cache))))
              for cam in ('center', 'right')]
    return {'max_target_pixel_shift': max(shifts) if shifts else None,
            'mean_gray_l1': float(np.mean(visual)),
            'time_gap_s': second['time_s']-first['time_s']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--labels', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Preserve previous audit output')
    by_episode = defaultdict(list)
    for line in (args.labels/'rows.jsonl').open():
        row = json.loads(line)
        if row['family'] == 'sfp':
            by_episode[row['episode']].append(row)
    if len(by_episode) != 10:
        raise ValueError(f'Expected ten complete scene groups, found {len(by_episode)}')
    results = {str(step): {'pairs': [], 'coverage_mm': [], 'coverage_s': [],
                           'coverage_by_slots': {str(k): [] for k in HISTORY_SLOTS},
                           'selected_anchor_counts': [], 'episode_route_mm': []}
               for step in STEPS_MM}
    fixed = {str(lag): [] for lag in LAGS_S}
    still = []
    cache = {}
    for episode, rows in by_episode.items():
        rows.sort(key=lambda r:r['time_s'])
        xyz = np.asarray([r['state'][:3] for r in rows], float)
        cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))] * 1000
        times = np.asarray([r['time_s'] for r in rows], float)
        for i in range(1, len(rows)):
            if cumulative[i]-cumulative[i-1] < 1:
                still.append(times[i]-times[i-1])
        for lag in LAGS_S:
            for i in range(1, len(rows)):
                j = int(np.searchsorted(times, times[i]-lag, side='right')-1)
                if j >= 0 and i > j:
                    fixed[str(lag)].append(float(cumulative[i]-cumulative[j]))
        for step in STEPS_MM:
            key = str(step)
            anchors = [0]
            for i in range(1, len(rows)):
                if cumulative[i]-cumulative[anchors[-1]] >= step:
                    anchors.append(i)
            if anchors[-1] != len(rows)-1:
                anchors.append(len(rows)-1)
            results[key]['selected_anchor_counts'].append(len(anchors))
            results[key]['episode_route_mm'].append(float(cumulative[-1]))
            for first, second in zip(anchors, anchors[1:]):
                metric = pair_metrics(rows[first], rows[second], cache)
                metric.update(episode=episode, distance_mm=float(cumulative[second]-cumulative[first]))
                results[key]['pairs'].append(metric)
            for n, i in enumerate(anchors):
                for k in HISTORY_SLOTS:
                    first_k = anchors[max(0,n-k+1)]
                    results[key]['coverage_by_slots'][str(k)].append(float(cumulative[i]-cumulative[first_k]))
                first = anchors[max(0,n-9)]
                results[key]['coverage_mm'].append(float(cumulative[i]-cumulative[first]))
                results[key]['coverage_s'].append(float(times[i]-times[first]))
    summary = {'schema':'sfp_distance_history_offline_audit/v1',
               'source_labels':str(args.labels), 'episodes':len(by_episode),
               'history_slots':10, 'candidate_step_mm':list(STEPS_MM),
               'history_slot_candidates':list(HISTORY_SLOTS),
               'time_lag_actual_distance_mm':{key:{'p10':percentile(x,10),'median':percentile(x,50),'p90':percentile(x,90)}
                                              for key,x in fixed.items()},
               'sub_1mm_between_half_second_rows':len(still),
               'total_half_second_pairs':sum(len(v)-1 for v in by_episode.values()),
               'distance':{},
               'simulator_port_pixels_diagnostic_only':True,
               'images_original_unmodified':True,
               'autonomous_route_not_evaluated':True,
               'final_scenes_opened':False}
    for step, data in results.items():
        pairs = data['pairs']; shifts=[p['max_target_pixel_shift'] for p in pairs if p['max_target_pixel_shift'] is not None]
        summary['distance'][step] = {'anchor_pairs':len(pairs),
            'anchors_per_episode_median':percentile(data['selected_anchor_counts'],50),
            'episode_route_mm_median':percentile(data['episode_route_mm'],50),
            'ten_slot_route_span_mm_median':percentile(data['coverage_mm'],50),
            'ten_slot_route_span_mm_p90':percentile(data['coverage_mm'],90),
            'route_span_by_slots_mm':{k:{'median':percentile(v,50),'p90':percentile(v,90)} for k,v in data['coverage_by_slots'].items()},
            'ten_slot_time_span_s_median':percentile(data['coverage_s'],50),
            'adjacent_target_pixel_shift_p50':percentile(shifts,50),
            'adjacent_target_pixel_shift_p90':percentile(shifts,90),
            'adjacent_target_pixel_shift_p95':percentile(shifts,95),
            'adjacent_target_pixel_shift_over_160_fraction':round(sum(v>160 for v in shifts)/len(shifts),4) if shifts else None,
            'adjacent_gray_l1_median':percentile([p['mean_gray_l1'] for p in pairs],50),
            'adjacent_time_gap_s_p95':percentile([p['time_gap_s'] for p in pairs],95)}
    args.output.mkdir(parents=True)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (args.output/'anchor_pairs.jsonl').open('w') as f:
        for step,data in results.items():
            for row in data['pairs']:
                f.write(json.dumps({'step_mm':int(step),**row},separators=(',',':'))+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
