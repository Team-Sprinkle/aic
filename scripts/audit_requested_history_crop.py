#!/usr/bin/env python3
"""Matched read-only history ablation and geometry gate for requested crops."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate, reprojection_residual_px
from train_requested_history_crop import (HistoryCrop, SpatialHistoryCrop, evaluate,
                                          rows_from, schedule)
from train_two_family_port_locator import CAMERAS


def read_jsonl(path):
    return [json.loads(s) for s in path.open()]


def summarize(predictions, labels, optics, source):
    lookup = {(r['episode'], round(r['time_s'], 3)): r for r in labels}
    by_episode = defaultdict(list)
    accepted = []
    wrong = []
    contained = []
    for p in predictions:
        row = lookup[(p['episode'], round(p['time_s'], 3))]
        if source == 'history':
            xy = {c: p['prediction_px'][j] for j, c in enumerate(CAMERAS)}
            probability = p['probability']
        else:
            xy = {c: p['views'][c]['xy'] for c in CAMERAS}
            probability = [p['views'][c]['prob'] for c in CAMERAS]
        accepted_flag = False
        reproj = None
        if min(probability) >= .5:
            cameras = {c: pose_matrix(row['state']) @ optics[c] for c in CAMERAS}
            try:
                point = triangulate(xy, cameras)
                reproj = float(reprojection_residual_px(point, xy, cameras))
                accepted_flag = bool(np.isfinite(point).all() and reproj <= 5.)
            except np.linalg.LinAlgError:
                pass
        rec = {'episode': p['episode'], 'time_s': p['time_s'],
               'accepted': accepted_flag, 'reprojection_px': reproj}
        by_episode[p['episode']].append(rec)
        if not accepted_flag:
            continue
        accepted.append(rec)
        target = next(t for t in row['targets'] if t['identity'] == row['selected_identity'])
        err = []
        alt = []
        for c in CAMERAS:
            label = target['cameras'][c]
            err.append(float(np.linalg.norm(np.asarray(xy[c]) - label[:2])) if label[3]
                       else float('inf'))
            alt.append(min((float(np.linalg.norm(np.asarray(xy[c]) - t['cameras'][c][:2]))
                            for t in row['targets'] if t['identity'] != row['selected_identity']
                            and t['cameras'][c][3]), default=float('inf')))
        if any(a < e for a, e in zip(alt, err)):
            wrong.append(rec)
        contained.append(bool(max(err) <= 60.))
    early = 0
    any_episode = 0
    per_episode = []
    for episode, items in sorted(by_episode.items()):
        items.sort(key=lambda r: r['time_s'])
        first = next((r for r in items if r['accepted']), None)
        any_episode += first is not None
        early += bool(first and first['time_s'] - items[0]['time_s'] <= 3.)
        per_episode.append({'episode': episode, 'first_accept_s':
                            first['time_s'] - items[0]['time_s'] if first else None})
    return {'rows': len(predictions), 'episodes': len(by_episode), 'accepted': len(accepted),
            'wrong_port_any_view_posthoc': len(wrong),
            'all_view_160px_crops_with_20px_margin': sum(contained),
            'all_view_crop_fraction': float(np.mean(contained)) if contained else 0.,
            'early_episodes_3s': early, 'any_episodes': any_episode,
            'per_episode': per_episode}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--stress-labels', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--selected-stress', type=Path, required=True)
    args = parser.parse_args()
    rows = rows_from(args.labels) + rows_from(args.stress_labels)
    count_main = len(rows_from(args.labels))
    dev = [i for i, r in enumerate(rows[:count_main]) if r['split'] == 'validation']
    stress = list(range(count_main, len(rows)))
    manifest = json.loads((args.run / 'manifest.json').read_text())
    feature_dir = Path(manifest.get('feature_cache', args.run))
    features = np.load(feature_dir / 'features.npy', mmap_mode='r')
    checkpoint = torch.load(args.run / 'best.pt', map_location='cpu', weights_only=False)
    model = (SpatialHistoryCrop(coordinate_output='spatial_history_xy' in checkpoint['architecture'])
             if 'spatial_history' in checkpoint['architecture']
             else HistoryCrop()).cuda().eval()
    model.load_state_dict(checkpoint['model'])
    original = schedule(rows)
    no_history = {i: [i] * 4 for i in range(len(rows))}
    calibration = json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics = {c: np.asarray(calibration['tcp_to_optical'][c], float) for c in CAMERAS}
    report = {}
    for name, indices in [('development', dev), ('stress', stress)]:
        labels = [rows[i] for i in indices]
        report[name] = {}
        for mode, times in [('history', original), ('current_only', no_history)]:
            path = args.run / f'{name}_{mode}_ablation.jsonl'
            evaluate(model, rows, features, times, indices, 'cuda', path)
            report[name][mode] = summarize(read_jsonl(path), labels, optics, 'history')
    selected = read_jsonl(args.selected_stress)
    report['stress']['frozen_selected_only'] = summarize(selected, rows[count_main:], optics, 'selected')
    report['history_effect_stress_wrong_delta'] = (report['stress']['history']['wrong_port_any_view_posthoc'] -
                                                    report['stress']['current_only']['wrong_port_any_view_posthoc'])
    report['final_scenes_opened'] = False
    (args.run / 'matched_geometry_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
