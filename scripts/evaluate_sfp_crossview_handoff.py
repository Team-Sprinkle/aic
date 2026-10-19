#!/usr/bin/env python3
"""Score a fixed, observation-only SFP two-view identity and geometry gate.

The candidate pixels and native-crop scores are frozen predictions. Scored
opening labels are read only after the accept decision, for grouped reporting.
The threshold is fixed from fitting episodes before either development split.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate, reprojection_residual_px


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reprojection-threshold-px', type=float, default=25.)
    args = p.parse_args()
    if args.output.exists():
        p.error('Preserve prior output')
    rows = [json.loads(s) for s in args.labels.open()]
    manifest = [json.loads(s) for s in args.manifest.open()]
    predictions = [json.loads(s) for s in args.predictions.open()]
    offsets = {c: np.asarray(v) for c, v in
               json.loads(args.calibration.read_text())['tcp_to_optical'].items()}
    by_row_camera = {(r['row'], r['camera']): r for r in manifest}
    groups = defaultdict(list)
    for pred in predictions:
        rec = manifest[pred['record']]
        if rec['camera'] != pred['camera']:
            raise ValueError('Prediction and manifest camera mismatch')
        groups[(rec['episode'], rec['split'], rec['row'],
                pred['request_port'], pred['candidate_slot'])].append(pred)
    selection_episodes = {f'card{card}_port{card % 2}_cards3_validation'
                          for card in range(5)}
    records = []
    for (episode, split, row, request, slot), preds in sorted(groups.items()):
        if len(preds) != 3 or {q['camera'] for q in preds} != set(offsets):
            continue
        subset = ('fit' if split == 'train' else
                  'checkpoint_selection' if episode in selection_episodes else
                  'independent_development')
        cameras = {c: pose_matrix(rows[row]['state'][:7]) @ offset
                   for c, offset in offsets.items()}
        pixels = {q['camera']: by_row_camera[row, q['camera']]['candidate_xy'][slot]
                  for q in preds}
        try:
            point = triangulate(pixels, cameras)
            residual = reprojection_residual_px(point, pixels, cameras)
        except np.linalg.LinAlgError:
            residual = float('inf')
        votes = sum(q['crop_score'] >= .5 for q in preds)
        baseline = votes == 3
        accepted = votes >= 2 and residual <= args.reprojection_threshold_px
        truth = all(q['label'] == 1 for q in preds)
        records.append({'episode': episode, 'split': subset, 'row': row,
                        'requested_port': request, 'candidate_slot': slot,
                        'positive_posthoc': truth, 'native_crop_votes': votes,
                        'reprojection_residual_px': residual,
                        'baseline_all_views_accept': baseline,
                        'crossview_accept': accepted,
                        'candidate_pixels': pixels})
    report = {'schema': 'sfp_crossview_handoff_gate/v1',
              'runtime_inputs': ['RGB-predicted candidate pixels',
                                 'native-crop identity probabilities',
                                 'measured TCP', 'fixed camera calibration'],
              'runtime_true_port_geometry': False,
              'fixed_reprojection_threshold_px': args.reprojection_threshold_px,
              'minimum_positive_camera_votes': 2,
              'subsets': {}, 'records': records, 'final_scenes_opened': False}
    for split in ('fit', 'checkpoint_selection', 'independent_development'):
        subset = [r for r in records if r['split'] == split]
        truth = [r for r in subset if r['positive_posthoc']]
        wrong = [r for r in subset if not r['positive_posthoc']]
        report['subsets'][split] = {
            'episodes': len({r['episode'] for r in subset}),
            'positive_triplets': len(truth), 'negative_triplets': len(wrong),
            'baseline_tp': sum(r['baseline_all_views_accept'] for r in truth),
            'baseline_fp': sum(r['baseline_all_views_accept'] for r in wrong),
            'crossview_tp': sum(r['crossview_accept'] for r in truth),
            'crossview_fp': sum(r['crossview_accept'] for r in wrong),
            'positive_reprojection_p95_px': float(np.percentile(
                [r['reprojection_residual_px'] for r in truth], 95)) if truth else None,
            'positive_reprojection_max_px': max(
                (r['reprojection_residual_px'] for r in truth), default=None),
            'accepted_per_episode': dict(Counter(
                r['episode'] for r in subset if r['crossview_accept']))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['subsets'], indent=2))


if __name__ == '__main__':
    main()
