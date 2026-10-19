#!/usr/bin/env python3
"""Posthoc SC two/three-view geometry audit on existing scored RGB frames."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from evaluate_sc_native_triangulation import triangulate, reprojection_residual_px


def stats(rows, key):
    v = np.asarray([r[key] for r in rows], dtype=float)
    return dict(n=len(v), p50=float(np.median(v)), p95=float(np.percentile(v, 95))) if len(v) else dict(n=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--labels', nargs='+', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve previous result')
    optics = {c: np.asarray(m, dtype=float) for c, m in
              json.loads(a.calibration.read_text())['tcp_to_optical'].items()}
    predicted = {(r['image'], r['camera'], int(r['target_port'])): r
                 for r in json.loads(a.audit.read_text())['records']}
    groups = defaultdict(dict)
    for path in a.labels:
        for line in path.open():
            label = json.loads(line)
            key = (label['image'], label['camera'], int(label['target_port']))
            if key in predicted:
                groups[(label['episode_id'], label['sim_ns'])][label['camera']] = \
                    (label, predicted[key])
    rows = []
    for (episode, sim_ns), group in groups.items():
        if not all(c in group for c in ('center', 'left', 'right')):
            continue
        pixels = {c: group[c][1]['predicted_selected_xy_native'] for c in group}
        truth = np.asarray(group['center'][0]['relative_opening_pose_tcp'][:3])
        record = dict(episode=episode, sim_ns=sim_ns,
                      exposed={c: bool(group[c][0]['geometric_visible_fraction'] >= .7)
                               for c in group})
        for name, subset in [('center_right', ('center', 'right')),
                             ('center_left', ('center', 'left')),
                             ('left_right', ('left', 'right')),
                             ('all_three', ('center', 'left', 'right'))]:
            estimate = triangulate({c: pixels[c] for c in subset}, optics)
            record[name + '_error_mm'] = float(np.linalg.norm(estimate-truth)*1000)
            if name == 'center_right':
                record['left_reprojection_residual_px'] = reprojection_residual_px(
                    estimate, pixels, optics)
        rows.append(record)
    both = [r for r in rows if r['exposed']['center'] and r['exposed']['right']]
    all_visible = [r for r in both if r['exposed']['left']]
    gate = [r for r in both if r['left_reprojection_residual_px'] <= 10]
    report = dict(schema='aic_sc_multiview_rejection_diagnostic/v1',
                  all_triplets=len(rows), center_right_exposed=len(both),
                  all_three_exposed=len(all_visible),
                  center_right_error_mm=stats(both, 'center_right_error_mm'),
                  center_left_error_mm=stats(both, 'center_left_error_mm'),
                  all_three_error_mm=stats(both, 'all_three_error_mm'),
                  exploratory_gate_px=10,
                  exploratory_gate_accepted=len(gate),
                  exploratory_gate_rejected=len(both)-len(gate),
                  exploratory_gate_accepted_error_mm=stats(gate, 'center_right_error_mm'),
                  exploratory_gate_rejected_left_hidden=sum(not r['exposed']['left']
                                                            for r in both if r not in gate),
                  rows=rows, geometry_only_posthoc=True,
                  threshold_not_promoted=True, final_scenes_opened=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in report if k != 'rows'}))


if __name__ == '__main__':
    main()
