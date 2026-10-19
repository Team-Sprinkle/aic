#!/usr/bin/env python3
"""Compare frozen RGB locator to audited SC aperture projection, grouped by scene."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def metrics(rows):
    error = np.asarray([r['pixel_error'] for r in rows])
    return dict(n=len(rows), p50_px=float(np.median(error)),
                p95_px=float(np.percentile(error, 95)),
                over_50_px=int(np.sum(error > 50)),
                over_100_px=int(np.sum(error > 100))) if len(error) else dict(n=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--labels', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve existing result')
    source = json.loads(a.audit.read_text())['records']
    geometry = {}
    for path in a.labels:
        for line in path.open():
            row = json.loads(line)
            key = (row['image'], row['camera'], int(row['target_port']))
            if key in geometry:
                raise ValueError(f'Duplicate aperture projection: {key}')
            geometry[key] = np.asarray(row['projected_opening_px'], dtype=float).mean(0)
    rows = []
    for record in source:
        key = (record['image'], record['camera'], int(record['target_port']))
        if key not in geometry:
            raise KeyError(f'Missing independent aperture projection: {key}')
        truth = geometry[key]
        pred = np.asarray(record['predicted_selected_xy_native'], dtype=float)
        visible_proxy = record['geometric_visible_fraction'] >= .7
        inframe_proxy = record['inframe_fraction'] >= .7
        rows.append(dict(episode=record['episode'], scene_group=record['scene_group'],
                         camera=record['camera'], target_port=record['target_port'],
                         image=record['image'], aperture_xy=truth.tolist(),
                         predicted_xy=pred.tolist(),
                         pixel_error=float(np.linalg.norm(pred-truth)),
                         geometric_exposure_proxy=visible_proxy,
                         projected_inframe_proxy=inframe_proxy))
    groups = defaultdict(list)
    for r in rows:
        if r['geometric_exposure_proxy']:
            groups[f"camera/{r['camera']}"].append(r)
            groups[f"episode/{r['episode']}"].append(r)
            groups[f"episode_camera/{r['episode']}/{r['camera']}"].append(r)
    report = dict(schema='aic_sc_locator_geometry_disagreement/v1',
                  audit=str(a.audit), label_sources=[str(x) for x in a.labels],
                  all_rows=metrics(rows), exposed_rows=metrics([r for r in rows if r['geometric_exposure_proxy']]),
                  grouped_exposed={k:metrics(v) for k, v in sorted(groups.items())},
                  rows=rows, true_visibility_certified=False,
                  privileged_projection_used_posthoc_only=True,
                  reserved_final_opened=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:report[k] for k in ('all_rows', 'exposed_rows', 'grouped_exposed')}))


if __name__ == '__main__':
    main()
