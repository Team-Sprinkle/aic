#!/usr/bin/env python3
"""Build an unlabeled original-RGB review batch; never infer visibility from projection."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np


def spread(rows, count):
    if not rows:
        return []
    indices = np.linspace(0, len(rows) - 1, min(count, len(rows)), dtype=int)
    return [rows[int(i)] for i in indices]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sfp', type=Path, required=True)
    p.add_argument('--sc', type=Path, required=True)
    p.add_argument('--sc-geometry', nargs='+', type=Path, required=True,
                   help='Independent projected aperture records for review centering only')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Keep old batch intact; use a new output path')
    a.output.mkdir(parents=True)
    sfp = [json.loads(x) for x in a.sfp.open()]
    sc = json.loads(a.sc.read_text())['records']
    geometry = {}
    for path in a.sc_geometry:
        for line in path.open():
            row = json.loads(line)
            points = np.asarray(row['projected_opening_px'], dtype=float)
            geometry[(row['image'], row['camera'], row['target_port'])] = points.mean(0).tolist()
    sfp_groups = defaultdict(list)
    for row in sfp:
        sfp_groups[row['scene_group']].append(row)
    selected = []
    for group, rows in sorted(sfp_groups.items()):
        # Two route phases from each independent episode; three cameras rotate.
        for j, row in enumerate(spread(sorted(rows, key=lambda r: r['time_s']), 2)):
            cam = ('center', 'left', 'right')[(len(selected) + j) % 3]
            target = next(x for x in row['targets'] if x['identity'] == row['selected_identity'])
            xy = target['cameras'][cam][:2]
            selected.append(dict(family='sfp', episode=row['episode'], scene_group=group,
                                 split=row['split'], camera=cam, image=row['images'][cam],
                                 selected_identity=row['selected_identity'], time_s=row['time_s'],
                                 audit_center_xy=xy, crop_center_source='privileged_projection_review_only',
                                 review_visibility=None))
    selected = spread(selected, 50)
    sc_buckets = defaultdict(list)
    for row in sc:
        category = ('hidden' if row['geometric_visible_fraction'] < .05 else
                    'partial' if row['geometric_visible_fraction'] < .7 else 'exposed')
        sc_buckets[category].append(row)
    for category, cap in [('hidden', 20), ('partial', 10), ('exposed', 20)]:
        for row in spread(sc_buckets[category], cap):
            key = (row['image'], row['camera'], row['target_port'])
            if key not in geometry:
                raise KeyError(f'No projected requested opening for {key}')
            selected.append(dict(family='sc', episode=row['episode'], scene_group=row['scene_group'],
                                 split='development', camera=row['camera'], image=row['image'],
                                 selected_identity=f"sc_port_{row['target_port']}",
                                 audit_center_xy=geometry[key],
                                 crop_center_source='privileged_projection_review_only',
                                 predicted_selected_xy_native=row['predicted_selected_xy_native'],
                                 geometric_exposure_proxy=row['geometric_visible_fraction'],
                                 review_visibility=None))
    # Images in review panels carry no marker or geometry label. The crop center
    # and proxy are metadata, visible only after the initial human review.
    for i, row in enumerate(selected):
        path = Path(row['image'])
        if not path.is_file():
            raise FileNotFoundError(path)
        rgb = cv2.imread(str(path))
        if rgb is None:
            raise ValueError(path)
        x, y = row['audit_center_xy']
        whole = cv2.resize(rgb, (320, 284), interpolation=cv2.INTER_AREA)
        crop = cv2.warpAffine(rgb, np.array([[1, 0, 160 - x], [0, 1, 142 - y]], np.float32),
                              (320, 284), borderMode=cv2.BORDER_CONSTANT)
        panel = np.hstack([whole, crop])
        cv2.putText(panel, f"{i:03d} {row['family']} {row['camera']} {row['selected_identity']}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 255), 2)
        row['panel'] = f'panels/{i:03d}.jpg'
        (a.output/'panels').mkdir(exist_ok=True)
        cv2.imwrite(str(a.output/row['panel']), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with (a.output/'review_manifest.jsonl').open('w') as file:
        for row in selected:
            file.write(json.dumps(row) + '\n')
    summary = dict(schema='aic_selected_visibility_review_batch/v1', rows=len(selected),
                   sfp=sum(r['family'] == 'sfp' for r in selected),
                   sc=sum(r['family'] == 'sc' for r in selected),
                   visibility_labels_certified=False,
                   geometry_never_deployable=True,
                   reserved_final_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
