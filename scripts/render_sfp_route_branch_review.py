#!/usr/bin/env python3
"""Render original-image paired route review without geometry overlays."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve existing result')
    audit = json.loads(a.audit.read_text())
    cache = [json.loads(line) for line in (a.cache / 'rows.jsonl').open()]
    chosen = []
    for scene in sorted({r['scene'] for r in audit['records'] if r['split'] == 'validation'}):
        candidates = [r for r in audit['records'] if r['scene'] == scene and
                      r['same_observation'] and r['future_path_delta_mm'] is not None]
        if not candidates:
            continue
        chosen.append(max(candidates, key=lambda r: r['future_path_delta_mm']))
    needed = {(int(cache[r[key]]['episode']), int(cache[r[key]]['frame']))
              for r in chosen for key in ('index_a', 'index_b')}
    labels = {}
    for line in a.labels.open():
        row = json.loads(line)
        key = (int(row['episode_index']), int(row['frame_index']))
        if key in needed:
            labels[key] = row
    missing = needed - set(labels)
    if missing:
        raise RuntimeError(f'Missing labels for {sorted(missing)[:6]}')
    a.output.mkdir(parents=True)
    manifest = []
    for n, record in enumerate(chosen):
        panels = []
        for side, key in (('A', 'index_a'), ('B', 'index_b')):
            source = cache[record[key]]
            label = labels[int(source['episode']), int(source['frame'])]
            cams = []
            for camera in ('center', 'left', 'right'):
                image = cv2.imread(label['images'][camera], cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(label['images'][camera])
                image = cv2.resize(image, (384, 342), interpolation=cv2.INTER_AREA)
                cv2.rectangle(image, (0, 0), (384, 34), (0, 0, 0), -1)
                cv2.putText(image, f'{side} card{record["request_" + side.lower()][0]} '
                                   f'port{record["request_" + side.lower()][1]} {camera}',
                            (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
                cams.append(image)
            panels.append(np.hstack(cams))
        sheet = np.vstack(panels)
        out = a.output / f'validation_scene_{n:02d}.jpg'
        cv2.imwrite(str(out), sheet)
        manifest.append({'scene': record['scene'], 'image': str(out),
                         'time_s': record['time_s'],
                         'request_a': record['request_a'], 'request_b': record['request_b'],
                         'same_observation': record['same_observation'],
                         'tcp_difference_mm': record['tcp_difference_mm'],
                         'future_path_delta_mm': record['future_path_delta_mm'],
                         'source_original_images': {
                             side: labels[int(cache[record[key]]['episode']),
                                          int(cache[record[key]]['frame'])]['images']
                             for side, key in (('A', 'index_a'), ('B', 'index_b'))}})
    (a.output / 'manifest.json').write_text(json.dumps({'schema': 'aic_sfp_route_branch_review/v1',
        'records': manifest, 'no_geometry_overlays': True, 'final_scenes_opened': False}, indent=2) + '\n')
    print(json.dumps({'sheets': len(manifest), 'output': str(a.output)}))


if __name__ == '__main__':
    main()
