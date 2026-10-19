#!/usr/bin/env python3
"""Select original SC wrist RGB for grouped, three-way visibility review.

Scored geometry determines review sampling and an audit crop only. It is not
a visibility label or a deployable crop-selection method.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np


CAMERAS = ('center', 'left', 'right')


def choose(rows, target_mm, require_inframe=None):
    selected = [r for r in rows if require_inframe is None or
                r['projection_inframe'] == require_inframe]
    if not selected:
        return None
    return min(selected, key=lambda r: (abs(r['tcp_distance_mm']-target_mm),
                                        abs(r['time_s']-20)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--coarse', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--groups-per-port', type=int, default=10)
    p.add_argument('--contact-distance-mm', type=float, default=6.0)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve previous batch')
    rows = [json.loads(line) for line in a.labels.open()]
    predictions = {(r['row'], r['camera']): r for r in
                   (json.loads(line) for line in a.coarse.open()) if r['family'] == 'sc'}
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if row['family'] != 'sc':
            continue
        selected = next(x for x in row['targets'] if x['identity'] == row['selected_identity'])
        distance = np.linalg.norm(np.asarray(selected['position_base_m'])-row['state'][:3])*1000
        for cam in CAMERAS:
            target = selected['cameras'][cam]
            coarse = predictions[(i, cam)]['predicted'][row['selected_identity']]
            groups[row['scene_group']].append({
                'episode': row['episode'], 'scene_group': row['scene_group'],
                'original_split': row['split'], 'target': row['selected_identity'],
                'camera': cam, 'time_s': row['time_s'], 'image': row['images'][cam],
                'tcp_distance_mm': float(distance), 'projection_xy': target[:2],
                'projection_inframe': bool(target[3]),
                'frozen_inframe_score': coarse['projected_inframe_probability'],
                'review_label': None})
    by_port = defaultdict(list)
    for group, items in groups.items():
        by_port[items[0]['target']].append(group)
    selected_groups = []
    for port in ('sc_port_0', 'sc_port_1'):
        candidates = sorted(by_port[port])
        if len(candidates) < a.groups_per_port:
            raise ValueError(f'Only {len(candidates)} scene groups for {port}')
        indices = np.linspace(0, len(candidates)-1, a.groups_per_port, dtype=int)
        selected_groups.extend(candidates[int(i)] for i in indices)
    chosen = []
    for group_index, group in enumerate(selected_groups):
        items = groups[group]
        # A route view, a near view, and an almost-contact view. Rotate the
        # camera so all three cameras appear across complete scene groups.
        for phase_index, (phase, distance) in enumerate(
                (('route', 250), ('approach', 90), ('contact', a.contact_distance_mm))):
            camera = CAMERAS[(group_index+phase_index) % 3]
            options = [r for r in items if r['camera'] == camera]
            row = choose(options, distance, phase != 'route')
            if row is None:
                row = choose(options, distance)
            if row is None:
                raise ValueError((group, phase))
            chosen.append(dict(row, review_phase=phase))
    a.output.mkdir(parents=True)
    panels = a.output/'panels'
    panels.mkdir()
    for i, row in enumerate(chosen):
        image = cv2.imread(row['image'])
        if image is None or image.shape[:2] != (1024, 1152):
            raise ValueError(row['image'])
        full = cv2.resize(image, (360, 320), interpolation=cv2.INTER_AREA)
        x, y = row['projection_xy']
        crop = cv2.warpAffine(image, np.array([[1, 0, 180-x], [0, 1, 160-y]], np.float32),
                              (360, 320), borderMode=cv2.BORDER_CONSTANT)
        panel = np.hstack([full, crop])
        cv2.putText(panel, f"{i:03d} {row['target']} {row['camera']} {row['review_phase']}",
                    (5, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 210, 255), 2)
        cv2.imwrite(str(panels/f'{i:03d}.jpg'), panel, [cv2.IMWRITE_JPEG_QUALITY, 91])
        row['panel'] = str((panels/f'{i:03d}.jpg').resolve())
    with (a.output/'review_manifest.jsonl').open('w') as file:
        for row in chosen:
            file.write(json.dumps(row)+'\n')
    summary = {'schema': 'aic_grouped_sc_visibility_review/v1',
               'rows': len(chosen), 'complete_scene_groups': len(set(selected_groups)),
               'port0_groups': a.groups_per_port, 'port1_groups': a.groups_per_port,
               'contact_distance_target_mm': a.contact_distance_mm,
               'all_review_labels_unset': True, 'geometry_for_review_only': True,
               'final_scenes_opened': False}
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
