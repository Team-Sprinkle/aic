#!/usr/bin/env python3
"""Prefer observation-only direct three-view port detection, else rail crop.

True target geometry is read only for offline coverage audit, never selection.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from train_two_family_port_locator import CAMERAS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--direct-candidates', type=Path, required=True)
    p.add_argument('--rail-proposals', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve previous merge')
    candidates = {(r['episode'], r['time_s']): r for r in
                  map(json.loads, a.direct_candidates.open())}
    labels = {(r['episode'], r['time_s']): r for r in
              map(json.loads, (a.labels/'rows.jsonl').open())}
    output = []
    counts = defaultdict(int)
    episodes = defaultdict(list)
    for row in map(json.loads, (a.rail_proposals/'rows.jsonl').open()):
        key = (row['episode'], row['time_s'])
        cand = candidates[key]['candidates'][row['requested']]
        crop = row['crop']
        direct = (cand['predicted_base_m'] is not None and
                  len(cand['views']) == 3 and
                  cand['reprojection_px'] is not None and
                  cand['reprojection_px'] <= 5)
        if direct:
            crop = {'source_card': int(row['requested'].split('_')[2]),
                    'source': 'native_full_row_direct',
                    'source_reprojection_px': cand['reprojection_px'],
                    'estimated_center_base_m': cand['predicted_base_m'],
                    'center_px': {c: cand['views'][c] for c in CAMERAS},
                    'crop_halfsize_px': 240}
            counts['direct'] += 1
        elif crop:
            crop = dict(crop, source='observed_rail_fallback')
            counts['rail'] += 1
        else:
            counts['none'] += 1
        item = {k: v for k, v in row.items() if k != 'audit_only'}
        item['crop'] = crop
        scored = {}
        target = next(t for t in labels[key]['targets']
                      if t['identity'] == row['requested'])
        if crop:
            for cam in CAMERAS:
                uv = target['cameras'][cam]
                offset = (np.asarray(uv[:2])-
                          np.asarray(crop['center_px'][cam])) if uv[3] else None
                scored[cam] = {'projected_inframe': bool(uv[3]),
                               'target_inside_crop': bool(offset is not None and
                                                          np.max(np.abs(offset)) <= 240),
                               'center_error_px': float(np.linalg.norm(offset))
                               if offset is not None else None}
        item['audit_only'] = scored
        output.append(item)
        episodes[row['episode']].append(item)
    a.output.mkdir(parents=True)
    with (a.output/'rows.jsonl').open('w') as f:
        for row in output:
            f.write(json.dumps(row, separators=(',', ':'))+'\n')
    summary = {'schema': 'sfp_direct_or_rail_proposals/v1',
               'rows': len(output), 'sources': dict(counts),
               'all_three_target_inside': sum(bool(r['crop']) and
                   all(v['target_inside_crop'] for v in r['audit_only'].values())
                   for r in output),
               'episode_groups': len(episodes),
               'runtime_true_geometry_input': False,
               'true_geometry_used_only_for_audit': True,
               'final_scenes_opened': False}
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
