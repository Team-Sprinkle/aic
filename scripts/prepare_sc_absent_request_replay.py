#!/usr/bin/env python3
"""Prepare same-RGB, different-request SC negatives from saved development resets."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--start-s', type=float, required=True)
    p.add_argument('--end-s', type=float, required=True)
    p.add_argument('--count', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve prior replay')
    stream = json.loads((a.scene/'scored_stream.json').read_text())
    selected = int(stream['task']['target_module_name'].rsplit('_', 1)[-1])
    absent = 1-selected
    if f'task_board|task_board/sc_port_{absent}' in stream['edges']:
        raise ValueError('Alternate port exists in this scene; not an absent-request negative')
    labels = [json.loads(line) for line in
              (a.scene/'original_rgb_projection_v1/rows.jsonl').open()]
    by_time = {}
    for row in labels:
        if a.start_s <= row['sim_ns']/1e9 <= a.end_s:
            by_time.setdefault(row['sim_ns'], {})[row['camera']] = row
    times = [t for t, cameras in sorted(by_time.items()) if len(cameras) == 3]
    if len(times) < a.count:
        raise ValueError(f'Only {len(times)} complete camera triplets')
    times = [times[int(i)] for i in np.linspace(0, len(times)-1, a.count)]
    audit = []
    review = []
    for t in times:
        rows = by_time[t]
        for camera in ('center','left','right'):
            row = rows[camera]
            audit.append({'episode':a.scene.name, 'image':row['image'],
                          'target_port':absent, 'camera':camera,
                          'geometric_visible_fraction':None})
        review.append({'image':rows['center']['image'],
                       'review_label':'not_identifiable_provisional',
                       'label_reason':'requested_sc_port_absent_from_scene_tf',
                       'same_rgb_original_request':selected,
                       'counterfactual_request':absent,
                       'scene_group_id':rows['center']['scene_group_id']})
    a.output.mkdir(parents=True)
    (a.output/'audit.json').write_text(json.dumps({'records':audit},indent=2)+'\n')
    (a.output/'review.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in review))
    (a.output/'summary.json').write_text(json.dumps({
        'scene':str(a.scene), 'source_episode':a.scene.name,
        'scene_group_id':review[0]['scene_group_id'],
        'counterfactual_requested_port':absent,
        'requested_port_absent_from_scored_tf':True,
        'complete_triplets':len(review), 'times_s':[t/1e9 for t in times],
        'same_original_rgb_as_positive_review':True,
        'geometry_used_to_certify_absence_only':True,
        'final_scenes_opened':False},indent=2)+'\n')
    print(json.dumps({'triplets':len(review), 'absent_port':absent,
                      'scene_group_id':review[0]['scene_group_id']}))


if __name__ == '__main__':
    main()
