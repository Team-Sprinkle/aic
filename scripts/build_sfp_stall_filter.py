#!/usr/bin/env python3
"""Build an immutable training-row mask for sustained failed-actuation windows.

Do not splice time, synthesize new actions, or modify the original images.
History still uses all observations; only the action loss omits these rows.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def windows(rows, minimum_s):
    time = np.asarray([r[0] for r in rows])
    xyz = np.asarray([r[1] for r in rows])
    cmd = np.asarray([r[2] for r in rows])
    delta = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 1000
    failed = (delta < .1) & (cmd[:-1] > 5.)
    out = []; start = None
    for i, active in enumerate(failed):
        if active and start is None:
            start = i
        if start is not None and (not active or i == len(failed)-1):
            end = i if not active else i+1
            if time[end] - time[start] >= minimum_s:
                out.append([float(time[start] - .1), float(time[end] + .1)])
            start = None
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--minimum-seconds", type=float, default=1.)
    a = p.parse_args()
    if a.output.exists():
        p.error("Preserve earlier mask")
    groups = defaultdict(list)
    with a.labels.open() as f:
        for line in f:
            r = json.loads(line)
            if r['split'] != 'train':
                continue
            source = r.get('source_scored_stream', '')
            obs = np.asarray(r['observed_tcp_pose_port_frame'][:3])
            target = np.asarray(r['teacher_tcp_target_poses_port_frame'][0][:3])
            groups[int(r['episode_index'])].append((float(r['timestamp_s']),
                r['state'][:3], float(np.linalg.norm(target-obs)*1000), source))
    intervals = {}
    for ep, rows in groups.items():
        found = windows(rows, a.minimum_seconds)
        if found:
            intervals[ep] = found
    indices = []; by_source = defaultdict(int)
    with (a.cache/'rows.jsonl').open() as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            if r['family'] != 'sfp' or r['split'] != 'train':
                continue
            ep = int(r['episode']); t = float(r['time_s'])
            if any(lo <= t <= hi for lo, hi in intervals.get(ep, [])):
                indices.append(i)
                source = groups[ep][0][3]
                by_source['scored' if source else 'legacy'] += 1
    pair = json.loads(a.pairs.read_text())
    paired_indices = {j for r in pair['train_pairs'] for j in (r['left_index'],r['right_index'])}
    overlap = len(paired_indices & set(indices))
    result = dict(schema='aic_sfp_stall_loss_filter/v1',
        labels=str(a.labels),cache=str(a.cache),paired_manifest=str(a.pairs),
        rule='measured TCP step <0.1mm and first teacher target >5mm for >=1s at raw cadence; pad 0.1s',
        minimum_seconds=a.minimum_seconds,affected_episodes=len(intervals),
        affected_intervals=sum(map(len,intervals.values())),
        excluded_training_cache_rows=len(indices),excluded_by_source=dict(by_source),
        paired_contrast_indices_excluded=overlap,
        intervals_by_episode={str(k):v for k,v in sorted(intervals.items())},
        indices=indices,validation_rows_excluded=0,
        source_edited=False,final_scenes_opened=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('affected_episodes','affected_intervals',
        'excluded_training_cache_rows','excluded_by_source','paired_contrast_indices_excluded')}))


if __name__ == '__main__':
    main()
