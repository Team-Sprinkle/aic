#!/usr/bin/env python3
"""Measure how often scored SC teacher setpoints exceed an actor cap.

This is a posthoc label audit. It does not change the trained actor or feed
teacher geometry to autonomous inference.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--caps-mm', type=float, nargs='+', default=[5, 20, 50])
    args = parser.parse_args()
    rows = []
    for source in args.labels:
        for line in source.open():
            row = json.loads(line)
            if row['split'] != 'train':
                continue
            current = np.asarray(row['observed_tcp_pose_opening_frame'][:3], dtype=float)
            desired = np.asarray(row['teacher_tcp_target_poses_opening_frame'][0][:3], dtype=float)
            rows.append({'source': str(source), 'episode_index': row['episode_index'],
                         'frame': row['frame'], 'gap_mm': (desired-current)*1000})
    groups = {
        'all': rows,
        'early_frame_0_50': [r for r in rows if r['frame'] <= 50],
        'transport_frame_20_40': [r for r in rows if 20 <= r['frame'] <= 40],
        'later_frame_100_plus': [r for r in rows if r['frame'] >= 100],
    }
    summary = {}
    for name, subset in groups.items():
        gaps = np.stack([r['gap_mm'] for r in subset])
        max_component = np.max(np.abs(gaps), axis=1)
        summary[name] = {
            'rows': len(subset),
            'episodes': len({(r['source'], r['episode_index']) for r in subset}),
            'max_component_gap_mm_p50_p90_p95_p99': np.quantile(
                max_component, [.5, .9, .95, .99]).tolist(),
            'fraction_any_component_exceeds_cap': {
                str(cap): float(np.mean(max_component > cap)) for cap in args.caps_mm},
            'absolute_xyz_gap_mm_p95': np.quantile(np.abs(gaps), .95, axis=0).tolist(),
        }
    report = {'schema': 'shared_sc_target_cap_audit/v1',
              'sources': [str(p) for p in args.labels],
              'scope': 'scored SC training label setpoints versus observed TCP; a port-frame component difference is a conservative proxy for TCP-frame component clipping, since the frames rotate',
              'summary': summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
