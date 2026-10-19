#!/usr/bin/env python3
"""Audit observation-only static-port temporal fusion on grouped SFP scenes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def stats(values):
    a = np.asarray(values)
    return {'n': len(a), 'median_mm': float(np.median(a)) if len(a) else None,
            'p95_mm': float(np.quantile(a, .95)) if len(a) else None,
            'within_0p5mm': float(np.mean(a <= .5)) if len(a) else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--selection-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = json.loads(a.geometry.read_text())
    label = {(r['episode'], round(float(r['time_s']), 4)): r
             for r in map(json.loads, (a.labels / 'rows.jsonl').open())}
    selector = set(json.loads(a.selection_manifest.read_text())['native_selection_episodes'])
    groups = defaultdict(list)
    for record in report['records']:
        groups[record['episode']].append(record)
    result = {}
    for window in (0., 1., 2., 5.):
        for threshold in (None, 3., 5.):
            by = defaultdict(list)
            for episode, rows in groups.items():
                accepted = []
                for r in sorted(rows, key=lambda x: x['time_s']):
                    t = float(r['time_s'])
                    if threshold is None or r['reprojection_residual_px'] <= threshold:
                        accepted.append((t, np.asarray(r['estimated_position_base_m'], float)))
                    history = [point for u, point in accepted if t-u <= window+1e-6]
                    if not history:
                        continue
                    estimated = np.median(np.stack(history), axis=0)
                    truth = np.asarray(r['truth_position_base_m'], float)
                    source = label[episode, round(t, 4)]
                    target = next(q for q in source['targets'] if q['identity'] == source['selected_identity'])
                    local = Rotation.from_quat(target['orientation_base_xyzw']).inv().apply(estimated-truth)*1000
                    split = 'selection' if episode in selector else 'test'
                    phase = 'near' if r['tcp_distance_to_port_mm'] < 100 else 'all'
                    by[split, phase].append(float(np.linalg.norm(local[:2])))
                    by[split, phase, 'xyz'].append(float(np.linalg.norm(estimated-truth)*1000))
                    if phase == 'near':
                        by[split, 'all'].append(float(np.linalg.norm(local[:2])))
            key = f'window_{window:g}s_reprojection_{threshold if threshold is not None else "none"}'
            result[key] = {str(k): stats(v) for k, v in by.items()}
    output = {'schema': 'aic_sfp_temporal_static_port_audit/v1', 'geometry': str(a.geometry),
              'selection_episodes': sorted(selector), 'candidate_results': result,
              'runtime_geometry_input': False, 'scoring_geometry_only': True,
              'reserved_final_opened': False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2) + '\n')
    for key, result in result.items():
        print(key, result.get("('selection', 'near')"), result.get("('test', 'near')"))


if __name__ == '__main__':
    main()
