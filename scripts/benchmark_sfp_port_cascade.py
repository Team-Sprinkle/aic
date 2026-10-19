#!/usr/bin/env python3
"""Benchmark complete RGB SFP perception on episode-grouped native replay views."""
import argparse
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sfp_port_cascade import SfpPortCascade


def stats(values):
    if not values:
        return {'n': 0, 'median': None, 'p95': None, 'p99': None}
    return {'n': len(values), 'median': float(np.median(values)),
            'p95': float(np.quantile(values, .95)), 'p99': float(np.quantile(values, .99))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--coarse', type=Path, required=True)
    p.add_argument('--broad', type=Path, required=True)
    p.add_argument('--fine', type=Path, required=True)
    p.add_argument('--orientation', type=Path, required=True)
    p.add_argument('--orientation-secondary',type=Path)
    p.add_argument('--pair-orientation',action='store_true',
                   help='Use both RGB-refined SFP openings plus fixed mount roll for orientation')
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rows-per-episode', type=int, default=12)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve existing benchmark')
    cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(4)
    model = SfpPortCascade(a.coarse, a.broad, a.fine, a.orientation, a.calibration,
                           orientation_path_secondary=a.orientation_secondary,
                           pair_orientation=a.pair_orientation,device=a.device)
    rows = [json.loads(line) for line in (a.labels / 'rows.jsonl').open()]
    groups = defaultdict(list)
    for row in rows:
        if row['split'] == 'validation':
            groups[row['episode']].append(row)
    results = []
    for episode, group in groups.items():
        selected_rows = [group[int(i)] for i in np.linspace(0, len(group)-1,
                          min(len(group), a.rows_per_episode))]
        for row in selected_rows:
            native = {c: cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                      for c, path in row['images'].items()}
            small = {}
            for camera, image in native.items():
                reduced = cv2.resize(image, (288, 256), interpolation=cv2.INTER_AREA)
                _, encoded = cv2.imencode('.jpg', reduced[..., ::-1],
                                          [cv2.IMWRITE_JPEG_QUALITY, 95])
                small[camera] = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR),
                                              cv2.COLOR_BGR2RGB)
            card = int(row['selected_identity'].split('_')[2])
            port = int(row['selected_identity'].split('_')[4])
            task = [1., 0., float(port == 0), float(port == 1),
                    *[float(i == card) for i in range(5)], 1.]
            start = time.perf_counter()
            predicted, info = model.predict(native, small, row['state'], task)
            latency = (time.perf_counter() - start) * 1000
            target = next(t for t in row['targets'] if t['identity'] == row['selected_identity'])
            item = {'episode': episode, 'time_s': row['time_s'],
                    'selected_identity': row['selected_identity'], 'latency_ms': latency,
                    'accepted': predicted is not None, 'diagnostic': info}
            item['truth_base_position_m'] = target['position_base_m']
            item['truth_base_quat_xyzw'] = target['orientation_base_xyzw']
            if predicted is not None:
                item['predicted_base_position_m'] = predicted[:3, 3].tolist()
                item['predicted_base_quat_xyzw'] = Rotation.from_matrix(
                    predicted[:3, :3]).as_quat().tolist()
                truth = np.asarray(target['position_base_m'])
                delta = Rotation.from_quat(target['orientation_base_xyzw']).inv().apply(
                    predicted[:3, 3] - truth) * 1000
                angle = Rotation.from_matrix(predicted[:3, :3]) * Rotation.from_quat(
                    target['orientation_base_xyzw']).inv()
                item.update(lateral_mm=float(np.linalg.norm(delta[:2])),
                            axial_mm=float(abs(delta[2])),
                            orientation_deg=float(np.rad2deg(angle.magnitude())))
            results.append(item)
    metrics = {}
    selection = [r for r in results if (int(r['selected_identity'].split('_')[2]) +
                 int(r['selected_identity'].split('_')[4])) % 2 == 0]
    test = [r for r in results if r not in selection]
    for split, subset in {'selection': selection, 'untouched_test': test,
                          'all_validation': results}.items():
        metrics[split] = {'episodes': len({r['episode'] for r in subset}),
                          'accepted_fraction': (sum(r['accepted'] for r in subset) / len(subset)
                                                if subset else None),
                          'latency_ms': stats([r['latency_ms'] for r in subset]),
                          **{key: stats([r[key] for r in subset if r['accepted']])
                             for key in ('lateral_mm', 'axial_mm', 'orientation_deg')}}
    locks = []
    for episode, group in groups.items():
        history = deque(maxlen=5)
        for row in [item for item in results if item['episode'] == episode]:
            if not row['accepted']:
                continue
            history.append(row)
            if len(history)<5:continue
            points=np.asarray([item['predicted_base_position_m'] for item in history])
            center=np.median(points,axis=0)
            rotations=Rotation.from_quat([item['predicted_base_quat_xyzw'] for item in history])
            mean=rotations.mean()
            spread_mm=float(np.max(np.linalg.norm(points-center,axis=1))*1000)
            spread_deg=float(np.max((mean.inv()*rotations).magnitude())*180/np.pi)
            if spread_mm>.5 or spread_deg>5:continue
            true_rotation=Rotation.from_quat(row['truth_base_quat_xyzw'])
            local=true_rotation.inv().apply(center-np.asarray(row['truth_base_position_m']))*1000
            locks.append({'episode':episode,'first_lock_time_s':row['time_s'],
                          'lateral_mm':float(np.linalg.norm(local[:2])),
                          'axial_mm':float(abs(local[2])),
                          'orientation_deg':float(np.rad2deg((mean.inv()*true_rotation).magnitude())),
                          'spread_mm':spread_mm,'spread_deg':spread_deg})
            break
    metrics['static_visual_lock']={'episodes':len(locks),
                                   'of_episodes':len(groups),
                                   'lateral_mm':stats([r['lateral_mm'] for r in locks]),
                                   'axial_mm':stats([r['axial_mm'] for r in locks]),
                                   'orientation_deg':stats([r['orientation_deg'] for r in locks])}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({'schema': 'sfp_rgb_cascade_benchmark/v1',
        'commands': ' '.join(sys.argv), 'episode_rows': results,
        'static_visual_locks':locks,'metrics': metrics,
        'runtime_true_geometry': False, 'truth_scoring_only': True,
        'visibility_certified': False, 'reserved_final_opened': False}, indent=2)+'\n')
    print(json.dumps(metrics))


if __name__ == '__main__':
    main()
