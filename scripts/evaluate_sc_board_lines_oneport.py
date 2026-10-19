#!/usr/bin/env python3
"""Replay the frozen SC board-rail yaw estimator on newer one-port resets.

Only RGB, measured TCP, fixed training calibration and frozen RGB model outputs
form the estimate. Scored port rotation is read after estimation for metrics.
"""
import argparse
from collections import defaultdict, deque
import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from audit_sc_port_targets import pose_matrix
from evaluate_sc_board_lines_yaw import camera_yaw
from run_sc_native_crop_orientation_ablation import oneport_examples
from train_sc_oneport_orientation_adaptation import Model, oneport_data, stats
from train_two_family_port_locator import Locator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oneport-glob', required=True)
    p.add_argument('--locator', type=Path, required=True)
    p.add_argument('--orientation', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--board-mount', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.output.exists():
        p.error('Preserve prior output')
    torch.set_num_threads(4)
    paths = [Path(x) for x in sorted(glob.glob(args.oneport_glob))]
    if not paths:
        p.error('No one-port resets')
    source_rows = {path.parent.parent.name:
                   {row['image']: row for row in map(json.loads, path.open())}
                   for path in paths}
    cal = json.loads(args.calibration.read_text())
    offsets = {c: np.asarray(v, float) for c, v in cal['tcp_to_optical'].items()}
    mount = Rotation.from_quat(json.loads(args.board_mount.read_text())['board_to_port_xyzw'])
    locator = Locator(None).to(args.device).eval()
    locator.load_state_dict(torch.load(args.locator, map_location='cpu', weights_only=False)['model'])
    x, y, meta = oneport_data(paths, locator, cal, args.device)
    examples = oneport_examples(paths, locator, args.device)
    if len(examples) != len(meta):
        raise ValueError('Input row identity mismatch')
    saved = torch.load(args.orientation, map_location='cpu', weights_only=False)
    model = Model(saved['model']['mean'], saved['model']['std']).to(args.device).eval()
    model.load_state_dict(saved['model'])
    with torch.inference_mode():
        base = np.concatenate([model(torch.from_numpy(x[i:i+128]).to(args.device)).cpu().numpy()
                               for i in range(0, len(x), 128)])
    histories = defaultdict(lambda: deque(maxlen=40))
    last = defaultdict(lambda: -float('inf'))
    records = []
    for i, (example, item) in enumerate(zip(examples, meta)):
        if example['episode'] != item['episode'] or example['center_sim_ns'] != item['center_sim_ns']:
            raise ValueError('One-port row order mismatch')
        prior = (Rotation.from_matrix(base[i]) * mount.inv()).as_euler('xyz')[2]
        camera_results = {}
        for j, c in enumerate(('center', 'left', 'right')):
            # Match the per-camera measured TCP at its image timestamp.
            image = example['images'][j]
            # The grouped frame finder selects the nearest timestamp per
            # camera; recover that camera's own state by its saved image path.
            camera_row = source_rows[item['episode']][image]
            base_optical = pose_matrix(camera_row['measured_tcp_base_pose']) @ offsets[c]
            camera_results[c] = camera_yaw(image, example['centers'][j], base_optical, prior)
        good = [v for v in camera_results.values() if v is not None and v['mad_deg'] <= 1.]
        candidate = float(np.median([v['yaw_rad'] for v in good])) if good else None
        spread = math.degrees(max(v['yaw_rad'] for v in good) - min(v['yaw_rad'] for v in good)) if good else None
        accepted = len(good) >= 2 and spread <= 1.
        t = item['center_sim_ns'] / 1e9
        episode = item['episode']
        if accepted and t - last[episode] >= .5:
            histories[episode].append(candidate)
            last[episode] = t
        estimate = (float(np.median(histories[episode])) if len(histories[episode]) >= 2
                    else candidate if candidate is not None else prior)
        pred_rotation = Rotation.from_euler('z', estimate) * mount
        truth_rotation = Rotation.from_matrix(y[i])
        error = math.degrees((pred_rotation.inv() * truth_rotation).magnitude())
        base_error = math.degrees((Rotation.from_matrix(base[i]).inv() * truth_rotation).magnitude())
        records.append({'episode': episode, 'group': item['group'], 'sim_ns': item['center_sim_ns'],
                        'distance_mm': item['distance_mm'], 'base_error_deg': base_error,
                        'board_line_full_orientation_error_deg': error,
                        'board_yaw_deg': math.degrees(estimate),
                        'candidate_deg': math.degrees(candidate) if candidate is not None else None,
                        'good_cameras': len(good), 'view_spread_deg': spread,
                        'history_length': len(histories[episode]), 'accepted': bool(accepted),
                        'images': example['images']})
    near = [r for r in records if r['distance_mm'] < 100]
    report = {'schema': 'sc_board_lines_oneport_replay/v1', 'input_reset_groups': len(paths),
              'near_triplets': len(near), 'near_episodes': len({r['episode'] for r in near}),
              'frozen_base_orientation': stats([r['base_error_deg'] for r in near]),
              'board_line_full_orientation': stats([r['board_line_full_orientation_error_deg'] for r in near]),
              'history_initialized_rate': sum(r['history_length'] >= 2 for r in near) / len(near) if near else None,
              'by_episode': {ep: stats([r['board_line_full_orientation_error_deg'] for r in near
                                        if r['episode'] == ep]) for ep in sorted({r['episode'] for r in near})},
              'records': records, 'runtime_true_geometry': False, 'final_scenes_opened': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
