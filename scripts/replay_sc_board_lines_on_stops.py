#!/usr/bin/env python3
"""Replay observation-only board-rail posture verification on frozen SC stops."""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from audit_sc_port_targets import pose_matrix
from evaluate_sc_board_lines_yaw import camera_yaw
from train_two_family_port_locator import Locator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stop-replay', type=Path, required=True)
    p.add_argument('--audit-root', type=Path, required=True)
    p.add_argument('--locator', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--board-mount', type=Path, required=True)
    p.add_argument('--tcp-in-port-prior', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve old replay')
    torch.set_num_threads(4)
    rows = json.loads(a.stop_replay.read_text())['rows']
    optics = {c: np.asarray(v) for c, v in
              json.loads(a.calibration.read_text())['tcp_to_optical'].items()}
    mount = Rotation.from_quat(json.loads(a.board_mount.read_text())['board_to_port_xyzw'])
    nominal = Rotation.from_quat(json.loads(a.tcp_in_port_prior.read_text())
                                 ['nominal_tcp_in_port_orientation_xyzw'])
    locator = Locator(None).to(a.device).eval()
    locator.load_state_dict(torch.load(a.locator, map_location='cpu', weights_only=False)['model'])
    output = []
    for row in rows:
        audit = a.audit_root / row['trace']
        trace = [json.loads(x) for x in (audit / 'runtime_trace.jsonl').open()]
        proposals = [x for x in trace if x['event'] == 'proposal']
        proposal = min(proposals, key=lambda x: abs(x['sim_time'] - row['sim_time']))
        state = np.asarray(proposal['state'][:7], float)
        images = [cv2.imread(x) for x in row['images']]
        if any(im is None for im in images):
            raise FileNotFoundError(row['images'])
        small = [cv2.cvtColor(cv2.resize(im, (288, 256), interpolation=cv2.INTER_AREA),
                              cv2.COLOR_BGR2RGB) for im in images]
        rgb = torch.from_numpy(np.stack(small)).permute(0, 3, 1, 2).to(a.device).float()/255
        hint = np.zeros((3, 5), np.float32);hint[:, 0] = 1
        for j in range(3):
            hint[j, 2+j] = 1
        with torch.inference_mode():
            points, _ = locator(rgb, torch.from_numpy(hint).to(a.device))
        points = points.float().cpu().numpy() * [1151, 1023]
        prior = (Rotation.from_quat(row['predicted_port_quat_xyzw']) * mount.inv()).as_euler('xyz')[2]
        cameras = ('center', 'left', 'right')
        results = []
        for j, c in enumerate(cameras):
            base_optical = pose_matrix(state) @ optics[c]
            results.append(camera_yaw(row['images'][j], points[j, row['requested_port']],
                                      base_optical, prior))
        good = [v for v in results if v is not None and v['mad_deg'] <= 1.]
        spread = math.degrees(max(v['yaw_rad'] for v in good) - min(v['yaw_rad'] for v in good)) if good else None
        valid = len(good) >= 2 and spread <= 1.
        yaw = float(np.median([v['yaw_rad'] for v in good])) if valid else None
        port = Rotation.from_euler('z', yaw) * mount if valid else None
        intended = port * nominal if valid else None
        measured = Rotation.from_quat(state[3:7])
        deviation = math.degrees((intended.inv() * measured).magnitude()) if valid else None
        output.append({'trace': row['trace'], 'posthoc_false_stop': row['posthoc_false_stop'],
                       'valid_rail_observation': valid, 'good_cameras': len(good),
                       'view_spread_deg': spread, 'predicted_tcp_to_measured_deg': deviation,
                       'three_degree_accept': bool(valid and deviation <= 3.),
                       'prior_head_predicted_tcp_to_measured_deg': row['predicted_tcp_to_measured_deg'],
                       'images': row['images']})
    report = {'schema': 'sc_board_lines_frozen_stop_replay/v1', 'rows': output,
              'valid_retained': sum(x['three_degree_accept'] and not x['posthoc_false_stop'] for x in output),
              'valid_total': sum(not x['posthoc_false_stop'] for x in output),
              'false_accepted': sum(x['three_degree_accept'] and x['posthoc_false_stop'] for x in output),
              'runtime_true_geometry': False, 'final_scenes_opened': False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
