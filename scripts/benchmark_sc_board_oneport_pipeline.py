#!/usr/bin/env python3
"""Time the SC locator, orientation prior, and board-rail replay path offline."""
import argparse
import glob
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from audit_sc_port_targets import pose_matrix
from evaluate_sc_board_lines_yaw import camera_yaw
from evaluate_sc_native_triangulation import triangulate
from train_sc_oneport_orientation_adaptation import Model, grouped_triplets
from train_two_family_port_locator import Locator, NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oneport-glob', required=True)
    p.add_argument('--locator', type=Path, required=True)
    p.add_argument('--orientation', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--board-mount', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve old timing')
    torch.set_num_threads(4);cv2.setNumThreads(1)
    paths = [Path(s) for s in sorted(glob.glob(a.oneport_glob))]
    samples = grouped_triplets(paths)
    locator = Locator(None).to(a.device).eval()
    locator.load_state_dict(torch.load(a.locator, map_location='cpu', weights_only=False)['model'])
    saved = torch.load(a.orientation, map_location='cpu', weights_only=False)
    orientation = Model(saved['model']['mean'], saved['model']['std']).to(a.device).eval()
    orientation.load_state_dict(saved['model'])
    optics = {c: np.asarray(v) for c, v in json.loads(a.calibration.read_text())['tcp_to_optical'].items()}
    mount = Rotation.from_quat(json.loads(a.board_mount.read_text())['board_to_port_xyzw'])
    times = []
    for _, views in samples:
        if a.device.startswith('cuda'):torch.cuda.synchronize(a.device)
        t0 = time.perf_counter()
        cameras = ('center', 'left', 'right')
        images = {c: cv2.imread(views[c]['image']) for c in cameras}
        small = [cv2.cvtColor(cv2.resize(images[c], (288, 256), interpolation=cv2.INTER_AREA),
                              cv2.COLOR_BGR2RGB) for c in cameras]
        rgb = torch.from_numpy(np.stack(small)).permute(0, 3, 1, 2).to(a.device).float()/255
        meta = np.zeros((3, 5), np.float32);meta[:, 0] = 1
        for j in range(3):meta[j, 2+j] = 1
        with torch.inference_mode():
            points, flags, features = locator(rgb, torch.from_numpy(meta).to(a.device),
                                               return_features=True)
        points = points.float().cpu().numpy()*[1151, 1023]
        flags = flags.float().sigmoid().cpu().numpy()
        poses = {c: pose_matrix(views[c]['measured_tcp_base_pose'])@optics[c] for c in cameras}
        xyz = [];mask = []
        for k in range(len(NAMES)):
            pixels = {c: points[j, k] for j, c in enumerate(cameras) if flags[j, k] >= .5}
            valid = False
            if len(pixels) >= 2:
                try:
                    point = triangulate(pixels, poses);valid = np.isfinite(point).all()
                except np.linalg.LinAlgError:pass
            xyz.extend(point.tolist() if valid else [0., 0., 0.]);mask.append(float(valid))
        port = int(views['center']['target_port'])
        code = np.zeros(len(NAMES));code[port] = 1
        state = views['center']['measured_tcp_base_pose']
        feature = np.r_[xyz, mask, code, state, features.float().cpu().numpy().reshape(-1)].astype(np.float32)
        with torch.inference_mode():
            rot = orientation(torch.from_numpy(feature).to(a.device)[None]).float().cpu().numpy()[0]
        prior = (Rotation.from_matrix(rot)*mount.inv()).as_euler('xyz')[2]
        results = [camera_yaw(views[c]['image'], points[j,port], poses[c], prior)
                   for j,c in enumerate(cameras)]
        if a.device.startswith('cuda'):torch.cuda.synchronize(a.device)
        times.append((time.perf_counter()-t0)*1000)
    measured = times[5:]
    report = {'schema': 'sc_board_oneport_offline_latency/v1', 'samples': len(times),
              'warmup_excluded': 5, 'p50_ms': float(np.percentile(measured, 50)),
              'p95_ms': float(np.percentile(measured, 95)),
              'p99_ms': float(np.percentile(measured, 99)),
              'scope': 'three native JPEG decodes, low-resolution locator, selected-port orientation head, calibrated geometry, three board-rail searches; excludes ROS, actor and command conversion',
              'final_scenes_opened': False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':main()
