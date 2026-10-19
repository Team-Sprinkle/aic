#!/usr/bin/env python3
"""Benchmark in-memory RGB preprocessing, coarse locator, native crop, and 3D.

This excludes camera transport, JPEG decoding, ROS, the transport actor, and
the insertion actor. It is a component benchmark, not complete live latency.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from evaluate_sc_native_triangulation import triangulate
from train_sc_native_keypoints import PretrainedKeypointRefiner
from train_two_family_port_locator import Locator, LOOKUP


CAMERAS = ('center', 'left', 'right')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows', type=Path, required=True)
    p.add_argument('--coarse', type=Path, required=True)
    p.add_argument('--adapted', type=Path, required=True)
    p.add_argument('--refiner', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--target-port', type=int, choices=(0, 1), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=100)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve previous benchmark')
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    rows = defaultdict(dict)
    for row in (json.loads(line) for line in a.rows.open()):
        if row['target_port'] == a.target_port and row['projected_inframe']:
            rows[row['sim_ns']][row['camera']] = row
    triplets = []
    for sim_ns, views in sorted(rows.items()):
        if set(views) == set(CAMERAS):
            triplets.append({cam: cv2.imread(views[cam]['image'], cv2.IMREAD_COLOR)
                             for cam in CAMERAS})
    if not triplets or any(im is None for group in triplets for im in group.values()):
        raise ValueError('No complete original-RGB triplets')
    locator = Locator(None).to(a.device).eval()
    state = torch.load(a.coarse, map_location='cpu', weights_only=False)['model']
    repair = torch.load(a.adapted, map_location='cpu', weights_only=False)
    if Path(repair['parent_checkpoint']).resolve() != a.coarse.resolve():
        raise ValueError('Coarse checkpoint mismatch')
    state['decoder.4.weight'] = repair['last_weight']
    state['decoder.4.bias'] = repair['last_bias']
    locator.load_state_dict(state)
    saved = torch.load(a.refiner, map_location='cpu', weights_only=False)
    if saved['crop_size'] != 160:
        raise ValueError('Unexpected crop size')
    refiner = PretrainedKeypointRefiner(None, False, point_count=1).to(a.device).eval()
    refiner.load_state_dict(saved['model'])
    optics = {k: np.asarray(v, dtype=float) for k, v in
              json.loads(a.calibration.read_text())['tcp_to_optical'].items()}

    def infer(group):
        pixels = {}
        for cam in CAMERAS:
            image = group[cam]
            small = cv2.cvtColor(cv2.resize(image, (288, 256),
                                             interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(small.copy()).permute(2, 0, 1)[None].to(a.device).float()/255
            meta = torch.tensor([[1, 0, int(cam=='center'), int(cam=='left'),
                                  int(cam=='right')]], device=a.device).float()
            with torch.inference_mode():
                xy, _ = locator(x, meta)
            center = xy[0, LOOKUP[f'sc_port_{a.target_port}']].float().cpu().numpy()*[1151, 1023]
            left, top = np.round(center-80).astype(int).tolist()
            crop = cv2.warpAffine(image, np.array([[1, 0, -left], [0, 1, -top]], np.float32),
                                  (160, 160), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            inp = torch.from_numpy(crop.copy()).permute(2, 0, 1)[None].to(a.device).float()/255
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                point = refiner(inp).float().cpu().numpy().reshape(2)
            pixels[cam] = point*159+[left, top]
        return triangulate(pixels, optics)

    for i in range(min(10, len(triplets))):
        infer(triplets[i])
    torch.cuda.synchronize()
    times = []
    for i in range(a.samples):
        start = time.perf_counter()
        infer(triplets[i % len(triplets)])
        torch.cuda.synchronize()
        times.append((time.perf_counter()-start)*1000)
    arr = np.asarray(times)
    result = {'schema': 'aic_sc_selected_port_chain_latency/v1',
              'source': str(a.rows), 'complete_image_triplets': len(triplets),
              'samples': len(times), 'target_port': a.target_port,
              'p50_ms': float(np.percentile(arr, 50)),
              'p95_ms': float(np.percentile(arr, 95)),
              'p99_ms': float(np.percentile(arr, 99)),
              'max_ms': float(arr.max()),
              'includes': ['in-memory original RGB resize', 'three coarse locator passes',
                           'RGB-selected native crop', 'three crop refiner passes',
                           'three-view triangulation'],
              'excludes': ['camera transport', 'JPEG decoding', 'ROS',
                           'transport actor', 'insertion actor'],
              'complete_live_latency_gate_passed': False,
              'final_scenes_opened': False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
