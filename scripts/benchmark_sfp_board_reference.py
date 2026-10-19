#!/usr/bin/env python3
"""Measure three-camera JPEG decode, resize and board-reference forward latency."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from train_sfp_board_reference import BoardReference, BoardReferenceLocal, CAMS, SIZE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--warmups', type=int, default=20)
    p.add_argument('--repeats', type=int, default=200)
    args = p.parse_args()
    if args.output.exists():
        p.error('Preserve existing output')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    kind = state.get('head', 'pooled')
    model = (BoardReferenceLocal(state['init']) if kind == 'slot_local'
             else BoardReference(state['init']))
    model.load_state_dict(state['model'])
    model.cuda().eval()
    torch.set_num_threads(4)
    rows = [json.loads(line) for line in (args.labels / 'rows.jsonl').open()
            if json.loads(line)['split'] == 'validation']
    rows = rows[:min(len(rows), 20)]
    if not rows:
        raise ValueError('No development rows')
    times = []
    with torch.inference_mode():
        for i in range(args.warmups + args.repeats):
            start = time.perf_counter()
            row = rows[i % len(rows)]
            images = []
            for cam in CAMS:
                frame = cv2.imread(row['images'][cam], cv2.IMREAD_COLOR)
                if frame is None:
                    raise FileNotFoundError(row['images'][cam])
                frame = cv2.cvtColor(cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA),
                                     cv2.COLOR_BGR2RGB)
                images.append(np.transpose(frame.copy(), (2, 0, 1)))
            x = torch.from_numpy(np.stack(images)[None]).cuda().float() / 255
            with torch.autocast('cuda', dtype=torch.bfloat16):
                model(x)
            torch.cuda.synchronize()
            if i >= args.warmups:
                times.append((time.perf_counter() - start) * 1000)
    report = {'schema': 'sfp_board_reference_local_latency/v1',
              'checkpoint': str(args.checkpoint), 'head': kind,
              'camera_count': 3, 'source': 'development_nonfinal_rgb',
              'includes': ['JPEG disk read', 'OpenCV resize', 'RGB conversion',
                           'host-to-GPU copy', 'model forward', 'GPU synchronization'],
              'excludes': ['ROS transport', 'controller', 'actor', 'precise pose refiner'],
              'repeats': len(times), 'latency_ms': {
                  'p50': float(np.percentile(times, 50)),
                  'p95': float(np.percentile(times, 95)),
                  'p99': float(np.percentile(times, 99)),
                  'mean': float(np.mean(times))}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
