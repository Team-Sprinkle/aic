#!/usr/bin/env python3
"""Test the frozen native SC crop refiner on new scored original-RGB views.

The crop is centered only on each RGB locator output. The projected aperture
is read afterward for scoring and never used for crop selection.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from train_sc_native_keypoints import PretrainedKeypointRefiner


def stats(values):
    a = np.asarray(values, dtype=float)
    return {'n': len(a), 'p50_px': float(np.median(a)),
            'p95_px': float(np.percentile(a, 95)),
            'over_50_px': int((a > 50).sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coarse', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve previous result')
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    source = json.loads(a.coarse.read_text())
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    size = int(saved['crop_size'])
    if size != 160 or saved.get('condition_target_port', False):
        raise ValueError('Unexpected frozen refiner architecture')
    model = PretrainedKeypointRefiner(None, False, point_count=1).to(a.device).eval()
    model.load_state_dict(saved['model'])
    records = []
    with torch.inference_mode():
        for row in source['rows']:
            image = cv2.imread(row['image'], cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (1024, 1152):
                raise ValueError(f"Missing or invalid RGB: {row['image']}")
            center = np.asarray(row['predicted_xy'], dtype=float)
            truth = np.asarray(row['projected_opening_xy'], dtype=float)
            left, top = (np.round(center-size/2).astype(int)).tolist()
            crop = cv2.warpAffine(image, np.array([[1, 0, -left], [0, 1, -top]], np.float32),
                                  (size, size), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to(a.device).float()/255
            with torch.autocast(a.device.split(':')[0], dtype=torch.bfloat16,
                                enabled=a.device.startswith('cuda')):
                point = model(x).float().cpu().numpy().reshape(2)
            predicted = point*(size-1)+[left, top]
            item = {k: row[k] for k in ('camera', 'sim_ns', 'image', 'projected_opening_xy')}
            item.update(checkpoint=row['checkpoint']+'_refined', predicted_xy=predicted.tolist(),
                        parent_coarse=row['checkpoint'], crop_origin=[left, top],
                        truth_in_crop_diagnostic=bool(0 <= truth[0]-left < size and
                                                      0 <= truth[1]-top < size),
                        error_px=float(np.linalg.norm(predicted-truth)))
            records.append(item)
    by_camera = defaultdict(dict)
    for checkpoint in ('original_refined', 'adapted_refined'):
        for camera in ('center', 'left', 'right'):
            rows = [r for r in records if r['checkpoint'] == checkpoint and r['camera'] == camera]
            by_camera[checkpoint][camera] = stats([r['error_px'] for r in rows])
    result = {'schema': 'aic_sc_independent_native_refiner/v1',
              'coarse_source': str(a.coarse), 'refiner_checkpoint': str(a.checkpoint),
              'reviewed_visible_window_s': source['reviewed_visible_window_s'],
              'by_camera': by_camera,
              'truth_inside_observation_selected_crop':
                  {ck: sum(r['truth_in_crop_diagnostic'] for r in records if r['checkpoint'] == ck)
                   for ck in ('original_refined', 'adapted_refined')},
              'rows': records, 'crop_selected_by_rgb_only': True,
              'visibility_certified': False, 'final_scenes_opened': False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'by_camera': by_camera,
                      'truth_inside_observation_selected_crop':
                          result['truth_inside_observation_selected_crop']}, indent=2))


if __name__ == '__main__':
    main()
