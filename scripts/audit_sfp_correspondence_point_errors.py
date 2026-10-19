#!/usr/bin/env python3
"""Inspect held-out point-pair errors of a frozen SFP correspondence verifier."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from train_sfp_candidate_correspondence_probe import (
    CAMERAS, Correspondence, frozen_maps, load_labeled_views,
)
from train_two_family_port_locator import Locator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--sfp-cache', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    a = p.parse_args()
    if a.output.exists(): p.error('Preserve prior audit')
    a.output.mkdir(parents=True)
    torch.set_num_threads(4)
    records, images = load_labeled_views(a.labels, a.sfp_cache, 4)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    locator = Locator(None).to(a.device).eval()
    locator.load_state_dict(torch.load(saved['locator'], map_location='cpu', weights_only=False)['model'])
    model = Correspondence().to(a.device).eval()
    model.load_state_dict(saved['model'])
    ids = [i for i, r in enumerate(records) if r['split'] == 'validation']
    visual = frozen_maps(locator, images[ids], a.device).to(a.device).float()
    state = torch.tensor([records[i]['state'] for i in ids], device=a.device, dtype=torch.float32)
    state = (state-saved['state_mean'].to(a.device))/saved['state_std'].to(a.device)
    camera = torch.tensor([[float(records[i]['camera'] == c) for c in CAMERAS] for i in ids], device=a.device)
    request = torch.tensor([[float(records[i]['selected'] == name) for name in saved['names']]
                            for i in ids], device=a.device)
    results = []
    with torch.inference_mode():
        for kind in ('true', 'neighbor'):
            point = torch.tensor([records[i]['true_xy' if kind == 'true' else 'negative_xy']
                                  for i in ids], device=a.device, dtype=torch.float32)
            condition = torch.cat((request, camera, state, point), 1)
            prob = []
            for j in range(0, len(ids), 128):
                prob.extend(model(visual[j:j+128], condition[j:j+128], point[j:j+128])
                            .sigmoid().cpu().tolist())
            for j, score in enumerate(prob):
                incorrect = score < .5 if kind == 'true' else score >= .5
                if not incorrect: continue
                r = records[ids[j]]
                image = cv2.cvtColor(images[ids[j]].copy(), cv2.COLOR_RGB2BGR)
                true_xy = np.asarray(r['true_xy'])*[287,255]
                wrong_xy = np.asarray(r['negative_xy'])*[287,255]
                cv2.circle(image, tuple(np.rint(true_xy).astype(int)), 6, (0,255,0), 2)
                cv2.circle(image, tuple(np.rint(wrong_xy).astype(int)), 6, (0,0,255), 2)
                name = f'{kind}_{len(results):03d}.png'
                cv2.imwrite(str(a.output/name), image)
                results.append({'kind': kind, 'score': score, 'episode': r['episode'],
                    'scene_group': r['group'], 'camera': r['camera'],
                    'requested': r['selected'], 'neighbor': r['negative_identity'],
                    'true_xy_288': true_xy.tolist(), 'neighbor_xy_288': wrong_xy.tolist(),
                    'point_separation_px_288': float(np.linalg.norm(true_xy-wrong_xy)),
                    'image': str(a.output/name), 'cache_image_key': r['image_key']})
    payload = {'schema': 'sfp_correspondence_grouped_point_error_audit/v1',
        'checkpoint': str(a.checkpoint), 'threshold': .5,
        'validation_groups': len({records[i]['group'] for i in ids}),
        'validation_views': len(ids), 'errors': results,
        'geometry_input_at_runtime': False, 'projection_labels_for_audit_only': True,
        'final_scenes_opened': False}
    (a.output/'audit.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({'groups': payload['validation_groups'], 'views': len(ids),
                      'errors': results}, indent=2))


if __name__ == '__main__': main()
