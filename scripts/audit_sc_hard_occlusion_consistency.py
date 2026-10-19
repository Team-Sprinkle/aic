#!/usr/bin/env python3
"""Replay known SC hidden-target triplets through the RGB-only crop cascade."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluate_sc_native_triangulation import triangulate, reprojection_residual_px
from train_sc_native_keypoints import PretrainedKeypointRefiner
from train_two_family_port_locator import Locator, LOOKUP


CAMERAS = ('center', 'left', 'right')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--indices', nargs='+', type=int, required=True)
    p.add_argument('--coarse', type=Path, required=True)
    p.add_argument('--adapted', type=Path,
                   help='Optional decoder-only repair; omit to evaluate a saved full checkpoint')
    p.add_argument('--refiner', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve previous result')
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    review = [json.loads(line) for line in a.review.open()]
    audit = json.loads(a.audit.read_text())['records']
    by = defaultdict(dict)
    for row in audit:
        name = Path(row['image']).name
        sim_ns = int(name.split('_')[1])
        by[(row['episode'], sim_ns, int(row['target_port']))][row['camera']] = row
    locator = Locator(None).to(a.device).eval()
    state = torch.load(a.coarse, map_location='cpu', weights_only=False)['model']
    if a.adapted:
        saved = torch.load(a.adapted, map_location='cpu', weights_only=False)
        if Path(saved['parent_checkpoint']).resolve() != a.coarse.resolve():
            raise ValueError('Different coarse parent')
        state['decoder.4.weight'] = saved['last_weight']
        state['decoder.4.bias'] = saved['last_bias']
    locator.load_state_dict(state)
    refiner = PretrainedKeypointRefiner(None, False, point_count=1).to(a.device).eval()
    refiner.load_state_dict(torch.load(a.refiner, map_location='cpu', weights_only=False)['model'])
    optics = {c: np.asarray(x, dtype=float) for c, x in
              json.loads(a.calibration.read_text())['tcp_to_optical'].items()}
    results = []
    with torch.inference_mode():
        for index in a.indices:
            selected = review[index]
            matching = next(row for row in audit if row['image'] == selected['image'])
            sim_ns = int(Path(selected['image']).name.split('_')[1])
            group = by[(matching['episode'], sim_ns, int(matching['target_port']))]
            points = {}
            details = {}
            for cam, row in group.items():
                rgb = cv2.imread(row['image'])
                small = cv2.cvtColor(cv2.resize(rgb, (288, 256),
                                                 interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                x = torch.from_numpy(small.copy()).permute(2, 0, 1)[None].to(a.device).float()/255
                meta = torch.tensor([[1, 0, int(cam=='center'), int(cam=='left'),
                                      int(cam=='right')]], device=a.device).float()
                xy, flags = locator(x, meta)
                slot = LOOKUP[f"sc_port_{row['target_port']}"]
                center = xy[0, slot].float().cpu().numpy()*[1151, 1023]
                score = float(flags[0, slot].sigmoid().cpu())
                left, top = np.round(center-80).astype(int).tolist()
                crop = cv2.warpAffine(rgb, np.array([[1, 0, -left], [0, 1, -top]], np.float32),
                                      (160, 160), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                inp = torch.from_numpy(crop.copy()).permute(2, 0, 1)[None].to(a.device).float()/255
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    point = refiner(inp).float().cpu().numpy().reshape(2)
                refined = point*159+[left, top]
                points[cam] = refined
                details[cam] = {'frozen_visibility_score': score,
                                'coarse_xy': center.tolist(), 'refined_xy': refined.tolist(),
                                'geometric_exposure_proxy': row['geometric_visible_fraction']}
            residual = (float(reprojection_residual_px(triangulate(points, optics), points, optics))
                        if len(points) >= 2 else None)
            results.append({'review_index': index, 'episode': matching['episode'],
                            'sim_ns': sim_ns, 'target_port': int(matching['target_port']),
                            'review_label': selected.get('review_label',
                                                         'not_identifiable_provisional'),
                            'views': len(points), 'by_camera': details,
                            'three_view_reprojection_px': residual,
                            'exploratory_all_three_under_2px':
                                len(points) == 3 and all(x['frozen_visibility_score'] >= .5
                                                         for x in details.values()) and residual <= 2})
    out = {'schema': 'aic_sc_hard_occlusion_consistency/v1',
           'manual_review_single_reviewer_provisional': True,
           'reprojection_threshold_px_exploratory': 2,
           'exploratory_accepted_count':
               sum(r['exploratory_all_three_under_2px'] for r in results),
           'complete_three_view_false_accepts':
               sum(r['exploratory_all_three_under_2px'] and
                   r['review_label'] == 'not_identifiable_provisional' for r in results),
           'rows': results, 'locator_checkpoint':str(a.coarse),
           'decoder_repair':str(a.adapted) if a.adapted else None,
           'geometry_used_only_for_postrun_review': True,
           'no_deployment_promotion': True, 'final_scenes_opened': False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({'accepted': out['exploratory_accepted_count'],
                      'false_accepts': out['complete_three_view_false_accepts'],
                      'rows': [{k:r[k] for k in ('review_index','views',
                                                   'three_view_reprojection_px',
                                                   'exploratory_all_three_under_2px')}
                               for r in results]}, indent=2))


if __name__ == '__main__':
    main()
