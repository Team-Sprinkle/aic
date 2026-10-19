#!/usr/bin/env python3
"""Test a frozen and adapted RGB locator on a new manually reviewed SC view."""

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from train_two_family_port_locator import Locator, LOOKUP


def stats(rows, name):
    values = np.asarray([r[name] for r in rows], dtype=float)
    return dict(n=len(values), p50_px=float(np.median(values)),
                p95_px=float(np.percentile(values, 95)),
                over_50_px=int(np.sum(values > 50))) if len(values) else dict(n=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--adapted', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--visible-start-s', type=float, default=8.7)
    p.add_argument('--visible-end-s', type=float, default=23.7)
    p.add_argument('--target-port', type=int, choices=(0, 1), default=0)
    p.add_argument('--cameras', nargs='+', choices=('center','left','right'),
                   default=['left'])
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve previous result')
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    source = [json.loads(line) for line in a.rows.open()]
    # This visibility window was fixed by inspecting the original RGB montage
    # before either checkpoint was evaluated on this new scene.
    reviewed = [r for r in source if r['camera'] in a.cameras and
                a.visible_start_s <= r['sim_ns']/1e9 <= a.visible_end_s and
                r['projected_inframe'] and r['target_port'] == a.target_port]
    if len(reviewed) < 10:
        raise ValueError('Too few independently reviewed visible images')
    model = Locator(None).to(a.device).eval()
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'])
    original = copy.deepcopy(model.state_dict())
    adapted = torch.load(a.adapted, map_location='cpu', weights_only=False)
    if Path(adapted['parent_checkpoint']).resolve() != a.checkpoint.resolve():
        raise ValueError('Adapted head has different parent checkpoint')
    head = copy.deepcopy(original)
    head['decoder.4.weight'] = adapted['last_weight']
    head['decoder.4.bias'] = adapted['last_bias']
    predictions = []
    for label, state in [('original', original), ('adapted', head)]:
        model.load_state_dict(state)
        model.eval()
        for row in reviewed:
            image = cv2.imread(row['image'])
            image = cv2.cvtColor(cv2.resize(image, (288, 256),
                                            interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(image.copy()).permute(2,0,1)[None].to(a.device).float()/255
            meta = torch.tensor([[1,0,int(row['camera']=='center'),
                                  int(row['camera']=='left'),
                                  int(row['camera']=='right')]], device=a.device).float()
            with torch.inference_mode():
                xy, flags = model(x, meta)
            slot = LOOKUP[f'sc_port_{a.target_port}']
            pred = xy[0, slot].float().cpu().numpy()*[1151,1023]
            inframe_score = float(flags[0, slot].sigmoid().cpu())
            truth = np.asarray(row['projected_opening_xy'])
            predictions.append(dict(checkpoint=label, camera=row['camera'], sim_ns=row['sim_ns'],
                                    image=row['image'], predicted_xy=pred.tolist(),
                                    frozen_inframe_score=inframe_score,
                                    projected_opening_xy=truth.tolist(),
                                    error_px=float(np.linalg.norm(pred-truth))))
    output = dict(schema='aic_sc_port0_left_independent_evaluation/v1',
                  source=str(a.rows), original_checkpoint=str(a.checkpoint),
                  adapted_head=str(a.adapted), scene_group_id=reviewed[0]['scene_group_id'],
                  reviewed_visible_window_s=[a.visible_start_s,a.visible_end_s],
                  target_port=a.target_port,
                  reviewed_cameras=a.cameras,
                  original=stats([r for r in predictions if r['checkpoint']=='original'],'error_px'),
                  adapted=stats([r for r in predictions if r['checkpoint']=='adapted'],'error_px'),
                  by_camera={camera:{checkpoint:stats([r for r in predictions
                                                       if r['checkpoint']==checkpoint and
                                                       r['camera']==camera],'error_px')
                                     for checkpoint in ('original','adapted')}
                             for camera in a.cameras},
                  rows=predictions, manual_review_scope='Requested SC opening visually exposed in saved montage for the selected cameras and time window; no claim for occluded frames',
                  autonomous_actor_geometry_input=False, final_scenes_opened=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k: output[k] for k in ('original','adapted','reviewed_visible_window_s')}))


if __name__ == '__main__':
    main()
