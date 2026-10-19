#!/usr/bin/env python3
"""Audit causal SC identity anchors on saved one-port development approaches.

This does not train, authorize handoff, or use geometry as an inference input.
Projection rows supply timestamps and posthoc labels only; the frozen locator
and identity head see RGB, camera/request code, and measured TCP pose.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from train_sc_spatial_identity_probe import SelectedHead
from train_two_family_port_locator import Locator

CAMERAS = ('center', 'left', 'right')


def load_triplets(paths):
    output = []
    for path in paths:
        by_camera = defaultdict(list)
        for line in path.open():
            row = json.loads(line)
            by_camera[row['camera']].append(row)
        if set(by_camera) != set(CAMERAS):
            raise ValueError(f'Missing camera in {path}')
        for cam in CAMERAS:
            by_camera[cam].sort(key=lambda x: int(x['sim_ns']))
        for center in by_camera['center']:
            stamp = int(center['sim_ns'])
            views = {'center': center}
            for cam in CAMERAS[1:]:
                views[cam] = min(by_camera[cam],
                                 key=lambda row: abs(int(row['sim_ns'])-stamp))
            max_offset = max(abs(int(v['sim_ns'])-stamp) for v in views.values())
            if max_offset > 75_000_000:
                continue
            first = center
            if any(v['target_port'] != first['target_port'] for v in views.values()):
                raise ValueError('Inconsistent requested port in triplet')
            group = (path.parents[1].name if path.parent.name == 'projection_state_v2'
                     else path.parent.name)
            output.append({'group': group, 'sim_ns': stamp,
                           'requested_port': int(first['target_port']),
                           'views': views, 'tcp': first['measured_tcp_base_pose'],
                           'max_camera_offset_ms': max_offset/1e6})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('rows', nargs='+', type=Path)
    p.add_argument('--locator', required=True, type=Path)
    p.add_argument('--identity', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--batch', type=int, default=24)
    p.add_argument('--both-ports-present', action='store_true',
                   help='Label the alternate request as a present neighbor, not absent')
    a = p.parse_args()
    if a.output.exists():
        p.error('Refusing to overwrite existing audit')
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    triplets = load_triplets(a.rows)
    saved = torch.load(a.identity, map_location='cpu', weights_only=False)
    if saved['variant'] != 'spatial':
        raise ValueError('Requires spatial identity head')
    locator = Locator(None).to(a.device).eval()
    locator.load_state_dict(torch.load(a.locator, map_location='cpu', weights_only=False)['model'])
    head = SelectedHead(True, 256, 12).to(a.device).eval()
    head.load_state_dict(saved['model'])
    samples = [(item, cam) for item in triplets for cam in CAMERAS]
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(samples), a.batch):
            batch = samples[start:start+a.batch]
            images = []
            for item, cam in batch:
                image = cv2.imread(item['views'][cam]['image'])
                if image is None:
                    raise FileNotFoundError(item['views'][cam]['image'])
                images.append(cv2.cvtColor(cv2.resize(image, (288, 256),
                                            interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(a.device).float()/255
            meta = torch.zeros((len(batch), 5), device=a.device)
            for j, (_, cam) in enumerate(batch):
                meta[j, 0] = 1
                meta[j, 2+CAMERAS.index(cam)] = 1
            points, flags, _, spatial = locator(x, meta, return_features=True,
                                                 return_spatial=True)
            conditions = []
            for item, cam in batch:
                state = (torch.as_tensor(item['tcp'], device=a.device, dtype=torch.float32)
                         - saved['state_mean'].to(a.device))/saved['state_std'].to(a.device)
                if saved.get('mask_tcp_orientation'):
                    state[3:7] = 0
                for req in (0, 1):
                    c = torch.zeros(12, device=a.device)
                    c[CAMERAS.index(cam)] = 1
                    c[3:10] = state
                    c[10+req] = 1
                    conditions.append(c)
            conditions = torch.stack(conditions)
            scores = head(spatial.repeat_interleave(2, 0).float(), conditions).sigmoid()
            scores = scores.reshape(len(batch), 2).cpu().numpy()
            points = points.float().cpu().numpy()
            flags = flags.float().sigmoid().cpu().numpy()
            for j, (item, cam) in enumerate(batch):
                predictions.append({'group': item['group'], 'sim_ns': item['sim_ns'],
                                    'camera': cam, 'requested_port': item['requested_port'],
                                    'max_camera_offset_ms': item['max_camera_offset_ms'],
                                    'image': item['views'][cam]['image'],
                                    'tcp': item['tcp'], 'identity_scores': scores[j].tolist(),
                                    'locator_flags': flags[j].tolist(),
                                    'locator_pixels': (points[j]*[1151, 1023]).tolist(),
                                    'projected_inframe': bool(item['views'][cam]['projected_inframe'])})
    grouped = defaultdict(list)
    for row in predictions:
        grouped[(row['group'], row['sim_ns'])].append(row)
    triplet_rows = []
    for (group, stamp), views in sorted(grouped.items()):
        views = sorted(views, key=lambda x: CAMERAS.index(x['camera']))
        req = views[0]['requested_port']
        triplet_rows.append({'group': group, 'sim_ns': stamp, 'requested_port': req,
                             'tcp': views[0]['tcp'],
                             'max_camera_offset_ms': views[0]['max_camera_offset_ms'],
                             'scores_requested': [v['identity_scores'][req] for v in views],
                             'scores_absent': [v['identity_scores'][1-req] for v in views],
                             'flags_requested': [v['locator_flags'][req] for v in views],
                             'flags_absent': [v['locator_flags'][1-req] for v in views],
                             'projected_inframe': [v['projected_inframe'] for v in views],
                             'requested_pixels': [v['locator_pixels'][req] for v in views],
                             'images': [v['image'] for v in views],
                             'absent_pixels': [v['locator_pixels'][1-req] for v in views]})
    metrics = {}
    for group in sorted({r['group'] for r in triplet_rows}):
        rows = [r for r in triplet_rows if r['group'] == group]
        clear = [r for r in rows if all(r['projected_inframe'])]
        metrics[group] = {'triplets': len(rows), 'projected_triplets': len(clear),
                          'requested_all_view_accept': sum(min(r['scores_requested']) >= .5 for r in clear),
                          ('other_port_all_view_accept' if a.both_ports_present
                           else 'absent_all_view_accept'):
                              sum(min(r['scores_absent']) >= .5 for r in rows),
                          'first_requested_anchor_ns': next((r['sim_ns'] for r in rows
                              if min(r['scores_requested']) >= .5 and all(r['projected_inframe'])), None)}
    result = {'schema': 'aic_sc_sparse_history_identity_audit/v1',
              'label_caution': 'projected_inframe is not reviewed visibility',
              'one_port_absent_request_assumption': not a.both_ports_present,
              'identity': str(a.identity), 'locator': str(a.locator),
              'input_rows': [str(x) for x in a.rows], 'metrics': metrics,
              'triplets': triplet_rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
