#!/usr/bin/env python3
"""Bounded 576x512 original-RGB SFP row locator on grouped scene splits.

All port geometry is a supervised label only. At inference the model receives
RGB and camera ID; the requested task code selects one of its ten heatmaps.
Projected-in-frame is not visual visibility.
"""
import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from train_two_family_port_locator import CAMERAS, LOOKUP, Locator, evaluate

cv2.setNumThreads(1)


class NativeRows(Dataset):
    def __init__(self, rows, indices, random_camera):
        self.rows = rows
        self.indices = indices
        self.random_camera = random_camera

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, ordinal):
        index, forced = self.indices[ordinal]
        row = self.rows[index]
        ci = random.randrange(3) if self.random_camera else forced
        cam = CAMERAS[ci]
        image = cv2.imread(row['images'][cam])
        if image is None:
            raise FileNotFoundError(row['images'][cam])
        image = cv2.cvtColor(cv2.resize(image, (576, 512),
                                     interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        points = np.zeros((12, 2), np.float32)
        present = np.zeros(12, np.float32)
        inframe = np.zeros(12, np.float32)
        for target in row['targets']:
            j = LOOKUP[target['identity']]
            uv = target['cameras'][cam]
            present[j] = 1.
            if uv[3]:
                inframe[j] = 1.
                points[j] = [uv[0] / 1151., uv[1] / 1023.]
        meta = np.zeros(5, np.float32)
        meta[1] = 1.
        meta[2+ci] = 1.
        selected = LOOKUP[row['selected_identity']]
        target = next(t for t in row['targets']
                      if t['identity'] == row['selected_identity'])
        distance = float(np.linalg.norm(np.asarray(target['position_base_m']) -
                                        np.asarray(row['state'][:3])))
        return (torch.from_numpy(image.copy()).permute(2, 0, 1),
                torch.from_numpy(meta), torch.from_numpy(points),
                torch.from_numpy(present), torch.from_numpy(inframe),
                selected, distance, index, ci)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--init', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=1800)
    p.add_argument('--batch-size', type=int, default=6)
    p.add_argument('--minutes', type=float, default=45.)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve previous run')
    if a.updates > 2400:
        p.error('Bounded update cap is 2400')
    rows = [json.loads(s) for s in (a.labels / 'rows.jsonl').open()]
    train = [(i, None) for i, r in enumerate(rows) if r['split'] == 'train']
    dev = [(i, c) for i, r in enumerate(rows) if r['split'] == 'validation'
           for c in range(3)]
    train_groups = {rows[i]['scene_group'] for i, _ in train}
    dev_groups = {rows[i]['scene_group'] for i, _ in dev}
    if train_groups & dev_groups:
        raise ValueError('Scene-group leak')
    if len(train_groups) != 66 or len(dev_groups) != 10:
        raise ValueError('Unexpected grouped split')
    random.seed(20261005)
    np.random.seed(20261005)
    torch.manual_seed(20261005)
    torch.set_num_threads(4)
    a.output.mkdir(parents=True)
    train_data = NativeRows(rows, train, True)
    dev_data = NativeRows(rows, dev, False)
    loader = DataLoader(train_data, batch_size=a.batch_size,
                        sampler=RandomSampler(train_data, replacement=True,
                                              num_samples=a.updates*a.batch_size),
                        num_workers=4, pin_memory=True, persistent_workers=True,
                        multiprocessing_context='spawn')
    validation = DataLoader(dev_data, batch_size=12, num_workers=4,
                            pin_memory=True, persistent_workers=True,
                            multiprocessing_context='spawn')
    model = Locator(None).cuda()
    model.load_state_dict(torch.load(a.init, map_location='cpu',
                                     weights_only=False)['model'])
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=4e-5, weight_decay=1e-4)
    history = []
    best = float('inf')
    best_step = None
    start = time.monotonic()
    stop = 'update_cap'
    complete = 0
    for step, data in enumerate(loader, 1):
        if step > a.updates:
            break
        if time.monotonic() - start > a.minutes*60:
            stop = 'wall_budget'
            break
        image, meta, points, present, inframe, selected, _, _, _ = data
        image = image.cuda(non_blocking=True).float()/255.
        meta = meta.cuda(non_blocking=True)
        points = points.cuda(non_blocking=True)
        present = present.cuda(non_blocking=True)
        inframe = inframe.cuda(non_blocking=True)
        selected = selected.cuda(non_blocking=True)
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            xy, flag, heat = model(image, meta, return_heatmap=True)
        weights = inframe.clone()*.3
        weights[torch.arange(len(selected), device='cuda'), selected] *= 10.
        coord = F.smooth_l1_loss(xy.float(), points, beta=.01,
                                 reduction='none').sum(-1)
        h, w = heat.shape[-2:]
        cell = ((points[..., 1]*(h-1)).round().long().clamp(0, h-1)*w +
                (points[..., 0]*(w-1)).round().long().clamp(0, w-1))
        spatial = F.cross_entropy((heat.float()*4).flatten(0, 1).flatten(1),
                                  cell.flatten(), reduction='none').view_as(inframe)
        loss = ((coord + .02*spatial)*weights).sum()/weights.sum().clamp_min(1)
        flag_mask = torch.zeros_like(present)
        flag_mask[:, 2:] = 1.
        flag_loss = F.binary_cross_entropy_with_logits(flag.float(), inframe,
                                                        reduction='none')
        loss += .05*(flag_loss*flag_mask).sum()/flag_mask.sum().clamp_min(1)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        complete = step
        if step % 300 == 0:
            report = evaluate(model, validation, 'cuda')
            wrong = report['selected_point_closer_to_alternate_fraction'].get('sfp', 1.)
            selected_error = report['errors'].get("('sfp', 'selected', 'all')", {})
            p95 = selected_error.get('p95_px', 1e6)
            score = 10000*wrong + p95
            entry = {'step': step, 'loss': float(loss.detach()),
                     'score': score, 'development': report}
            history.append(entry)
            print(json.dumps({'step': step, 'loss': entry['loss'],
                              'wrong_view_fraction': wrong,
                              'selected_p95_px': p95}), flush=True)
            torch.save({'model': model.state_dict(), 'step': step,
                        'input_size': [576, 512],
                        'runtime_true_geometry_input': False},
                       a.output / f'checkpoint_step{step}.pt')
            if score < best:
                best = score
                best_step = step
                torch.save({'model': model.state_dict(), 'step': step,
                            'input_size': [576, 512],
                            'runtime_true_geometry_input': False},
                           a.output / 'best.pt')
    if best_step is None:
        raise RuntimeError('No checkpoint saved')
    summary = {'schema': 'sfp_native_row_locator/v1',
               'train_episodes': len(train_groups),
               'development_episodes': len(dev_groups),
               'train_image_triplets': len(train),
               'development_image_triplets': len(dev)//3,
               'best_step': best_step, 'updates_completed': complete,
               'stop_reason': stop, 'input_size': [576, 512],
               'selected': next(x for x in history if x['step'] == best_step),
               'runtime_true_geometry_input': False,
               'true_visibility_supervision': False,
               'final_scenes_opened': False}
    (a.output / 'history.json').write_text(json.dumps(history, indent=2)+'\n')
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
