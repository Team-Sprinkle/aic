#!/usr/bin/env python3
"""Bounded 3D-consistent SFP target cue from frozen board-slot image features."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from audit_sc_port_targets import pose_matrix
from build_sc_native_pose_labels import FOCAL, WIDTH, HEIGHT
from train_sfp_slot_feature_port_cue import CAMS, Rows, SlotPortCue


class Rows3D(Rows):
    def __getitem__(self, item):
        images, state, _, _, index = super().__getitem__(item)
        positions = np.zeros((5, 2, 3), np.float32)
        present = np.zeros((5, 2), np.float32)
        for target in self.rows[index]['targets']:
            bits = target['identity'].split('_')
            card, port = int(bits[2]), int(bits[4])
            positions[card, port] = target['position_base_m']
            present[card, port] = 1.
        return images, state, torch.from_numpy(positions), torch.from_numpy(present), index


class Shared3DCue(SlotPortCue):
    def __init__(self, board_checkpoint):
        super().__init__(board_checkpoint)
        del self.port_head
        self.port_head = nn.Sequential(nn.LayerNorm(660), nn.Linear(660, 192), nn.GELU(),
                                       nn.Linear(192, 96), nn.GELU(), nn.Linear(96, 4))

    def forward(self, images, state):
        batch = images.shape[0]
        with torch.no_grad():
            (ends, _, occupancy, _), local, global_feature = self.board(
                images, return_slot_features=True)
        fraction = torch.linspace(0, 1, 5, device=images.device)[None, None, :, None]
        slot = ends[:, :, :1] * (1 - fraction) + ends[:, :, 1:] * fraction
        slot = slot.permute(0, 2, 1, 3).reshape(batch, 5, 6)
        features = torch.cat((
            local[:, :, None].expand(-1, -1, 2, -1),
            global_feature[:, None, None].expand(-1, 5, 2, -1),
            state[:, None, None].expand(-1, 5, 2, -1),
            slot[:, :, None].expand(-1, -1, 2, -1),
            F.one_hot(torch.arange(5, device=images.device), 5).float()[None, :, None]
            .expand(batch, -1, 2, -1),
            F.one_hot(torch.arange(2, device=images.device), 2).float()[None, None]
            .expand(batch, 5, -1, -1)), -1)
        if features.shape[-1] != 660:
            raise ValueError(f'Feature size {features.shape[-1]}')
        raw = self.port_head(features)
        xyz = state[:, None, None, :3] + .6 * torch.tanh(raw[..., :3])
        sigma = .003 + .1 * F.softplus(raw[..., 3])
        return xyz, sigma, occupancy


def projects_all_views(point, state, optics):
    base_tcp = pose_matrix(state)
    for cam in CAMS:
        inv = np.linalg.inv(base_tcp @ optics[cam])
        optical = inv[:3, :3] @ point + inv[:3, 3]
        if optical[2] <= .02:
            return False
        x = WIDTH / 2 + FOCAL * optical[0] / optical[2]
        y = HEIGHT / 2 + FOCAL * optical[1] / optical[2]
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return False
    return True


@torch.inference_mode()
def evaluate(model, rows, loader, optics, save=None):
    model.eval()
    records = []
    for images, state, _, _, indices in loader:
        images = images.cuda(non_blocking=True).float() / 255.
        state = state.cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            xyz, sigma, occupancy = model(images, state)
        xyz = xyz.float().cpu().numpy()
        sigma = sigma.float().cpu().numpy()
        occupancy = occupancy.float().sigmoid().cpu().numpy()
        for bi, index in enumerate(indices.tolist()):
            row = rows[index]
            bits = row['selected_identity'].split('_')
            card, port = int(bits[2]), int(bits[4])
            point = xyz[bi, card, port]
            true = next(t for t in row['targets'] if t['identity'] == row['selected_identity'])
            nearest = min(row['targets'], key=lambda t: np.linalg.norm(
                point - np.asarray(t['position_base_m'])))
            error = float(np.linalg.norm(point - np.asarray(true['position_base_m'])) * 1000)
            uncertainty = float(sigma[bi, card, port] * 1000)
            accepted = bool(occupancy[bi, card] >= .5 and uncertainty <= 20 and
                            projects_all_views(point, row['state'], optics))
            records.append({'episode': row['episode'], 'scene_group': row['scene_group'],
                            'time_s': row['time_s'], 'requested': row['selected_identity'],
                            'predicted_base_m': point.tolist(), 'error_mm': error,
                            'predicted_uncertainty_mm': uncertainty,
                            'requested_occupancy_prob': float(occupancy[bi, card]),
                            'nearest_identity_label_only': nearest['identity'],
                            'wrong': nearest['identity'] != row['selected_identity'],
                            'accepted': accepted})
    by_episode = defaultdict(list)
    for record in records:
        by_episode[record['scene_group']].append(record)
    early = 0
    for group in by_episode.values():
        group.sort(key=lambda x: x['time_s'])
        early += any(x['accepted'] and not x['wrong'] and
                     x['time_s'] - group[0]['time_s'] <= 3 for x in group)
    accepted = [x for x in records if x['accepted']]
    result = {'rows': len(records), 'episodes': len(by_episode),
              'accepted': len(accepted), 'wrong_accepted': sum(x['wrong'] for x in accepted),
              'early_correct_episodes': early,
              'all_3d_median_mm': float(np.median([x['error_mm'] for x in records])),
              'all_3d_p95_mm': float(np.percentile([x['error_mm'] for x in records], 95)),
              'accepted_3d_p95_mm': float(np.percentile(
                  [x['error_mm'] for x in accepted], 95)) if accepted else None,
              'uncertainty_20mm_precision': (sum(x['error_mm'] <= 20 for x in accepted) /
                                              len(accepted)) if accepted else None}
    if save:
        with save.open('w') as target:
            for record in records:
                target.write(json.dumps(record) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--board-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=2000)
    parser.add_argument('--minutes', type=float, default=30)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve previous output')
    if args.updates > 2000 or args.minutes > 30:
        parser.error('Bounded 2000-update/30-minute diagnostic')
    rows = [json.loads(line) for line in (args.labels / 'rows.jsonl').open()]
    train = [i for i, row in enumerate(rows) if row['split'] == 'train']
    dev = [i for i, row in enumerate(rows) if row['split'] == 'validation']
    train_groups = {rows[i]['scene_group'] for i in train}
    dev_groups = {rows[i]['scene_group'] for i in dev}
    if len(train_groups) != 70 or len(dev_groups) != 10 or train_groups & dev_groups:
        raise ValueError('Unexpected complete-reset split')
    random.seed(20261007)
    np.random.seed(20261007)
    torch.manual_seed(20261007)
    torch.set_num_threads(4)
    frequency = defaultdict(int)
    for i in train:
        frequency[rows[i]['scene_group']] += 1
    weights = [1 / frequency[rows[i]['scene_group']] for i in train]
    sampler = WeightedRandomSampler(weights, args.updates * 6, replacement=True)
    loader = DataLoader(Rows3D(rows, train), batch_size=6, sampler=sampler,
                        num_workers=2, pin_memory=True, persistent_workers=True,
                        multiprocessing_context='spawn')
    dev_loader = DataLoader(Rows3D(rows, dev), batch_size=6, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True,
                            multiprocessing_context='spawn')
    calibration = json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics = {cam: np.asarray(calibration['tcp_to_optical'][cam]) for cam in CAMS}
    model = Shared3DCue(args.board_checkpoint).cuda()
    optimizer = torch.optim.AdamW(model.port_head.parameters(), lr=2e-4, weight_decay=1e-4)
    args.output.mkdir(parents=True)
    history = []
    best_score = float('inf')
    best_step = None
    start = time.monotonic()
    reason = 'update_cap'
    for step, (images, state, truth, present, _) in enumerate(loader, 1):
        if step > args.updates:
            break
        if time.monotonic() - start > args.minutes * 60:
            reason = 'time_cap'
            break
        images = images.cuda(non_blocking=True).float() / 255.
        state = state.cuda(non_blocking=True)
        truth = truth.cuda(non_blocking=True)
        present = present.cuda(non_blocking=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred, sigma, _ = model(images, state)
        distance = torch.linalg.vector_norm(pred.float() - truth, dim=-1)
        sigma = sigma.float()
        coordinate = F.smooth_l1_loss(pred.float(), truth, beta=.02,
                                      reduction='none').sum(-1)
        calibrated = distance / sigma + 3 * torch.log(sigma)
        loss = ((coordinate + .005 * calibrated) * present).sum() / present.sum().clamp_min(1)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.port_head.parameters(), 1.)
        optimizer.step()
        if step % 250 == 0:
            report = evaluate(model, rows, dev_loader, optics,
                              args.output / f'predictions_step{step}.jsonl')
            score = (1000 * report['wrong_accepted'] +
                     40 * (10 - report['early_correct_episodes']) +
                     (report['accepted_3d_p95_mm'] or 1000))
            history.append({'step': step, 'loss': float(loss.detach()),
                            'score': score, 'development': report})
            print(json.dumps(history[-1]), flush=True)
            if score < best_score:
                best_score, best_step = score, step
                torch.save({'port_head': model.port_head.state_dict(),
                            'board_checkpoint': str(args.board_checkpoint),
                            'step': step, 'geometry_runtime_input': False},
                           args.output / 'best.pt')
    if best_step is None:
        raise RuntimeError('No checkpoint before stop')
    selected = next(x['development'] for x in history if x['step'] == best_step)
    summary = {'schema': 'sfp_slot_feature_port_3d/v1',
               'train_groups': len(train_groups), 'development_groups': len(dev_groups),
               'steps_completed': step, 'stop_reason': reason, 'best_step': best_step,
               'selected': selected,
               'gate_passed': bool(selected['wrong_accepted'] == 0 and
                                   selected['early_correct_episodes'] >= 9 and
                                   selected['accepted_3d_p95_mm'] is not None and
                                   selected['accepted_3d_p95_mm'] <= 20),
               'final_scenes_opened': False, 'runtime_geometry_input': False,
               'board_checkpoint': str(args.board_checkpoint)}
    (args.output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
