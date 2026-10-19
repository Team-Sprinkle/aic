#!/usr/bin/env python3
"""One bounded requested-port cue from frozen RGB board-slot features."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate, reprojection_residual_px
from train_sfp_board_reference import BoardReferenceLocal


CAMS = ('center', 'left', 'right')


class Rows(Dataset):
    def __init__(self, rows, indices):
        self.rows, self.indices = rows, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        index = self.indices[item]
        row = self.rows[index]
        images = []
        points = np.zeros((3, 5, 2, 2), np.float32)
        inframe = np.zeros((3, 5, 2), np.float32)
        for ci, cam in enumerate(CAMS):
            image = cv2.imread(row['images'][cam], cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(row['images'][cam])
            image = cv2.cvtColor(cv2.resize(image, (576, 512), interpolation=cv2.INTER_AREA),
                                 cv2.COLOR_BGR2RGB)
            images.append(np.transpose(image.copy(), (2, 0, 1)))
        for target in row['targets']:
            bits = target['identity'].split('_')
            card, port = int(bits[2]), int(bits[4])
            for ci, cam in enumerate(CAMS):
                point = target['cameras'][cam]
                if point[3]:
                    points[ci, card, port] = (point[0] / 1151., point[1] / 1023.)
                    inframe[ci, card, port] = 1.
        return (torch.from_numpy(np.stack(images)),
                torch.from_numpy(np.asarray(row['state'][:7], np.float32)),
                torch.from_numpy(points), torch.from_numpy(inframe), index)


class SlotPortCue(nn.Module):
    def __init__(self, board_checkpoint):
        super().__init__()
        board_cp = torch.load(board_checkpoint, map_location='cpu', weights_only=False)
        self.board = BoardReferenceLocal(Path(board_cp['init']))
        self.board.load_state_dict(board_cp['model'])
        self.board.requires_grad_(False)
        self.port_head = nn.Sequential(nn.LayerNorm(659), nn.Linear(659, 192), nn.GELU(),
                                       nn.Linear(192, 96), nn.GELU(), nn.Linear(96, 3))

    def train(self, mode=True):
        super().train(mode)
        self.board.eval()
        return self

    def forward(self, images, state):
        batch = images.shape[0]
        with torch.no_grad():
            (ends, _, occupancy, _), local, global_feature = self.board(
                images, return_slot_features=True)
        fraction = torch.linspace(0, 1, 5, device=images.device)[None, None, :, None]
        slots = ends[:, :, :1] * (1 - fraction) + ends[:, :, 1:] * fraction
        local = local[:, None, :, None].expand(-1, 3, -1, 2, -1)
        global_feature = global_feature[:, None, None, None].expand(-1, 3, 5, 2, -1)
        state = state[:, None, None, None].expand(-1, 3, 5, 2, -1)
        cam = F.one_hot(torch.arange(3, device=images.device), 3).float()[None, :, None, None]
        cam = cam.expand(batch, -1, 5, 2, -1)
        card = F.one_hot(torch.arange(5, device=images.device), 5).float()[None, None, :, None]
        card = card.expand(batch, 3, -1, 2, -1)
        port = F.one_hot(torch.arange(2, device=images.device), 2).float()[None, None, None]
        port = port.expand(batch, 3, 5, -1, -1)
        slot = slots[:, :, :, None].expand(-1, -1, -1, 2, -1)
        features = torch.cat((local, global_feature, state, slot, card, port, cam), -1)
        if features.shape[-1] != 659:
            raise ValueError(f'Feature size {features.shape[-1]}')
        result = self.port_head(features)
        xy = slot + .4 * torch.tanh(result[..., :2])
        return xy, result[..., 2], occupancy


@torch.inference_mode()
def evaluate(model, rows, loader, optics, save=None):
    model.eval()
    records = []
    for images, state, _, _, indices in loader:
        images = images.cuda(non_blocking=True).float() / 255.
        state = state.cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred, logit, occupied = model(images, state)
        pred = pred.float().cpu().numpy()
        prob = logit.float().sigmoid().cpu().numpy()
        occupied = occupied.float().sigmoid().cpu().numpy()
        for bi, index in enumerate(indices.tolist()):
            row = rows[index]
            bits = row['selected_identity'].split('_')
            card, port = int(bits[2]), int(bits[4])
            pixels = {cam: (pred[bi, ci, card, port] * [1151, 1023]).tolist()
                      for ci, cam in enumerate(CAMS)}
            rec = {'episode': row['episode'], 'scene_group': row['scene_group'],
                   'time_s': row['time_s'], 'requested': row['selected_identity'],
                   'accepted': False, 'wrong': False,
                   'requested_occupancy_prob': float(occupied[bi, card]),
                   'view_probability': {cam: float(prob[bi, ci, card, port])
                                        for ci, cam in enumerate(CAMS)}}
            if rec['requested_occupancy_prob'] < .5 or any(
                    rec['view_probability'][cam] < .5 for cam in CAMS):
                rec['reason'] = 'flag_or_occupancy'
            else:
                cameras = {cam: pose_matrix(row['state']) @ optics[cam] for cam in CAMS}
                try:
                    point = triangulate(pixels, cameras)
                    residual = reprojection_residual_px(point, pixels, cameras)
                except np.linalg.LinAlgError:
                    point, residual = None, float('inf')
                rec['reprojection_px'] = float(residual)
                if point is None or not np.isfinite(point).all() or residual > 5:
                    rec['reason'] = 'reprojection'
                else:
                    nearest = min(row['targets'], key=lambda x: np.linalg.norm(
                        point - np.asarray(x['position_base_m'])))
                    truth = next(x for x in row['targets']
                                 if x['identity'] == row['selected_identity'])
                    rec.update(accepted=True, reason='accepted',
                               predicted_base_m=point.tolist(),
                               error_mm=float(np.linalg.norm(
                                   point - np.asarray(truth['position_base_m'])) * 1000),
                               nearest_identity_label_only=nearest['identity'],
                               wrong=nearest['identity'] != row['selected_identity'])
            records.append(rec)
    by_episode = defaultdict(list)
    for record in records:
        by_episode[record['scene_group']].append(record)
    early = 0
    for group in by_episode.values():
        group.sort(key=lambda x: x['time_s'])
        early += any(x['accepted'] and not x['wrong'] and
                     x['time_s'] - group[0]['time_s'] <= 3 for x in group)
    accepted = [x for x in records if x['accepted']]
    result = {'rows': len(records), 'episodes': len(by_episode), 'accepted': len(accepted),
              'wrong': sum(x['wrong'] for x in accepted), 'early_correct_episodes': early,
              'accepted_3d_p95_mm': float(np.percentile(
                  [x['error_mm'] for x in accepted], 95)) if accepted else None}
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
        raise ValueError(f'Unexpected complete-reset split {len(train_groups)}/{len(dev_groups)}')
    random.seed(20261007)
    np.random.seed(20261007)
    torch.manual_seed(20261007)
    torch.set_num_threads(4)
    frequency = defaultdict(int)
    for i in train:
        frequency[rows[i]['scene_group']] += 1
    weights = [1 / frequency[rows[i]['scene_group']] for i in train]
    sampler = WeightedRandomSampler(weights, args.updates * 6, replacement=True)
    loader = DataLoader(Rows(rows, train), batch_size=6, sampler=sampler,
                        num_workers=2, pin_memory=True, persistent_workers=True,
                        multiprocessing_context='spawn')
    dev_loader = DataLoader(Rows(rows, dev), batch_size=6, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True,
                            multiprocessing_context='spawn')
    calibration = json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics = {cam: np.asarray(calibration['tcp_to_optical'][cam]) for cam in CAMS}
    model = SlotPortCue(args.board_checkpoint).cuda()
    optimizer = torch.optim.AdamW(model.port_head.parameters(), lr=2e-4, weight_decay=1e-4)
    args.output.mkdir(parents=True)
    history = []
    best_score = float('inf')
    best_step = None
    start = time.monotonic()
    reason = 'update_cap'
    for step, (images, state, points, mask, _) in enumerate(loader, 1):
        if step > args.updates:
            break
        if time.monotonic() - start > args.minutes * 60:
            reason = 'time_cap'
            break
        images = images.cuda(non_blocking=True).float() / 255.
        state = state.cuda(non_blocking=True)
        points = points.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred, logit, _ = model(images, state)
        coordinate = F.smooth_l1_loss(pred.float(), points, beta=.02,
                                      reduction='none').sum(-1)
        loss = (coordinate * mask).sum() / mask.sum().clamp_min(1)
        loss = loss + .1 * F.binary_cross_entropy_with_logits(logit.float(), mask)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.port_head.parameters(), 1.)
        optimizer.step()
        if step % 250 == 0:
            report = evaluate(model, rows, dev_loader, optics,
                              args.output / f'predictions_step{step}.jsonl')
            score = (1000 * report['wrong'] + 40 * (10 - report['early_correct_episodes']) +
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
    summary = {'schema': 'sfp_slot_feature_port_cue/v1',
               'train_groups': len(train_groups), 'development_groups': len(dev_groups),
               'steps_completed': step, 'stop_reason': reason, 'best_step': best_step,
               'selected': selected,
               'gate_passed': bool(selected['wrong'] == 0 and
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
