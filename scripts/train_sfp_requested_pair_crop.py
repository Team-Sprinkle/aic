#!/usr/bin/env python3
"""Bounded original-image crop model for both ports of the requested SFP card.

Training geometry is label-only. Runtime crops are observation-derived.
No checkpoint or acceptance gate is promoted by this script.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import cv2  # Resolve Pixi image libraries before importing Torch.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate, reprojection_residual_px
from train_sfp_observed_rail_crop import crop_native
from train_target_conditioned_port_locator import ConditionalLocator
from train_two_family_port_locator import CAMERAS


def read_jsonl(path):
    return [json.loads(line) for line in path.open()]


def key(row):
    return row['episode'], round(row['time_s'], 3)


def assemble_rows(labels_path, train_proposals, fresh_proposals, val_proposals):
    labels = read_jsonl(labels_path / 'rows.jsonl')
    lookup = {key(row): row for row in labels}
    result = []
    for split, source in [('train', train_proposals), ('train', fresh_proposals),
                          ('validation', val_proposals)]:
        for proposal in read_jsonl(source / 'rows.jsonl'):
            label = lookup.get(key(proposal))
            if label is None or label['fit_split'] != split or not proposal['crop']:
                continue
            result.append({'label': label, 'crop': proposal['crop'], 'split': split})
    train_groups = {r['label']['scene_group'] for r in result if r['split'] == 'train'}
    val_groups = {r['label']['scene_group'] for r in result if r['split'] == 'validation'}
    if train_groups & val_groups:
        raise ValueError('Complete reset group leakage')
    if len(train_groups) < 50 or len(val_groups) != 10:
        raise ValueError(f'Unexpected crop group counts: {len(train_groups)}, {len(val_groups)}')
    first_time = defaultdict(lambda: float('inf'))
    for r in labels:
        if r['fit_split'] == 'validation':
            first_time[r['episode']] = min(first_time[r['episode']], r['time_s'])
    return result, first_time, len(train_groups), len(val_groups)


class PairCrops(Dataset):
    def __init__(self, rows, indices):
        self.rows = rows
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        ri, forced_camera = self.indices[index]
        row = self.rows[ri]
        label = row['label']
        camera_index = random.randrange(3) if forced_camera is None else forced_camera
        camera = CAMERAS[camera_index]
        source = cv2.imread(label['images'][camera], cv2.IMREAD_COLOR)
        if source is None:
            raise FileNotFoundError(label['images'][camera])
        center = np.asarray(row['crop']['center_px'][camera], np.float32)
        image = crop_native(source, center)
        card = int(label['selected_identity'].split('_')[2])
        xy = np.full((2, 2), .5, np.float32)
        inside = np.zeros(2, np.float32)
        for port in range(2):
            target = next(t for t in label['targets']
                          if t['identity'] == f'sfp_card_{card}_port_{port}')
            pixel = target['cameras'][camera]
            if pixel[3]:
                local = .5 + (np.asarray(pixel[:2]) - center) / 480
                if np.all((0 <= local) & (local <= 1)):
                    xy[port] = local
                    inside[port] = 1
        meta = np.zeros(5, np.float32)
        meta[1] = 1
        meta[2 + camera_index] = 1
        return (torch.from_numpy(image.copy()).permute(2, 0, 1),
                torch.from_numpy(meta), card, torch.from_numpy(xy),
                torch.from_numpy(inside), ri, camera_index)


class RequestedPair(nn.Module):
    def __init__(self, init):
        super().__init__()
        previous = ConditionalLocator(None)
        previous.load_state_dict(torch.load(init, map_location='cpu', weights_only=False)['model'])
        self.backbone = previous.backbone
        self.identity = nn.Embedding(5, 8)
        self.decoder = nn.Sequential(
            nn.Conv2d(64 + 256 + 5 + 8, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 2, 1))
        self.flags = nn.Sequential(nn.Linear(256 + 5 + 8, 128), nn.GELU(),
                                   nn.Linear(128, 2))
        self.register_buffer('mean', torch.tensor((.485, .456, .406))[None, :, None, None])
        self.register_buffer('std', torch.tensor((.229, .224, .225))[None, :, None, None])

    def train(self, mode=True):
        super().train(mode)
        self.backbone.bn1.eval()
        self.backbone.layer1.eval()
        self.backbone.layer2.eval()
        return self

    def forward(self, rgb, meta, card):
        x = (rgb - self.mean) / self.std
        x = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        early = self.backbone.layer1(x)
        deep = self.backbone.layer3(self.backbone.layer2(early))
        condition = torch.cat((meta, self.identity(card)), -1)
        features = torch.cat((early, F.interpolate(deep, size=early.shape[-2:],
                         mode='bilinear', align_corners=False),
                         condition[:, :, None, None].expand(-1, -1, *early.shape[-2:])), 1)
        heat = self.decoder(features)
        batch, _, height, width = heat.shape
        probability = F.softmax(heat.reshape(batch, 2, -1).float() * 4, dim=-1)
        yy, xx = torch.meshgrid(torch.linspace(0, 1, height, device=rgb.device),
                                torch.linspace(0, 1, width, device=rgb.device), indexing='ij')
        points = torch.stack(((probability * xx.flatten()).sum(-1),
                              (probability * yy.flatten()).sum(-1)), -1)
        flags = self.flags(torch.cat((deep.mean((-1, -2)), condition), -1))
        return points, flags, heat


@torch.inference_mode()
def evaluate(model, rows, loader, optics, first_time, save):
    model.eval()
    views = defaultdict(dict)
    for image, meta, card, _, _, row_index, camera_index in loader:
        xy, logits, _ = model(image.cuda().float() / 255., meta.cuda(), card.cuda())
        xy = xy.cpu().numpy()
        probability = logits.sigmoid().cpu().numpy()
        for i in range(len(xy)):
            views[int(row_index[i])][CAMERAS[int(camera_index[i])]] = {
                'xy': xy[i].tolist(), 'probability': probability[i].tolist()}
    records = []
    for ri, by_camera in sorted(views.items()):
        item = rows[ri]
        label = item['label']
        card = int(label['selected_identity'].split('_')[2])
        port = int(label['selected_identity'].split('_')[-1])
        rec = {'episode': label['episode'], 'scene_group': label['scene_group'],
               'time_s': label['time_s'], 'requested': label['selected_identity'],
               'accepted': False, 'reason': None, 'predicted_base_m': None,
               'error_mm_label_only': None, 'nearest_identity_label_only': None}
        if len(by_camera) != 3:
            rec['reason'] = 'missing_camera'
            records.append(rec)
            continue
        poses = {c: pose_matrix(label['state']) @ optics[c] for c in CAMERAS}
        points = []
        reprojections = []
        for p in (0, 1):
            if any(by_camera[c]['probability'][p] < .5 for c in CAMERAS):
                rec['reason'] = 'port_not_confident'
                break
            pixels = {c: (np.asarray(item['crop']['center_px'][c]) +
                          (np.asarray(by_camera[c]['xy'][p]) - .5) * 480).tolist()
                      for c in CAMERAS}
            try:
                point = triangulate(pixels, poses)
                residual = reprojection_residual_px(point, pixels, poses)
            except np.linalg.LinAlgError:
                rec['reason'] = 'triangulation'
                break
            if not np.isfinite(point).all() or residual > 5:
                rec['reason'] = 'cross_view'
                break
            points.append(point)
            reprojections.append(float(residual))
        if len(points) == 2:
            delta = points[1] - points[0]
            gap = float(np.linalg.norm(delta) * 1000)
            rec['pair_gap_mm'] = gap
            rec['pair_direction_x_mm'] = float(-delta[0] * 1000)
            rec['pair_reprojection_px'] = max(reprojections)
            if not 15 <= gap <= 35 or -delta[0] * 1000 < 15:
                rec['reason'] = 'pair_order_or_spacing'
            else:
                rec['accepted'] = True
                rec['reason'] = 'accepted'
                rec['predicted_base_m'] = points[port].tolist()
                # The next fields are posthoc scores only.
                nearest = min(label['targets'], key=lambda t: np.linalg.norm(
                    points[port] - np.asarray(t['position_base_m'])))
                requested = next(t for t in label['targets']
                                 if t['identity'] == label['selected_identity'])
                rec['nearest_identity_label_only'] = nearest['identity']
                rec['error_mm_label_only'] = float(np.linalg.norm(
                    points[port] - np.asarray(requested['position_base_m'])) * 1000)
        records.append(rec)
    with save.open('w') as output:
        for record in records:
            output.write(json.dumps(record, separators=(',', ':')) + '\n')
    accepted = [r for r in records if r['accepted']]
    wrong = [r for r in accepted if r['nearest_identity_label_only'] != r['requested']]
    by_episode = defaultdict(list)
    for r in records:
        by_episode[r['episode']].append(r)
    early = sum(any(r['accepted'] and r['time_s'] - first_time[episode] <= 3
                    for r in group) for episode, group in by_episode.items())
    return {'proposed_rows': len(records), 'accepted': len(accepted), 'wrong': len(wrong),
            'early_episodes': early,
            'episodes_any': sum(any(r['accepted'] for r in group)
                                for group in by_episode.values()),
            'accepted_p95_mm': float(np.quantile([r['error_mm_label_only'] for r in accepted], .95))
            if accepted else None,
            'reasons': dict((reason, sum(r['reason'] == reason for r in records))
                            for reason in sorted({r['reason'] for r in records})),
            'no_true_geometry_in_acceptance': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--train-proposals', type=Path, required=True)
    parser.add_argument('--fresh-proposals', type=Path, required=True)
    parser.add_argument('--development-proposals', type=Path, required=True)
    parser.add_argument('--init', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=1200)
    parser.add_argument('--minutes', type=float, default=60)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve previous fit')
    if args.updates > 1200 or args.minutes > 60:
        parser.error('Budget is at most 1,200 updates / 60 minutes')
    args.output.mkdir(parents=True)
    rows, first_time, train_groups, val_groups = assemble_rows(
        args.labels, args.train_proposals, args.fresh_proposals, args.development_proposals)
    train_indices = [(i, None) for i, r in enumerate(rows) if r['split'] == 'train']
    val_indices = [(i, c) for i, r in enumerate(rows) if r['split'] == 'validation'
                   for c in range(3)]
    random.seed(20261004)
    np.random.seed(20261004)
    torch.manual_seed(20261004)
    torch.set_num_threads(4)
    data = PairCrops(rows, train_indices)
    sample = RandomSampler(data, replacement=True, num_samples=args.updates * args.batch_size)
    train_loader = DataLoader(data, batch_size=args.batch_size, sampler=sample,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(PairCrops(rows, val_indices), batch_size=12,
                            num_workers=0, pin_memory=True)
    calibration = json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics = {c: np.asarray(calibration['tcp_to_optical'][c], float) for c in CAMERAS}
    model = RequestedPair(args.init).cuda()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                   lr=6e-5, weight_decay=1e-4)
    history = []
    best = float('inf')
    best_step = 0
    start = time.monotonic()
    stop_reason = 'update_cap'
    completed = 0
    for step, batch in enumerate(train_loader, 1):
        if time.monotonic() - start > args.minutes * 60:
            stop_reason = 'wall_budget'
            break
        image, meta, card, truth, inside, *_ = batch
        image = image.cuda(non_blocking=True).float() / 255.
        meta = meta.cuda(non_blocking=True)
        card = card.cuda(non_blocking=True)
        truth = truth.cuda(non_blocking=True)
        inside = inside.cuda(non_blocking=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred, flags, heat = model(image, meta, card)
        height, width = heat.shape[-2:]
        cell = (truth[:, :, 1] * (height - 1)).round().long().clamp(0, height - 1) * width
        cell += (truth[:, :, 0] * (width - 1)).round().long().clamp(0, width - 1)
        coordinate = F.smooth_l1_loss(pred, truth, beta=.01, reduction='none').sum(-1)
        spatial = F.cross_entropy(heat.float().reshape(-1, height * width) * 4,
                                  cell.reshape(-1), reduction='none').reshape(len(image), 2)
        location = ((coordinate + .02 * spatial) * inside).sum() / inside.sum().clamp_min(1)
        pair_mask = inside[:, 0] * inside[:, 1]
        pair_vector = F.smooth_l1_loss(pred[:, 1] - pred[:, 0],
                                       truth[:, 1] - truth[:, 0], beta=.01,
                                       reduction='none').sum(-1)
        loss = location + .3 * (pair_vector * pair_mask).sum() / pair_mask.sum().clamp_min(1)
        loss += .1 * F.binary_cross_entropy_with_logits(flags.float(), inside)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite pair loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        completed = step
        if step % 200 == 0:
            report = evaluate(model, rows, val_loader, optics, first_time,
                              args.output / f'predictions_step{step}.jsonl')
            score = (1000 * report['wrong'] + 20 * (10 - report['early_episodes']) +
                     (report['accepted_p95_mm'] if report['accepted_p95_mm'] is not None else 1e6))
            item = {'step': step, 'loss': float(loss.detach()), 'score': score,
                    'development': report}
            history.append(item)
            print(json.dumps(item), flush=True)
            if score < best:
                best = score
                best_step = step
                torch.save({'model': model.state_dict(), 'step': step,
                            'architecture': 'requested_card_two_port_native_crop_v1',
                            'runtime_true_geometry_input': False}, args.output / 'best.pt')
    selected = next((x for x in history if x['step'] == best_step), None)
    if selected is None:
        raise RuntimeError('Stopped before first development checkpoint')
    metrics = selected['development']
    summary = {'schema': 'sfp_requested_pair_crop/v1',
               'train_groups': train_groups, 'development_groups': val_groups,
               'train_proposals': len(train_indices),
               'development_proposals': len(val_indices) // 3,
               'completed_updates': completed, 'stop_reason': stop_reason,
               'best_step': best_step, 'selected': selected,
               'development_gate_passed': bool(metrics['wrong'] == 0 and
                   metrics['early_episodes'] >= 8 and metrics['accepted_p95_mm'] is not None and
                   metrics['accepted_p95_mm'] <= 10),
               'original_crop_size_px': 480, 'network_size_px': 320,
               'runtime_true_geometry_input': False, 'final_scenes_opened': False}
    (args.output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
