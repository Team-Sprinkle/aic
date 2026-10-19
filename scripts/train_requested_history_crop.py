#!/usr/bin/env python3
"""Bounded causal three-view history -> requested-port crop experiment.

Simulator port geometry is used only for training labels and posthoc scoring.
Inference inputs are RGB feature maps, public requested-port code, timestamps,
and measured relative TCP motion. This model emits one requested port per view.
"""
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

from audit_sc_port_targets import pose_matrix
from train_target_conditioned_port_locator import ConditionalLocator
from train_two_family_port_locator import CAMERAS, LOOKUP

WIDTH, HEIGHT = 1152, 1024
HISTORY_SECONDS = (0., 1.5, 3., 6.)


def rows_from(path):
    return [json.loads(line) for line in (path / 'rows.jsonl').open()]


def schedule(rows):
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[row['episode']].append(i)
    result = []
    for indices in grouped.values():
        indices.sort(key=lambda j: rows[j]['time_s'])
        for i in indices:
            now = rows[i]['time_s']
            result.append((i, [max((j for j in indices if rows[j]['time_s'] <= now - dt + .05),
                                   key=lambda j: rows[j]['time_s'], default=indices[0])
                               for dt in HISTORY_SECONDS]))
    return dict(result)


def relative_pose(current, past):
    rel = np.linalg.inv(pose_matrix(current)) @ pose_matrix(past)
    return np.concatenate((rel[:3, 3] / .2, rel[:3, :2].reshape(-1))).astype(np.float32)


def cache_features(rows, path, checkpoint, device):
    feature_path = path / 'features.npy'
    manifest_path = path / 'features_manifest.json'
    manifest = [{'episode': r['episode'], 'time_s': r['time_s'], 'scene_group': r['scene_group'],
                 'images': r['images']} for r in rows]
    if feature_path.exists() and manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Feature cache row/image mismatch')
        return np.load(feature_path, mmap_mode='r')
    model = ConditionalLocator(None).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    backbone = model.backbone
    features = np.lib.format.open_memmap(feature_path, mode='w+', dtype=np.float16,
                                          shape=(len(rows), 3, 256, 32, 36))
    with torch.inference_mode():
        for start in range(0, len(rows), 12):
            batch = []
            for row in rows[start:start + 12]:
                for camera in CAMERAS:
                    image = cv2.imread(row['images'][camera], cv2.IMREAD_COLOR)
                    if image is None:
                        raise FileNotFoundError(row['images'][camera])
                    image = cv2.cvtColor(cv2.resize(image, (576, 512), interpolation=cv2.INTER_AREA),
                                         cv2.COLOR_BGR2RGB)
                    batch.append(torch.from_numpy(image.copy()).permute(2, 0, 1))
            rgb = torch.stack(batch).to(device).float() / 255.
            for offset in range(0, len(rgb), 12):
                x = (rgb[offset:offset + 12] - model.mean) / model.std
                x = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(x))))
                x = backbone.layer3(backbone.layer2(backbone.layer1(x)))
                n = len(x)
                features[start + offset // 3:start + (offset + n) // 3] = (
                    x.detach().cpu().numpy().astype(np.float16).reshape(-1, 3, 256, 32, 36))
            if (start // 12) % 20 == 0:
                print(json.dumps({'cached': min(start + 12, len(rows)), 'total': len(rows)}), flush=True)
    features.flush()
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return np.load(feature_path, mmap_mode='r')


class HistoryCrop(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Embedding(12, 256)
        self.camera = nn.Embedding(3, 256)
        self.time_pose = nn.Sequential(nn.Linear(10, 128), nn.GELU(), nn.Linear(128, 256))
        self.blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=512, dropout=.1, batch_first=True), 2)
        self.spatial = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(128, 1, 1))
        self.flag = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, frames, motion, target):
        # frames: B, four causal times, three cameras, 256, 32, 36.
        b = frames.shape[0]
        pooled = frames.float().mean((-1, -2))
        cams = self.camera(torch.arange(3, device=frames.device))[None, None]
        temporal = self.time_pose(motion)[:, :, None, :]
        tokens = (pooled + cams + temporal).reshape(b, 12, 256)
        query = self.identity(target)[:, None]
        encoded = self.blocks(torch.cat((query, tokens), 1))
        q = encoded[:, 0, None].expand(-1, 3, -1)
        current = frames[:, 0].float()
        context = q + encoded[:, 1:4]
        spatial = torch.cat((current, context[:, :, :, None, None].expand(-1, -1, -1, 32, 36)), 2)
        heat = self.spatial(spatial.reshape(b * 3, 512, 32, 36)).reshape(b, 3, 32, 36)
        prob = F.softmax((heat * 4).flatten(2), dim=-1)
        yy, xx = torch.meshgrid(torch.linspace(0, 1, 32, device=frames.device),
                                torch.linspace(0, 1, 36, device=frames.device), indexing='ij')
        xy = torch.stack(((prob * xx.flatten()).sum(-1), (prob * yy.flatten()).sum(-1)), -1)
        flag = self.flag(torch.cat((context, pooled[:, 0]), -1)).squeeze(-1)
        return xy, flag, heat


class SpatialHistoryCrop(nn.Module):
    """Retain a 4x5 image grid per historical camera instead of pooling it away."""
    def __init__(self, coordinate_output=False):
        super().__init__()
        self.coordinate_output = coordinate_output
        self.identity = nn.Embedding(12, 256)
        self.camera = nn.Embedding(3, 256)
        self.position = nn.Embedding(20, 256)
        self.time_pose = nn.Sequential(nn.Linear(10, 128), nn.GELU(), nn.Linear(128, 256))
        self.blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=512, dropout=.1, batch_first=True), 2)
        self.spatial = nn.Sequential(nn.Conv2d(514 if coordinate_output else 512,
                                               128, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(128, 1, 1))
        self.flag = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, frames, motion, target):
        b = len(frames)
        grid = F.adaptive_avg_pool2d(frames.float().reshape(b * 12, 256, 32, 36),
                                      (4, 5)).reshape(b, 4, 3, 256, 20)
        grid = grid.permute(0, 1, 2, 4, 3)
        cams = self.camera(torch.arange(3, device=frames.device))[None, None, :, None]
        positions = self.position(torch.arange(20, device=frames.device))[None, None, None]
        temporal = self.time_pose(motion)[:, :, None, None]
        tokens = (grid + cams + positions + temporal).reshape(b, 240, 256)
        queries = self.identity(target)[:, None, :] + self.camera(
            torch.arange(3, device=frames.device))[None]
        contexts = self.blocks(torch.cat((queries, tokens), 1))[:, :3]
        current = frames[:, 0].float()
        joined = torch.cat((current,
                            contexts[:, :, :, None, None].expand(-1, -1, -1, 32, 36)), 2)
        if self.coordinate_output:
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, 32, device=frames.device),
                                    torch.linspace(-1, 1, 36, device=frames.device), indexing='ij')
            coordinates = torch.stack((xx, yy))[None, None].expand(b, 3, -1, -1, -1)
            joined = torch.cat((joined, coordinates), 2)
        heat = self.spatial(joined.reshape(b * 3, joined.shape[2], 32, 36)).reshape(b, 3, 32, 36)
        prob = F.softmax((heat * 4).flatten(2), dim=-1)
        yy, xx = torch.meshgrid(torch.linspace(0, 1, 32, device=frames.device),
                                torch.linspace(0, 1, 36, device=frames.device), indexing='ij')
        xy = torch.stack(((prob * xx.flatten()).sum(-1), (prob * yy.flatten()).sum(-1)), -1)
        flag = self.flag(torch.cat((contexts, current.mean((-1, -2))), -1)).squeeze(-1)
        return xy, flag, heat


def sample_batch(rows, features, times, indices, batch_size, device, training):
    chosen = random.choices(indices, k=batch_size) if training else indices
    frame_indices = np.asarray([times[i] for i in chosen], np.int64)
    frames = torch.from_numpy(np.asarray(features[frame_indices], dtype=np.float32)).to(device)
    motion = np.zeros((len(chosen), 4, 10), np.float32)
    targets = []
    points = np.full((len(chosen), 3, 2), .5, np.float32)
    labels = np.zeros((len(chosen), 3), np.float32)
    for k, i in enumerate(chosen):
        row = rows[i]
        if training and random.random() < .4:
            # Include alternate present ports and absent slots as task-code negatives.
            target_name = f'sfp_card_{random.randrange(5)}_port_{random.randrange(2)}'
        else:
            target_name = row['selected_identity']
        targets.append(LOOKUP[target_name])
        target = next((t for t in row['targets'] if t['identity'] == target_name), None)
        for h, j in enumerate(frame_indices[k]):
            motion[k, h, :9] = relative_pose(row['state'], rows[j]['state'])
            motion[k, h, 9] = (row['time_s'] - rows[j]['time_s']) / 6.
        if target is None:
            continue
        for c, camera in enumerate(CAMERAS):
            uv = target['cameras'][camera]
            if uv[3]:
                labels[k, c] = 1.
                points[k, c] = [uv[0] / (WIDTH - 1), uv[1] / (HEIGHT - 1)]
    return (frames, torch.from_numpy(motion).to(device),
            torch.tensor(targets, device=device), torch.from_numpy(points).to(device),
            torch.from_numpy(labels).to(device), chosen)


@torch.inference_mode()
def evaluate(model, rows, features, times, indices, device, save=None):
    model.eval()
    records = []
    for start in range(0, len(indices), 16):
        f, m, t, p, y, chosen = sample_batch(rows, features, times,
                                              indices[start:start + 16], 16, device, False)
        xy, logits, _ = model(f, m, t)
        for k, i in enumerate(chosen):
            r = rows[i]
            prediction = xy[k].cpu().numpy() * [WIDTH - 1, HEIGHT - 1]
            probability = logits[k].sigmoid().cpu().numpy()
            actual = p[k].cpu().numpy() * [WIDTH - 1, HEIGHT - 1]
            valid = y[k].cpu().numpy().astype(bool)
            errors = np.linalg.norm(prediction - actual, axis=-1)
            other = []
            for c, camera in enumerate(CAMERAS):
                alternatives = [v['cameras'][camera][:2] for v in r['targets']
                                if v['identity'] != r['selected_identity'] and v['cameras'][camera][3]]
                other.append(min((float(np.linalg.norm(prediction[c] - q)) for q in alternatives),
                                 default=float('inf')))
            records.append({'episode': r['episode'], 'scene_group': r['scene_group'],
                            'time_s': r['time_s'], 'requested': r['selected_identity'],
                            'prediction_px': prediction.tolist(), 'probability': probability.tolist(),
                            'label_projection_inframe': valid.tolist(),
                            'target_error_px_label_only': errors.tolist(),
                            'nearest_other_px_label_only': other})
    # Frozen operating rule: all three flags >=.5; requested opening must lie
    # within 60 px of each 160 px crop center for posthoc full-port margin.
    by_ep = defaultdict(list)
    accepted = []
    wrong = []
    crops = []
    for r in records:
        r['propose'] = bool(min(r['probability']) >= .5)
        if r['propose']:
            accepted.append(r)
            if any(np.asarray(r['nearest_other_px_label_only']) <
                   np.asarray(r['target_error_px_label_only'])):
                wrong.append(r)
            crops.append(all(r['label_projection_inframe']) and
                         max(r['target_error_px_label_only']) <= 60)
        by_ep[r['episode']].append(r)
    early = 0
    any_episode = 0
    for items in by_ep.values():
        items.sort(key=lambda r: r['time_s'])
        first = next((r for r in items if r['propose']), None)
        any_episode += first is not None
        early += bool(first and first['time_s'] - items[0]['time_s'] <= 3)
    report = {'episodes': len(by_ep), 'rows': len(records), 'proposals': len(accepted),
              'wrong_port_any_view_posthoc': len(wrong), 'all_view_160px_crops_with_20px_margin': sum(crops),
              'all_view_crop_fraction': float(np.mean(crops)) if crops else 0.,
              'early_episodes_3s': early, 'any_episodes': any_episode}
    if save:
        with save.open('w') as stream:
            for row in records:
                stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--stress-labels', type=Path, required=True)
    parser.add_argument('--init', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--minutes', type=float, default=60.)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--spatial-history', action='store_true')
    parser.add_argument('--spatial-history-xy', action='store_true')
    parser.add_argument('--feature-cache', type=Path,
                        help='Reuse a feature cache only when its exact row/image manifest matches')
    args = parser.parse_args()
    if args.output.exists(): parser.error('Output exists')
    if args.updates > 1200: parser.error('Bounded experiment: at most 1200 updates')
    args.output.mkdir(parents=True)
    random.seed(20261004); np.random.seed(20261004); torch.manual_seed(20261004)
    torch.set_num_threads(8)
    rows = rows_from(args.labels) + rows_from(args.stress_labels)
    groups = {split: {r['scene_group'] for r in rows if r.get('split') == split}
              for split in ('train', 'validation')}
    if groups['train'] & groups['validation']: raise ValueError('Scene/reset overlap')
    train = [i for i, r in enumerate(rows) if r['split'] == 'train']
    base_dev = [i for i, r in enumerate(rows[:len(rows_from(args.labels))]) if r['split'] == 'validation']
    stress = list(range(len(rows) - len(rows_from(args.stress_labels)), len(rows)))
    manifest = {'train_scene_groups': sorted({rows[i]['scene_group'] for i in train}),
                'development_scene_groups': sorted({rows[i]['scene_group'] for i in base_dev}),
                'stress_scene_groups': sorted({rows[i]['scene_group'] for i in stress}),
                'history_seconds': HISTORY_SECONDS, 'init': str(args.init),
                'target_code_used': True, 'runtime_geometry_input': False,
                'crop_size_px': 160, 'stress_is_previously_inspected_not_final': True}
    if set(manifest['train_scene_groups']) & set(manifest['stress_scene_groups']):
        raise ValueError('Train/stress reset overlap')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    feature_dir = args.feature_cache or args.output
    features = cache_features(rows, feature_dir, args.init, 'cuda')
    manifest['feature_cache'] = str(feature_dir)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    times = schedule(rows)
    model = (SpatialHistoryCrop(coordinate_output=args.spatial_history_xy)
             if args.spatial_history or args.spatial_history_xy else HistoryCrop()).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    best = float('inf'); best_step = 0; history = []; start = time.monotonic()
    for step in range(1, args.updates + 1):
        if time.monotonic() - start > args.minutes * 60: break
        model.train()
        batch = sample_batch(rows, features, times, train, args.batch_size, 'cuda', True)
        f, m, t, p, y, _ = batch
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16): xy, flag, heat = model(f, m, t)
        pixel = F.smooth_l1_loss(xy.float(), p, beta=.02, reduction='none').sum(-1)
        cells = (p[..., 1] * 31).round().long() * 36 + (p[..., 0] * 35).round().long()
        ce = F.cross_entropy((heat.float() * 4).reshape(-1, 32 * 36), cells.flatten(),
                             reduction='none').reshape_as(y)
        loss = ((pixel + .01 * ce) * y).sum() / y.sum().clamp_min(1)
        loss += .2 * F.binary_cross_entropy_with_logits(flag.float(), y)
        if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        if step % 100 == 0:
            dev = evaluate(model, rows, features, times, base_dev, 'cuda')
            score = dev['wrong_port_any_view_posthoc'] * 10 + (10 - dev['early_episodes_3s']) * 5
            score += 10 * (1 - dev['all_view_crop_fraction'])
            history.append({'step': step, 'loss': float(loss.detach()), 'selection_score': score,
                            'development': dev})
            print(json.dumps(history[-1]), flush=True)
            if score < best:
                best = score; best_step = step
                torch.save({'model': model.state_dict(), 'step': step,
                            'architecture': ('requested_three_view_spatial_history_xy_crop_v1'
                                             if args.spatial_history_xy else
                                             'requested_three_view_spatial_history_crop_v1'
                                             if args.spatial_history else
                                             'requested_three_view_causal_history_crop_v1'),
                            'runtime_geometry_input': False}, args.output / 'best.pt')
    if not best_step: raise RuntimeError('No development checkpoint selected')
    model.load_state_dict(torch.load(args.output / 'best.pt', weights_only=False)['model'])
    development = evaluate(model, rows, features, times, base_dev, 'cuda',
                           args.output / 'development_predictions.jsonl')
    stress_result = evaluate(model, rows, features, times, stress, 'cuda',
                             args.output / 'stress_predictions.jsonl')
    summary = {'schema': 'requested_history_crop_v1', 'best_step': best_step,
               'completed_updates': step, 'train_episodes': len({rows[i]['episode'] for i in train}),
               'development': development, 'previously_inspected_stress': stress_result,
               'promotion': False, 'final_scenes_opened': False}
    (args.output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
