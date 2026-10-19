#!/usr/bin/env python3
"""Bounded requested-port 3D coarse crop from causal three-view history."""
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

from audit_sc_port_targets import pose_matrix
from build_sc_native_pose_labels import project
from train_requested_history_crop import (HISTORY_SECONDS, cache_features, relative_pose,
                                          rows_from, schedule)
from train_two_family_port_locator import CAMERAS, LOOKUP


class HistoryPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Embedding(12, 256)
        self.camera = nn.Embedding(3, 256)
        self.time_pose = nn.Sequential(nn.Linear(10, 128), nn.GELU(), nn.Linear(128, 256))
        self.blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=512, dropout=.1, batch_first=True), 2)
        self.point = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 3))
        self.available = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, frames, motion, target):
        pooled = frames.float().mean((-1, -2))
        cams = self.camera(torch.arange(3, device=frames.device))[None, None]
        tokens = (pooled + cams + self.time_pose(motion)[:, :, None]).reshape(len(frames), 12, 256)
        fused = self.blocks(torch.cat((self.identity(target)[:, None], tokens), 1))[:, 0]
        return self.point(fused) * .2, self.available(fused).squeeze(-1)


def batch(rows, features, times, indices, size, training):
    chosen = random.choices(indices, k=size) if training else indices
    indices4 = np.asarray([times[i] for i in chosen], np.int64)
    frames = torch.from_numpy(np.asarray(features[indices4], dtype=np.float32)).cuda()
    motion = np.zeros((len(chosen), 4, 10), np.float32)
    names = []; points = np.zeros((len(chosen), 3), np.float32)
    present = np.zeros(len(chosen), np.float32)
    for k, i in enumerate(chosen):
        row = rows[i]
        name = (f'sfp_card_{random.randrange(5)}_port_{random.randrange(2)}'
                if training and random.random() < .4 else row['selected_identity'])
        names.append(LOOKUP[name])
        for h, j in enumerate(indices4[k]):
            motion[k, h, :9] = relative_pose(row['state'], rows[j]['state'])
            motion[k, h, 9] = (row['time_s'] - rows[j]['time_s']) / 6.
        target = next((t for t in row['targets'] if t['identity'] == name), None)
        if target is not None:
            base_tcp = pose_matrix(row['state'])
            points[k] = (np.linalg.inv(base_tcp) @ np.r_[target['position_base_m'], 1.])[:3]
            present[k] = 1.
    return frames, torch.from_numpy(motion).cuda(), torch.tensor(names, device='cuda'), \
           torch.from_numpy(points).cuda(), torch.from_numpy(present).cuda(), chosen


@torch.inference_mode()
def evaluate(model, rows, features, times, indices, optics, out=None):
    model.eval(); recs = []
    for start in range(0, len(indices), 32):
        f, m, t, _, _, chosen = batch(rows, features, times, indices[start:start + 32], 32, False)
        pred, logits = model(f, m, t)
        points = pred.cpu().numpy(); probs = logits.sigmoid().cpu().numpy()
        for k, i in enumerate(chosen):
            row = rows[i]; base_tcp = pose_matrix(row['state'])
            base_point = (base_tcp @ np.r_[points[k], 1.])[:3]
            requested = next(t for t in row['targets'] if t['identity'] == row['selected_identity'])
            point_transform = np.eye(4)
            point_transform[:3, 3] = base_point
            camera_projections = {c: project(point_transform, base_tcp @ optics[c]) for c in CAMERAS}
            errors = []; nearest = []
            for c in CAMERAS:
                uv = camera_projections[c]
                gt = requested['cameras'][c]
                errors.append(float(np.linalg.norm(np.asarray(uv[:2]) - gt[:2])) if gt[3]
                              else float('inf'))
                nearest.append(min((float(np.linalg.norm(np.asarray(uv[:2]) -
                                               alt['cameras'][c][:2])) for alt in row['targets']
                                    if alt['identity'] != row['selected_identity'] and
                                    alt['cameras'][c][3]), default=float('inf')))
            accepted = bool(probs[k] >= .5 and all(camera_projections[c][3] for c in CAMERAS))
            recs.append({'episode': row['episode'], 'scene_group': row['scene_group'],
                         'time_s': row['time_s'], 'requested': row['selected_identity'],
                         'predicted_base_m': base_point.tolist(),
                         'crop_center_px': {c: camera_projections[c][:2] for c in CAMERAS},
                         'available_score': float(probs[k]), 'accepted': accepted,
                         'target_error_px_label_only': errors,
                         'nearest_other_px_label_only': nearest,
                         'position_error_mm_label_only': float(np.linalg.norm(
                             base_point - np.asarray(requested['position_base_m'])) * 1000)})
    by_ep = defaultdict(list)
    accepted = []; wrong = []; contained = []
    for row in recs:
        by_ep[row['episode']].append(row)
        if row['accepted']:
            accepted.append(row)
            if any(a < e for a, e in zip(row['nearest_other_px_label_only'],
                                          row['target_error_px_label_only'])):
                wrong.append(row)
            contained.append(max(row['target_error_px_label_only']) <= 60)
    early = 0; any_ep = 0
    for ep, items in by_ep.items():
        items.sort(key=lambda r: r['time_s'])
        first = next((r for r in items if r['accepted']), None)
        early += bool(first and first['time_s'] - items[0]['time_s'] <= 3.)
        any_ep += first is not None
    result = {'episodes': len(by_ep), 'rows': len(recs), 'accepted': len(accepted),
              'wrong_port_any_view_posthoc': len(wrong), 'early_episodes_3s': early,
              'any_episodes': any_ep,
              'all_view_160px_crops_with_20px_margin': sum(contained),
              'all_view_crop_fraction': float(np.mean(contained)) if contained else 0.,
              'accepted_3d_error_p50_mm': float(np.median([r['position_error_mm_label_only']
                                                           for r in accepted])) if accepted else None,
              'accepted_3d_error_p95_mm': float(np.quantile([r['position_error_mm_label_only']
                                                             for r in accepted], .95)) if accepted else None}
    if out:
        with out.open('w') as stream:
            for r in recs: stream.write(json.dumps(r, separators=(',', ':')) + '\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--stress-labels', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=1000)
    p.add_argument('--minutes', type=float, default=60.)
    a = p.parse_args()
    if a.output.exists(): p.error('Preserve existing output')
    if a.updates > 1200: p.error('Bounded experiment: at most 1200 updates')
    a.output.mkdir(parents=True)
    random.seed(20261004); np.random.seed(20261004); torch.manual_seed(20261004)
    torch.set_num_threads(8)
    original = rows_from(a.labels); stress_rows = rows_from(a.stress_labels)
    rows = original + stress_rows
    train = [i for i, r in enumerate(original) if r['split'] == 'train']
    dev = [i for i, r in enumerate(original) if r['split'] == 'validation']
    stress = list(range(len(original), len(rows)))
    scene_groups = [{rows[i]['scene_group'] for i in group} for group in (train, dev, stress)]
    if any(scene_groups[i] & scene_groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Reset group overlap')
    features = np.load(a.cache / 'features.npy', mmap_mode='r')
    if len(features) != len(rows): raise ValueError('Cache row mismatch')
    times = schedule(rows)
    calibration = json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics = {c: np.asarray(calibration['tcp_to_optical'][c], float) for c in CAMERAS}
    manifest = {'train_scene_groups': sorted(scene_groups[0]),
                'development_scene_groups': sorted(scene_groups[1]),
                'stress_scene_groups': sorted(scene_groups[2]),
                'feature_cache': str(a.cache), 'history_seconds': HISTORY_SECONDS,
                'runtime_geometry_input': False, 'final_scenes_opened': False}
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    model = HistoryPoint().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    best = float('inf'); best_step = 0; history = []; start = time.monotonic()
    for step in range(1, a.updates + 1):
        if time.monotonic() - start > a.minutes * 60: break
        f, m, t, target, present, _ = batch(rows, features, times, train, 16, True)
        model.train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16): pred, available = model(f, m, t)
        position = F.smooth_l1_loss(pred.float() * 100, target * 100, beta=.5,
                                    reduction='none').sum(-1)
        loss = (position * present).sum() / present.sum().clamp_min(1)
        loss += .5 * F.binary_cross_entropy_with_logits(available.float(), present)
        if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        if step % 100 == 0:
            report = evaluate(model, rows, features, times, dev, optics)
            score = (report['wrong_port_any_view_posthoc'] * 10 +
                     (10 - report['early_episodes_3s']) * 5 +
                     10 * (1 - report['all_view_crop_fraction']))
            history.append({'step': step, 'loss': float(loss.detach()),
                            'selection_score': score, 'development': report})
            print(json.dumps(history[-1]), flush=True)
            if score < best:
                best = score; best_step = step
                torch.save({'model': model.state_dict(), 'step': step,
                            'architecture': 'requested_three_view_causal_history_3d_v1',
                            'runtime_geometry_input': False}, a.output / 'best.pt')
    if not best_step: raise RuntimeError('No checkpoint selected')
    model.load_state_dict(torch.load(a.output / 'best.pt', weights_only=False)['model'])
    development = evaluate(model, rows, features, times, dev, optics,
                           a.output / 'development_predictions.jsonl')
    stress_result = evaluate(model, rows, features, times, stress, optics,
                             a.output / 'stress_predictions.jsonl')
    summary = {'schema': 'requested_history_3d_v1', 'best_step': best_step,
               'completed_updates': step, 'train_episodes': len(scene_groups[0]),
               'development': development, 'previously_inspected_stress': stress_result,
               'promotion': False, 'final_scenes_opened': False}
    (a.output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
