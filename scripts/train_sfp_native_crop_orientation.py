#!/usr/bin/env python3
"""Frozen native RGB contact-crop orientation probe; geometry is label only.

Crop centers are predictions of the deployed RGB cascade, never simulator TF.
Split/checkpoint selection is by whole Gazebo episode. The five second-half
validation episodes are reported separately and never used for selection.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.nn import functional as F

from train_sc_native_keypoints import PretrainedKeypointRefiner

CAMERAS = ('center', 'left', 'right')


def rotation6(x):
    a = F.normalize(x[..., :3], dim=-1)
    b = F.normalize(x[..., 3:] - (x[..., 3:] * a).sum(-1, keepdim=True) * a, dim=-1)
    return torch.stack((a, b, torch.cross(a, b, dim=-1)), -1)


class Head(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(size, 512), nn.SiLU(), nn.Linear(512, 256),
                                 nn.SiLU(), nn.Linear(256, 6))

    def forward(self, x):
        return rotation6(self.net(x))


def extract(model, images):
    z = (images - model.mean) / model.std
    z = model.backbone.maxpool(model.backbone.relu(model.backbone.bn1(model.backbone.conv1(z))))
    z = model.backbone.layer1(z)
    mid = model.backbone.layer2(z)
    deep = model.backbone.layer3(mid)
    return torch.cat((F.adaptive_avg_pool2d(mid, 2).flatten(1),
                      F.adaptive_avg_pool2d(deep, 1).flatten(1)), 1)


def measure(pred, truth, ids):
    d = np.rad2deg((Rotation.from_matrix(pred).inv() * Rotation.from_matrix(truth)).magnitude())
    return {'n': len(d), 'episodes': len(set(ids)), 'median_deg': float(np.median(d)),
            'p95_deg': float(np.quantile(d, .95)),
            'per_episode_median_deg': {e: float(np.median(d[np.asarray(ids) == e])) for e in sorted(set(ids))}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--labels', type=Path, required=True)
    ap.add_argument('--fine-predictions', type=Path, required=True)
    ap.add_argument('--fine-checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--updates', type=int, default=2000)
    ap.add_argument('--minutes', type=float, default=30)
    a = ap.parse_args()
    if a.output.exists(): ap.error('Preserve existing output')
    a.output.mkdir(parents=True)
    torch.set_num_threads(8); torch.manual_seed(20261002); np.random.seed(20261002)
    rows = [json.loads(s) for s in (a.labels / 'rows.jsonl').open()]
    predictions = {(q['row'], q['camera']): q for q in map(json.loads, a.fine_predictions.open())}
    checkpoint = torch.load(a.fine_checkpoint, map_location='cpu', weights_only=False)
    backbone = PretrainedKeypointRefiner(None, False, point_count=1).cuda().eval()
    backbone.load_state_dict(checkpoint['model'])
    features = []; truth = []; episodes = []; split = []; identities = []; rejected = defaultdict(int)
    import time
    start = time.monotonic()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i, row in enumerate(rows):
            if time.monotonic() - start > a.minutes * 60: raise TimeoutError('Feature cache budget exceeded')
            target = next(t for t in row['targets'] if t['identity'] == row['selected_identity'])
            crops = []; xy = []
            for camera in CAMERAS:
                q = predictions[i, camera]['predicted'][row['selected_identity']]
                center = np.asarray(q['xy'], float)
                if q['projected_inframe_probability'] < .5: rejected['projected_out_of_frame'] += 1
                rgb = cv2.imread(row['images'][camera], cv2.IMREAD_COLOR)
                if rgb is None: raise FileNotFoundError(row['images'][camera])
                left, top = np.round(center - 80).astype(int)
                crop = cv2.warpAffine(rgb, np.array([[1, 0, -left], [0, 1, -top]], np.float32),
                                      (160, 160), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                crops.append(torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).copy()).permute(2, 0, 1))
                xy.extend((center / (1151, 1023)).tolist())
            image = torch.stack(crops).cuda().float() / 255
            f = extract(backbone, image).float().cpu().numpy().reshape(-1)
            code = np.zeros(10, np.float32); code[int(row['selected_identity'].split('_')[2]) * 2 +
                                           int(row['selected_identity'].split('_')[4])] = 1
            tcp = Rotation.from_quat(row['state'][3:7]); port = Rotation.from_quat(target['orientation_base_xyzw'])
            relative = (tcp.inv() * port).as_matrix()
            features.append(np.r_[f, xy, code, row['state'][3:7]].astype(np.float32))
            truth.append(relative); episodes.append(row['episode']); split.append(row['split'])
            identities.append(row['selected_identity'])
            if (i + 1) % 200 == 0: print(json.dumps({'cached_rows': i + 1, 'total': len(rows)}), flush=True)
    x = np.stack(features); y = np.stack(truth).astype(np.float32)
    np.savez_compressed(a.output / 'frozen_features.npz', features=x, rotation=y,
                        episodes=np.asarray(episodes), splits=np.asarray(split))
    episode_names = sorted({e for e, s in zip(episodes, split) if s == 'validation'})
    selection = set(e for e in episode_names if e.endswith('port0_cards3_validation'))
    if len(selection) != 5: raise ValueError(f'Unexpected selection split: {selection}')
    tr = np.array([i for i, s in enumerate(split) if s == 'train'])
    va = np.array([i for i, e in enumerate(episodes) if e in selection])
    te = np.array([i for i, e in enumerate(episodes) if e in set(episode_names)-selection])
    if not len(tr) or not len(va) or not len(te): raise ValueError('Empty episode split')
    ep_counts = {e: sum(episodes[i] == e for i in tr) for e in set(episodes[i] for i in tr)}
    weights = np.array([1 / ep_counts[episodes[i]] for i in tr]); weights /= weights.sum()
    mean = x[tr].mean(0); std = np.maximum(x[tr].std(0), .01)
    xx = torch.tensor((x - mean) / std, device='cuda'); yy = torch.tensor(y, device='cuda')
    model = Head(x.shape[1]).cuda(); optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    best = float('inf'); best_step = 0; history = []; stale = 0
    for step in range(1, a.updates + 1):
        batch = np.random.choice(tr, size=128, p=weights)
        model.train(); optimizer.zero_grad(set_to_none=True)
        prediction = model(xx[batch]); loss = (prediction - yy[batch]).square().mean()
        loss.backward(); optimizer.step()
        if step % 100 == 0:
            model.eval()
            with torch.inference_mode(): vpred = model(xx[va]).cpu().numpy()
            result = measure(vpred, y[va], [episodes[i] for i in va])
            history.append({'step': step, 'loss': float(loss), 'selection': result})
            print(json.dumps({'step': step, 'selection_p95_deg': result['p95_deg']}), flush=True)
            if result['p95_deg'] < best - .05:
                best = result['p95_deg']; best_step = step; stale = 0
                torch.save({'model': model.state_dict(), 'mean': mean, 'std': std,
                            'schema': 'sfp_native_crop_orientation/v1',
                            'runtime_geometry_input': False}, a.output / 'best.pt')
            else: stale += 1
            if step >= 500 and stale >= 8: break
    best_ckpt = torch.load(a.output / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(best_ckpt['model']); model.eval()
    with torch.inference_mode():
        val_pred = model(xx[va]).cpu().numpy(); test_pred = model(xx[te]).cpu().numpy()
    summary = {'schema': 'sfp_native_crop_orientation_probe/v1', 'best_step': best_step,
               'updates_completed': step, 'train_episodes': len(set(episodes[i] for i in tr)),
               'selection': measure(val_pred, y[va], [episodes[i] for i in va]),
               'development_test': measure(test_pred, y[te], [episodes[i] for i in te]),
               'selection_episodes': sorted(selection),
               'development_test_episodes': sorted(set(episodes[i] for i in te)),
               'rejected': dict(rejected), 'crop_source': str(a.fine_predictions),
               'fine_checkpoint': str(a.fine_checkpoint), 'runtime_geometry_input': False,
               'reserved_final_opened': False}
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (a.output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
