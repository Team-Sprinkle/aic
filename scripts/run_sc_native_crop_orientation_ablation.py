#!/usr/bin/env python3
"""Matched SC original-RGB crop orientation diagnostic on grouped development resets.

The saved RGB locator chooses every crop. Scored geometry supplies labels and
posthoc crop/error audits only. This script never modifies the deployed actor.
"""
import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.nn import functional as F

from train_sc_oneport_orientation_adaptation import (
    CAMERAS, Model, grouped_triplets, oneport_data, original_data, stats,
)
from train_sc_native_keypoints import PretrainedKeypointRefiner
from train_two_family_port_locator import Locator


def select_indices(rows, x, cap, seed):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if row['family'] != 'sc':
            continue
        target = next(t for t in row['targets'] if t['identity'] == row['selected_identity'])
        if np.linalg.norm(np.asarray(target['position_base_m']) - row['state'][:3]) >= .1:
            continue
        if not all(target['cameras'][c][3] for c in CAMERAS):
            continue
        groups[(row['split'], row['scene_group'])].append(i)
    out = []
    for key, indices in sorted(groups.items()):
        out += sorted(rng.choice(indices, min(len(indices), cap), replace=False).tolist())
    return out


class CropEncoder(nn.Module):
    def __init__(self, checkpoint, spatial=False):
        super().__init__()
        self.spatial = spatial
        self.refiner = PretrainedKeypointRefiner(None, False, point_count=1)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.refiner.load_state_dict(saved['model'])
        self.refiner.eval()

    def forward(self, rgb):
        model = self.refiner
        x = (rgb - model.mean) / model.std
        x = model.backbone.maxpool(model.backbone.relu(model.backbone.bn1(model.backbone.conv1(x))))
        x = model.backbone.layer1(x)
        middle = model.backbone.layer2(x)
        deep = model.backbone.layer3(middle)
        features = [F.adaptive_avg_pool2d(middle, 2).flatten(1),
                    F.adaptive_avg_pool2d(deep, 1).flatten(1)]
        if self.spatial:
            features.append(F.adaptive_avg_pool2d(deep, 2).flatten(1))
        return torch.cat(features, 1)


def features_from_crops(images, centers, encoder, device, crop_size, batch_size=96):
    crop_tensors = []
    for image, center in zip(images, centers):
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image)
        left, top = np.round(np.asarray(center) - crop_size / 2).astype(int)
        crop = cv2.warpAffine(bgr, np.asarray([[1, 0, -left], [0, 1, -top]], np.float32),
                              (crop_size, crop_size), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
        crop_tensors.append(torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).copy())
                              .permute(2, 0, 1))
    result = []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(crop_tensors), batch_size):
            batch = torch.stack(crop_tensors[start:start+batch_size]).to(device).float() / 255
            result.append(encoder(batch).float().cpu().numpy())
    values = np.concatenate(result)
    return values.reshape(-1, 3 * values.shape[-1])


def original_examples(rows, selected, pred):
    examples = []
    for i in selected:
        row = rows[i]
        identity = row['selected_identity']
        target = next(t for t in row['targets'] if t['identity'] == identity)
        centers = [pred[i][c]['predicted'][identity]['xy'] for c in CAMERAS]
        true = [target['cameras'][c][:2] for c in CAMERAS]
        examples.append({'source': 'ordinary', 'row': i, 'group': row['scene_group'],
                         'episode': row['episode'], 'split': row['split'],
                         'port': int(identity[-1]), 'images': [row['images'][c] for c in CAMERAS],
                         'centers': centers, 'true_xy': true})
    return examples


def oneport_examples(paths, locator, device):
    examples = []
    for path in paths:
        for _, views in grouped_triplets([path]):
            images = []
            for c in CAMERAS:
                bgr = cv2.imread(views[c]['image'])
                if bgr is None:
                    raise FileNotFoundError(views[c]['image'])
                images.append(cv2.cvtColor(cv2.resize(bgr, (288, 256), interpolation=cv2.INTER_AREA),
                                            cv2.COLOR_BGR2RGB))
            meta = np.zeros((3, 5), np.float32)
            meta[:, 0] = 1
            for j in range(3):
                meta[j, 2+j] = 1
            rgb = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(device).float() / 255
            with torch.inference_mode():
                pts, _ = locator(rgb, torch.from_numpy(meta).to(device))
            port = int(views['center']['target_port'])
            centers = (pts[:, port].float().cpu().numpy() * [1151, 1023]).tolist()
            examples.append({'source': 'oneport', 'group': views['center']['scene_group_id'],
                             'episode': views['center']['episode_id'],
                             'split': 'train' if path.parent.parent.name.startswith('train_') else 'development',
                             'port': port, 'images': [views[c]['image'] for c in CAMERAS],
                             'centers': centers,
                             'true_xy': [views[c]['projected_opening_xy'] for c in CAMERAS],
                             'center_sim_ns': views['center']['sim_ns']})
    return examples


class Residual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.std = nn.Parameter(torch.ones(dim), requires_grad=False)
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.SiLU(), nn.Linear(256, 128),
                                 nn.SiLU(), nn.Linear(128, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net((x - self.mean) / self.std).tanh() * .35


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--labels', type=Path, required=True)
    ap.add_argument('--sc-predictions', type=Path, required=True)
    ap.add_argument('--sfp-predictions', type=Path, required=True)
    ap.add_argument('--visual-features', type=Path, required=True)
    ap.add_argument('--oneport-train-glob', required=True)
    ap.add_argument('--oneport-dev-glob', required=True)
    ap.add_argument('--locator', type=Path, required=True)
    ap.add_argument('--initial', type=Path, required=True)
    ap.add_argument('--refiner', type=Path, required=True)
    ap.add_argument('--calibration', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--crop-size', type=int, default=160)
    ap.add_argument('--per-original-group', type=int, default=16)
    ap.add_argument('--updates', type=int, default=800)
    ap.add_argument('--spatial', action='store_true',
                    help='Keep a 2x2 deep feature grid per view instead of only global deep pooling')
    ap.add_argument('--audit-only', action='store_true')
    a = ap.parse_args()
    if a.output.exists():
        ap.error('Output exists: keep experiments immutable')
    a.output.mkdir(parents=True)
    torch.manual_seed(20261003)
    np.random.seed(20261003)
    torch.set_num_threads(4)
    calibration = json.loads(a.calibration.read_text())
    rows = [json.loads(s) for s in a.labels.open()]
    selected = select_indices(rows, None, a.per_original_group, 20261003)
    pred = defaultdict(dict)
    for q in map(json.loads, a.sc_predictions.open()):
        pred[q['row']][q['camera']] = q
    original = original_examples(rows, selected, pred)
    train_paths = [Path(p) for p in sorted(glob.glob(a.oneport_train_glob))]
    dev_paths = [Path(p) for p in sorted(glob.glob(a.oneport_dev_glob))]
    if len(train_paths) != 8 or len(dev_paths) != 4:
        raise ValueError('Unexpected oneport scene count')
    locator = Locator(None).to(a.device).eval()
    locator.load_state_dict(torch.load(a.locator, map_location='cpu', weights_only=False)['model'])
    special = oneport_examples(train_paths+dev_paths, locator, a.device)
    examples = original+special
    train_groups = {q['group'] for q in examples if q['split'] == 'train'}
    dev_groups = {q['group'] for q in examples if q['split'] != 'train'}
    if train_groups & dev_groups:
        raise ValueError('Reset-group leakage')
    offsets = [np.linalg.norm(np.asarray(q['centers'])-np.asarray(q['true_xy']), axis=1)
               for q in examples]
    offsets = np.stack(offsets)
    def pixel_stats(values):
        return {'n': int(len(values)), 'median_px': float(np.median(values)),
                'p95_px': float(np.quantile(values, .95))}
    audit = {'schema': 'sc_native_crop_orientation_audit/v1',
             'examples': len(examples), 'train_groups': len(train_groups),
             'dev_groups': len(dev_groups), 'crop_size': a.crop_size,
             'crop_center_error_px': {source: pixel_stats(offsets[[i for i,q in enumerate(examples)
                 if q['source'] == source]].ravel()) for source in ('ordinary','oneport')},
             'opening_inside_crop_fraction': {source: float(np.mean(offsets[[i for i,q in enumerate(examples)
                 if q['source'] == source]] < a.crop_size/2)) for source in ('ordinary','oneport')},
             'oneport_group_counts': dict(__import__('collections').Counter(q['episode'] for q in special))}
    (a.output/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    (a.output/'manifest.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in examples))
    print(json.dumps(audit), flush=True)
    if a.audit_only:
        return
    # The existing V2 head is frozen. Fit only a small native-crop residual.
    original_x, original_y, _ = original_data(a.labels, a.sc_predictions,
        a.sfp_predictions, a.visual_features, calibration)
    original_x = original_x[selected]
    original_y = original_y[selected]
    one_x, one_y, one_meta = oneport_data(train_paths+dev_paths, locator, calibration, a.device)
    if len(one_meta) != len(special):
        raise ValueError('Oneport extraction order mismatch')
    x = np.concatenate((original_x, one_x))
    truth = np.concatenate((original_y, one_y))
    saved = torch.load(a.initial, map_location='cpu', weights_only=False)
    base = Model(saved['model']['mean'], saved['model']['std']).to(a.device).eval()
    base.load_state_dict(saved['model'])
    with torch.inference_mode():
        base_rotation = np.concatenate([base(torch.from_numpy(x[i:i+256]).to(a.device))
                                        .float().cpu().numpy() for i in range(0,len(x),256)])
    target = (Rotation.from_matrix(base_rotation).inv()*Rotation.from_matrix(truth)).as_rotvec().astype(np.float32)
    encoder = CropEncoder(a.refiner, a.spatial).to(a.device).eval()
    flat_images = [image for q in examples for image in q['images']]
    flat_centers = [center for q in examples for center in q['centers']]
    crop_features = features_from_crops(flat_images, flat_centers, encoder, a.device, a.crop_size)
    centers = np.asarray([q['centers'] for q in examples], np.float32).reshape(len(examples), -1)
    # Measured TCP is already included in the frozen baseline. Avoid accidental
    # use of any target pose as a feature; crop centers and requested port suffice.
    port = np.asarray([q['port'] for q in examples], np.float32)[:,None]
    feat = np.concatenate((crop_features, centers / np.tile([1151,1023], 3), port), axis=1).astype(np.float32)
    fit = np.asarray([i for i,q in enumerate(examples) if q['split']=='train'], int)
    dev = np.asarray([i for i,q in enumerate(examples) if q['split']=='development'], int)
    original_val = np.asarray([i for i,q in enumerate(examples) if q['split']=='validation'], int)
    if not len(fit) or not len(dev):
        raise ValueError('Missing grouped data')
    model = Residual(feat.shape[1]).to(a.device)
    model.mean.data = torch.from_numpy(feat[fit].mean(0)).to(a.device)
    model.std.data = torch.from_numpy(np.maximum(feat[fit].std(0), .02)).to(a.device)
    xx = torch.from_numpy(feat).to(a.device)
    yy = torch.from_numpy(target).to(a.device)
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=2e-4, weight_decay=1e-3)
    rng = np.random.default_rng(20261003)
    history = []
    def score(indices):
        with torch.inference_mode():
            correction = model(xx[indices]).cpu().numpy()
        pred = Rotation.from_matrix(base_rotation[indices])*Rotation.from_rotvec(correction)
        errors = np.rad2deg((pred.inv()*Rotation.from_matrix(truth[indices])).magnitude())
        before = np.rad2deg((Rotation.from_matrix(base_rotation[indices]).inv()*
                           Rotation.from_matrix(truth[indices])).magnitude())
        return {'baseline': stats(before), 'crop': stats(errors),
                'by_episode': {ep: stats(errors[[j for j,i in enumerate(indices)
                    if examples[i]['episode']==ep]]) for ep in sorted({examples[i]['episode'] for i in indices})}}
    baseline = score(dev)
    best = float('inf')
    best_step = 0
    for step in range(1, a.updates+1):
        batch = rng.choice(fit, min(128,len(fit)), replace=True)
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = F.smooth_l1_loss(model(xx[batch]), yy[batch], beta=.02)
        loss.backward(); optimizer.step()
        if step % 100 == 0:
            model.eval(); result = score(dev)
            history.append({'step': step, 'loss': float(loss.detach()),
                            'development': result['crop']})
            print(json.dumps(history[-1]), flush=True)
            if result['crop']['p95_deg'] < best - .02:
                best = result['crop']['p95_deg']; best_step = step
                torch.save({'model': model.state_dict(), 'step': step,
                            'schema': 'sc_native_crop_orientation_residual/v1',
                            'runtime_geometry_input': False,
                            'spatial': a.spatial}, a.output/'best.pt')
    model.load_state_dict(torch.load(a.output/'best.pt', map_location='cpu', weights_only=False)['model'])
    model.eval()
    summary = {'schema': 'sc_native_crop_orientation_ablation/v1', 'audit': audit,
               'fit_examples': len(fit), 'development_examples': len(dev),
               'ordinary_validation_examples': len(original_val), 'best_step': best_step,
               'development': score(dev), 'ordinary_validation': score(original_val),
               'history': history, 'crop_source': 'RGB joint locator',
               'spatial': a.spatial,
               'truth_used_for_runtime_crop': False, 'final_scenes_opened': False}
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
