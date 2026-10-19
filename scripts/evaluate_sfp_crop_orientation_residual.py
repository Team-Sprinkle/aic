#!/usr/bin/env python3
"""Bounded frozen-crop residual on top of the existing RGB orientation prior."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn

from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate
from train_sfp_native_crop_orientation import measure
from train_two_family_port_locator import CAMERAS, NAMES
from train_two_family_port_orientation import Model


class Residual(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(d, 256), nn.SiLU(), nn.Linear(256, 128),
                                    nn.SiLU(), nn.Linear(128, 3))
        nn.init.zeros_(self.layers[-1].weight); nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        return self.layers(x)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--all-candidates', type=Path, required=True)
    p.add_argument('--visual-features', type=Path, required=True)
    p.add_argument('--crop-features', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--prior-primary', type=Path, required=True)
    p.add_argument('--prior-secondary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error('Preserve previous run')
    a.output.mkdir(parents=True)
    torch.set_num_threads(8); torch.manual_seed(20261002); np.random.seed(20261002)
    rows = [json.loads(s) for s in (a.labels / 'rows.jsonl').open()]
    pred = defaultdict(dict)
    for q in map(json.loads, a.all_candidates.open()):
        if q['row'] < len(rows): pred[q['row']][q['camera']] = q
    visual = np.load(a.visual_features, mmap_mode='r')[:len(rows)]
    crop = np.load(a.crop_features)['features']
    cal = json.loads(a.calibration.read_text())
    optics = {c: np.asarray(cal['tcp_to_optical'][c], float) for c in CAMERAS}
    x = []; gt = []; episodes = []; splits = []; prior_pred = []
    priors = []
    for path in (a.prior_primary, a.prior_secondary):
        saved = torch.load(path, map_location='cpu', weights_only=False)['model']
        net = Model(saved['mean'], saved['std']).eval(); net.load_state_dict(saved)
        priors.append(net)
    for i, r in enumerate(rows):
        base_tcp = pose_matrix(r['state'][:7]); cams = {c: base_tcp @ optics[c] for c in CAMERAS}
        points = []; mask = []
        for name in NAMES:
            pixels = {c: pred[i][c]['predicted'][name]['xy'] for c in CAMERAS
                      if pred[i][c]['predicted'][name]['projected_inframe_probability'] >= .5}
            if len(pixels) >= 2:
                try:
                    point = triangulate(pixels, cams); valid = float(np.isfinite(point).all())
                except np.linalg.LinAlgError: valid = 0.
            else: valid = 0.
            points.extend(point.tolist() if valid else [0., 0., 0.]); mask.append(valid)
        code = np.zeros(len(NAMES), np.float32); code[NAMES.index(r['selected_identity'])] = 1
        base_feature = np.r_[points, mask, code, r['state'][:7],
                             np.asarray(visual[i], np.float32).reshape(-1)].astype(np.float32)
        with torch.inference_mode():
            rotations = [m(torch.from_numpy(base_feature)[None]).numpy()[0] for m in priors]
        first = Rotation.from_matrix(rotations[0]); second = Rotation.from_matrix(rotations[1])
        prior = first * Rotation.from_rotvec((first.inv() * second).as_rotvec() * .5)
        target = next(t for t in r['targets'] if t['identity'] == r['selected_identity'])
        truth = Rotation.from_quat(target['orientation_base_xyzw'])
        x.append(crop[i]); gt.append((prior.inv() * truth).as_rotvec())
        prior_pred.append(prior.as_matrix()); episodes.append(r['episode']); splits.append(r['split'])
    x = np.asarray(x, np.float32); y = np.asarray(gt, np.float32)
    prior_pred = np.asarray(prior_pred); target = (Rotation.from_matrix(prior_pred) * Rotation.from_rotvec(y)).as_matrix()
    val_eps = sorted({e for e, s in zip(episodes, splits) if s == 'validation'})
    selection = set(e for e in val_eps if e.endswith('port0_cards3_validation'))
    train = np.array([i for i,s in enumerate(splits) if s == 'train'])
    val = np.array([i for i,e in enumerate(episodes) if e in selection])
    test = np.array([i for i,e in enumerate(episodes) if e in set(val_eps)-selection])
    mean = x[train].mean(0); std = x[train].std(0).clip(min=.01)
    xx = torch.tensor((x-mean)/std, device='cuda'); yy = torch.tensor(y, device='cuda')
    model = Residual(x.shape[1]).cuda(); opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    counts = {e:sum(episodes[i]==e for i in train) for e in set(episodes[i] for i in train)}
    weight = np.array([1/counts[episodes[i]] for i in train]); weight /= weight.sum()
    best = float('inf'); best_step = 0; stale = 0; history = []
    for step in range(1,1501):
        ids = np.random.choice(train,128,p=weight)
        model.train(); opt.zero_grad(set_to_none=True)
        delta = model(xx[ids]); loss = (delta-yy[ids]).square().mean()+.15*delta.square().mean()
        loss.backward(); opt.step()
        if step % 100 == 0:
            model.eval()
            with torch.inference_mode(): vv = model(xx[val]).cpu().numpy()
            predicted = (Rotation.from_matrix(prior_pred[val])*Rotation.from_rotvec(vv)).as_matrix()
            report = measure(predicted,target[val],[episodes[i] for i in val])
            history.append({'step':step,'selection':report})
            print(json.dumps({'step':step,'selection_p95_deg':report['p95_deg']}),flush=True)
            if report['p95_deg'] < best-.05:
                best = report['p95_deg']; best_step = step; stale=0
                torch.save({'model':model.state_dict(),'mean':mean,'std':std,
                            'runtime_geometry_input':False},a.output/'best.pt')
            else: stale+=1
            if step>=500 and stale>=7:break
    saved=torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model']);model.eval()
    results = {}
    with torch.inference_mode():
        for name,idx in [('selection',val),('development_test',test)]:
            residual=model(xx[idx]).cpu().numpy()
            predicted=(Rotation.from_matrix(prior_pred[idx])*Rotation.from_rotvec(residual)).as_matrix()
            results[name]={'prior':measure(prior_pred[idx],target[idx],[episodes[i] for i in idx]),
                           'residual':measure(predicted,target[idx],[episodes[i] for i in idx])}
    summary={'schema':'sfp_crop_orientation_residual/v1','best_step':best_step,
             'train_episodes':len(set(episodes[i] for i in train)),**results,
             'selection_episodes':sorted(selection),
             'development_test_episodes':sorted(set(episodes[i] for i in test)),
             'runtime_geometry_input':False,'reserved_final_opened':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
