#!/usr/bin/env python3
"""Bounded frozen-backbone SC port-0/left heatmap repair on grouped scenes."""

import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
import yaml

from train_two_family_port_locator import Locator, LOOKUP


DEFAULT_TRAIN = {'port0_four_shiftplus', 'port0_two_mixed'}


def metrics(rows, predicted):
    result = defaultdict(list)
    for row, xy in zip(rows, predicted):
        error = float(np.linalg.norm(np.asarray(xy)-row['truth_xy']))
        result[(row['episode'], row['camera'])].append(error)
    return {f'{k[0]}/{k[1]}': dict(n=len(v), p50_px=float(np.median(v)),
                                  p95_px=float(np.percentile(v, 95)),
                                  over_50_px=int(np.sum(np.asarray(v)>50)))
            for k, v in sorted(result.items())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--labels', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=300)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--train-groups', nargs='+', default=sorted(DEFAULT_TRAIN))
    p.add_argument('--test-groups', nargs='*', default=[],
                   help='Optional genuinely independent scene groups; omit for train-only replay')
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve previous result')
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    np.random.seed(20261002)
    labels = []
    for path in a.labels:
        labels.extend(json.loads(line) for line in path.open())
    train_groups = set(a.train_groups)
    test_groups = set(a.test_groups)
    if train_groups & test_groups:
        raise ValueError('Train/test episode names overlap')
    episodes = {r['episode_id'] for r in labels}
    if episodes != train_groups | test_groups:
        raise ValueError(f'Unexpected grouped split {episodes}')
    scene_hashes = {}
    for r in labels:
        episode = r['episode_id']
        if episode in scene_hashes:
            continue
        audit = Path(r['image']).parent.parent
        config = yaml.safe_load((audit/'eval_config.yaml').read_text())
        trial = next(iter(config['trials'].values()))
        scene_hashes[episode] = hashlib.sha256(
            json.dumps(trial['scene'], sort_keys=True).encode()).hexdigest()
    train_hashes = {scene_hashes[e] for e in train_groups}
    test_hashes = {scene_hashes[e] for e in test_groups}
    if train_hashes & test_hashes:
        raise ValueError('Train/test share a complete scene configuration')
    rows = []
    for r in labels:
        if r['geometric_visible_fraction'] < .7:
            continue
        truth = np.asarray(r['projected_opening_px'], dtype=float).mean(0)
        if not (0 <= truth[0] < 1152 and 0 <= truth[1] < 1024):
            continue
        image = cv2.imread(r['image'])
        if image is None:
            raise FileNotFoundError(r['image'])
        image = cv2.cvtColor(cv2.resize(image, (288, 256), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2RGB)
        rows.append(dict(episode=r['episode_id'], camera=r['camera'],
                         port=int(r['target_port']), truth_xy=truth,
                         image=torch.from_numpy(image.copy()).permute(2, 0, 1)))
    train = [r for r in rows if r['episode'] in train_groups and r['camera'] == 'left' and r['port'] == 0]
    if len(train) < 30:
        raise ValueError('Too few independent exposed train examples')
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model = Locator(None).to(a.device).eval()
    model.load_state_dict(saved['model'])
    original = copy.deepcopy(model.state_dict())
    for param in model.parameters():
        param.requires_grad_(False)
    last = model.decoder[-1]
    last.weight.requires_grad_(True)
    last.bias.requires_grad_(True)
    opt = torch.optim.Adam([last.weight, last.bias], lr=3e-4)
    train_x = torch.stack([r['image'] for r in train]).to(a.device).float()/255
    train_meta = torch.tensor([[1, 0, 0, 1, 0]]*len(train), device=a.device).float()
    truth = torch.tensor(np.stack([r['truth_xy'] for r in train]), device=a.device).float()
    rng = np.random.default_rng(20261002)
    losses = []
    for step in range(1, a.updates+1):
        ii = torch.as_tensor(rng.integers(0, len(train), 16), device=a.device)
        _, _, heat = model(train_x[ii], train_meta[ii], return_heatmap=True)
        _, _, h, w = heat.shape
        x = (truth[ii, 0]/1151*(w-1)).round().long().clamp(0, w-1)
        y = (truth[ii, 1]/1023*(h-1)).round().long().clamp(0, h-1)
        target = y*w+x
        loss = F.cross_entropy((heat[:, LOOKUP['sc_port_0']]*4).flatten(1), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            losses.append(dict(update=step, heatmap_ce=float(loss.detach().cpu())))
    def predict(state):
        model.load_state_dict(state)
        model.eval()
        values = []
        with torch.inference_mode():
            for r in rows:
                x = r['image'][None].to(a.device).float()/255
                meta = torch.tensor([[1, 0, int(r['camera']=='center'),
                                      int(r['camera']=='left'), int(r['camera']=='right')]],
                                    device=a.device).float()
                xy, _ = model(x, meta)
                index = LOOKUP[f"sc_port_{r['port']}"]
                values.append((xy[0, index].float().cpu().numpy()*[1151, 1023]).tolist())
        return values
    fitted = copy.deepcopy(model.state_dict())
    before = predict(original)
    after = predict(fitted)
    test_rows = [i for i, r in enumerate(rows) if r['episode'] in test_groups]
    test_truth = [rows[i] for i in test_rows]
    baseline = metrics(test_truth, [before[i] for i in test_rows])
    adapted = metrics(test_truth, [after[i] for i in test_rows])
    train_idx = [i for i, r in enumerate(rows) if r['episode'] in train_groups]
    report = dict(schema='aic_sc_port0_left_decoder_diagnostic/v1',
                  checkpoint=str(a.checkpoint), train_groups=sorted(train_groups),
                  untouched_test_groups=sorted(test_groups), train_left_exposed=len(train),
                  scene_configuration_sha256=scene_hashes,
                  updates=a.updates, frozen_backbone=True, last_conv_only=True,
                  train_before=metrics([rows[i] for i in train_idx], [before[i] for i in train_idx]),
                  train_after=metrics([rows[i] for i in train_idx], [after[i] for i in train_idx]),
                  test_before=baseline, test_after=adapted, loss_history=losses,
                  label='geometric aperture projection; true visibility not certified',
                  promoted=False, final_scenes_opened=False)
    a.output.mkdir(parents=True)
    torch.save(dict(last_weight=fitted['decoder.4.weight'].cpu(),
                    last_bias=fitted['decoder.4.bias'].cpu(),
                    parent_checkpoint=str(a.checkpoint), diagnostic_only=True),
               a.output/'decoder_only.pt')
    (a.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('train_left_exposed', 'test_before', 'test_after')}))


if __name__ == '__main__':
    main()
