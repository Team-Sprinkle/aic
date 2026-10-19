#!/usr/bin/env python3
"""Bounded frozen-feature SC arrival-head repair using existing on-policy paths.

All fitted examples exclude every episode group in the seven-path frozen
comparison. The output is a diagnostic, never an automatically promoted stop.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


THRESHOLDS = (.3, .5, .7, .8, .9)


def event_scores(rows: list[dict], probabilities: np.ndarray,
                 threshold: float, streak_required: int = 2,
                 min_time_s: float = 15.) -> dict:
    by_audit = defaultdict(list)
    for row, probability in zip(rows, probabilities):
        by_audit[row['audit']].append((row, float(probability)))
    outcomes = []
    for audit, values in by_audit.items():
        values.sort(key=lambda pair: pair[0]['sim_time_s'])
        safe_exists = any(row['coarse_safe'] for row, _ in values)
        streak = 0
        chosen = None
        for row, probability in values:
            streak = streak+1 if row['sim_time_s'] >= min_time_s and probability >= threshold else 0
            if streak >= streak_required:
                chosen = dict(time_s=row['sim_time_s'], probability=probability,
                              coarse_safe=row['coarse_safe'], position_mm=row['position_mm'],
                              angle_deg=row['angle_deg'], max_route_mm=row['max_route_mm'])
                break
        outcomes.append(dict(audit=audit, episode=values[0][0]['episode'],
                             safe_exists=safe_exists, proposal=chosen))
    return dict(accepted_safe=sum(x['proposal'] is not None and x['proposal']['coarse_safe']
                                  for x in outcomes),
                false_proposal=sum(x['proposal'] is not None and not x['proposal']['coarse_safe']
                                   for x in outcomes),
                missed_safe=sum(x['safe_exists'] and x['proposal'] is None for x in outcomes),
                audits=len(outcomes), outcomes=outcomes)


def score_key(result: dict) -> tuple:
    # A false pause is undesirable but can later be rejected by perception.
    # False acceptance into insertion is handled by the separate verifier.
    return (result['false_proposal']*3+result['missed_safe'],
            result['false_proposal'], -result['accepted_safe'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--frozen-comparison', type=Path, required=True)
    parser.add_argument('--handoff', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=600)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve previous result; choose a new output')
    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    rows = [json.loads(line) for line in (args.features/'rows.jsonl').open()]
    x = np.load(args.features/'embeddings.npy').astype(np.float32)
    if len(rows) != len(x) or x.shape[1] != 256:
        raise ValueError('Frozen embedding shape mismatch')
    frozen = json.loads(args.frozen_comparison.read_text())
    frozen_groups = {row['episode'] for row in frozen['episodes']}
    frozen_audits = {str(Path(row['audit']).resolve()) for row in frozen['episodes']}
    groups = sorted({row['episode'] for row in rows if row['episode'] not in frozen_groups})
    if len(groups) < 4:
        raise ValueError(f'Insufficient independent training groups: {groups}')
    # Choose validation groups before examining the fitted results. If a
    # category is missing in validation, report it rather than mixing resets.
    validation_groups = set(groups[::max(1, len(groups)//2)][:2])
    train_idx = np.asarray([i for i, row in enumerate(rows)
                            if row['episode'] not in frozen_groups and
                            row['episode'] not in validation_groups], dtype=int)
    val_idx = np.asarray([i for i, row in enumerate(rows)
                          if row['episode'] in validation_groups], dtype=int)
    test_idx = np.asarray([i for i, row in enumerate(rows)
                           if row['episode'] in frozen_groups and
                           str(Path(row['audit']).resolve()) in frozen_audits], dtype=int)
    if not len(train_idx) or not len(val_idx) or not len(test_idx):
        raise ValueError('Empty grouped split')
    y = np.asarray([row['coarse_safe'] for row in rows], dtype=np.float32)
    if not (y[train_idx].any() and (1-y[train_idx]).any()):
        raise ValueError('Training data requires positive and negative labels')
    mean = x[train_idx].mean(0)
    std = x[train_idx].std(0).clip(min=.05)
    normalized = torch.as_tensor((x-mean)/std)
    truth = torch.as_tensor(y)
    checkpoint = torch.load(args.handoff, map_location='cpu', weights_only=False)
    original_weight = checkpoint['model']['handoff.weight'].float()
    original_bias = checkpoint['model']['handoff.bias'].float()
    original = (torch.as_tensor(x)@original_weight.T+original_bias).sigmoid().numpy().ravel()
    original_test = {str(t):event_scores([rows[i] for i in test_idx], original[test_idx], t)
                     for t in THRESHOLDS}
    best = None
    histories = []
    rng = np.random.default_rng(20261002)
    for decay in (1e-3, 1e-2, 1e-1):
        head = nn.Linear(256, 1)
        # Transform the old head into standardized-feature coordinates.
        with torch.no_grad():
            head.weight.copy_(original_weight*torch.as_tensor(std)[None])
            head.bias.copy_(original_bias+torch.sum(original_weight*torch.as_tensor(mean)))
        optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=decay)
        positive = train_idx[y[train_idx] > .5]
        negative = train_idx[y[train_idx] <= .5]
        for step in range(args.updates):
            indices = np.r_[rng.choice(positive, 64, replace=True),
                            rng.choice(negative, 64, replace=True)]
            optimizer.zero_grad(set_to_none=True)
            output = head(normalized[indices]).flatten()
            loss = nn.functional.binary_cross_entropy_with_logits(output, truth[indices])
            loss += decay*sum((weight**2).sum() for weight in head.parameters())/256
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            probabilities = head(normalized).flatten().sigmoid().numpy()
        validation = {str(t):event_scores([rows[i] for i in val_idx],
                                          probabilities[val_idx], t)
                      for t in THRESHOLDS}
        threshold = min(THRESHOLDS, key=lambda t:score_key(validation[str(t)]))
        key = score_key(validation[str(threshold)])
        histories.append(dict(decay=decay, threshold=threshold,
                              validation=validation[str(threshold)], score=key))
        if best is None or key < best['key']:
            best = dict(decay=decay, threshold=threshold, key=key,
                        probabilities=probabilities.copy(),
                        weight=head.weight.detach().numpy().copy(),
                        bias=head.bias.detach().numpy().copy())
    selected_test = event_scores([rows[i] for i in test_idx],
                                 best['probabilities'][test_idx], best['threshold'])
    selected_validation = event_scores([rows[i] for i in val_idx],
                                       best['probabilities'][val_idx], best['threshold'])
    args.output.mkdir(parents=True)
    torch.save(dict(weight=torch.as_tensor(best['weight']),bias=torch.as_tensor(best['bias']),
                    mean=torch.as_tensor(mean),std=torch.as_tensor(std),
                    threshold=best['threshold'],source_checkpoint=str(args.handoff),
                    diagnostic_only=True),args.output/'head.pt')
    report = dict(schema='aic_onpolicy_arrival_head_diagnostic/v1',
                  source_features=str(args.features), updates=args.updates,
                  train_groups=sorted(set(groups)-validation_groups),
                  validation_groups=sorted(validation_groups),
                  frozen_test_groups=sorted(frozen_groups),
                  train_rows=len(train_idx), train_positive=int(y[train_idx].sum()),
                  validation_rows=len(val_idx),validation_positive=int(y[val_idx].sum()),
                  frozen_rows=len(test_idx),frozen_positive=int(y[test_idx].sum()),
                  candidates=histories, selected_decay=best['decay'],
                  selected_threshold=best['threshold'],
                  selected_validation=selected_validation,
                  selected_frozen_test=selected_test,
                  original_frozen_at_selected_threshold=original_test[str(best['threshold'])],
                  original_frozen_by_threshold=original_test,
                  action_backbone_unchanged=True,
                  head_not_deployed=True,reserved_final_opened=False)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in
                      ('train_groups','validation_groups','frozen_test_groups',
                       'train_rows','train_positive','selected_decay','selected_threshold')},
                     indent=2),flush=True)
    for name in ('selected_validation','selected_frozen_test',
                 'original_frozen_at_selected_threshold'):
        result=report[name]
        print(name, {key:result[key] for key in
                     ('accepted_safe','false_proposal','missed_safe','audits')}, flush=True)


if __name__ == '__main__':
    main()
