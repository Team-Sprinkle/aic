#!/usr/bin/env python3
"""Frozen-feature early-route probe on complete scene-grouped SFP splits."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import lsqr


def metrics(pred, truth):
    err = np.linalg.norm((pred - truth) * 1000, axis=1)
    magnitude = np.linalg.norm(truth * 1000, axis=1)
    cosine = np.sum(pred * truth, axis=1) / np.maximum(1e-9, np.linalg.norm(pred, axis=1) *
                                                       np.linalg.norm(truth, axis=1))
    return {'rows': len(err), 'error_mm_p50': float(np.median(err)),
            'error_mm_p95': float(np.percentile(err, 95)),
            'direction_cosine_median': float(np.median(cosine[magnitude >= 10]))
            if np.any(magnitude >= 10) else None}


def fit_ridge(x, y, weight, alpha=100):
    center = np.average(x, axis=0, weights=weight)
    scale = np.sqrt(np.average((x - center) ** 2, axis=0, weights=weight))
    scale[scale < 1e-5] = 1
    target_mean = np.average(y, axis=0, weights=weight)
    design = (x - center) / scale
    root = np.sqrt(weight / np.mean(weight))
    # LSQR's damp is sqrt(alpha); this avoids a large dense normal matrix.
    coefficients = np.stack([lsqr(design * root[:, None], (y[:, j] - target_mean[j]) * root,
                                  damp=np.sqrt(alpha), atol=1e-6, btol=1e-6,
                                  iter_lim=500)[0] for j in range(3)], axis=1)
    return center, scale, target_mean, coefficients


def predict(model, x):
    center, scale, target_mean, coefficients = model
    return ((x - center) / scale) @ coefficients + target_mean


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--paired-audit', type=Path, required=True)
    p.add_argument('--holdout-scenes', type=Path,
                   help='Diagnostic-only scene resplit; parent policy may have seen these scenes')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-output', type=Path,
                   help='Save the frozen visual+TCP-pose+task ridge for independent teacher scoring')
    p.add_argument('--baseline-model-output', type=Path,
                   help='Save the TCP-pose+task ridge for the independent matched control')
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve existing result')
    if a.model_output and a.model_output.exists():
        p.error('Preserve existing model')
    if a.baseline_model_output and a.baseline_model_output.exists():
        p.error('Preserve existing baseline model')
    rows = [json.loads(s) for s in (a.cache / 'rows.jsonl').open()]
    features = np.load(a.cache / 'features.npy', mmap_mode='r')
    episodes = defaultdict(list)
    for i, r in enumerate(rows):
        if r['family'] == 'sfp':
            episodes[(r['split'], r['episode'])].append(i)
    samples = {}
    for (_, episode), ids in episodes.items():
        ids.sort(key=lambda j: rows[j]['time_s'])
        times = np.asarray([rows[j]['time_s'] for j in ids])
        for i in ids:
            r = rows[i]
            if not 0 <= r['time_s'] <= 1.6:
                continue
            j = ids[int(np.argmin(np.abs(times - (r['time_s'] + 1))))]
            if abs(rows[j]['time_s'] - (r['time_s'] + 1)) > .2:
                continue
            motion = np.asarray(rows[j]['state'][:3], dtype=float) - r['state'][:3]
            state = np.asarray(r['state'], dtype=float)
            task = np.asarray(r['task'], dtype=float)
            visual = np.asarray(features[i], dtype=float)
            samples[i] = {'split': r['split'], 'episode': episode,
                          'scene': r['scene'], 'state_task': np.concatenate([state, task]),
                          'pose_task': np.concatenate([state[:7], task]),
                          'visual_state_task': np.concatenate([visual, state, task]),
                          'visual_pose_task': np.concatenate([visual, state[:7], task]),
                          'future_motion': motion}
    holdout = set(json.loads(a.holdout_scenes.read_text())['scenes']) if a.holdout_scenes else None
    train_ids = [i for i, r in samples.items() if r['split'] == 'train' and
                 (holdout is None or r['scene'] not in holdout)]
    val_ids = [i for i, r in samples.items() if (r['split'] == 'validation' if holdout is None
                                                else r['scene'] in holdout)]
    if {samples[i]['scene'] for i in train_ids} & {samples[i]['scene'] for i in val_ids}:
        raise RuntimeError('Scene leakage')
    counts = defaultdict(int)
    for i in train_ids:
        r = rows[i]
        task = np.asarray(r['task'])
        counts[(int(task[4:9].argmax()), int(task[2:4].argmax()))] += 1
    weight = np.asarray([1 / counts[(int(np.asarray(rows[i]['task'])[4:9].argmax()),
                                     int(np.asarray(rows[i]['task'])[2:4].argmax()))]
                         for i in train_ids], dtype=float)
    y_train = np.asarray([samples[i]['future_motion'] for i in train_ids])
    y_val = np.asarray([samples[i]['future_motion'] for i in val_ids])
    outputs = {}
    preds = {}
    for mode in ('pose_task', 'visual_pose_task', 'state_task', 'visual_state_task'):
        x_train = np.asarray([samples[i][mode] for i in train_ids])
        x_val = np.asarray([samples[i][mode] for i in val_ids])
        model = fit_ridge(x_train, y_train, weight)
        if mode == 'pose_task' and a.baseline_model_output:
            a.baseline_model_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(a.baseline_model_output, center=model[0], scale=model[1],
                     target_mean=model[2], coefficients=model[3],
                     input_name='TCP_pose7_then_task10', future_horizon_s=1.0)
        if mode == 'visual_pose_task' and a.model_output:
            a.model_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(a.model_output, center=model[0], scale=model[1],
                     target_mean=model[2], coefficients=model[3],
                     input_name='frozen_ImageNet_3x512_then_TCP_pose7_then_task10',
                     future_horizon_s=1.0)
        pred = predict(model, x_val)
        preds[mode] = {i: pred[k] for k, i in enumerate(val_ids)}
        outputs[mode] = metrics(pred, y_val)
        # Task-code permutation is an inference control on the same held-out images and state.
        shuffled = x_val.copy()
        for k in range(len(shuffled)):
            shuffled[k, -10:] = x_val[(k + 17) % len(shuffled), -10:]
        outputs[mode]['shuffled_task_code'] = metrics(predict(model, shuffled), y_val)
        if mode.startswith('visual_'):
            visual_shuffled = x_val.copy()
            visual_shuffled[:, :-len(samples[val_ids[0]]['pose_task' if mode == 'visual_pose_task'
                                                                   else 'state_task'])] = np.roll(
                visual_shuffled[:, :-len(samples[val_ids[0]]['pose_task' if mode == 'visual_pose_task'
                                                                       else 'state_task'])], 17, axis=0)
            outputs[mode]['shuffled_visual_scene'] = metrics(predict(model, visual_shuffled), y_val)
    mean = np.tile(np.average(y_train, axis=0, weights=weight), (len(y_val), 1))
    outputs['constant_mean'] = metrics(mean, y_val)
    audit = json.loads(a.paired_audit.read_text())
    paired = [r for r in audit['records'] if (r['split'] == 'validation' if holdout is None
                                           else r['scene'] in holdout) and
              r['same_observation'] and r['index_a'] in samples and r['index_b'] in samples]
    pair_results = {}
    for mode in ('pose_task', 'visual_pose_task', 'state_task', 'visual_state_task'):
        record = []
        for r in paired:
            a_idx, b_idx = r['index_a'], r['index_b']
            true_delta = samples[a_idx]['future_motion'] - samples[b_idx]['future_motion']
            predicted_delta = preds[mode][a_idx] - preds[mode][b_idx]
            true_size = np.linalg.norm(true_delta) * 1000
            pred_size = np.linalg.norm(predicted_delta) * 1000
            cosine = float(np.dot(true_delta, predicted_delta) /
                           max(1e-9, np.linalg.norm(true_delta) * np.linalg.norm(predicted_delta)))
            record.append({'scene': r['scene'], 'index_a': a_idx, 'index_b': b_idx,
                           'true_branch_delta_mm': true_size, 'pred_branch_delta_mm': pred_size,
                           'branch_delta_error_mm': float(np.linalg.norm(true_delta - predicted_delta) * 1000),
                           'direction_cosine': cosine})
        separable = [r for r in record if r['true_branch_delta_mm'] >= 10]
        pair_results[mode] = {'pairs': len(record), 'scenes': len({r['scene'] for r in record}),
                              'separable_pairs_ge_10mm': len(separable),
                              'separable_scenes': len({r['scene'] for r in separable}),
                              'correct_branch_fraction_cosine_ge_point_7': float(np.mean(
                                  [r['direction_cosine'] >= .7 for r in separable])) if separable else None,
                              'branch_delta_error_mm_p50': float(np.median(
                                  [r['branch_delta_error_mm'] for r in separable])) if separable else None,
                              'records': record}
    result = {'schema': 'aic_sfp_early_route_frozen_probe/v1',
              'cache': str(a.cache), 'paired_audit': str(a.paired_audit),
              'train_rows': len(train_ids), 'validation_rows': len(val_ids),
              'train_scenes': len({samples[i]['scene'] for i in train_ids}),
              'validation_scenes': len({samples[i]['scene'] for i in val_ids}),
              'future_horizon_s': 1.0, 'sample_time_s_max': 1.6,
              'probe_only_holdout_scenes': str(a.holdout_scenes) if a.holdout_scenes else None,
              'ridge_alpha': 100, 'task_balanced_training': True,
              'metrics': outputs, 'paired_validation': pair_results,
              'teacher_geometry_actor_input': False, 'final_scenes_opened': False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({**{k: v for k, v in result.items() if k not in ('metrics', 'paired_validation')},
                      'metrics': outputs,
                      'paired_validation': {k: {q: val for q, val in v.items() if q != 'records'}
                                            for k, v in pair_results.items()}}, indent=2))


if __name__ == '__main__':
    main()
