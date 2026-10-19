#!/usr/bin/env python3
"""Train-only fixed pixel-bias calibration for frozen SFP rail predictions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


CAMS = ('center', 'left', 'right')
SCALE = np.asarray([1151., 1023.])


def read(path):
    return [json.loads(line) for line in path.open()]


def check(prediction, row, split):
    if prediction['scene_group'] != row['scene_group'] or \
            prediction['episode'] != row['episode'] or \
            abs(prediction['time_s'] - row['time_s']) > 1e-6 or row['split'] != split:
        raise ValueError('Prediction/label join or split mismatch')


def errors(predictions, labels):
    result = defaultdict(list)
    for prediction in predictions:
        row = labels[prediction['row']]
        for ci, cam in enumerate(CAMS):
            for end, slot in enumerate((0, 4)):
                label = row['rail_projections_training_label_only'][cam][slot]
                if not label[3] or prediction['endpoint_inframe_prob'][ci][end] < .5:
                    continue
                pixel = np.asarray(prediction['endpoint_xy_normalized'][ci][end]) * SCALE
                result[(cam, end)].append(pixel - np.asarray(label[:2]))
    return result


def stats(values):
    return {'rows': len(values), 'median_px': float(np.median(values)),
            'p95_px': float(np.percentile(values, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--train-predictions', type=Path, required=True)
    parser.add_argument('--development-predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve prior output')
    all_rows = read(args.labels / 'rows.jsonl')
    train_rows = [x for x in all_rows if x['split'] == 'train']
    train_predictions = read(args.train_predictions)
    development_predictions = read(args.development_predictions)
    if len({x['scene_group'] for x in train_rows}) != 70 or len(train_predictions) != len(train_rows):
        raise ValueError('Unexpected 70-group train prediction cache')
    for prediction in train_predictions:
        check(prediction, train_rows[prediction['row']], 'train')
    for prediction in development_predictions:
        check(prediction, all_rows[prediction['row']], 'validation')
        if all_rows[prediction['row']]['autonomous_actor_diagnostic']:
            raise ValueError('Development predictions must exclude actor diagnostics')
    train_errors = errors(train_predictions, train_rows)
    offsets = {}
    for cam in CAMS:
        for end in (0, 1):
            residuals = np.asarray(train_errors[(cam, end)])
            if len(residuals) < 100:
                raise ValueError(f'Insufficient train samples for {cam}/{end}')
            offsets[(cam, end)] = np.median(residuals, axis=0)
    original = errors(development_predictions, all_rows)
    corrected = []
    for prediction in development_predictions:
        row = dict(prediction)
        coords = np.asarray(prediction['endpoint_xy_normalized'], float).copy()
        for ci, cam in enumerate(CAMS):
            for end in (0, 1):
                coords[ci, end] -= offsets[(cam, end)] / SCALE
        row['endpoint_xy_normalized'] = coords.tolist()
        row['train_only_fixed_bias_applied'] = True
        corrected.append(row)
    adjusted = errors(corrected, all_rows)
    summary = {'schema': 'sfp_rail_train_bias/v1',
               'train_groups': 70, 'development_groups': 10,
               'offset_predicted_minus_label_px': {
                   f'{cam}_{end}': offsets[(cam, end)].tolist()
                   for cam in CAMS for end in (0, 1)},
               'development_pixel_error': {
                   f'{cam}_{end}': {'before': stats(np.linalg.norm(original[(cam, end)], axis=1)),
                                   'after': stats(np.linalg.norm(adjusted[(cam, end)], axis=1))}
                   for cam in CAMS for end in (0, 1)},
               'train_only_geometry_calibration': True,
               'development_geometry_used_for_scoring_only': True,
               'final_scenes_opened': False}
    args.output.mkdir(parents=True)
    with (args.output / 'predictions.jsonl').open('w') as target:
        for row in corrected:
            target.write(json.dumps(row) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['development_pixel_error'], indent=2))


if __name__ == '__main__':
    main()
