#!/usr/bin/env python3
"""Score frozen rail predictions against projected labels and save RGB overlays."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

CAMS = ('center', 'left', 'right')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--overlays-per-split', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    args.output.mkdir(parents=True)
    rows = [json.loads(line) for line in (args.labels / 'rows.jsonl').open()]
    preds = [json.loads(line) for line in args.predictions.open()]
    errors = defaultdict(list)
    flags = defaultdict(list)
    scenes = set()
    by_scene = defaultdict(list)
    for pred in preds:
        row = rows[pred['row']]
        if pred['scene_group'] != row['scene_group']:
            raise ValueError('Prediction/label row mismatch')
        by_scene[row['scene_group']].append((pred, row))
        scenes.add(row['scene_group'])
        for ci, cam in enumerate(CAMS):
            for j, rail in enumerate((0, 4)):
                gt = row['rail_projections_training_label_only'][cam][rail]
                inframe = bool(gt[3])
                probability = pred['endpoint_inframe_prob'][ci][j]
                flags['all'].append((inframe, probability))
                if not inframe:
                    continue
                xy = pred['endpoint_xy_normalized'][ci][j]
                err = float(np.linalg.norm(np.array([xy[0]*1151, xy[1]*1023]) - gt[:2]))
                errors['all'].append(err)
                errors[cam].append(err)
                if row['card_count'] <= 2:
                    errors['sparse'].append(err)
                else:
                    errors['dense'].append(err)
    # Early geometry is the important part of the camera-check use case.
    for scene, items in by_scene.items():
        items.sort(key=lambda pair: pair[0]['time_s'])
        first = items[0][0]['time_s']
        for pred, row in items:
            if pred['time_s'] - first > 3:
                continue
            for ci, cam in enumerate(CAMS):
                for j, rail in enumerate((0, 4)):
                    gt = row['rail_projections_training_label_only'][cam][rail]
                    if gt[3]:
                        xy = pred['endpoint_xy_normalized'][ci][j]
                        errors['early'].append(float(np.linalg.norm(
                            np.array([xy[0]*1151, xy[1]*1023]) - gt[:2])))
    metrics = {'schema': 'sfp_board_rail_reference_score/v1',
               'predictions': str(args.predictions), 'rows': len(preds),
               'reset_groups': len(scenes),
               'endpoint_pixel_error': {
                   name: {'n': len(values), 'median': float(np.median(values)),
                          'p90': float(np.percentile(values, 90)),
                          'mean': float(np.mean(values))}
                   for name, values in errors.items() if values},
               'inframe': {'n': len(flags['all']),
                           'accuracy_at_half': float(np.mean([int(t == (p >= .5))
                                                           for t, p in flags['all']]))}}
    for scene, items in sorted(by_scene.items())[:args.overlays_per_split]:
        pred, row = min(items, key=lambda pair: pair[0]['time_s'])
        for ci, cam in enumerate(CAMS):
            image = cv2.imread(row['images'][cam])
            if image is None:
                raise FileNotFoundError(row['images'][cam])
            for j, rail in enumerate((0, 4)):
                gt = row['rail_projections_training_label_only'][cam][rail]
                xy = pred['endpoint_xy_normalized'][ci][j]
                pp = (int(round(xy[0]*1151)), int(round(xy[1]*1023)))
                cv2.circle(image, pp, 9, (0, 0, 255), 3)
                cv2.putText(image, f'PRED rail{rail}', (max(0, pp[0]-50), max(20, pp[1]-14)),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
                if gt[3]:
                    gp = (int(round(gt[0])), int(round(gt[1])))
                    cv2.circle(image, gp, 6, (0, 255, 0), 2)
            cv2.putText(image, 'RED=model RGB prediction  GREEN=projected CAD label',
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 3)
            cv2.putText(image, 'RED=model RGB prediction  GREEN=projected CAD label',
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 1)
            cv2.imwrite(str(args.output / f'{scene[:12]}_{cam}.jpg'), image)
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
