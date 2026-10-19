#!/usr/bin/env python3
"""Render posthoc SFP pose markers from frozen RGB predictions and labels."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = {(r['episode'], round(float(r['time_s']), 4)): r
            for r in map(json.loads, (a.labels / 'rows.jsonl').open())}
    predictions = [r for r in map(json.loads, a.predictions.open()) if
                   r['split_within_validation'] == 'test']
    for r in predictions:
        r['error_px'] = float(np.linalg.norm(np.asarray(r['predicted_xy']) - r['truth_xy_scoring_only']))
    predictions.sort(key=lambda r: r['error_px'])
    chosen = []
    for q in (.50, .90, .95, .99):
        record = predictions[int(q * (len(predictions) - 1))]
        row = rows[record['episode'], round(float(record['time_s']), 4)]
        image = cv2.imread(row['images'][record['camera']], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(row['images'][record['camera']])
        for target in row['targets']:
            x, y, _, inframe = target['cameras'][record['camera']]
            if inframe:
                color = (0, 255, 0) if target['identity'] == row['selected_identity'] else (0, 180, 255)
                cv2.circle(image, (round(x), round(y)), 9, color, 2)
        x, y = map(round, record['predicted_xy'])
        cv2.drawMarker(image, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 25, 3)
        description = f"q{round(q*100)} {record['episode']} {record['camera']} err={record['error_px']:.1f}px"
        cv2.rectangle(image, (0, 0), (1152, 60), (0, 0, 0), -1)
        cv2.putText(image, description, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
        cv2.putText(image, 'red=model  green=selected GT  orange=alternate GT  GT overlay only',
                    (15, 50), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
        chosen.append(image)
    collage = np.vstack((np.hstack(chosen[:2]), np.hstack(chosen[2:])))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.output), collage)
    print(json.dumps({'output': str(a.output), 'test_views': len(predictions),
                      'quantile_errors_px': {str(q): predictions[int(q*(len(predictions)-1))]['error_px']
                                             for q in (.50, .90, .95, .99)},
                      'ground_truth_used_for_overlay_only': True}))


if __name__ == '__main__':
    main()
