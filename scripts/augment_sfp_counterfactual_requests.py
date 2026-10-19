#!/usr/bin/env python3
"""Create training-only same-image alternate SFP requests from observed candidates.

Simulator target lists select supervised labels only. The runtime candidate
generator still receives solely the requested task code and observed RGB.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidates', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve prior manifest')
    labels = {(r['episode'], r['time_s']): r for r in
              map(json.loads, (a.labels / 'rows.jsonl').open())}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    unique_images = set()
    with a.output.open('w') as out:
        for row in map(json.loads, a.candidates.open()):
            key = (row['episode'], row['time_s'])
            label = labels[key]
            if label['split'] != 'train':
                raise ValueError('Counterfactual requests must use training groups only')
            unique_images.add(key)
            for target in label['targets']:
                identity = target['identity']
                if not identity.startswith('sfp_card_'):
                    continue
                item = {**row, 'requested': identity}
                out.write(json.dumps(item, separators=(',', ':')) + '\n')
                total += 1
    print(json.dumps({'manifest': str(a.output), 'rows': total,
                      'unique_image_triplets': len(unique_images),
                      'use_of_true_geometry': 'training target list only'}))


if __name__ == '__main__':
    main()
