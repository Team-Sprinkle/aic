#!/usr/bin/env python3
"""Freeze a diagnostic-only 3/4/5-card paired scene holdout."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve existing holdout')
    audit = json.loads(a.audit.read_text())
    train_scenes = {r['scene'] for r in audit['records'] if r['split'] == 'train'}
    sources = {}
    for line in a.labels.open():
        r = json.loads(line)
        scene = r['scene_sha256']
        if scene not in train_scenes or scene in sources:
            continue
        source = r.get('source_scene_yaml') or str(Path(r['images']['center']).parents[1] / 'eval_config.yaml')
        sources[scene] = source
    groups = {3: [], 4: [], 5: []}
    for scene, path in sources.items():
        cfg = yaml.safe_load(Path(path).read_text())
        board = next(iter(cfg['trials'].values()))['scene']['task_board']
        cards = sum(bool(board[f'nic_rail_{i}']['entity_present']) for i in range(5))
        if cards in groups:
            groups[cards].append({'scene': scene, 'card_count': cards, 'source_config': path,
                                  'selection_hash': hashlib.sha256(scene.encode()).hexdigest()})
    chosen = []
    for count, records in groups.items():
        if len(records) < 5:
            p.error(f'Only {len(records)} paired scenes with {count} cards')
        chosen.extend(sorted(records, key=lambda x: x['selection_hash'])[:5])
    result = {'schema': 'aic_sfp_branch_probe_holdout/v1',
              'purpose': 'diagnostic-only hard-card generalization; parent actor previously trained on these scenes',
              'scenes': [x['scene'] for x in chosen], 'records': chosen,
              'source_counts': {str(k): len(v) for k, v in groups.items()},
              'final_scenes_opened': False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'holdout_scenes': len(chosen), 'source_counts': result['source_counts']}))


if __name__ == '__main__':
    main()
