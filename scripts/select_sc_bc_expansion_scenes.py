#!/usr/bin/env python3
"""Select a deterministic, card/port-balanced subset of generated SC starts."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--counts', type=int, nargs='+', required=True)
    parser.add_argument('--per-cell', type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.source / 'source_manifest.json').read_text())
    if not manifest['training_only']:
        parser.error('Source must be training-only')
    cells = defaultdict(list)
    for trial, row in manifest['sources'].items():
        cells[(row['nic_card_count'], row['target_module'])].append(trial)
    chosen = []
    for count in args.counts:
        for target in ('sc_port_0', 'sc_port_1'):
            sample = cells[(count, target)][:args.per_cell]
            if len(sample) < args.per_cell:
                parser.error(f'Only {len(sample)} scenes for {count} cards, {target}')
            chosen.extend(sample)
    chosen = sorted(chosen)
    args.output.mkdir(parents=True)
    engine = yaml.safe_load((args.source / 'engine_config.yaml').read_text())
    engine['trials'] = {trial: engine['trials'][trial] for trial in chosen}
    engine_path = args.output / 'engine_config.yaml'
    engine_path.write_text(yaml.safe_dump(engine, sort_keys=False))
    pool_manifest = args.source / 'source_manifest.json'
    selected = dict(manifest, source_pool=str(args.source),
                    source_pool_manifest=str(pool_manifest),
                    source_pool_manifest_sha256=hashlib.sha256(pool_manifest.read_bytes()).hexdigest(),
                    engine_config=str(engine_path),
                    engine_sha256=hashlib.sha256(engine_path.read_bytes()).hexdigest(),
                    sources={trial: manifest['sources'][trial] for trial in chosen})
    (args.output / 'source_manifest.json').write_text(json.dumps(selected, indent=2) + '\n')
    collection = json.loads((args.source / 'collection_config.json').read_text())
    collection['source_episode_indices'] = {
        trial: collection['source_episode_indices'][trial] for trial in chosen}
    collection['source_manifest'] = str(args.output / 'source_manifest.json')
    (args.output / 'collection_config.json').write_text(json.dumps(collection, indent=2) + '\n')
    print(json.dumps({'selected': len(chosen), 'counts': args.counts,
                      'per_cell': args.per_cell, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
