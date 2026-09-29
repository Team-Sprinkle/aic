#!/usr/bin/env python3
"""Create training-only SC retry scenes from scored partial teacher attempts."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trial-base', type=int, required=True)
    parser.add_argument('--route', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.source / 'source_manifest.json').read_text())
    if not manifest['training_only']:
        parser.error('Source must be training-only')
    attempts = json.loads((args.source / 'chunk_results.json').read_text())
    scores = {}
    for attempt in attempts:
        if not attempt['complete']:
            continue
        path = (Path(attempt['root']) / 'eval_collection/collection_config/'
                'attempt_0001/scoring.yaml')
        report = yaml.safe_load(path.read_text())
        for old in attempt['trials']:
            if old in scores:
                raise ValueError(f'Duplicate scored trial: {old}')
            scores[old] = float(report[old]['tier_3']['score'])
    engine = yaml.safe_load((args.source / 'engine_config.yaml').read_text())
    remapped = {}
    provenance = {}
    for old in manifest['sources']:
        if old not in scores or scores[old] == 75:
            continue
        new = f'trial_{args.trial_base + len(remapped)}'
        remapped[new] = engine['trials'][old]
        provenance[new] = dict(manifest['sources'][old], original_trial=old,
                               original_tier3=scores[old], retry_route=args.route)
    if not remapped:
        raise ValueError('No scored partial trials to retry')
    args.output.mkdir(parents=True)
    engine['trials'] = remapped
    engine_path = args.output / 'engine_config.yaml'
    engine_path.write_text(yaml.safe_dump(engine, sort_keys=False))
    new_manifest = dict(manifest, source_run=str(args.source),
                        source_manifest_sha256=hashlib.sha256(
                            (args.source / 'source_manifest.json').read_bytes()).hexdigest(),
                        engine_config=str(engine_path),
                        engine_sha256=hashlib.sha256(engine_path.read_bytes()).hexdigest(),
                        retry_route=args.route, trial_base=args.trial_base, sources=provenance)
    (args.output / 'source_manifest.json').write_text(json.dumps(new_manifest, indent=2) + '\n')
    collection = json.loads((args.source / 'collection_config.json').read_text())
    collection['source_manifest'] = str(args.output / 'source_manifest.json')
    collection['source_episode_indices'] = {
        name: {'source_split': 'new_development_train'} for name in remapped}
    (args.output / 'collection_config.json').write_text(json.dumps(collection, indent=2) + '\n')
    (args.output / 'retry_mapping.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'retry_scenes': len(remapped), 'route': args.route,
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
