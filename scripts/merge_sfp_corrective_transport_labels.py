#!/usr/bin/env python3
"""Append bounded, audited corrective teacher labels to the frozen SFP BC set."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--corrective', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve existing merged output')
    args.output.mkdir(parents=True)
    groups = defaultdict(set)
    episodes = defaultdict(set)
    counts = Counter()
    keys = set()
    corrections = json.loads((args.corrective.parent / 'manifest.json').read_text())
    if not corrections['corrective_teacher'] or corrections['split'] != 'train':
        raise ValueError('Corrective source must be audited train-only materialization')
    with (args.output / 'sfp_labels.jsonl').open('w') as out:
        for source, path in [('base', args.base), ('corrective', args.corrective)]:
            for line in path.open():
                row = json.loads(line)
                split = row['split']
                if source == 'corrective':
                    if (split != 'train' or row.get('privileged_teacher_after_takeover') is not True
                            or row.get('actor_prefix_used_as_supervision') is not False
                            or row['geometry_label_only'] is not True):
                        raise ValueError('Corrective data failed causal/provenance admission')
                key = (int(row['episode_index']), int(row['frame_index']))
                if key in keys:
                    raise ValueError(f'Duplicate episode/frame: {key}')
                keys.add(key)
                task = row['task_vector']
                target = (int(max(range(5), key=lambda c: task[4 + c])),
                          int(max(range(2), key=lambda p: task[2 + p])))
                if source == 'corrective' and target != (
                        int(row['selected_card']), int(row['selected_port'])):
                    raise ValueError('Corrective target code and recorded target disagree')
                groups[split].add(row['scene_sha256'])
                episodes[(source, split, target)].add(row['episode_index'])
                counts[(source, split, target)] += 1
                out.write(line)
    if groups['train'] & groups['validation']:
        raise ValueError('Train/development scene leak')
    if len({target for source, split, target in episodes if source == 'corrective'}) != 10:
        raise ValueError('Corrective labels do not cover all ten SFP requests')
    manifest = dict(schema='aic_sfp_corrective_transport_merge/v1',
                    sources_sha256={str(path): digest(path) for path in (args.base, args.corrective)},
                    rows_by_source_split_target={str(k): v for k, v in sorted(counts.items())},
                    episodes_by_source_split_target={str(k): len(v) for k, v in sorted(episodes.items())},
                    scene_groups={k: len(v) for k, v in groups.items()},
                    train_development_scene_overlap=0,
                    corrective_actor_prefix_used_as_supervision=False,
                    corrective_privileged_teacher=True,
                    final_scenes_opened=False)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(rows=sum(counts.values()), corrective_episodes=sum(
        len(v) for k, v in episodes.items() if k[0] == 'corrective'))))


if __name__ == '__main__':
    main()
