#!/usr/bin/env python3
"""Capture a generated SC training batch in restartable small Gazebo groups.

Only records scored teacher attempts. Admission to BC is a separate audit.
"""

import argparse
import json
import subprocess
from pathlib import Path

import yaml


PYTHON = '.pixi/envs/default/bin/python'
EVAL = 'scripts/evaluate_act_checkpoints_runtime.py'


def evaluate(source, names, label, lift_m):
    root = Path(f'{source}_{label}')
    root.mkdir(exist_ok=True)
    config = yaml.safe_load((source / 'engine_config.yaml').read_text())
    config['trials'] = {name: config['trials'][name] for name in names}
    (root / 'engine_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    original = json.loads((source / 'collection_config.json').read_text())
    original['source_episode_indices'] = {
        name: original['source_episode_indices'][name] for name in names}
    original['source_parent_run'] = str(source)
    (root / 'collection_config.json').write_text(json.dumps(original, indent=2) + '\n')
    (root / 'episodes').mkdir(exist_ok=True)
    command = [PYTHON, EVAL, '--run-dir', str(root),
               '--workspace-host', '/data1/chmin/yj/ws_aic/src/aic',
               '--workspace-container', '/data1/chmin/yj/ws_aic/src/aic',
               '--artifact-host-root', '/var/tmp/chmin_aic_20260918_act',
               '--artifact-container-root', '/var/tmp/chmin_aic_20260918_act',
               '--checkpoint-glob', 'collection_config.json',
               '--engine-config', str(root / 'engine_config.yaml'),
               '--container', 'aic_act_all_eval_20260918',
               '--policy-module', 'aic_example_policies.ros.CollectCorrectiveCheatCode',
               '--diagnostic-ground-truth', '--corrective-data-dir', str(root / 'episodes'),
               '--corrective-perturbation-scale', '0',
               '--corrective-sc-preroute-lift-m', str(lift_m),
               '--corrective-student-probability', '0',
               '--corrective-execution-frame', 'base_link',
               '--corrective-native-image-every-n-frames', '0',
               '--command-mode', 'absolute_pose', '--command-frame', 'base_link',
               '--policy-device', 'cuda:0', '--max-runtime-sec', '90',
               '--engine-timeout-sec', '5000', '--eval-subdir', 'eval_collection',
               '--record-rollout', '--once-existing']
    (root / 'run.json').write_text(json.dumps(command, indent=2) + '\n')
    with (root / 'eval.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                check=False)
    try:
        report = json.loads((root / 'eval.log').read_text().splitlines()[-1])
        complete = bool(report['evaluation_complete'] and report['engine_returncode'] == 0)
    except (ValueError, KeyError, IndexError):
        complete = False
    return {'root': str(root), 'trials': names, 'exit_code': result.returncode,
            'complete': complete}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--chunk-size', type=int, default=4)
    parser.add_argument('--sc-preroute-lift-m', type=float, default=0.0)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error('--chunk-size must be positive')
    if not 0 <= args.sc_preroute_lift_m <= .15:
        parser.error('--sc-preroute-lift-m must be in [0, .15] meters')
    manifest = json.loads((args.source / 'source_manifest.json').read_text())
    if not manifest['training_only']:
        parser.error('Refusing to run a sealed development batch as teacher collection')
    names = list(manifest['sources'])
    results = []
    for start in range(0, len(names), args.chunk_size):
        group = names[start:start + args.chunk_size]
        label = f'part{start + 1:03d}_{start + len(group):03d}'
        result = evaluate(args.source, group, label, args.sc_preroute_lift_m)
        results.append(result)
        print(json.dumps(result), flush=True)
        if not result['complete'] and len(group) > 1:
            for index, name in enumerate(group, start + 1):
                retry = evaluate(args.source, [name], f'isolated{index:03d}',
                                 args.sc_preroute_lift_m)
                results.append(retry)
                print(json.dumps(retry), flush=True)
        (args.source / 'chunk_results.json').write_text(json.dumps(results, indent=2) + '\n')
    (args.source / 'collection_done').write_text(json.dumps({
        'source_trials': len(names), 'capture_attempts': len(results),
        'complete_attempts': sum(row['complete'] for row in results),
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
