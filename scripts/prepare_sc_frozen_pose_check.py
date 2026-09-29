#!/usr/bin/env python3
"""Freeze SC RGB estimator artifacts before generating fresh development scenes."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed-a', type=int, required=True)
    p.add_argument('--seed-b', type=int, required=True)
    p.add_argument('--trial-base', type=int, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    root = Path('/var/tmp/chmin_aic_20260918_act')
    files = {
        'locator': root/'sc_native_broadened_20260924/locator_v2/locator.pt',
        'opening': root/'sc_native_yaw_aug_20260924/keypoint_conditioned/refiner.pt',
        'orientation': root/'sc_native_yaw_aug_20260924/landmarks_wide/landmarks.pt',
        'tip': root/'sc_physical_tip_augmented_20260924/tip_only_refiner/tip_refiner.pt',
        'camera_training_edges': root/'sc_native_yaw_aug_20260924/edges_joined.json',
        'board_to_port': Path('configs/hierarchical_recovery/sc_board_to_port_training_fixed.json'),
        'port_tracker_code': Path('scripts/evaluate_sc_pairwise_port_tracker.py'),
        'yaw_code': Path('scripts/evaluate_sc_board_lines_yaw.py'),
        'tip_code': Path('scripts/evaluate_sc_causal_physical_tip.py'),
        'relative_code': Path('scripts/evaluate_sc_relative_pose_fusion.py'),
    }
    bundle = {'schema': 'sc_frozen_pose_check/v1',
              'created_before_new_scene_generation': True,
              'utc': datetime.now(timezone.utc).isoformat(),
              'seeds': [a.seed_a, a.seed_b],
              'trial_base': a.trial_base,
              'estimator_settings': {'reject_search_boundary': True,
                                     'max_history_jump_deg': 180.,
                                     'port_tracker_defaults': True},
              'hashes': {name: {'path': str(path), 'sha256': sha(path)}
                         for name, path in files.items()}}
    (a.output/'frozen_bundle.json').write_text(json.dumps(bundle, indent=2)+'\n')
    configs = []
    generator = Path('aic_engine/scripts/generate_random_trials_config.py')
    for seed, suffix in ((a.seed_a, 'a'), (a.seed_b, 'b')):
        path = a.output / f'generated_{suffix}.yaml'
        subprocess.run(['.pixi/envs/default/bin/python', str(generator),
                        '--output', str(path), '--num_trials', '4',
                        '--episodes_per_setup', '1', '--seed', str(seed),
                        '--profile', 'qualification_eval_like',
                        '--sfp_to_nic_weight', '0', '--sc_to_sc_weight', '1'], check=True)
        configs.append((path, yaml.safe_load(path.read_text())))
    merged = {key: value for key, value in configs[0][1].items() if key != 'trials'}
    source = {}
    merged['trials'] = {}
    for batch_index, (path, config) in enumerate(configs):
        for trial_index, (old_key, trial) in enumerate(config['trials'].items()):
            key = f'trial_{a.trial_base + batch_index*4 + trial_index:06d}'
            if trial['tasks']['task_1']['target_module_name'].startswith('nic_card_mount_'):
                raise ValueError('SC-only frozen suite generated an SFP task')
            merged['trials'][key] = trial
            source[key] = {'source_yaml': str(path), 'source_trial': old_key,
                           'source_yaml_sha256': sha(path),
                           'target_module': trial['tasks']['task_1']['target_module_name']}
    (a.output/'engine_config.yaml').write_text(yaml.safe_dump(merged, sort_keys=False))
    (a.output/'collection_config.json').write_text(json.dumps({
        'kind': 'fresh_frozen_SC_pose_development_check',
        'policy': 'privileged teacher only for image/metric capture; not BC or autonomous evidence',
        'source': source}, indent=2)+'\n')
    print(json.dumps({'scenes': len(source), 'ports': [x['target_module'] for x in source.values()],
                      'output': str(a.output)}))


if __name__ == '__main__':
    main()
