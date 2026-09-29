#!/usr/bin/env python3
"""Generate fresh qualification-like SC teacher starts for training only."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--trial-base', type=int, required=True)
    parser.add_argument('--num-trials', type=int, default=8)
    parser.add_argument('--purpose', choices=('training','development'),default='training')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    generated = args.output / 'generated.yaml'
    subprocess.run(['.pixi/envs/default/bin/python',
                    'aic_engine/scripts/generate_random_trials_config.py',
                    '--output', str(generated), '--num_trials', str(args.num_trials),
                    '--episodes_per_setup', '1', '--seed', str(args.seed),
                    '--profile', 'qualification_eval_like',
                    '--sfp_to_nic_weight', '0', '--sc_to_sc_weight', '1'], check=True)
    config = yaml.safe_load(generated.read_text())
    sources = {}
    renamed = {}
    for index, (old, trial) in enumerate(config['trials'].items()):
        target = trial['tasks']['task_1']['target_module_name']
        if target.startswith('nic_card_mount_'):
            raise ValueError(f'Not an SC task: {old}')
        key = f'trial_{args.trial_base + index:06d}'
        renamed[key] = trial
        sources[key] = {'source_trial': old, 'target_module': target,
                        'source_yaml_sha256': sha(generated),
                        'nic_card_count': sum(
                            bool(item.get('entity_present'))
                            for name, item in trial['scene']['task_board'].items()
                            if name.startswith('nic_rail_'))}
    config['trials'] = renamed
    engine = args.output / 'engine_config.yaml'
    engine.write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = {'schema': 'sc_bc_expansion_batch/v1',
                'training_only': args.purpose == 'training',
                'purpose': args.purpose,
                'seed': args.seed, 'trial_base': args.trial_base,
                'generator_profile': 'qualification_eval_like',
                'generated_yaml': str(generated), 'generated_sha256': sha(generated),
                'engine_config': str(engine), 'engine_sha256': sha(engine),
                'sources': sources,
                'admission': ('Only scored full insertion and causal, clean teacher streams may enter BC'
                              if args.purpose == 'training' else
                              'Development scenes are sealed from BC and checkpoint selection')}
    (args.output / 'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (args.output / 'collection_config.json').write_text(json.dumps({
        'kind': ('sc_bc_expansion_teacher_capture' if args.purpose == 'training'
                 else 'sc_bc_expansion_development_scene_reservation'),
        'policy': 'privileged CheatCode teacher for training data only; never actor inference',
        'source_manifest': str(args.output / 'source_manifest.json'),
        'source_episode_indices': {key: {'source_split': ('new_development_train'
                                                       if args.purpose == 'training' else
                                                       'new_development_validation')}
                                   for key in sources}}, indent=2)+'\n')
    print(json.dumps({'trials': len(renamed), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
