#!/usr/bin/env python3
"""Run at most six train-only same-board alternate-request CheatCode controls.

Sequential and resumable. Checks the rootless key quota before every launch,
and stops after three trials unless at least two fully insert. It never opens
reserved final scenes or uses recorded source actions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from run_sfp_cheatcode_target_pilot import BASE, ROOT, score


def key_slot_available():
    uid = os.getuid()
    for line in Path('/proc/key-users').read_text().splitlines():
        if line.strip().startswith(f'{uid}:'):
            # /proc/key-users fields: uid, usage, instantiated/total,
            # quota-keys/maxkeys, quota-bytes/maxbytes. Docker is constrained
            # by the fourth field, not instantiated/total.
            used, limit = line.split()[3].split('/')
            return int(used) < int(limit), f'{used}/{limit}'
    raise RuntimeError(f'No /proc/key-users entry for UID {uid}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', default='1')
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    if plan['schema'] != 'sfp_same_board_paired_request_plan/v1' or not plan['train_only']:
        p.error('Expected the verified train-only paired-request plan')
    if len(plan['pairs']) != 6 or plan['final_scenes_opened']:
        p.error('Unexpected scene count or final-scene status')
    a.output.mkdir(parents=True, exist_ok=True)
    progress_path = a.output / 'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        'schema': 'sfp_paired_request_pilot/v1', 'plan': str(a.plan.resolve()),
        'final_scenes_opened': False, 'results': {},
    }
    if progress['plan'] != str(a.plan.resolve()):
        p.error('Existing progress belongs to a different plan')
    for index, pair in enumerate(plan['pairs']):
        if index == 3:
            pilot = [progress['results'].get(row['name'], {}).get('official_score', {})
                     for row in plan['pairs'][:3]]
            if sum(float(item.get('tier3', 0)) >= 75 for item in pilot) < 2:
                print(json.dumps({'stopped': 'first_three_gate', 'scores': pilot}), flush=True)
                return
        if pair['name'] in progress['results']:
            continue
        available, quota = key_slot_available()
        if not available:
            print(json.dumps({'stopped': 'rootless_user_key_quota', 'quota': quota,
                              'next_scene': pair['name']}), flush=True)
            return
        scene = Path(pair['scene_dir'])
        if not all((scene / item).exists() for item in
                   ('eval_config.yaml', 'world_audit.sdf', 'bridge_audit.yaml')):
            raise FileNotFoundError(f'Incomplete scene: {scene}')
        actual_hash = hashlib.sha256((scene / 'eval_config.yaml').read_bytes()).hexdigest()
        if actual_hash != pair['trial_yaml_sha256'] or pair['split'] != 'train':
            raise ValueError(f'Changed or nontraining paired scene: {scene}')
        name = pair['name'] + '_alternate'
        audit = a.output / ('teacher_' + name)
        env = os.environ.copy()
        env.update(DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock',
                   AIC_TRANSPORT_EVAL_GPU=a.gpu,
                   AIC_TRANSPORT_EVAL_POLICY_GPU=a.gpu,
                   AIC_TRANSPORT_EVAL_SCENE_DIR=str(scene.resolve()),
                   AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
                   AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.CheatCode',
                   AIC_TRANSPORT_EVAL_TEACHER_PREFIX='true',
                   AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='70',
                   AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC='200')
        command = ['bash', 'scripts/run_clean_transport_eval.sh', name,
                   str(BASE), str(audit)]
        with (a.output / (name + '.launch.log')).open('w') as log:
            try:
                result = subprocess.run(command, cwd=ROOT, env=env,
                                        stdout=log, stderr=subprocess.STDOUT,
                                        timeout=350, check=False)
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                returncode = None
        record = {'request': pair['alternate_request'], 'scene': str(scene),
                  'audit': str(audit), 'command': command,
                  'returncode': returncode, 'official_score': score(audit)}
        progress['results'][pair['name']] = record
        progress_path.write_text(json.dumps(progress, indent=2) + '\n')
        print(json.dumps({'scene': pair['name'], **record}), flush=True)
        if returncode != 0 or not record['official_score']['available']:
            print(json.dumps({'stopped': 'launch_or_score_failure'}), flush=True)
            return
    full = sum(float(row['official_score'].get('tier3', 0)) >= 75
               for row in progress['results'].values())
    print(json.dumps({'completed': len(progress['results']), 'full_insertions': full,
                      'six_scene_gate_passed': full >= 4}), flush=True)


if __name__ == '__main__':
    main()
