#!/usr/bin/env python3
"""Collect native SFP RGB in missing one/two-card TRAIN settings, sequentially.

Uses existing canonical train reset configurations only. The official CheatCode
sets the task; port geometry is for later offline labels, never an actor input.
Four fixed pilot cases must yield at least three full insertions before --full.
"""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import yaml

from run_sfp_cheatcode_target_pilot import BASE, TEMPLATE, score


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'outputs/trajectory_datasets/expert_verified/manifest.json'
RAW = ROOT / 'outputs/trajectory_datasets/clean_including_no_insert_trajs'
DEFAULT_OUTPUT = Path('/home/nas/chmin/aic_archives/2026-10-05_sfp_sparse_camera_train_v1')
PILOT = {('cards1_slot0_port0'), ('cards1_slot4_port1'),
         ('cards2_slot1_port0'), ('cards2_slot4_port1')}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases(include_controls=False):
    manifest = json.loads(MANIFEST.read_text())
    if manifest['schema_version'] != 1:
        raise ValueError('Unexpected expert manifest')
    sources = defaultdict(list)
    for episode in manifest['episodes']:
        if (episode.get('split') != 'train' or episode.get('nic_count') not in (1, 2)
                or episode.get('task', {}).get('task_family') != 'sfp_to_nic'
                or not episode.get('source')):
            continue
        path = RAW / episode['source'] / 'trials' / f"{episode['trial_id']}.yaml"
        if not path.is_file():
            continue
        config = yaml.safe_load(path.read_text())
        trial = next(iter(config['trials'].values()))
        board = trial['scene']['task_board']
        slots = tuple(i for i in range(5) if board[f'nic_rail_{i}']['entity_present'])
        if len(slots) == episode['nic_count']:
            sources[slots].append((path, episode, config))
    planned = []
    for slot in range(5):
        layout = (slot,)
        if not sources[layout]:
            raise ValueError(f'Missing canonical train scene {layout}')
        path, episode, config = min(sources[layout], key=lambda r: str(r[0]))
        for port in (0, 1):
            planned.append((f'cards1_slot{slot}_port{port}', layout, slot, port,
                            path, episode, config))
    for slot in range(1, 5):
        layout = (0, slot)
        if not sources[layout]:
            raise ValueError(f'Missing canonical train scene {layout}')
        path, episode, config = min(sources[layout], key=lambda r: str(r[0]))
        for port in (0, 1):
            planned.append((f'cards2_slot{slot}_port{port}', layout, slot, port,
                            path, episode, config))
        if include_controls and slot in (1, 4):
            planned.append((f'cards2_slot{slot}_original_card0_port1', layout, 0, 1,
                            path, episode, config))
    return planned


def key_quota():
    line = next((line for line in Path('/proc/key-users').read_text().splitlines()
                 if line.lstrip().startswith(f'{os.getuid()}:')), None)
    if line:
        count, limit = line.split()[3].split('/')
        if int(count) >= int(limit):
            raise RuntimeError(f'Rootless key quota {count}/{limit}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--full', action='store_true', help='Continue only after pilot gate')
    p.add_argument('--controls', action='store_true', help='Run only original-card controls in a separate output')
    a = p.parse_args()
    planned = cases(a.controls)
    manifest = {'schema': 'sfp_sparse_camera_train_collection/v1',
                'source_expert_manifest': str(MANIFEST),
                'source_expert_manifest_sha256': sha(MANIFEST),
                'pilot_cases': [] if a.controls else sorted(PILOT),
                'pilot_full_insertion_gate': 'at least 3 of 4' if not a.controls else 'not applicable',
                'all_cases': [{'name': name, 'source_slots': list(layout),
                               'selected_card': card, 'selected_port': port,
                               'source_trial': str(path), 'source_sha256': sha(path),
                               'source_split': episode['split'],
                               'source_scene_sha256': episode['scene_sha256']}
                              for name, layout, card, port, path, episode, _ in planned],
                'gpu': 1, 'rootless': True, 'max_sim_seconds': 70,
                'final_scenes_opened': False}
    a.output.mkdir(parents=True, exist_ok=True)
    mp = a.output / 'collection_manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise ValueError('Frozen sparse collection plan differs')
    mp.write_text(json.dumps(manifest, indent=2) + '\n')
    pp = a.output / 'progress.json'
    progress = json.loads(pp.read_text()) if pp.exists() else {'results': {}}
    pilot_success = sum(progress['results'].get(name, {}).get('official_score', {}).get('tier3', 0) >= 75
                        for name in PILOT)
    pilot_done = all(name in progress['results'] for name in PILOT)
    if a.full and not a.controls and pilot_done and pilot_success < 3:
        raise RuntimeError(f'Pilot gate failed: {pilot_success}/4')
    if a.controls:
        selected = [row for row in planned if 'original_card0' in row[0]]
    else:
        selected = [row for row in planned if row[0] in PILOT] if not a.full else planned
    for name, layout, card, port, source, episode, original in selected:
        if name in progress['results']:
            continue
        if a.full and not a.controls and name not in PILOT and not (pilot_done and pilot_success >= 3):
            raise RuntimeError(f'Cannot expand before passing pilot: {pilot_success}/4')
        scene = a.output / 'scenes' / name
        audit = a.output / 'rollouts' / name
        if scene.exists() or audit.exists():
            raise FileExistsError(f'Existing partial case {name}')
        key_quota()
        config = copy.deepcopy(original)
        task = next(iter(next(iter(config['trials'].values()))['tasks'].values()))
        task['target_module_name'] = f'nic_card_mount_{card}'
        task['port_name'] = f'sfp_port_{port}'
        scene.mkdir(parents=True)
        (scene / 'eval_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        for item in ('world_audit.sdf', 'bridge_audit.yaml'):
            shutil.copy2(TEMPLATE / item, scene / item)
        env = {**os.environ,
               'DOCKER_HOST': f'unix:///run/user/{os.getuid()}/docker.sock',
               'AIC_TRANSPORT_EVAL_GPU': '1',
               'AIC_TRANSPORT_EVAL_POLICY_GPU': '1',
               'AIC_TRANSPORT_EVAL_SCENE_DIR': str(scene.resolve()),
               'AIC_TRANSPORT_EVAL_POLICY': 'aic_example_policies.ros.CheatCode',
               'AIC_TRANSPORT_EVAL_TEACHER_PREFIX': 'true',
               'AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA': '.06',
               'AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC': '70',
               'AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC': '200'}
        command = ['bash', 'scripts/run_clean_transport_eval.sh', name,
                   str(BASE), str(audit)]
        with (a.output / f'{name}.launch.log').open('w') as log:
            try:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=320)
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                returncode = None
        item = {'name': name, 'source_trial': str(source), 'source_scene_sha256': episode['scene_sha256'],
                'source_slots': list(layout), 'selected_card': card, 'selected_port': port,
                'scene': str(scene), 'audit': str(audit), 'command': command,
                'returncode': returncode, 'official_score': score(audit),
                'frame_manifest_exists': (audit / 'frame_manifest.csv').is_file()}
        progress['results'][name] = item
        pp.write_text(json.dumps(progress, indent=2) + '\n')
        print(json.dumps({'name': name, 'score': item['official_score'],
                          'frames': item['frame_manifest_exists']}), flush=True)
        pilot_done = all(n in progress['results'] for n in PILOT)
        pilot_success = sum(progress['results'].get(n, {}).get('official_score', {}).get('tier3', 0) >= 75
                            for n in PILOT)
    summary = {'pilot_completed': pilot_done, 'pilot_full_insertions': pilot_success,
               'pilot_gate_pass': pilot_done and pilot_success >= 3,
               'completed': len(progress['results']),
               'full_insertions': sum(r['official_score'].get('tier3', 0) >= 75
                                      for r in progress['results'].values()),
               'final_scenes_opened': False}
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
