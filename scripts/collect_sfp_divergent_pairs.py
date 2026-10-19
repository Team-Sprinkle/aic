#!/usr/bin/env python3
"""Collect one alternate requested route on each of 20 verified train resets.

Only the task request changes. The original successful trajectory and scene
configuration are immutable. No source action array is used as a label here.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import yaml

from run_sfp_cheatcode_target_pilot import BASE, ROOT, TEMPLATE, score


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan(labels, output):
    episodes = {}
    for line in labels.open():
        row = json.loads(line)
        episodes.setdefault(row['episode_index'], row)
    by_count = defaultdict(list)
    for row in episodes.values():
        if row['split'] != 'train' or row['episode_index'] < 2400000:
            continue
        source = Path(row['source_scored_stream']).parent / 'eval_config.yaml'
        cfg = yaml.safe_load(source.read_text())
        trial = next(iter(cfg['trials'].values()))
        board = trial['scene']['task_board']
        present = [i for i in range(5) if board[f'nic_rail_{i}']['entity_present']]
        if len(present) < 2:
            continue
        original = int(row['selected_card'])
        alternate = max(present, key=lambda i: abs(i-original))
        if abs(alternate-original) < 2:
            continue
        by_count[len(present)].append((row, source, cfg, alternate))
    selected = []
    # Equal count coverage; prefer far-apart requested cards and distinct
    # original request identities inside each count.
    for count in range(2, 6):
        items = sorted(by_count[count], key=lambda x: (
            -abs(x[3]-int(x[0]['selected_card'])),
            int(x[0]['selected_card']), int(x[0]['selected_port']),
            int(x[0]['episode_index'])))
        seen_requests = set()
        chosen = []
        for item in items:
            key = (int(item[0]['selected_card']), int(item[0]['selected_port']))
            if key not in seen_requests:
                chosen.append(item)
                seen_requests.add(key)
            if len(chosen) == 5:
                break
        for item in items:
            if len(chosen) == 5:
                break
            if item not in chosen:
                chosen.append(item)
        if len(chosen) < 5:
            raise ValueError(f'Only {len(chosen)} divergent train resets for {count} cards')
        selected.extend(chosen)
    scenes = []
    for row, source, cfg, alternate in selected:
        original = [int(row['selected_card']), int(row['selected_port'])]
        desired = [alternate, 1-original[1]]
        trial = next(iter(cfg['trials'].values()))
        task = next(iter(trial['tasks'].values()))
        task['target_module_name'] = f'nic_card_mount_{desired[0]}'
        task['port_name'] = f'sfp_port_{desired[1]}'
        name = f"episode{row['episode_index']}_card{original[0]}p{original[1]}_to_card{desired[0]}p{desired[1]}"
        folder = output / 'scenes' / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'eval_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        for file in ('world_audit.sdf', 'bridge_audit.yaml'):
            shutil.copy2(TEMPLATE / file, folder / file)
        scenes.append(dict(name=name, original_episode=row['episode_index'],
                           source_scene_hash=row['scene_sha256'], source_config=str(source),
                           original_request=original, alternate_request=desired,
                           card_count=len([i for i in range(5) if trial['scene']['task_board'][f'nic_rail_{i}']['entity_present']]),
                           scene_dir=str(folder), config_sha256=sha(folder / 'eval_config.yaml')))
    return dict(schema='aic_sfp_divergent_pairs/v1', train_only=True,
                source_agent_actions_used=False, final_scenes_opened=False,
                first_five_min_full=4, planned_count=20, scenes=scenes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', default='1')
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    if a.gpu not in tuple(str(i) for i in range(8)):
        p.error('GPU must be 0..7')
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output / 'plan.json'
    if path.exists():
        planned = json.loads(path.read_text())
    else:
        planned = make_plan(a.labels, a.output)
        path.write_text(json.dumps(planned, indent=2) + '\n')
    if (planned['schema'] not in ('aic_sfp_divergent_pairs/v1',
                                 'aic_sfp_divergent_pairs/v2') or
            planned['final_scenes_opened'] or
            (planned['schema'].endswith('/v1') and not planned['train_only'])):
        raise ValueError('Invalid immutable plan')
    if a.plan_only:
        print(json.dumps(dict(planned=len(planned['scenes']), output=str(path))), flush=True)
        return
    progress_path = a.output / 'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'results': {}}
    for number, spec in enumerate(planned['scenes']):
        if number == 5 and sum(progress['results'][x['name']]['official_score'].get('tier3', 0) >= 75
                               for x in planned['scenes'][:5]) < planned['first_five_min_full']:
            print(json.dumps(dict(stopped='pilot_success_gate')), flush=True)
            break
        name = spec['name']
        if name in progress['results']:
            continue
        scene = Path(spec['scene_dir'])
        if sha(scene / 'eval_config.yaml') != spec['config_sha256']:
            raise ValueError('Changed scene config')
        audit = a.output / 'rollouts' / name
        env = os.environ.copy()
        env.update(DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock',
                   AIC_TRANSPORT_EVAL_GPU=a.gpu, AIC_TRANSPORT_EVAL_POLICY_GPU=a.gpu,
                   AIC_TRANSPORT_EVAL_SCENE_DIR=str(scene.resolve()),
                   AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
                   AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.CheatCode',
                   AIC_TRANSPORT_EVAL_TEACHER_PREFIX='true',
                   AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='70',
                   AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC='200')
        command = ['bash', 'scripts/run_clean_transport_eval.sh', name, str(BASE), str(audit)]
        with (a.output / (name+'.launch.log')).open('w') as log:
            try:
                run = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=350)
            except subprocess.TimeoutExpired:
                run = None
        result = dict(returncode=run.returncode if run else None,
                      official_score=score(audit), audit=str(audit), command=command)
        progress['results'][name] = result
        progress_path.write_text(json.dumps(progress, indent=2) + '\n')
        print(json.dumps(dict(number=number, name=name, **result)), flush=True)
        if run is None or run.returncode or not result['official_score']['available']:
            print(json.dumps(dict(stopped='infrastructure_error', name=name)), flush=True)
            break
    print(json.dumps(dict(completed=len(progress['results']),
                          full=sum(x['official_score'].get('tier3', 0) >= 75
                                   for x in progress['results'].values()))), flush=True)


if __name__ == '__main__':
    main()
