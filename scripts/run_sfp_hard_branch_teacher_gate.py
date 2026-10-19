#!/usr/bin/env python3
"""Collect at most six independent hard-card paired CheatCode controls."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from collect_sfp_diverse_teacher import SOURCE_ROOT, PATTERNS, config, digest, board_count
from run_sfp_cheatcode_target_pilot import TEMPLATE
from run_sfp_paired_actor_live_three import extract_teacher, run_case


def teacher_score(root):
    path = root / 'results/scoring.yaml'
    if not path.exists():
        return {'available': False}
    scored = yaml.safe_load(path.read_text())
    trial = next(v for k, v in scored.items() if k.startswith('trial_'))
    return {'available': True, 'tier3': float(trial.get('tier_3', {}).get('score', 0)),
            'contact': float(trial.get('tier_2', {}).get('categories', {}).get('contacts', {}).get('score', 0))}


def build_plan(labels, previous_plan, output):
    blocked = {json.loads(s)['scene_sha256'] for s in labels.open()}
    previous = json.loads(previous_plan.read_text())
    if previous.get('final_scenes_opened'):
        raise ValueError('Previous plan opened final scenes')
    blocked.update(x['scene_digest'] for x in previous['candidate_scenes'])
    pool = {3: [], 4: [], 5: []}
    for pattern in PATTERNS:
        for path in SOURCE_ROOT.glob(pattern):
            cfg = config(str(path))
            trial = next(iter(cfg['trials'].values()))
            board = trial['scene']['task_board']
            count = board_count(board)
            if count not in pool:
                continue
            task = next(iter(trial['tasks'].values()))
            card = int(task['target_module_name'].rsplit('_', 1)[-1])
            port = int(task['port_name'].rsplit('_', 1)[-1])
            occupied = [i for i in range(5) if board[f'nic_rail_{i}']['entity_present']]
            if card not in occupied:
                continue
            other = max((i for i in occupied if i != card), key=lambda i: abs(i - card))
            if abs(card - other) < 2:
                continue
            scene = digest({'scene': trial['scene'], 'robot': cfg.get('robot')})
            if scene in blocked:
                continue
            pool[count].append((path, scene, card, port, other))
    selected = []
    used = set()
    for count in (3, 4, 5):
        for path, scene, card, port, other in sorted(
                pool[count], key=lambda row: hashlib.sha256(str(row[0]).encode()).hexdigest()):
            if scene in used:
                continue
            used.add(scene)
            selected.append((count, path, scene, card, port, other))
            if sum(x[0] == count for x in selected) == 2:
                break
        if sum(x[0] == count for x in selected) != 2:
            raise ValueError(f'Not enough fresh {count}-card source scenes')
    specs = []
    for count, path, scene, card, port, other in selected:
        name = f'cards{count}_card{card}p{port}_to_card{other}p{1-port}_{path.parent.parent.name}_{path.stem}'
        variants = []
        for tag, selected_card, selected_port in (('original', card, port), ('alternate', other, 1-port)):
            cfg = config(str(path)).copy()
            # Reload to avoid mutating the cached source config.
            cfg = yaml.safe_load(Path(path).read_text())
            trial = next(iter(cfg['trials'].values()))
            task = next(iter(trial['tasks'].values()))
            task['target_module_name'] = f'nic_card_mount_{selected_card}'
            task['port_name'] = f'sfp_port_{selected_port}'
            if digest({'scene': trial['scene'], 'robot': cfg.get('robot')}) != scene:
                raise AssertionError('Task edit altered scene/reset')
            folder = output / 'scenes' / name / tag
            folder.mkdir(parents=True)
            (folder / 'eval_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
            for filename in ('world_audit.sdf', 'bridge_audit.yaml'):
                shutil.copy2(TEMPLATE / filename, folder / filename)
            variants.append({'tag': tag, 'request': [selected_card, selected_port],
                             'scene_dir': str(folder),
                             'config_sha256': hashlib.sha256((folder / 'eval_config.yaml').read_bytes()).hexdigest()})
        specs.append({'name': name, 'card_count': count, 'scene_digest': scene,
                      'source_trial': str(path), 'variants': variants})
    return {'schema': 'aic_sfp_hard_branch_teacher_gate/v1', 'pairs': specs,
            'maximum_pairs': 6, 'maximum_rollouts': 12,
            'gate': 'at least 4 pairs both tier3>=75, contact>=0, causal early one-second motion difference>=10mm',
            'train_labels_admitted': False, 'final_scenes_opened': False}


def pose_at(stream, t):
    i = min(range(len(stream)), key=lambda k: abs(stream[k][0] / 1e9 - t))
    stamp, item = stream[i]
    if abs(stamp / 1e9 - t) > .11:
        return None
    pose = item['tcp_pose']
    xyz = np.asarray([pose['position'][k] for k in ('x', 'y', 'z')])
    quat = np.asarray([pose['orientation'][k] for k in ('x', 'y', 'z', 'w')])
    return xyz, quat


def causal_branch(a, b):
    left = json.loads((a / 'scored_stream.json').read_text())
    right = json.loads((b / 'scored_stream.json').read_text())
    lc = left['streams']['/aic_controller/controller_state']
    rc = right['streams']['/aic_controller/controller_state']
    start = max(min(c['sim_ns'] for c in left['commands'] if c['kind'] == 'pose'),
                min(c['sim_ns'] for c in right['commands'] if c['kind'] == 'pose')) / 1e9
    matches = []
    for dt in np.arange(.0, 1.61, .2):
        t = start + dt
        p, q = pose_at(lc, t), pose_at(rc, t)
        pf, qf = pose_at(lc, t + 1), pose_at(rc, t + 1)
        if p is None or q is None or pf is None or qf is None:
            continue
        gap = float(np.linalg.norm(p[0] - q[0]) * 1000)
        angle = float((Rotation.from_quat(p[1]).inv() * Rotation.from_quat(q[1])).magnitude() * 180 / np.pi)
        future = float(np.linalg.norm((pf[0] - p[0]) - (qf[0] - q[0])) * 1000)
        matches.append({'relative_time_s': float(dt), 'tcp_gap_mm': gap,
                        'orientation_gap_deg': angle, 'future_route_difference_mm': future,
                        'same_observation': bool(gap <= 2 and angle <= 2)})
    valid = [x for x in matches if x['same_observation']]
    return {'matches': matches, 'best_same_observation': max(valid, key=lambda x: x['future_route_difference_mm'])
            if valid else None,
            'passes_10mm': bool(valid and max(x['future_route_difference_mm'] for x in valid) >= 10)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--previous-live-plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', default='7')
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output / 'plan.json'
    if path.exists():
        plan = json.loads(path.read_text())
    else:
        plan = build_plan(a.labels, a.previous_live_plan, a.output)
        path.write_text(json.dumps(plan, indent=2) + '\n')
    if plan['schema'] != 'aic_sfp_hard_branch_teacher_gate/v1' or plan['final_scenes_opened']:
        p.error('Invalid frozen plan')
    print(json.dumps({'pairs': len(plan['pairs']), 'card_counts': [x['card_count'] for x in plan['pairs']]}), flush=True)
    if a.plan_only:
        return
    progress_path = a.output / 'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'results': {}}
    for spec in plan['pairs']:
        name = spec['name']
        if name not in progress['results']:
            progress['results'][name] = {'variants': {}}
        result = progress['results'][name]
        for variant in spec['variants']:
            tag = variant['tag']
            if tag in result['variants']:
                continue
            scene = Path(variant['scene_dir'])
            if hashlib.sha256((scene / 'eval_config.yaml').read_bytes()).hexdigest() != variant['config_sha256']:
                raise ValueError('Frozen scene config changed')
            run_name = name + '_' + tag
            launched = run_case(a.output, run_name, scene, 'aic_example_policies.ros.CheatCode', None, a.gpu)
            audit = Path(launched['output'])
            scored = teacher_score(audit)
            result['variants'][tag] = {'launch': launched, 'score': scored}
            progress_path.write_text(json.dumps(progress, indent=2) + '\n')
            print(json.dumps({'pair': name, 'variant': tag, 'score': scored}), flush=True)
            if launched['returncode'] or not scored['available']:
                raise RuntimeError(f'Teacher launch failed: {name}/{tag}')
            extract_teacher(audit)
        if 'causal_branch' not in result:
            runs = result['variants']
            a_dir = Path(runs['original']['launch']['output'])
            b_dir = Path(runs['alternate']['launch']['output'])
            result['causal_branch'] = causal_branch(a_dir, b_dir)
            result['gate_passed'] = bool(result['causal_branch']['passes_10mm'] and
                all(runs[tag]['score']['tier3'] >= 75 and runs[tag]['score']['contact'] >= 0
                    for tag in ('original', 'alternate')))
            progress_path.write_text(json.dumps(progress, indent=2) + '\n')
            print(json.dumps({'pair': name, 'gate_passed': result['gate_passed'],
                              'branch': result['causal_branch']['best_same_observation']}), flush=True)
    passed = sum(bool(x.get('gate_passed')) for x in progress['results'].values())
    summary = {'schema': 'aic_sfp_hard_branch_teacher_gate_result/v1',
               'pairs_completed': len(progress['results']), 'pairs_passed': passed,
               'gate_passed': passed >= 4, 'maximum_pairs': 6,
               'final_scenes_opened': False, 'progress': str(progress_path)}
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
