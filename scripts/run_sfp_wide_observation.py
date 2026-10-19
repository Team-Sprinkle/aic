#!/usr/bin/env python3
"""Run three predeclared nonfinal SFP wide-view diagnostics sequentially."""
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1')
OUTPUT = Path('/home/nas/chmin/aic_archives/2026-10-04_sfp_wide_observation_v1')
CHECKPOINT = ROOT / 'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt'
CASES = ('card2_port0', 'card3_port0', 'card4_port1')


def key_quota():
    line = next((line for line in Path('/proc/key-users').read_text().splitlines()
                 if line.lstrip().startswith(f'{os.getuid()}:')), None)
    if line:
        count, limit = line.split()[3].split('/')
        if int(count) >= int(limit):
            raise RuntimeError(f'Rootless key quota exhausted: {count}/{limit}')


def main():
    plan = {row['name']: row for row in json.loads((SOURCE / 'plan.json').read_text())['scenes']}
    pilot = OUTPUT / 'card1_port0_v2' / 'wide_view_events.json'
    if not pilot.exists():
        raise RuntimeError('Safe pilot record missing')
    events = json.loads(pilot.read_text())['events']
    stages = {row['stage'] for row in events}
    if 'wide_view_hold' not in stages or 'return_z_settling' not in stages or 'unsafe_abort' in stages:
        raise RuntimeError('Safe pilot return gate did not pass')
    manifest = {'schema': 'aic_sfp_wide_view_fixed_plan/v1', 'cases': list(CASES),
                'source_scene_hashes': {name: plan[name]['reset_sha256'] for name in CASES},
                'pilot': str(pilot), 'checkpoint': str(CHECKPOINT),
                'policy_sha256': hashlib.sha256((ROOT / 'aic_example_policies/aic_example_policies/ros/CheatCodeWideViewDiagnostic.py').read_bytes()).hexdigest(),
                'gpu': 1, 'rootless': True, 'final_scenes_opened': False}
    path = OUTPUT / 'remaining_plan.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError('Frozen plan/source changed')
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    progress_path = OUTPUT / 'remaining_progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'results': {}}
    env = {**os.environ, 'DOCKER_HOST': f'unix:///run/user/{os.getuid()}/docker.sock',
           'AIC_TRANSPORT_EVAL_GPU': '1', 'AIC_TRANSPORT_EVAL_POLICY_GPU': '1',
           'AIC_TRANSPORT_EVAL_POLICY': 'aic_example_policies.ros.CheatCodeWideViewDiagnostic',
           'AIC_TRANSPORT_EVAL_TEACHER_PREFIX': 'true',
           'AIC_TRANSPORT_EVAL_ORACLE_CONTACTS': 'true',
           'AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA': '.06',
           'AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC': '90',
           'AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC': '250'}
    for name in CASES:
        audit = OUTPUT / name
        if name in progress['results']:
            if not (audit / 'wide_view_events.json').exists():
                raise RuntimeError(f'Missing completed events for {name}')
            continue
        if audit.exists():
            raise FileExistsError(audit)
        key_quota()
        env['AIC_TRANSPORT_EVAL_SCENE_DIR'] = plan[name]['scene']
        command = ['bash', 'scripts/run_clean_transport_eval.sh', f'wide_observation_{name}_v1',
                   str(CHECKPOINT), str(audit)]
        with (OUTPUT / f'{name}.launch.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, timeout=360, check=True)
        event_file = audit / 'wide_view_events.json'
        data = json.loads(event_file.read_text()) if event_file.exists() else {'events': []}
        stages = {r['stage'] for r in data['events']}
        row = {'scene_hash': plan[name]['reset_sha256'], 'audit': str(audit),
               'wide_hold': 'wide_view_hold' in stages,
               'returned': 'return_z_hold' in stages and 'unsafe_abort' not in stages,
               'unsafe_abort': 'unsafe_abort' in stages, 'command': command}
        progress['results'][name] = row
        progress_path.write_text(json.dumps(progress, indent=2) + '\n')
        print(json.dumps({'name': name, **row}), flush=True)


if __name__ == '__main__':
    main()
