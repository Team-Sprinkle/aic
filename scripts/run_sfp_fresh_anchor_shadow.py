#!/usr/bin/env python3
"""Resume ten frozen, sequential rootless SFP teacher camera captures."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from run_sfp_cheatcode_target_pilot import BASE, ROOT, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--skip-existing', action='store_true')
    args = ap.parse_args()
    root = args.plan.parent
    plan = json.loads(args.plan.read_text())
    progress_path = root/'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        'schema':'sfp_fresh_anchor_shadow_progress/v1', 'results':{}}
    env = {**os.environ,
           'DOCKER_HOST':f'unix:///run/user/{os.getuid()}/docker.sock',
           'AIC_TRANSPORT_EVAL_GPU':'1', 'AIC_TRANSPORT_EVAL_POLICY_GPU':'1',
           'AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA':'.06',
           'AIC_TRANSPORT_EVAL_POLICY':'aic_example_policies.ros.CheatCode',
           'AIC_TRANSPORT_EVAL_TEACHER_PREFIX':'true',
           'AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC':'70',
           'AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC':'200'}
    for spec in plan['scenes']:
        name = spec['name']
        audit = root/f'teacher_{name}'
        if name in progress['results']:
            audit = root/f'teacher_{name}'
            if not (audit/'scored_stream.json').exists():
                raise RuntimeError(f'Existing rollout needs host bag extraction first: {name}')
            continue
        if audit.exists():
            if not args.skip_existing or not (audit/'results/scoring.yaml').exists():
                raise FileExistsError(f'Existing audit needs inspection: {audit}')
            rc = 0
            cmd = ['already completed externally']
        else:
            key_row = next((line for line in Path('/proc/key-users').read_text().splitlines()
                            if line.lstrip().startswith(f'{os.getuid()}:')), None)
            if key_row is not None:
                # /proc/key-users: UID usage nkeys/nikeys qnkeys/maxkeys
                # qnbytes/maxbytes. The fourth field is the quota, not the
                # preceding instantiated-key count.
                count, limit = key_row.split()[3].split('/')
                if int(count) >= int(limit):
                    raise RuntimeError(f'Rootless Docker UID key quota full ({count}/{limit}); '
                                       'no new Gazebo container was launched')
            env['AIC_TRANSPORT_EVAL_SCENE_DIR'] = spec['scene']
            cmd = ['bash', 'scripts/run_clean_transport_eval.sh', name, str(BASE), str(audit)]
            log = root/f'{name}.launch.log'
            with log.open('w') as f:
                try:
                    result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f,
                                            stderr=subprocess.STDOUT, timeout=360)
                    rc = result.returncode
                except subprocess.TimeoutExpired:
                    # The launcher owns cleanup; a timeout requires explicit
                    # inspection before any next launch.
                    raise RuntimeError(f'Gazebo launch timed out: {name}; inspect container')
        record = dict(scene=spec, audit=str(audit), command=cmd,
                      returncode=rc, official_score=score(audit))
        progress['results'][name] = record
        progress_path.write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps(dict(name=name, score=record['official_score'], rc=rc)),flush=True)
        if rc != 0:
            raise RuntimeError(f'Failed Gazebo capture {name}; inspect saved log')
        if not (audit/'scored_stream.json').exists():
            extract = [str(ROOT/'.pixi/envs/default/bin/python'),
                       'scripts/extract_moveit_training_stream.py',str(audit),
                       str(audit/'scored_stream.json')]
            host_env = {**env,
                        'PYTHONPATH':'/var/tmp/chmin_aic_mcap_reader_py312',
                        'LD_LIBRARY_PATH':str(ROOT/'.pixi/envs/default/lib')+':'
                                          +os.environ.get('LD_LIBRARY_PATH','')}
            with (root/f'{name}.extract.log').open('w') as f:
                subprocess.run(extract,cwd=ROOT,env=host_env,stdout=f,stderr=subprocess.STDOUT,check=True)
            stream = json.loads((audit/'scored_stream.json').read_text())
            print(json.dumps(dict(name=name, cameras={k:len(v) for k,v in stream['cameras'].items()},
                                  commands=len(stream['commands']))),flush=True)
    print(json.dumps(dict(completed=len(progress['results']), total=len(plan['scenes']))),flush=True)


if __name__ == '__main__':
    main()
