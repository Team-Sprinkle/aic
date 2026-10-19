#!/usr/bin/env python3
"""Evaluate one frozen corrective BC candidate on the same three development resets."""
import argparse
import json
import os
from pathlib import Path
import subprocess


REQUESTS = ((0, 0), (2, 0), (4, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((args.collection / 'progress.json').read_text())['results']
    selected = {}
    for row in rows.values():
        key = (row['card'], row['port'])
        if key in REQUESTS:
            if key in selected:
                raise ValueError(f'Duplicate development request: {key}')
            selected[key] = row
    if set(selected) != set(REQUESTS):
        raise ValueError(f'Missing requests: {set(REQUESTS) - set(selected)}')
    progress_path = args.output / 'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    base = Path('outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt').resolve()
    resnet = Path('/var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth')
    for card, port in REQUESTS:
        key = f'card{card}_port{port}'
        if key in progress:
            continue
        source = selected[card, port]
        out = args.output / key
        if out.exists():
            raise ValueError(f'Existing untracked rollout: {out}')
        env = os.environ.copy()
        env.update(DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock',
                   AIC_TRANSPORT_EVAL_GPU=str(args.gpu),
                   AIC_TRANSPORT_EVAL_POLICY_GPU=str(args.gpu),
                   AIC_TRANSPORT_EVAL_SCENE_DIR=source['scene'],
                   AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.RunTemporalTransport',
                   AIC_TEMPORAL_TRANSPORT_CHECKPOINT=str(args.checkpoint.resolve()),
                   AIC_TEMPORAL_TRANSPORT_IMAGENET=str(resnet),
                   AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
                   AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='70',
                   AIC_SFP_DIAGNOSTIC_STOP='0')
        command = ['bash', 'scripts/run_clean_transport_eval.sh',
                   Path(source['scene']).name, str(base), str(out)]
        with (args.output / f'{key}.launcher.log').open('w') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=650, check=False)
        progress[key] = dict(card=card, port=port, scene=source['scene'],
                             teacher=source['audit'], checkpoint=str(args.checkpoint.resolve()),
                             output=str(out), command=command, returncode=result.returncode,
                             score_exists=(out / 'results/scoring.yaml').exists())
        progress_path.write_text(json.dumps(progress, indent=2) + '\n')
        print(json.dumps(dict(key=key, returncode=result.returncode,
                              score_exists=progress[key]['score_exists'])), flush=True)
        if result.returncode or not progress[key]['score_exists']:
            raise RuntimeError(f'Rollout failed: {key}; inspect {out}')


if __name__ == '__main__':
    main()
