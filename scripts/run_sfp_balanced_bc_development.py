#!/usr/bin/env python3
"""Run the frozen 3-scene SFP development comparison sequentially."""
import argparse
import json
import os
from pathlib import Path
import subprocess


REQUESTS = ((0, 0), (2, 0), (4, 1))
VARIANTS = {
    'baseline': 'outputs/experiments/2026-10-01_transport_five_way/long_history/best.pt',
    'ordinary': '/var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/ordinary_bc_best_v1.pt',
    'zero_previous': '/var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/zero_previous_bc_best_v1.pt',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=1)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((a.collection / 'progress.json').read_text())['results']
    selected = {}
    for row in rows.values():
        key = (row['card'], row['port'])
        if key in REQUESTS:
            if key in selected:
                raise ValueError(f'Duplicate development request: {key}')
            selected[key] = row
    if set(selected) != set(REQUESTS):
        raise ValueError(f'Missing development requests: {set(REQUESTS) - set(selected)}')
    progress_path = a.output / 'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    base = Path('outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt').resolve()
    resnet = Path('/var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth')
    for card, port in REQUESTS:
        source = selected[card, port]
        for variant, ckpt in VARIANTS.items():
            key = f'card{card}_port{port}_{variant}'
            if key in progress:
                continue
            out = a.output / key
            if out.exists():
                raise ValueError(f'Existing untracked rollout: {out}')
            env = os.environ.copy()
            env.update(AIC_TRANSPORT_EVAL_GPU=str(a.gpu),
                       AIC_TRANSPORT_EVAL_POLICY_GPU=str(a.gpu),
                       AIC_TRANSPORT_EVAL_SCENE_DIR=source['scene'],
                       AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.RunTemporalTransport',
                       AIC_TEMPORAL_TRANSPORT_CHECKPOINT=str(Path(ckpt).resolve()),
                       AIC_TEMPORAL_TRANSPORT_IMAGENET=str(resnet),
                       AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
                       AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='70',
                       AIC_SFP_DIAGNOSTIC_STOP='0')
            command = ['bash', 'scripts/run_clean_transport_eval.sh',
                       Path(source['scene']).name, str(base), str(out)]
            print(json.dumps(dict(key=key, command=command, scene=source['scene'])), flush=True)
            with (a.output / f'{key}.launcher.log').open('w') as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=650, check=False)
            progress[key] = dict(variant=variant, card=card, port=port,
                                 scene=source['scene'], teacher=source['audit'],
                                 checkpoint=str(Path(ckpt).resolve()), output=str(out),
                                 command=command, returncode=result.returncode,
                                 score_exists=(out/'results/scoring.yaml').exists())
            progress_path.write_text(json.dumps(progress, indent=2) + '\n')
            print(json.dumps(dict(key=key, returncode=result.returncode,
                                  score_exists=progress[key]['score_exists'])), flush=True)
            if result.returncode or not progress[key]['score_exists']:
                raise RuntimeError(f'Rollout failed: {key}; inspect {out}')


if __name__ == '__main__':
    main()
