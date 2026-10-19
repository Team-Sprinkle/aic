#!/usr/bin/env python3
"""Three-reset, fixed-time actor then privileged CheatCode transport gate."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import yaml


REQUESTS = ((0, 0), (2, 0), (4, 1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=1)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve previous/partial gate; choose a new output')
    a.output.mkdir(parents=True)
    sources = json.loads((a.development / 'progress.json').read_text())['results']
    selected = {(r['card'], r['port']): r for r in sources.values()
                if (r['card'], r['port']) in REQUESTS}
    if set(selected) != set(REQUESTS):
        raise ValueError(f'Missing requests: {set(REQUESTS) - set(selected)}')
    base = Path('outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt').resolve()
    progress = []
    for card, port in REQUESTS:
        source = selected[card, port]
        audit = a.output / f'card{card}_port{port}_actor2s_then_cheatcode'
        env = os.environ.copy()
        env.update(DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock',
                   AIC_TRANSPORT_EVAL_GPU=str(a.gpu),
                   AIC_TRANSPORT_EVAL_POLICY_GPU=str(a.gpu),
                   AIC_TRANSPORT_EVAL_SCENE_DIR=source['scene'],
                   AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.RunTemporalThenCheatCodeDiagnostic',
                   AIC_TRANSPORT_EVAL_TEACHER_PREFIX='true',
                   AIC_TEMPORAL_TRANSPORT_CHECKPOINT=str(a.checkpoint.resolve()),
                   AIC_TEMPORAL_TRANSPORT_IMAGENET='/var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth',
                   AIC_SFP_CORRECTIVE_ACTOR_DURATION_S='2',
                   AIC_SFP_CORRECTIVE_TEACHER_DURATION_S='30',
                   AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
                   AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='70',
                   AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC='300',
                   AIC_SFP_DIAGNOSTIC_STOP='0')
        command = ['bash', 'scripts/run_clean_transport_eval.sh',
                   Path(source['scene']).name, str(base), str(audit)]
        print(json.dumps(dict(card=card, port=port, command=command,
                              scene=source['scene'])), flush=True)
        with (a.output / f'card{card}_port{port}.launcher.log').open('w') as log:
            result = subprocess.run(command, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=650, check=False)
        event_path = audit / 'corrective_teacher_takeover.json'
        event = json.loads(event_path.read_text()) if event_path.is_file() else None
        buffer_path = audit / 'corrective_teacher_buffered_arrival.json'
        score_path = audit / 'results/scoring.yaml'
        contact = None
        if score_path.is_file():
            score = yaml.safe_load(score_path.read_text())
            trial = next(v for k, v in score.items() if k.startswith('trial_'))
            contact = float(trial.get('tier_2', {}).get('categories', {})
                            .get('contacts', {}).get('score', 0))
        row = dict(card=card, port=port, scene=source['scene'], audit=str(audit),
                   actor_checkpoint=str(a.checkpoint.resolve()),
                   command=command, returncode=result.returncode,
                   actor_prefix_completed=bool(event and event.get('actor_prefix_completed')),
                   teacher_outcome=event.get('teacher_outcome') if event else None,
                   safe_buffered_arrival=buffer_path.is_file(),
                   official_contact_score=contact,
                   passed=bool(result.returncode == 0 and buffer_path.is_file() and
                               contact is not None and contact >= 0))
        progress.append(row)
        (a.output / 'summary.json').write_text(json.dumps(dict(
            schema='aic_sfp_actor_state_corrective_teacher_gate/v1',
            planned_requests=[list(k) for k in REQUESTS],
            max_takeovers=3, actor_prefix_s=2, teacher_time_cap_s=30,
            runs=progress, passes=sum(r['passed'] for r in progress),
            gate_passed=(len(progress)==3 and sum(r['passed'] for r in progress)>=2),
            privileged_teacher_only_after_takeover=True,
            autonomous_success_claim=False, reserved_final_opened=False),indent=2)+'\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
