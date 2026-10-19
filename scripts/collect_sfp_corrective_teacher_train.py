#!/usr/bin/env python3
"""Bounded balanced actor-state -> CheatCode training controls, sequential Gazebo."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess

import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--balanced-train', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, choices=(10, 20), required=True)
    p.add_argument('--gpu', type=int, default=1)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    source_plan = json.loads((a.balanced_train / 'plan_per_target_5.json').read_text())
    source_progress = json.loads((a.balanced_train / 'progress.json').read_text())['results']
    by_target = defaultdict(list)
    for row in source_plan['scene_trials']:
        by_target[row['card'], row['port']].append(row)
    if len(by_target) != 10 or any(len(v) != 5 for v in by_target.values()):
        raise ValueError('Expected five prevalidated train scenes per target')
    selected = [by_target[key][repeat] for repeat in range(2)
                for key in sorted(by_target)]
    plan_path = a.output / 'plan20.json'
    planned = dict(schema='aic_sfp_corrective_teacher_train_plan/v1',
                   selected=selected, first_tranche=10, cap=20,
                   first_tranche_gate='at least 8/10 safe buffered arrivals without scored contact and valid causal labels',
                   source_split='train', reserved_final_opened=False)
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != planned:
            raise ValueError('Frozen plan changed')
    else:
        plan_path.write_text(json.dumps(planned, indent=2) + '\n')
    progress_path = a.output / 'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    base = Path('outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt').resolve()
    for row in selected[:a.limit]:
        tid = Path(row['source_trial']).stem
        if tid in progress:
            continue
        source = source_progress[tid]
        audit = a.output / f'card{row["card"]}_port{row["port"]}_{tid}'
        # Earlier feasibility-pilot controls predate the separate `scenes/`
        # field. Their immutable audit contains the same three scene files.
        scene = Path(source.get('scene') or source['audit']).resolve()
        env = os.environ.copy()
        env.update(DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock',
                   AIC_TRANSPORT_EVAL_GPU=str(a.gpu), AIC_TRANSPORT_EVAL_POLICY_GPU=str(a.gpu),
                   AIC_TRANSPORT_EVAL_SCENE_DIR=str(scene),
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
        command = ['bash','scripts/run_clean_transport_eval.sh',scene.name,str(base),str(audit)]
        print(json.dumps(dict(trial=tid, card=row['card'], port=row['port'],
                              scene=str(scene), command=command)),flush=True)
        with (a.output / f'{tid}.launcher.log').open('w') as log:
            try:
                result = subprocess.run(command, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=650, check=False)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = None
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
        accepted = bool(code == 0 and event and event.get('actor_prefix_completed') and
                        event.get('teacher_outcome') == 'safe_buffered_arrival' and
                        buffer_path.is_file() and contact is not None and contact >= 0)
        progress[tid] = dict(trial=tid, card=row['card'], port=row['port'],
                             source_scene=str(scene), audit=str(audit),
                             checkpoint=str(a.checkpoint.resolve()), command=command,
                             returncode=code, teacher_outcome=event.get('teacher_outcome') if event else None,
                             buffered=buffer_path.is_file(), contact_score=contact,
                             accepted=accepted)
        progress_path.write_text(json.dumps(progress, indent=2) + '\n')
        print(json.dumps(progress[tid]),flush=True)
    first = [progress.get(Path(row['source_trial']).stem) for row in selected[:10]]
    summary = dict(schema='aic_sfp_corrective_teacher_train_collection/v1',
                   completed=len(progress),accepted=sum(v['accepted'] for v in progress.values()),
                   first_tranche_complete=all(first),
                   first_tranche_accepted=sum(bool(v and v['accepted']) for v in first),
                   first_tranche_gate=all(first) and sum(bool(v and v['accepted']) for v in first)>=8,
                   cap=20,actor_prefix_s=2,teacher_cap_s=30,
                   privileged_after_takeover=True, autonomous_success_claim=False,
                   reserved_final_opened=False)
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
