#!/usr/bin/env python3
"""Bounded training-reset actor-to-CheatCode full-insertion diagnostic.

The actor sees only normal observations. The privileged teacher takes over
after the measured actor deviation; these runs are not autonomous successes.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import yaml

from run_sfp_cheatcode_target_pilot import BASE, ROOT, score

SOURCE = Path('/home/nas/chmin/aic_archives/2026-10-07_sfp_divergent_pairs_v1')
RESNET = Path('/var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=7)
    p.add_argument('--resume', action='store_true',
                   help='Reconcile completed audits and run only missing/invalid scenes')
    p.add_argument('--reconcile-only', action='store_true',
                   help='Read saved audits without starting a container')
    a = p.parse_args()
    plan = json.loads((SOURCE/'plan.json').read_text())
    previous = json.loads((SOURCE/'progress.json').read_text())['results']
    if not plan['train_only'] or plan['final_scenes_opened']:
        p.error('Source must be training-only with sealed final scenes')
    selected = []
    for count in (2, 3, 5):
        choices = [s for s in plan['scenes'] if s['card_count'] == count and
                   previous.get(s['name'], {}).get('official_score', {}).get('tier3', 0) >= 75]
        if len(choices) < 2:
            p.error(f'Need two successful teacher source resets with {count} cards')
        selected.extend(choices[:2])
    if a.output.exists() and not (a.resume or a.reconcile_only):
        p.error('Preserve existing gate; pass --resume to continue it')
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema='aic_sfp_later_actor_takeover_gate/v1',
                    source_plan=str(SOURCE/'plan.json'),
                    selected=selected, max_runs=6, train_only=True,
                    actor_elapsed_sim_s=[6, 8], teacher_cap_sim_s=60,
                    privileged_teacher_only_after_takeover=True,
                    autonomous_success_claim=False, final_scenes_opened=False,
                    checkpoint=str(a.checkpoint.resolve()), gpu=a.gpu)
    manifest_path = a.output/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            p.error('Frozen gate manifest differs; cannot resume')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    rows = []
    invalid = []
    for index, spec in enumerate(selected):
        duration = 6 if index % 2 == 0 else 8
        original_name = f"{spec['name']}_actor{duration}s_teacher_full"
        name = original_name
        original = a.output/original_name
        original_score = score(original)
        original_event_path = original/'corrective_teacher_takeover.json'
        original_event = (json.loads(original_event_path.read_text())
                          if original_event_path.exists() else {})
        if (original_score.get('message') == 'Task not completed.' and
                original_event.get('teacher_outcome') == 'teacher_returned_full_insert' and
                (original/'launcher.exit').exists()):
            invalid.append(dict(audit=str(original), reason='diagnostic returned false; scorer had no task end time'))
            name += '_scorefix'
        audit = a.output/name
        env = os.environ.copy()
        env.update(DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock',
                   AIC_TRANSPORT_EVAL_GPU=str(a.gpu),
                   AIC_TRANSPORT_EVAL_POLICY_GPU=str(a.gpu),
                   AIC_TRANSPORT_EVAL_SCENE_DIR=spec['scene_dir'],
                   AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.RunTemporalThenCheatCodeDiagnostic',
                   AIC_TRANSPORT_EVAL_TEACHER_PREFIX='true',
                   AIC_TEMPORAL_TRANSPORT_CHECKPOINT=str(a.checkpoint.resolve()),
                   AIC_TEMPORAL_TRANSPORT_IMAGENET=str(RESNET),
                   AIC_SFP_CORRECTIVE_ACTOR_DURATION_S=str(duration),
                   AIC_SFP_CORRECTIVE_TEACHER_DURATION_S='60',
                   AIC_SFP_CORRECTIVE_REQUIRE_FULL_INSERT='1',
                   AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
                   AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='90',
                   AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC='300',
                   AIC_SFP_DIAGNOSTIC_STOP='0')
        command = ['bash', 'scripts/run_clean_transport_eval.sh', name,
                   str(BASE), str(audit)]
        completed = (audit/'results/scoring.yaml').exists() and (audit/'launcher.exit').exists()
        if not completed:
            if a.reconcile_only:
                continue
            if audit.exists():
                raise RuntimeError(f'Partial audit exists; inspect before retrying: {audit}')
            with (a.output/f'{name}.launch.log').open('w') as log:
                try:
                    run = subprocess.run(command, cwd=ROOT, env=env,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         timeout=500, check=False)
                    rc = run.returncode
                except subprocess.TimeoutExpired:
                    rc = None
        else:
            rc = 0 if (audit/'launcher.exit').read_text().strip() == 'finished' else None
        takeover = audit/'corrective_teacher_takeover.json'
        event = json.loads(takeover.read_text()) if takeover.exists() else {}
        official = score(audit)
        scoring_path = audit/'results/scoring.yaml'
        if scoring_path.exists():
            scoring = yaml.safe_load(scoring_path.read_text())
            trial = next((v for k, v in scoring.items() if k.startswith('trial_')), {})
            official['contact_score'] = float(trial.get('tier_2', {}).get('categories', {})
                                              .get('contacts', {}).get('score', 0))
        row = dict(name=name, card_count=spec['card_count'], scene_digest=spec['source_scene_hash'],
                   audit=str(audit), command=command, actor_elapsed_sim_s=duration,
                   returncode=rc, teacher_outcome=event.get('teacher_outcome'),
                   takeover_sim_time_s=event.get('takeover_sim_time_s'),
                   official_score=official,
                   fully_inserted=bool(rc == 0 and official.get('tier3', 0) >= 75),
                   safe_full_insert=bool(rc == 0 and official.get('tier3', 0) >= 75 and
                                         official.get('contact_score', 0) >= 0))
        rows.append(row)
        (a.output/'summary_reconciled.json').write_text(json.dumps(dict(
            schema='aic_sfp_later_actor_takeover_gate/v2', runs=rows,
            invalid_scoring_replays=invalid,
            completed=len(rows), safe_full_insertions=sum(r['safe_full_insert'] for r in rows),
            gate_passed=len(rows) == 6 and sum(r['safe_full_insert'] for r in rows) >= 4,
            final_scenes_opened=False), indent=2)+'\n')
        print(json.dumps(row), flush=True)
        if rc is None:
            raise RuntimeError(f'Timed out; inspect live container for {name}')


if __name__ == '__main__':
    main()
