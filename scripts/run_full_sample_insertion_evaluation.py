#!/usr/bin/env python3
"""Finish transport analysis, then run a frozen all-scenario BC evaluation.

No training, tuning on released evaluation scenes, or reserved-final access.
Progress is resumable by completed per-run metrics; partial runs are preserved.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

from shared_recovery_eval_environment import environment, effective_environment


OLD = Path('/home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1')
POLICY = 'aic_example_policies.ros.RunSampleThenSharedInsertion'
START = '<!-- full-insertion-evaluation-live:start -->'
END = '<!-- full-insertion-evaluation-live:end -->'


def read(p):
    return json.loads(p.read_text()) if p.exists() else {}


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(p)


def digest(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def publish(repo, root, stage, **fields):
    state = dict(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(),
                 artifacts=str(root), training=False, RL=False, reserved_final_scenes_opened=False, **fields)
    save(root/'state.json', state)
    block = '\n'.join([START, '## Full insertion evaluation continuation', '',
        f'- Stage: `{stage}`. Artifacts: `{root}`.',
        '- Frozen rule transport → requested-port coarse/native refinement → shared 150 mm BC insertion actor.',
        '- Planned: 62 scenario rollouts (62 groups × one), plus three complete sample YAML runs and three complete released-evaluation YAML runs.',
        '- Hierarchical insertion recovery is rule-based; no new RL or training. Released evaluation is scoring only; reserved final scenes stay sealed.',
        '- Counts and scores: see `progress.json`, `results_by_scenario.json` and `configuration_scores.json` in the artifact folder.', END])
    for relative in ['docs/STATUS.md', 'docs/EXPERIMENTS.md', 'docs/WORLD_MODEL_NEXT_STEPS.md',
                     'outputs_README.md', 'docs/experiments/2026-10-16-full-pipeline-three-per-scenario.md']:
        p = repo/relative
        text = p.read_text() if p.exists() else ''
        if START in text:
            before, rest = text.split(START, 1)
            _, after = rest.split(END, 1)
            text = before + block + after
        else:
            text = block + '\n\n' + text
        p.write_text(text)


def run(command, log, repo, *, env=None, ros=False):
    if ros:
        prefix = ('source .pixi/envs/default/setup.bash\n'
                  'export PYTHONPATH=/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/host_reader_deps:"${PYTHONPATH:-}"\n')
        command = ['bash', '-c', prefix + shlex.join(list(map(str, command)))]
    with log.open('w') as f:
        subprocess.run(list(map(str, command)), cwd=repo, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


def analyze_transport(source, output):
    summary = read(source/'evaluation_runs/summary.json')
    rows = summary['results']
    groups = defaultdict(list)
    for row in rows:
        groups[row['domain'], row['family']].append(row)
    failures = []
    for row in rows:
        if row['qualified']:
            continue
        trace = [json.loads(line) for line in (Path(row['audit'])/'runtime_trace.jsonl').open()]
        failures.append(dict(case=row['case'], domain=row['domain'], family=row['family'], audit=row['audit'],
            rule_failure=row.get('rule_failure'), visual_reason=row.get('last_visual_reason'),
            acquired=row['rule_visual_acquisition'], wrong=row.get('wrong_accepted_port'),
            no_scored_harm=row['no_scored_harm'], terminal=row['terminal'],
            latency=row.get('inference_p50_p95_p99_ms'),
            recovery_events=[t for t in trace if t['event'] in (
                'transport_route_backtrack_requested', 'transport_measured_route_retreat_complete',
                'transport_retreat_abort', 'transport_recovery_budget_exhausted')]))
    report = dict(completed=len(rows), planned=130, qualified=sum(r['qualified'] for r in rows),
        groups=[dict(domain=k[0], family=k[1], episodes=len(v), qualified=sum(r['qualified'] for r in v),
                     wrong=sum(r.get('wrong_accepted_port') is True for r in v),
                     harm=sum(not r['no_scored_harm'] for r in v)) for k, v in groups.items()],
        failures=failures, insertion_evaluated=False, policy_changed=False,
        decision='Proceed to user-approved full insertion measurement; preserve acquisition failures in denominator.',
        limitation='Transport and conditional perception accuracy do not establish autonomous insertion. Trace analysis, not a claim of visually reviewed videos.')
    save(output/'transport_analysis.json', report)
    lines = ['# Completed frozen transport analysis', '', f"Qualified: {report['qualified']}/{len(rows)}. No insertion actor ran.", '',
             '| Domain | Task | Episodes | Qualified | Wrong | Harm |', '| --- | --- | --- | --- | --- | --- |']
    for g in report['groups']:
        lines.append('| ' + ' | '.join(str(g[k]) for k in ['domain', 'family', 'episodes', 'qualified', 'wrong', 'harm']) + ' |')
    lines += ['', '## Failures retained', '']
    for f in failures:
        lines.append(f"- `{f['case']}`: rule={f['rule_failure']}, visual={f['visual_reason']}, acquired={f['acquired']}; `{f['audit']}`.")
    lines += ['', report['decision'], report['limitation']]
    (output/'transport_analysis.md').write_text('\n'.join(lines) + '\n')
    return report


def prepare(repo, root, transport):
    frozen = root/'frozen'
    frozen.mkdir()
    spec = read(transport/'frozen/transport_spec.json')
    models = frozen/'models'
    models.mkdir()
    # Snapshot perception and both BC arms before any new evaluation outcome.
    for key, path in list(spec['bundle'].items()):
        src = Path(path)
        dst = models/(key + src.suffix)
        shutil.copyfile(src, dst)
        spec['bundle'][key] = str(dst)
    actors = {}
    validation_rank = {}
    for arm in ['fresh', 'visual_warm']:
        folder = OLD/f'shared_insertion_{arm}_full_v1'
        summary = read(folder/'summary.json')
        assert summary['updates_completed'] == 30000
        actors[arm] = str(models/(arm + '.pt'))
        shutil.copyfile(folder/'best.pt', actors[arm])
        save(frozen/(arm + '_training_summary.json'), summary)
        validation_rank[arm] = float(np.mean([
            summary['selected_validation'][family]['first_target_mm']['p95'] +
            summary['selected_validation'][family]['first_orientation_deg']['p95']
            for family in ['sc', 'sfp']]))
    selected = min(validation_rank, key=validation_rank.get)
    # Selection is fixed from held-out teacher-state validation, not official
    # scores or the upcoming autonomous episodes. Lower is only a heuristic.
    auxiliary = {}
    for key, src in dict(sc_grasp=Path('/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/sc_grasp_translation_observation_v1.pt'),
                         sfp_grasp=OLD/'SFP_native_grasp_keypoints_v1/best.pt',
                         sfp_grasp_approval=OLD/'SFP_free_approach_grasp_approval_v1.json').items():
        dst = models/(key + src.suffix)
        shutil.copyfile(src, dst)
        auxiliary[key] = str(dst)
    save(frozen/'transport_spec.json', spec)
    save(frozen/'selection.json', dict(actors=actors, selected=selected, validation_rank=validation_rank,
        selection='Mean over families of held-out first-target p95 mm + orientation p95 degrees; fixed before autonomous evaluation; no optimality claim.',
        recovery='hierarchical', auxiliary=auxiliary))
    # Retain the existing sample-only generator with exactly two explicit
    # changes: a fresh seed and one replicate. No official scene constants.
    source = (repo/'scripts/prepare_sample_conservative_transport.py').read_text()
    for before, after in [('else 2026101673)', 'else 2026101691)'), ('for rep in range(2):', 'for rep in range(1):')]:
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    generator = frozen/'prepare_one_replicate.py'
    generator.write_text(source)
    bank = root/'scenario_bank'
    preparation_env = dict(os.environ)
    preparation_env['PYTHONPATH'] = str(repo/'scripts') + os.pathsep + preparation_env.get('PYTHONPATH', '')
    run([sys.executable, generator, '--phase', 'evaluation', '--output', bank], root/'prepare.log', repo, env=preparation_env)
    plan = read(bank/'plan.json')
    assert len(plan['cases']) == 62
    counts = defaultdict(int)
    for c in plan['cases']:
        counts[c['family'], c['card_count'], c['port'], c['requested_card']] += 1
    assert len(counts) == 62 and set(counts.values()) == {1}
    plan.update(planned=62, bundle=spec['bundle'], evaluation_after_gate='62 frozen full pipeline runs; not transport-only.',
                gate='Measurement of all cases; retain ordinary physical failures. No automatic promotion, retraining or RL.',
                rule_spec=str(frozen/'transport_spec.json'))
    save(bank/'plan.json', plan)
    # Exact full YAMLs are copied byte-for-byte, with original robot home and
    # all three trials in original order. Extra cameras record diagnostics.
    template = Path(plan['cases'][0]['scene'])
    exact = []
    for label, src, expected in [
        ('sample', repo/'configs/evaluation_protocol/qualification_public_predeadline/sample_config.yaml',
         '21f6eb9ac0d71f7dddac3657c67c41b5fa212e5aaa1fd68528b08ec219585c8d'),
        ('released_eval', Path('/home/nas/chmin/aic_archives/2026-10-16_official_transport_recheck_v1/official_eval_config.yaml'),
         '9f987c07174481474eec2c4fa21faf7d84dae4743a037b809ed7945efb2e5ef5')]:
        assert digest(src) == expected, label
        cfg = yaml.safe_load(src.read_text())
        robot = (repo/'aic_description/urdf/ur_gz.urdf.xacro').read_text()
        for key, value in cfg['robot']['home_joint_positions'].items():
            robot, n = re.subn(rf'({re.escape(key)}=)(-?[0-9]+\.[0-9]+)', rf'\g<1>{value:.7f}', robot, count=1)
            assert n == 1
        for rep in range(3):
            name = f'exact_{label}_full_r{rep}'
            scene = root/'configuration_scenes'/name
            scene.mkdir(parents=True)
            shutil.copyfile(src, scene/'eval_config.yaml')
            (scene/'ur_llb_start.xacro').write_text(robot)
            for filename in ['world_audit.sdf', 'bridge_audit.yaml']:
                shutil.copyfile(template/filename, scene/filename)
            exact.append(dict(case=name, scene=str(scene), domain=label, replicate=rep,
                              configuration_sha256=expected, expected_trials=list(cfg['trials'])))
    save(root/'plan.json', dict(schema='full_sample_pipeline_evaluation/v1', scenarios=plan['cases'], exact_configurations=exact,
        scenario_groups=62, episodes_per_scenario=1, scenario_episodes=62, exact_config_runs=6,
        exact_config_episodes=18, selected_actor=selected, recovery='hierarchical',
        official_settings_used_for='scoring only after freeze; no tuning', reserved_final_scenes_opened=False,
        budget='Sequential one GPU; approximately 5–10 hours, dependent on task durations, startup and video rendering. No training.'))
    # Preserve all potentially imported experiment sources, then verify these
    # exact files before every run. New unrelated files do not invalidate it.
    files = list((repo/'aic_example_policies/aic_example_policies').rglob('*.py'))
    files += list((repo/'scripts').glob('*.py')) + list((repo/'scripts').glob('*.sh'))
    hashes = {}
    for p in files:
        dest = frozen/'sources'/p.relative_to(repo)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        hashes[str(p)] = digest(p)
    model_hashes = {str(p): digest(p) for p in models.iterdir()}
    scene_hashes = {str(p): digest(p) for folder in [bank/'scenes', root/'configuration_scenes'] for p in folder.rglob('*') if p.is_file()}
    save(frozen/'manifest.json', dict(source_hashes=hashes, model_hashes=model_hashes, scene_hashes=scene_hashes,
        spec_sha256=digest(frozen/'transport_spec.json'), selected_actor=selected, no_training=True))


def summarize(root, results, configurations):
    groups = defaultdict(list)
    for r in results:
        if r['stage'] == 'scenario':
            groups[r['family'], r['card_count'], r['port'], r['requested_card']].append(r)
    cells = [dict(family=k[0], cards=k[1], port=k[2], requested_card=k[3], episodes=len(v),
                  qualified=sum(x['qualified'] for x in v), physical_full_insertions=sum(x['physical_full_insertion'] for x in v),
                  actor_ran=sum(x['actor_ran'] for x in v),
                  tier3_mean=float(np.mean([x['raw_terminal']['official_tier3'] for x in v]))) for k, v in sorted(groups.items(), key=str)]
    save(root/'results_by_scenario.json', dict(cells=cells, completed=sum(c['episodes'] for c in cells), planned=62,
        caveat='One episode per group is a coverage screen, not a high-reliability certificate.'))
    config_report = {}
    for label in ['sample', 'released_eval']:
        rows = [c for c in configurations if c['domain'] == label]
        config_report[label] = dict(completed=len(rows), planned=3, results=rows,
            total_score_mean=float(np.mean([c['scoring']['total'] for c in rows])) if rows else None,
            total_score_min=min([c['scoring']['total'] for c in rows], default=None),
            total_score_max=max([c['scoring']['total'] for c in rows], default=None),
            note='Unchanged full YAML run with original trial order/home/time limits; local instrumented Gazebo, not cloud leaderboard certification.')
    save(root/'configuration_scores.json', config_report)
    save(root/'progress.json', dict(scenario_completed=sum(c['episodes'] for c in cells), scenario_planned=62,
        integration_completed=sum(r['stage'] == 'integration' for r in results), results=results,
        complete_config_runs=len(configurations), config_runs_planned=6, reserved_final_scenes_opened=False))
    lines = ['# Full pipeline evaluation', '', 'All numbers below are measured rollouts, not teacher trajectories.', '',
             '| Task | Cards | Port | NIC target | Episodes | Strict insertion | Physical insertion | Actor ran | Mean tier 3 |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for c in cells:
        lines.append('| ' + ' | '.join(str(c[k]) for k in ['family', 'cards', 'port', 'requested_card', 'episodes', 'qualified', 'physical_full_insertions', 'actor_ran', 'tier3_mean']) + ' |')
    lines += ['', '## Exact complete YAML scores', '']
    for label, c in config_report.items():
        lines.append(f"- {label}: {c['completed']}/3 complete runs; mean official total {c['total_score_mean']}; min/max {c['total_score_min']}/{c['total_score_max']}.")
    lines += ['', 'Official tier 3 and total scores are reported separately from the stricter causal physical geometry / no-harm / latency gate.',
              'The released evaluation configuration is scoring only. No tuning or automatic promotion follows. Reserved final scenes remain sealed.']
    (root/'results.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='5')
    a = parser.parse_args()
    repo = Path.cwd()
    root = a.output
    root.mkdir(parents=True, exist_ok=True)
    # A second invocation must not duplicate a queued/live coordinator.
    import fcntl
    lock = (root/'coordinator.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    publish(repo, root, 'waiting_for_frozen_transport_completion_and_analysis')
    while not (a.transport_root/'decision.json').exists():
        time.sleep(30)
    if not (a.transport_root/'evaluation_runs/summary.json').exists():
        raise RuntimeError('Transport ended without complete evaluation; inspect preserved state before continuing')
    analysis = analyze_transport(a.transport_root, root)
    assert analysis['completed'] == 130
    if not (root/'frozen/manifest.json').exists():
        if (root/'frozen').exists():
            raise RuntimeError('Partial freeze preserved: diagnose before restarting')
        prepare(repo, root, a.transport_root)
    plan = read(root/'plan.json')
    manifest = read(root/'frozen/manifest.json')
    selection = read(root/'frozen/selection.json')
    spec = read(root/'frozen/transport_spec.json')
    progress = read(root/'progress.json')
    results = progress.get('results', [])
    configurations = [v for group in read(root/'configuration_scores.json').values() for v in group.get('results', [])]

    def verify():
        for section in ['source_hashes', 'model_hashes', 'scene_hashes']:
            for path, expected in manifest[section].items():
                if digest(path) != expected:
                    raise RuntimeError('Frozen input changed: ' + path)
        assert digest(root/'frozen/transport_spec.json') == manifest['spec_sha256']

    def launch(c, arm, recovery, stage, *, complete_config=False):
        name = c['case'] + '__' + arm + '__' + recovery
        if complete_config:
            if any(x['case'] == c['case'] for x in configurations):
                return
        elif any(x['run_key'] == name for x in results):
            return
        verify()
        audit = root/'runs'/name
        if audit.exists():
            raise RuntimeError('Partial run preserved; inspect before retrying: ' + str(audit))
        audit.parent.mkdir(exist_ok=True)
        save(root/'current_job.json', dict(case=c, arm=arm, recovery=recovery, stage=stage, audit=str(audit)))
        casefile = audit.with_suffix('.case.json')
        save(casefile, c)
        env = environment(repo, OLD, spec['bundle'], c, policy=POLICY, domain='97',
                          actor=Path(selection['actors'][arm]), recovery=recovery)
        aux = selection['auxiliary']
        env.update(AIC_TRANSPORT_EVAL_GPU=a.gpu, AIC_TRANSPORT_EVAL_POLICY_GPU=a.gpu,
            AIC_TRANSPORT_EVAL_REPO=str(repo), AIC_TRANSPORT_EVAL_LAUNCH_SNAPSHOT='1',
            AIC_RULE_TRANSPORT_TEMPLATE=str(root/'frozen/transport_spec.json'),
            AIC_LOCKED_GRASP_CHECKPOINT=aux['sc_grasp'], AIC_SFP_GRASP_CHECKPOINT=aux['sfp_grasp'],
            AIC_SFP_GRASP_APPROVAL=aux['sfp_grasp_approval'])
        checkpoint = repo/'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt'
        launcher = repo/'scripts/run_shared_recovery_eval.sh'
        if complete_config:
            # The legacy launcher bounds a whole container to450s. Three
            # unchanged180s tasks require a whole-container watchdog >540s.
            generated = subprocess.check_output(['bash', str(launcher), '--print-generated-launcher'], cwd=repo, text=True)
            assert generated.count('deadline=$((SECONDS+450))') == 1
            generated = generated.replace('deadline=$((SECONDS+450))', 'deadline=$((SECONDS+1100))')
            launcher = audit.with_suffix('.full_configuration_launcher.sh')
            launcher.write_text(generated)
        command = ['bash', launcher, c['case'], checkpoint, audit]
        save(audit.with_suffix('.command.json'), dict(command=list(map(str, command)),
            environment={k: v for k, v in env.items() if k.startswith('AIC_')},
            whole_container_watchdog_s=1100 if complete_config else 450, task_time_limit_changed=False))
        run(command, audit.with_suffix('.log'), repo, env=env)
        actual = effective_environment(audit, env)
        scoring = yaml.safe_load((audit/'results/scoring.yaml').read_text())
        if complete_config:
            if set(scoring) - {'total'} != set(c['expected_trials']):
                raise RuntimeError('Incomplete full configuration score; do not silently sum isolated trials')
            configs = dict(case=c['case'], domain=c['domain'], replicate=c['replicate'], audit=str(audit),
                configuration_sha256=digest(audit/'eval_config.yaml'), scoring=scoring,
                effective_runtime_settings=actual)
            assert configs['configuration_sha256'] == c['configuration_sha256']
            configurations.append(configs)
            save(audit/'full_configuration_metrics.json', configs)
        else:
            run([sys.executable, repo/'scripts/extract_moveit_training_stream.py', audit, audit/'scored_stream.json'],
                audit/'extraction.log', repo, ros=True)
            run([sys.executable, repo/'scripts/score_shared150mm_rollout.py', '--run', audit, '--case', casefile,
                 '--arm', arm + '__' + recovery], audit/'scoring.log', repo, ros=True)
            row = read(audit/'shared150mm_metrics.json')
            row.update(run_key=name, stage=stage, effective_runtime_settings=actual, official_scoring=scoring)
            results.append(row)
        summarize(root, results, configurations)
        publish(repo, root, stage, scenario_completed=sum(r['stage'] == 'scenario' for r in results),
                configuration_runs_completed=len(configurations))
        # Every scenario gets at least one smooth video; failed isolated
        # rollouts are all filmed. Full multi-trial scoring preserves raw bags
        # and terminal RGB, avoiding the single-trial video cutter on resets.
        if not complete_config and (stage == 'integration' or c['replicate'] == 0 or not row['qualified']):
            run([sys.executable, repo/'scripts/render_transport_five_camera_video.py', audit,
                 '--fps', '20', '--grid', '--active-only'], audit/'video.log', repo)

    # Four matched, implementation-only integration checks. Ordinary physical
    # failures count and do not cancel the requested full measurement.
    for family in ['sc', 'sfp']:
        c = next(c for c in plan['scenarios'] if c['family'] == family and c['card_count'] == 1 and c['port'] == 0)
        c = dict(c, case='integration_' + c['case'], domain='integration')
        for arm in ['fresh', 'visual_warm']:
            launch(c, arm, 'none', 'integration')
    for c in plan['scenarios']:
        launch(c, selection['selected'], 'hierarchical', 'scenario')
    # Official scenes are last and cannot affect model/rule selection.
    for c in plan['exact_configurations']:
        launch(c, selection['selected'], 'hierarchical', 'configuration_scoring', complete_config=True)
    save(root/'decision.json', dict(status='full_evaluation_complete_requires_review', scenario_episodes=62,
        complete_configuration_runs=6, insertion_evaluated=True, no_training=True, no_RL=True,
        no_official_tuning=True, reserved_final_scenes_opened=False))
    publish(repo, root, 'complete_requires_review', scenario_completed=62, configuration_runs_completed=6)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Preserve exact failure; never replace an implementation error with
        # a zero model score or skip to official scoring with broken runtime.
        args = sys.argv
        if '--output' in args:
            folder = Path(args[args.index('--output') + 1])
            if folder.exists():
                save(folder/'execution_error.json', dict(type=type(error).__name__, message=str(error),
                    requires_diagnosis=True, training_started=False))
        raise
