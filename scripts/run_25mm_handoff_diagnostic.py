#!/usr/bin/env python3
"""Four frozen matched BC rollouts: independently test a25mm actor start.

No training, official-scene tuning, RL or silent edit to the150mm evaluation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from shared_recovery_eval_environment import environment, effective_environment


def save(p, d):
    p.write_text(json.dumps(d, indent=2)+'\n')


def run(command, log, repo, env=None, ros=False):
    if ros:
        prefix = ('source .pixi/envs/default/setup.bash\n'
                  'export PYTHONPATH=/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/host_reader_deps:"${PYTHONPATH:-}"\n')
        command = ['bash', '-c', prefix+shlex.join(list(map(str, command)))]
    with log.open('w') as f:
        subprocess.run(list(map(str, command)), cwd=repo, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    repo = Path.cwd()
    root = Path('/home/nas/chmin/aic_archives/2026-10-16_full_sample_insertion_evaluation_v1')
    if a.output.exists():
        raise FileExistsError('Preserve the existing diagnostic')
    a.output.mkdir(parents=True)
    plan = json.loads((root/'plan.json').read_text())
    selected = json.loads((root/'frozen/selection.json').read_text())
    spec = json.loads((root/'frozen/transport_spec.json').read_text())
    cases = [next(c for c in plan['scenarios'] if c['family'] == f and c['card_count'] == 1 and c['port'] == 0
                  and (f == 'sc' or c['requested_card'] == 0)) for f in ['sc', 'sfp']]
    sources = [repo/'aic_example_policies/aic_example_policies/ros'/n for n in [
        'RunSample25mmThenInsertion.py', 'RunSampleThenSharedInsertion.py', 'RunRuleThenSharedRecovery.py',
        'RunLockedPortLocal.py', 'RunSharedPortTCP.py']]+[Path(__file__)]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for path in sources:
        shutil.copyfile(path, a.output/path.name)
    save(a.output/'plan.json', dict(cases=cases, actors=selected['actors'], sources=hashes,
        hypothesis='A25mm visually/proprioceptively verified start removes failed150mm BC approach responsibility; compare both frozen actors.',
        controller='Public RGB lock and measured TCP;25mm nominal; no true geometry; approach force gates unchanged.',
        recovery='none to isolate initial BC; near-range hierarchical acquisition semantics require separate integration',
        stopping_rule='Four fixed development rollouts; no routine collection/training. If approach fails, diagnose its trace. If both actors remain erratic near25mm, audit model inputs/labels and train only a separately justified matched repair.',
        GPU=5, maximum_project_gpus=4, budget='Four <=180s tasks plus startup/video; approximately20–45min.',
        final_scenes_opened=False, old_evaluation_modified=False))
    results = []
    for c in cases:
        for arm in ['fresh', 'visual_warm']:
            assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == value for path, value in hashes.items())
            name = c['case']+'__'+arm+'__25mm'
            audit = a.output/name
            casefile = a.output/(name+'.case.json')
            save(casefile, c)
            env = environment(repo, Path('/home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1'),
                spec['bundle'], c, policy='aic_example_policies.ros.RunSample25mmThenInsertion', domain='98',
                actor=Path(selected['actors'][arm]), recovery='none')
            aux = selected['auxiliary']
            env.update(AIC_TRANSPORT_EVAL_GPU='5', AIC_TRANSPORT_EVAL_POLICY_GPU='5',
                AIC_RULE_TRANSPORT_TEMPLATE=str(root/'frozen/transport_spec.json'),
                AIC_LOCKED_GRASP_CHECKPOINT=aux['sc_grasp'], AIC_SFP_GRASP_CHECKPOINT=aux['sfp_grasp'],
                AIC_SFP_GRASP_APPROVAL=aux['sfp_grasp_approval'])
            command = ['bash', repo/'scripts/run_shared_recovery_eval.sh', name,
                repo/'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt', audit]
            save(a.output/'current_job.json', dict(case=c, arm=arm, audit=str(audit)))
            save(a.output/(name+'.command.json'), dict(command=list(map(str, command)),
                environment={k: v for k, v in env.items() if k.startswith('AIC_')}))
            run(command, a.output/(name+'.log'), repo, env)
            effective_environment(audit, env)
            run([sys.executable, repo/'scripts/extract_moveit_training_stream.py', audit, audit/'scored_stream.json'],
                audit/'extraction.log', repo, ros=True)
            run([sys.executable, repo/'scripts/score_shared150mm_rollout.py', '--run', audit, '--case', casefile,
                 '--arm', arm+'__25mm'], audit/'scoring.log', repo, ros=True)
            row = json.loads((audit/'shared150mm_metrics.json').read_text())
            trace = [json.loads(l) for l in (audit/'runtime_trace.jsonl').open()]
            ready = [t for t in trace if t['event'] == 'near_handoff_ready']
            row['near_handoff_ready'] = bool(ready)
            row['near_handoff_failure'] = [t for t in trace if t['event'] == 'near_handoff_failed']
            if ready:
                run([sys.executable, repo/'scripts/audit_wide_approach_teacher_terminal.py', audit,
                     '--terminal-event', 'terminal_before_return', '--output', audit/'independent_terminal_audit.json'],
                    audit/'terminal_audit.log', repo, ros=True)
                # Use the same raw MCAP audit at the retained checkpoint;
                # this reads truth only after the entire rollout has ended.
                expression = ('from pathlib import Path;import json;from audit_wide_approach_teacher_terminal import audit;'
                    f'p=Path({str(audit)!r});d=audit(p,terminal_event="near_handoff_ready");'
                    '(p/"near_handoff_geometry.json").write_text(json.dumps(d,indent=2)+"\\n")')
                run([sys.executable, '-c', expression], audit/'checkpoint_audit.log', repo,
                    env=dict(os.environ, PYTHONPATH=str(repo/'scripts')+os.pathsep+os.environ.get('PYTHONPATH', '')), ros=True)
                row['near_handoff_geometry'] = json.loads((audit/'near_handoff_geometry.json').read_text())
            results.append(row)
            save(a.output/'progress.json', dict(completed=len(results), planned=4, results=results,
                final_scenes_opened=False, training=False))
            run([sys.executable, repo/'scripts/render_transport_five_camera_video.py', audit, '--fps', '20', '--grid', '--active-only'],
                audit/'video.log', repo)
    save(a.output/'summary.json', dict(completed=4, results=results, new_training=False,
        next='Compare actual checkpoint geometry and autonomous actor behavior. No automatic broad sweep or training.'))


if __name__ == '__main__':
    main()
