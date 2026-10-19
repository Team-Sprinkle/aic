#!/usr/bin/env python3
"""Run frozen transport/acquisition only; preserve failures and grouped counts."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
from acquisition_gate import qualified_acquisition
from aggregate_frozen_pipeline_coverage import interval
from score_frozen_pipeline_coverage import score
from shared_recovery_eval_environment import environment, effective_environment


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prior-root', type=Path, required=True)
    a = p.parse_args(); repo = Path.cwd()
    plan = json.loads((a.output/'plan.json').read_text())
    previous = a.output/'progress.json'
    results = json.loads(previous.read_text())['results'] if previous.exists() else []
    completed = {r['case'] for r in results}

    def report(final=False):
        groups = defaultdict(list)
        for r in results:
            key = '/'.join(map(str, [r['domain'], r['family'], r['arm'], r['card_count'], r['port']]))
            groups[key].append(r)
        cells = {}
        for key, rows in groups.items():
            valid = [r for r in rows if not r.get('infrastructure_error')]
            k = sum(r['qualified'] for r in valid)
            cells[key] = dict(attempts=len(rows), valid=len(valid), qualified=k,
                correct_visual_acquisition=sum(bool(r.get('rule_visual_acquisition')) and r.get('wrong_accepted_port') is False for r in valid),
                wrong_accepts=sum(r.get('wrong_accepted_port') is True for r in valid),
                scored_harm=sum(r.get('no_scored_harm') is not True for r in valid),
                qualified_rate_Wilson95=interval(k,len(valid)),
                failures=dict(Counter(r.get('rule_failure') or r.get('last_visual_reason') or 'latency_or_geometry_gate' for r in valid if not r['qualified'])))
        out = dict(completed=len(results),planned=plan['planned'],results=results,groups=cells,
            insertion_actor_enabled=False, reserved_final_scenes_opened=False,
            complete=final and len(results)==plan['planned'], scope=plan['exact_scope'],stress_scope=plan['stress_scope'])
        write(a.output/('summary.json' if final else 'progress.json'),out)
        lines = ['# Official transport recheck',f'\nCompleted {len(results)}/{plan["planned"]}; transport/acquisition only, no insertion.\n',
                 '| Group: domain/family/arm/cards/port | Valid | Correct acquisition | Strict pass | Wrong | Harm |',
                 '| --- | --- | --- | --- | --- | --- |']
        for key, v in sorted(cells.items()):
            lines.append(f'| {key} | {v["valid"]} | {v["correct_visual_acquisition"]} | {v["qualified"]} | {v["wrong_accepts"]} | {v["scored_harm"]} |')
        lines += ['\nExact cases preserve each public trial but isolate it into its own launch. Range stress changes card count, target and rails and is not exact official evaluation.',
                  '\nFive repetitions have wide uncertainty. No inference about full insertion or later hidden settings is justified.']
        (a.output/'progress.md').write_text('\n'.join(lines)+'\n')

    for case in plan['cases']:
        if case['case'] in completed:
            continue
        for file, expected in {**plan['runtime_hashes'],**plan['frozen_checkpoint_hashes']}.items():
            if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=expected:
                write(a.output/'stopped.json',dict(reason='frozen_source_or_model_changed',file=file));return
        bundle = dict(plan['bundle'])
        if case['arm']=='baseline_SC_refiner':
            bundle['sc_native_refiner'] = plan['baseline_sc_checkpoint']
        policy = 'aic_example_policies.ros.'+('RunNative5RequestedRuleTransport' if case['family']=='sc' else 'RunSfpRequestedRuleTransport')
        env = environment(repo,a.prior_root,bundle,case,policy=policy,domain='90')
        # Passive read-only contact reporter retained; no privileged stream is subscribed by policy.
        run = a.output/'runs'/case['case'];run.parent.mkdir(exist_ok=True)
        write(a.output/'current_job.json',dict(case=case,started_unix=time.time(),run=str(run)))
        cmd = ['bash',str(repo/'scripts/run_shared_recovery_eval.sh'),case['case'],
               str(repo/'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt'),str(run)]
        write(run.with_suffix('.command.json'),dict(command=cmd,environment={k:v for k,v in env.items() if k.startswith('AIC_')}))
        try:
            with run.with_suffix('.log').open('w') as f:
                subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            actual = effective_environment(run,env)
            cmd = [sys.executable,str(repo/'scripts/extract_moveit_training_stream.py'),str(run),str(run/'scored_stream.json')]
            prefix = 'source .pixi/envs/default/setup.bash\nexport PYTHONPATH=/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/host_reader_deps:"${PYTHONPATH:-}"\n'
            with (run/'extraction.log').open('w') as f:
                subprocess.run(['bash','-c',prefix+shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT,check=True)
            row = score(run,case,case['arm'])
            trace = [json.loads(line) for line in (run/'runtime_trace.jsonl').open()]
            active = {r['sim_time'] for r in trace if r['event'] in ('rule_visual_inspection','SFP_rule_visual_inspection')}
            latency = [r['inference_ms'] for r in trace if r['event']=='proposal' and r['sim_time'] in active]
            row['all_proposals_p50_p95_p99_ms'] = row['inference_p50_p95_p99_ms']
            row['inference_p50_p95_p99_ms'] = np.percentile(latency,[50,95,99]).tolist() if latency else None
            row['qualified'] = qualified_acquisition(row)
            row.update(effective_runtime_settings=actual,domain=case['domain'],official_trial=case['official_trial'],arm=case['arm'])
            write(run/'transport_metrics.json',row)
        except Exception as exc:
            row = {k:case[k] for k in ('case','domain','family','arm','card_count','port')}
            row.update(qualified=False,infrastructure_error=repr(exc),run=str(run))
            results.append(row);report()
            write(a.output/'stopped.json',dict(reason='infrastructure_or_scoring_error',case=case['case'],error=repr(exc)))
            raise
        results.append(row);report()
        print(json.dumps(dict(case=case['case'],completed=len(results),qualified=row['qualified'],
            acquired=row['rule_visual_acquisition'],failure=row.get('rule_failure'))),flush=True)
        # Ordinary physical failures remain in the full coverage matrix. No tuning.
    report(final=True)


if __name__=='__main__':
    main()
