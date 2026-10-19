#!/usr/bin/env python3
"""Four matched SC cases, then three unchanged public samples, stopping at failure."""
import concurrent.futures
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import numpy as np

repo = Path.cwd()
py = repo/'.pixi/envs/default/bin/python'
root = Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1')
out = root/'V34_SC_matched_coverage'
out.mkdir(exist_ok=False)
state = out/'state.json'
policy = 'RunVisualConnectorServoRecoveryFastV34'

def read(path):
    return json.loads(path.read_text())

def save(data):
    tmp = state.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    tmp.replace(state)

def run(command, log, ros=False):
    command = list(map(str, command))
    log.with_suffix('.command.json').write_text(json.dumps(command, indent=2)+'\n')
    if ros:
        prefix = 'source .pixi/envs/default/setup.bash\nexport PYTHONPATH=/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/host_reader_deps:"${PYTHONPATH:-}"\n'
        command = ['bash', '-c', prefix+shlex.join(command)]
    with log.open('w') as stream:
        subprocess.run(command, cwd=repo, stdout=stream, stderr=subprocess.STDOUT, check=True)

def evaluate(case, gpu, domain):
    case_root = out/case['case']
    case_root.mkdir()
    (case_root/'evaluation_cases.json').write_text(json.dumps(dict(cases=[case]), indent=2)+'\n')
    variants = read(root/'V21_close_grasp_development/evaluation_variants.json')
    variants['variants'][0]['name'] = 'V34_context_tight_alignment_stiffness500_shared_codec'
    (case_root/'evaluation_variants.json').write_text(json.dumps(variants, indent=2)+'\n')
    bundle = case_root/'pose_bundle'
    run([py, repo/'scripts/prepare_SC_close_composite_bundle.py',
         '--rotation-fit', root/'spacefilled_near_fused_orientation_warm',
         '--rotation-review', root/'final80_orientation_review',
         '--translation-fit', root/'expanded_near_spatial_joint',
         '--translation-review', root/'expanded_near_translation_component_review',
         '--experimental-case', case['case'], '--independent-frozen-replay',
         root/'final80_frozen_rollout_generalization/summary.json', '--output', bundle], case_root/'bundle.log')
    shutil.copyfile(bundle/'selection.json', case_root/'close_pose_selection.json')
    run([py, repo/'scripts/run_visual_servo_close_grasp_probe.py', '--root', case_root,
         '--output', case_root/'runs', '--gpu', gpu, '--domain', domain,
         '--experimental-diagnostic', '--policy', policy], case_root/'rollout.log')
    row = read(case_root/'runs/summary.json')['results'][0]
    gate = case_root/'explicit_trial_gate.json'
    run([py, repo/'scripts/qualify_SC_explicit_close_trial.py', '--audit', row['audit'], '--output', gate], case_root/'gate.log')
    return dict(case=case['case'], gate=read(gate), physical=row['raw_terminal'], audit=row['audit'], gate_path=str(gate))

def timing(path):
    rows = [json.loads(line) for line in (path/'rollout/runtime_trace.jsonl').open()]
    stages = {}
    for name, event, key in [('transport', 'proposal', 'inference_ms'),
                             ('fresh', 'shared_perception_latency', 'total_ms'),
                             ('close', 'close_grasp_observation', 'complete_close_estimate_step_ms'),
                             ('servo', 'visual_servo_step', 'complete_control_step_ms')]:
        values = [r[key] for r in rows if r['event']==event and key in r]
        if name=='transport' and not values:
            # Historical transport latency is nested in proposal records.
            values = [r['proposal']['inference_ms'] for r in rows if isinstance(r.get('proposal'), dict) and 'inference_ms' in r['proposal']]
        stages[name] = dict(calls=len(values), p50_p95_p99_ms=np.percentile(values, [50,95,99]).tolist() if values else None)
    report = dict(stages=stages, p95_gate_passed=all(v['calls'] and v['p50_p95_p99_ms'][1]<300 for v in stages.values()))
    (path/'explicit_timing_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    return report

save(dict(stage='running_four_matched_SC_cases', policy=policy, final_scenes_sealed=True))
cases = read(root/'V17_fresh_SC_development/evaluation_cases.json')['cases']
results = []
failed = False
# Two isolated rootless workers; legacy GPU0 remains untouched.
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    pending = {pool.submit(evaluate, cases[i], str(3+i), str(137+i)): (3+i, 137+i) for i in range(2)}
    next_case = 2
    while pending:
        done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
        for future in done:
            gpu, domain = pending.pop(future)
            try:
                result = future.result()
            except Exception as exc:
                failed = True
                result = dict(error=repr(exc), gpu=gpu)
            results.append(result)
            failed |= not result.get('gate', {}).get('explicit_trial_pass', False)
            save(dict(stage='matched_case_failed_waiting_for_already_active_jobs' if failed else 'running_four_matched_SC_cases', results=results, remaining_unlaunched=len(cases)-next_case, final_scenes_sealed=True))
            if not failed and next_case<len(cases):
                pending[pool.submit(evaluate, cases[next_case], str(gpu), str(domain))] = (gpu, domain)
                next_case += 1
if failed:
    save(dict(stage='failed_gate_requires_targeted_diagnosis', results=results, final_scenes_sealed=True))
    raise SystemExit(0)

sample_results = []
for index in range(1, 4):
    path = root/('sample_V34_experimental' + ('' if index==1 else '_repeat'+str(index)))
    save(dict(stage='running_unchanged_public_sample', round=index, results=results, sample_results=sample_results, final_scenes_sealed=True))
    run([py, repo/'scripts/run_SC_close_grasp_full_sample.py', '--output', path,
         '--close-selection', root/'V21_sample_experimental_pose_bundle/selection.json',
         '--policy', policy, '--recovery', 'local', '--experimental-diagnostic',
         '--successful-pilot-gate', results[0]['gate_path']], root/(path.name+'.coordinator.log'))
    run([py, repo/'scripts/analyze_full_connector_sample_configuration.py', path], path/'analysis.log', ros=True)
    run([py, repo/'scripts/render_insertion_cycle_review.py', path], path/'review.log')
    summary = read(path/'analysis_summary.json')
    latency = timing(path)
    item = dict(root=str(path), total=summary['total'], strict_successes=summary['strict_physical_insertions'], timing=latency)
    sample_results.append(item)
    if summary['strict_physical_insertions']!=3 or not latency['p95_gate_passed']:
        save(dict(stage='sample_failed_requires_targeted_diagnosis', results=results, sample_results=sample_results, final_scenes_sealed=True))
        raise SystemExit(0)
save(dict(stage='four_SC_cases_and_three_public_samples_complete', results=results, sample_results=sample_results, model_globally_qualified=False, final_scenes_sealed=True))
