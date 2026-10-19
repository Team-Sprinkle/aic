#!/usr/bin/env python3
"""Finish active sample, score it, repeat twice only after the physical gate."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import numpy as np

repo=Path.cwd()
py=repo/'.pixi/envs/default/bin/python'
root=Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1')
state=root/'V21_three_sample_state.json'
if state.exists():
    raise FileExistsError('Inspect existing continuation; never duplicate evaluations')

def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None

def save(data):
    temp=state.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(state)

def run(command, log, ros=False):
    if ros:
        prefix='source .pixi/envs/default/setup.bash\nexport PYTHONPATH=/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/host_reader_deps:"${PYTHONPATH:-}"\n'
        command=['bash','-c',prefix+shlex.join(list(map(str,command)))]
    with log.open('w') as stream:
        subprocess.run(list(map(str,command)),cwd=repo,stdout=stream,stderr=subprocess.STDOUT,check=True)

def timing(path):
    rows=[json.loads(x) for x in (path/'rollout/runtime_trace.jsonl').open()]
    results={}
    for name,event,key in [('fresh_acquisition','shared_perception_latency','total_ms'),
                           ('close_grasp','close_grasp_observation','complete_close_estimate_step_ms'),
                           ('insertion_control','visual_servo_step','complete_control_step_ms')]:
        values=[r[key] for r in rows if r['event']==event]
        results[name]=dict(calls=len(values),p50_p95_p99_ms=np.percentile(values,[50,95,99]).tolist() if values else None)
    passed=all(v['calls'] and v['p50_p95_p99_ms'][1]<300 for v in results.values())
    report=dict(used_stage_timing=results,p95_gate_passed=passed,actor_not_used=True)
    (path/'explicit_timing_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

results=[]
for index,name in enumerate(['sample_V21_experimental','sample_V21_experimental_repeat2','sample_V21_experimental_repeat3']):
    path=root/name
    if index:
        save(dict(stage='running_next_unchanged_sample',round=index+1,results=results,maximum_rounds=3))
        command=[py,repo/'scripts/run_SC_close_grasp_full_sample.py','--output',path,
                 '--close-selection',root/'V21_sample_experimental_pose_bundle/selection.json',
                 '--recovery','local','--experimental-diagnostic','--successful-pilot-gate',
                 root/'V21_close_grasp_development/explicit_trial_gate.json']
        (root/(name+'.command.json')).write_text(json.dumps(list(map(str,command)))+'\n')
        run(command,root/(name+'.coordinator.log'))
    else:
        save(dict(stage='waiting_for_existing_sample_no_duplicate',round=1,maximum_rounds=3))
        while True:
            current=read(path/'state.json')
            if current and current['stage'].startswith('complete_'):break
            if current and current['stage']=='infrastructure_or_execution_failure_requires_audit':
                save(dict(stage='execution_failure_requires_targeted_diagnosis',results=results,failed_root=str(path)));raise SystemExit(1)
            time.sleep(20)
    save(dict(stage='causal_physical_and_video_analysis',round=index+1,results=results))
    if not (path/'analysis_summary.json').exists():
        if (path/'per_trial').exists():raise RuntimeError('Partial analysis exists; audit before resuming')
        run([py,repo/'scripts/analyze_full_connector_sample_configuration.py',path],path/'analysis.log',ros=True)
    run([py,repo/'scripts/render_insertion_cycle_review.py',path],path/'review.log')
    summary=read(path/'analysis_summary.json');latency=timing(path)
    result=dict(root=str(path),total=summary['total'],strict_successes=summary['strict_physical_insertions'],
                trials=summary['trials'],timing=latency,physical=[r['physical'] for r in summary['results']])
    results.append(result)
    if summary['strict_physical_insertions']!=3 or not latency['p95_gate_passed']:
        save(dict(stage='failed_gate_requires_targeted_diagnosis_no_routine_repeat',results=results,
                  maximum_rounds=3,final_scenes_sealed=True));raise SystemExit(0)
save(dict(stage='three_unchanged_public_samples_complete',results=results,experimental_only=True,
          model_original_validation_still_failed=True,final_scenes_sealed=True))
