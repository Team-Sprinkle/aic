#!/usr/bin/env python3
"""Freeze one targeted candidate; run and causally score the unchanged public sample."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--gpu',default='5')
p.add_argument('--domain',default='100')
p.add_argument('--cycle',type=int,required=True)
p.add_argument('--target-continuation',action='store_true',
               help='Continue beyond the initial five cycles toward the user-requested score target')
p.add_argument('--attempt',type=int,default=1,help='Preserve failed startup in its original folder')
p.add_argument('--policy',required=True)
p.add_argument('--sfp-selection',type=Path)
p.add_argument('--transport-spec',type=Path)
p.add_argument('--empty-row-selection',type=Path)
p.add_argument('--hypothesis',required=True)
a=p.parse_args()
if a.cycle<1 or a.attempt<1 or (a.cycle>5 and not a.target_continuation):
    p.error('Use a positive cycle; beyond five explicitly select --target-continuation')
repo=Path.cwd();py=repo/'.pixi/envs/default/bin/python'
root=Path('/home/nas/chmin/aic_archives/2026-10-18_five_repair_cycles_v1')
old=Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1')
label='cycle_'+str(a.cycle)+(('_attempt'+str(a.attempt)) if a.attempt>1 else '')
out=root/label
assert not out.exists(),'Do not overwrite or duplicate a completed/active cycle'
state=root/'state.json'
def save(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def run(cmd,log,ros=False):
    cmd=list(map(str,cmd));save(log.with_suffix('.command.json'),cmd)
    if ros:
        prefix='source .pixi/envs/default/setup.bash\nexport PYTHONPATH=/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/host_reader_deps:"${PYTHONPATH:-}"\n'
        cmd=['bash','-c',prefix+shlex.join(cmd)]
    with log.open('w') as stream:subprocess.run(cmd,cwd=repo,stdout=stream,stderr=subprocess.STDOUT,check=True)
cmd=[py,repo/'scripts/run_SC_close_grasp_full_sample.py','--output',out,'--policy',a.policy,'--gpu',a.gpu,'--domain',a.domain,
     '--close-selection',old/'V21_sample_experimental_pose_bundle/selection.json',
     '--recovery','local','--experimental-diagnostic','--successful-pilot-gate',
     old/'V34_SC_matched_coverage/fresh_SC_explicit_n0_p0_seed1813/explicit_trial_gate.json']
if a.sfp_selection:cmd+=['--sfp-orientation-selection',a.sfp_selection]
if a.transport_spec:cmd+=['--transport-spec',a.transport_spec]
if a.empty_row_selection:cmd+=['--empty-row-selection',a.empty_row_selection]
save(state,dict(stage='running_unchanged_public_sample',cycle=a.cycle,policy=a.policy,hypothesis=a.hypothesis,target=290,initial_requested_cycles=5,target_continuation=a.target_continuation,final_scenes_sealed=True))
run(cmd,root/(label+'.log'))
save(out/'repair_hypothesis.json',dict(cycle=a.cycle,hypothesis=a.hypothesis))
run([py,repo/'scripts/analyze_full_connector_sample_configuration.py',out],out/'analysis.log',ros=True)
run([py,repo/'scripts/render_insertion_cycle_review.py',out],out/'review.log')
run([py,repo/'scripts/analyze_public_cycle_gaps.py',out],out/'gap_analysis.log')
summary=json.loads((out/'analysis_summary.json').read_text())
trace=[json.loads(line) for line in (out/'rollout/runtime_trace.jsonl').open()]
latencies={}
for name,event,key in [('transport','proposal','inference_ms'),('fresh','shared_perception_latency','total_ms'),('close','close_grasp_observation','complete_close_estimate_step_ms'),('servo','visual_servo_step','complete_control_step_ms')]:
    values=[r[key] for r in trace if r['event']==event and key in r]
    latencies[name]=dict(calls=len(values),p50_p95_p99_ms=np.percentile(values,[50,95,99]).tolist() if values else None)
scene_values=[r['complete_scene_check_ms'] for r in trace if r['event']=='empty_row_stationary_scene_check']
if scene_values:latencies['stationary_scene_check']=dict(calls=len(scene_values),p50_p95_p99_ms=np.percentile(scene_values,[50,95,99]).tolist())
far_values=[r['complete_preparation_ms'] for r in trace if r['event']=='far_context_prepared_before_approach']
if far_values:latencies['far_context_preparation']=dict(calls=len(far_values),p50_p95_p99_ms=np.percentile(far_values,[50,95,99]).tolist())
qualified=summary['strict_physical_insertions']==3
latency_pass=all(v['calls'] and v['p50_p95_p99_ms'][1]<300 for v in latencies.values())
result=dict(cycle=a.cycle,policy=a.policy,hypothesis=a.hypothesis,score=summary['total'],strict_insertions=summary['strict_physical_insertions'],physical_all_passed=qualified,latencies=latencies,latency_gate_passed=latency_pass,target_reached=summary['total']>290 and qualified and latency_pass,final_scenes_sealed=True,artifact_root=str(out))
save(out/'cycle_result.json',result)
results=root/'cycle_results.json'
prior=json.loads(results.read_text()) if results.exists() else []
assert all(x['cycle']!=a.cycle for x in prior)
prior.append(result);save(results,prior)
save(state,dict(stage='target_reached' if result['target_reached'] else 'cycle_complete_requires_measured_next_repair',cycle=a.cycle,result=result,completed_cycles=len(prior),initial_requested_cycles=5,target_continuation=a.target_continuation,final_scenes_sealed=True))
print(json.dumps(result),flush=True)
