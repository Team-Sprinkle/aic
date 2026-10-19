#!/usr/bin/env python3
"""Independent causal full-depth score for the new shared autonomous actor.

Read simulator truth only after the rollout has returned. Never expose this
module, labels or oracle recorder output to deployment policy decisions.
"""
import argparse,json
from pathlib import Path
import numpy as np
from audit_wide_approach_teacher_terminal import audit
from score_frozen_pipeline_coverage import score


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--case',type=Path,required=True);p.add_argument('--arm',required=True);a=p.parse_args()
 c=json.load(a.case.open());trace=list(map(json.loads,(a.run/'runtime_trace.jsonl').open()))
 exceptions=[t for t in trace if t['event']=='pipeline_exception']
 if exceptions:
  (a.run/'integration_failure.json').write_text(json.dumps(dict(implementation_failure=True,events=exceptions,autonomous_BC_performance_valid=False),indent=2)+'\n')
  raise RuntimeError('Pipeline implementation exception: inspect preserved trace before further rollout')
 raw=audit(a.run,terminal_event='terminal_before_return')
 old=score(a.run,c,a.arm);terminal=next(t for t in reversed(trace) if t['event']=='terminal_before_return')
 if terminal.get('observation_source')!='retained_public_observation_before_return':raise ValueError('Missing retained terminal observation')
 # Independent legacy stream-based geometry must agree with raw pre-return
 # MCAP geometry; never hide a reset/timestamp mismatch behind a success flag.
 differences={k:abs(raw[r]-old['terminal'][k]) for k,r in [('axial_mm','depth_mm'),('lateral_mm','lateral_mm')]}
 if max(differences.values())>1:raise ValueError('Raw and extracted terminal geometry disagree: '+str(differences))
 locks=[t for t in trace if t['event']=='shared_visual_lock']
 truth=np.asarray(raw['measured_base_opening_matrix'])[:3,3]
 lock_errors=[float(np.linalg.norm(np.asarray(t['report']['predicted_opening_base_m'])-truth)*1000) for t in locks]
 phase_latency={}
 active={t['sim_time'] for t in trace if t['event'] in ('rule_visual_inspection','SFP_rule_visual_inspection')}
 local={t['sim_time'] for t in trace if t['event']=='locked_local_perception'}
 for phase,values in [('transport_acquisition',[t['inference_ms'] for t in trace if t['event']=='proposal' and t['sim_time'] in active]),('actor',[t['inference_ms'] for t in trace if t['event']=='proposal' and t['sim_time'] in local]),('fresh_acquisition',[t['total_ms'] for t in trace if t['event']=='shared_perception_latency'])]:
  phase_latency[phase]=dict(calls=len(values),p50_p95_p99_ms=np.percentile(values,[50,95,99]).tolist() if values else None)
 latency_ok=all(v['calls'] and v['p50_p95_p99_ms'][1]<300 for v in phase_latency.values())
 runtime_geometry=any(t.get('geometry_input') is True or t.get('runtime_true_geometry') is True for t in trace)
 actor_ran=bool(local);force_reference=any(t['event']=='initial_force_reference_frozen' for t in trace)
 lock_ok=bool(locks) and max(lock_errors)<=1 and old['wrong_accepted_port'] is False
 result=dict(schema='shared150mm_autonomous_rollout/v1',case=c['case'],family=c['family'],card_count=c['card_count'],port=c['port'],requested_card=c['requested_card'],domain=c['domain'],replicate=c['replicate'],arm=a.arm,audit=str(a.run),
  raw_terminal=raw,independent_terminal_differences_mm=differences,physical_full_insertion=raw['qualified'],
  actor_ran=actor_ran,initial_force_reference_recorded=force_reference,shared_visual_lock_errors_mm=lock_errors,requested_port_locks_passed=lock_ok,
  rule_wrong_accepted_port=old['wrong_accepted_port'],
  phase_latency=phase_latency,latency_passed=latency_ok,geometry_runtime_input=runtime_geometry,
  qualified=bool(raw['qualified'] and actor_ran and force_reference and lock_ok and latency_ok and not runtime_geometry),
  terminal=terminal,backoff_observations=sum(t['event']=='measured_retreat_complete' for t in trace),
  lateral_retries=sum(t['event']=='lateral_retry_complete' for t in trace),reacquisitions=max(0,len(locks)-1),
  supervisor_exploration_is_RL=False,final_scenes_opened=False,post_reset_geometry_used=False)
 (a.run/'shared150mm_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ('case','arm','physical_full_insertion','qualified','actor_ran','requested_port_locks_passed','latency_passed')}),flush=True)
 if old['wrong_accepted_port'] is True:
  raise RuntimeError('Wrong rule-stage port accepted; preserve this scored failure and stop branch')
 if not force_reference:
  raise RuntimeError('Initial public force reference failed; diagnose before further rollouts')
if __name__=='__main__':main()
