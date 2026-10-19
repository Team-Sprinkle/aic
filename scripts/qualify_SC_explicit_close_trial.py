#!/usr/bin/env python3
"""Controller-specific gate; preserve the generic BC-actor score unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--audit', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a=p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
metrics_path=a.audit/'shared150mm_metrics.json'
m=json.loads(metrics_path.read_text())
trace=[json.loads(line) for line in (a.audit/'runtime_trace.jsonl').open()]
latencies={k:v for k,v in m['phase_latency'].items() if k!='actor'}
for name,event,key in [('close_grasp','close_grasp_observation','complete_close_estimate_step_ms'),
                       ('insertion_control','visual_servo_step','complete_control_step_ms')]:
    values=[r[key] for r in trace if r['event']==event]
    latencies[name]=dict(calls=len(values),p50_p95_p99_ms=np.percentile(values,[50,95,99]).tolist() if values else None)
latency_pass=all(v['calls'] and v['p50_p95_p99_ms'][1]<300 for v in latencies.values())
updates=[r for r in trace if r['event']=='close_grasp_update_accepted']
checks=dict(physical_insertion_and_toolkit=m['physical_full_insertion'],
    all_used_stages_p95_below300ms=latency_pass,
    correct_visual_port=m['requested_port_locks_passed'],
    initial_force_reference=m['initial_force_reference_recorded'],
    no_runtime_geometry=not m['geometry_runtime_input'],
    learned_actor_not_used=not m['actor_ran'],
    close_update_applied=bool(updates),
    port_unchanged=all(r['port_pose_unchanged'] for r in updates),
    update_bounds=all(r['rotation_change_deg']<=3 and r['translation_change_mm']<=1 for r in updates))
report=dict(schema='explicit_controller_trial_gate/v1',case=m['case'],
    explicit_trial_pass=all(checks.values()),checks=checks,phase_latency=latencies,
    raw_metrics_sha256=hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
    raw_metrics_path=str(metrics_path),model_globally_qualified=False,automatic_promotion=False,
    generic_actor_score_unchanged=True,
    explanation='Generic shared150mm qualified flag requires learned actor execution and actor timing. This trial uses an explicit controller; score all actually used stages instead.')
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
