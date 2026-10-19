#!/usr/bin/env python3
"""Retained (~50Hz) public wrist loads around seating; no post-reset observations."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

root=Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1')
output=root/'SFP_seating_force_transient_audit';output.mkdir(exist_ok=False)
results=[]
for name in ['sample_V21_experimental','sample_V21_experimental_repeat2']:
    audit=root/name/'per_trial/trial_1'
    data=json.loads((audit/'scored_stream.json').read_text())
    trace=[json.loads(x) for x in (audit/'runtime_trace.jsonl').open()]
    reference=np.array(next(x for x in trace if x['event']=='initial_force_reference_frozen')['base_force_n'])
    terminal=next(x for x in reversed(trace) if x['event']=='terminal_before_return' and
                  x.get('observation_source')=='retained_public_observation_before_return')['sim_time']
    controllers=data['streams']['/aic_controller/controller_state']
    times=np.array([x[0] for x in controllers])
    rows=[x for x in data['streams']['/fts_broadcaster/wrench'] if terminal-6<=x[0]/1e9<=terminal]
    stamps=np.array([x[0] for x in rows]);indices=np.searchsorted(times,stamps,side='right')-1
    valid=(indices>=0)&((stamps-times[np.maximum(indices,0)])<=4_000_000)
    stamps=stamps[valid];indices=indices[valid];rows=[x for x,v in zip(rows,valid) if v]
    quaternions=[controllers[i][1]['tcp_pose']['orientation'] for i in indices]
    rotations=Rotation.from_quat([[q[k] for k in ['x','y','z','w']] for q in quaternions])
    force=np.array([[r[1]['wrench']['force'][k] for k in ['x','y','z']] for r in rows])
    load=rotations.apply(force)-reference
    norms=np.linalg.norm(load,axis=1)
    segments=[];start=None
    for i,high in enumerate(norms>=8):
        if high and start is None:start=i
        if start is not None and (not high or i==len(norms)-1):
            end=i if high else i-1
            segments.append(dict(start_s=stamps[start]/1e9,end_s=stamps[end]/1e9,
                span_ms=(stamps[end]-stamps[start])/1e6,samples=end-start+1,
                maximum_load_n=float(norms[start:end+1].max()),
                right_censored_at_terminal=bool(end==len(norms)-1)))
            start=None
    result=dict(run=name,terminal_time_s=terminal,public_force_samples=len(norms),
        load_p50_p95_max_n=np.r_[np.percentile(norms,[50,95]),norms.max()].tolist(),
        over8N_segments=segments,causal_controller_pose_age_limit_ms=4,
        no_post_return_or_post_reset_data=True,baseline_retared=False,
        caveat='These are load changes, not an oracle contact-force decomposition. The failed trajectory ends at abort, so that last excursion is right-censored.')
    results.append(result)
    np.savez(output/(name+'.npz'),time_s=stamps/1e9,load_change_n=norms,load_change_base_n=load)
(output/'summary.json').write_text(json.dumps(dict(results=results,no_policy_changes=True),indent=2)+'\n')
print(json.dumps(results),flush=True)
