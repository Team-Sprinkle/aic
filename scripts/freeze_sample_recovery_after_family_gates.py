#!/usr/bin/env python3
"""Retained policy specialization: adaptive SC route, unchanged SFP controller."""
from pathlib import Path
import hashlib
import json
import sys

source=Path(__file__).with_name('freeze_sample_rule_after_family_gates.py')
text=source.read_text()
changes={
 "sources=[repo/'aic_example_policies/aic_example_policies/ros/RunSampleConservativeTransport.py',":
 "sources=[repo/'aic_example_policies/aic_example_policies/ros/RunSampleTransportRecovery.py',repo/'aic_example_policies/aic_example_policies/ros/RunSampleConservativeTransport.py',",
 "selected_policy='RunSampleConservativeTransport',SC_side='opposite_public_row_corridor'":
 "selected_policy='RunSampleTransportRecovery',SC_side='baseline_then_measured_retreat_and_opposite_side'",
 "'--spec',str(frozen/'transport_spec.json'),'--gpu','5','--domain','93'":
 "'--spec',str(frozen/'transport_spec.json'),'--gpu','5','--domain','96','--policy','aic_example_policies.ros.RunSampleTransportRecovery'",
 "if not passed:\n":
 "pilot=Path('/home/nas/chmin/aic_archives/2026-10-16_sample_conservative_development_v3/SC_reverse_progress_repair_retest_v7')\n    evidence=json.loads((pilot/'summary.json').read_text())\n    trace=[json.loads(line) for line in (pilot/'development_sc_n1_p0_r0__94/runtime_trace.jsonl').open()]\n    recovery_exercised=any(r['event']=='transport_measured_route_retreat_complete' for r in trace)\n    passed=passed and evidence['results'][0]['qualified'] and recovery_exercised\n    save(a.output/'live_recovery_gate.json',dict(passed=passed,integration_pilot=str(pilot),measured_retreat_exercised=recovery_exercised,training=False))\n    if not passed:\n",
}
for old,new in changes.items():
    if text.count(old)!=1:raise ValueError('Base family gate changed: '+old)
    text=text.replace(old,new)
output=Path(sys.argv[sys.argv.index('--output')+1]);snapshot=output.with_suffix('.coordinator_snapshot.py')
if snapshot.exists():raise FileExistsError(snapshot)
snapshot.write_text(text)
output.with_suffix('.specialization.json').write_text(json.dumps(dict(base=str(source),base_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    snapshot=str(snapshot),snapshot_sha256=hashlib.sha256(text.encode()).hexdigest(),changes=list(changes),SC_gate='7/7strict plus actual live measured-retreat+handoff pilot',SFP_logic_unchanged=True),indent=2)+'\n')
exec(compile(text,str(snapshot),'exec'),dict(__name__='__main__',__file__=str(snapshot)))
