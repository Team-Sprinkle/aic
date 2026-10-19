#!/usr/bin/env python3
"""Only a successful SC development recovery screen can add a full sample run."""
import json,subprocess,time
from pathlib import Path
root=Path('/home/nas/chmin/aic_archives/2026-10-18_three_trial_repair_cycle_v1');repo=Path.cwd();python=repo/'.pixi/envs/default/bin/python'
state=root/'SC_recovery_followup_state.json'
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
prerequisites=[root/'SC_visual_recovery_diagnostic_v1/runs/summary.json',root/'repaired/analysis_summary.json']
while not all(p.exists() for p in prerequisites):
 save(state,dict(stage='waiting_for_SC_recovery_development_and_second_sample_score'));time.sleep(30)
d=json.loads(prerequisites[0].read_text());eligible=[]
for mode in ['local','hierarchical']:
 rows=[r for r in d['results'] if r['recovery_mode']==mode]
 passed=len(rows)==2 and all(r['physical_full_insertion'] and r['requested_port_locks_passed'] and r['raw_terminal']['no_scored_harm'] and r['phase_latency']['fresh_acquisition']['p50_p95_p99_ms'][1]<300 for r in rows)
 if passed:eligible.append(mode)
if not eligible:
 save(state,dict(stage='SC_recovery_gate_failed_preserve_diagnose',eligible_modes=[],automatic_promotion=False,finished_three_trial_scores=str(root/'score_comparison.json')))
else:
 mode='hierarchical' if 'hierarchical' in eligible else eligible[0]
 save(root/'SC_recovery_fullsample_selection.json',dict(selected_mode=mode,eligible_modes=eligible,reason='Both frozen development repeats pass full depth/lateral/orientation, noharm/requestedport and freshRGBp95<300ms',selection_scope='Diagnostic candidate, not population reliability or learned BC/RL',SC_estimator_changed=False))
 save(state,dict(stage='running_additional_validated_recovery_three_trial_sample',mode=mode))
 with (root/'SC_recovery_sample.log').open('w') as f:subprocess.run([str(python),'scripts/run_recovery_sample_configuration.py','--policy','RunVisualConnectorServoRecoveryV5','--recovery',mode,'--output',str(root/'recovery_candidate')],stdout=f,stderr=subprocess.STDOUT,check=True)
 cmd='source .pixi/envs/default/setup.bash\nexport PYTHONPATH=/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/host_reader_deps:"${PYTHONPATH:-}"\n'+str(python)+' scripts/analyze_full_connector_sample_configuration.py '+str(root/'recovery_candidate')
 with (root/'SC_recovery_sample_analysis.log').open('w') as f:subprocess.run(['bash','-c',cmd],stdout=f,stderr=subprocess.STDOUT,check=True)
 summary=json.loads((root/'recovery_candidate/analysis_summary.json').read_text());save(state,dict(stage='complete_needs_physical_video_review',score=summary['total'],combined_success=summary['strict_physical_insertions'],automatic_promotion=False))
