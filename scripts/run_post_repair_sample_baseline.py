#!/usr/bin/env python3
"""Finish the active recovery screen, then score the unchanged public sample."""
import json,subprocess,time
from pathlib import Path

repo=Path.cwd();python=repo/'.pixi/envs/default/bin/python'
r=Path('/home/nas/chmin/aic_archives/2026-10-17_insertion_control_repair_v1')
root=Path('/home/nas/chmin/aic_archives/2026-10-18_three_trial_repair_cycle_v1')
root.mkdir(exist_ok=False)
def save(name,d):(root/name).write_text(json.dumps(d,indent=2)+'\n')
save('plan.json',dict(user_request='Finish ongoing comparison, full3trialscore, diagnose andrepair, full3trialrescore.',sample='Unchanged original predeadline public sample_config.yaml,3trials,max300',first_policy='RunVisualConnectorServoDiagnosticV3',reason='Explicit image controller demonstrates development insertion whereas the learned context actors are0/8. This is an explicitly labeled controller substitution, not new BC learning.',RL=False,final_scenes_opened=False,automatic_retraining=False,stopping='One baseline and one repaired fullsample; targeted development fixes first. Preserve both failures and comparisons.'))
while not (r/'post_repair_backoff_comparison_v1/runs/summary.json').exists():
 save('state.json',dict(stage='waiting_for_frozen_backoff_comparison',active_jobs_interrupted=False));time.sleep(30)
save('state.json',dict(stage='running_first_unchanged_three_trial_sample'))
with (root/'baseline.log').open('w') as f:subprocess.run([str(python),'scripts/run_repaired_sample_configuration.py','--policy','RunVisualConnectorServoDiagnosticV3','--output',str(root/'baseline')],stdout=f,stderr=subprocess.STDOUT,check=True)
save('state.json',dict(stage='analyzing_first_sample_and_rendering_videos'))
command='source .pixi/envs/default/setup.bash\nexport PYTHONPATH=/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/host_reader_deps:"${PYTHONPATH:-}"\n'+str(python)+' scripts/analyze_full_connector_sample_configuration.py '+str(root/'baseline')
with (root/'baseline_analysis.log').open('w') as f:subprocess.run(['bash','-c',command],stdout=f,stderr=subprocess.STDOUT,check=True)
save('state.json',dict(stage='baseline_complete_needs_targeted_repair_then_rescore',summary=str(root/'baseline/analysis_summary.json')))
