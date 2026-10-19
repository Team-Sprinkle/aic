#!/usr/bin/env python3
"""After the baseline score, test the timing repair then rescore the same sample."""
import json,subprocess,time
from pathlib import Path
import yaml
repo=Path.cwd();python=repo/'.pixi/envs/default/bin/python'
root=Path('/home/nas/chmin/aic_archives/2026-10-18_three_trial_repair_cycle_v1')
while not root.exists():time.sleep(5)
state=root/'repair_followup_state.json'
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
while not (root/'baseline/analysis_summary.json').exists():
 save(state,dict(stage='waiting_for_first_three_trial_score_and_causal_analysis'));time.sleep(30)
old=Path('/home/nas/chmin/aic_archives/2026-10-17_insertion_control_repair_v1/visual_servo_fresh_SFP_coverage_v2')
report=json.loads((old/'runs/summary.json').read_text());cases=json.loads((old/'evaluation_cases.json').read_text())['cases'];failed={r['case'] for r in report['results'] if not r['physical_full_insertion']};names=failed|{cases[0]['case']}
output=root/'dwell_repair_development';output.mkdir(exist_ok=False)
chosen=[]
for case in cases:
 if case['case'] not in names:continue
 c=dict(case);c['configured_start_joints']=yaml.safe_load((Path(c['scene'])/'eval_config.yaml').read_text())['robot']['home_joint_positions'];chosen.append(c)
assert len(chosen)==5
save(output/'evaluation_cases.json',dict(cases=chosen));save(output/'evaluation_variants.json',dict(variants=json.loads((old/'evaluation_variants.json').read_text())['variants']))
save(output/'hypothesis.json',dict(change='Only public estimated-seat dwell0.5->2.0seconds. Same RGB models and control gains.',runtime_completion_event_used=False,baseline='Four prior toolkit partial outcomes plus one successful control.',gate='5/5strict combined success, no wrong-port accepts or scored harm',stopping='One matched five-scene test, then same unchanged complete public sample; preserve any failed diagnosis.'))
save(state,dict(stage='testing_seating_dwell_on_five_frozen_development_starts'))
with (output/'runner.log').open('w') as log:subprocess.run([str(python),'scripts/run_visual_servo_dwell_probe.py','--root',str(output),'--output',str(output/'runs')],stdout=log,stderr=subprocess.STDOUT,check=True)
d=json.loads((output/'runs/summary.json').read_text());passed=d['full']==5
policy='RunVisualConnectorServoDiagnosticV4' if passed else 'RunVisualConnectorServoDiagnosticV3'
save(output/'decision.json',dict(gate_passed=passed,successes=d['full'],episodes=5,selected_for_rescore=policy,SC_grasp_repair_promoted=False,reason='Only the validated seating-duration change may be selected; keep baseline if it fails.'))
save(state,dict(stage='running_second_unchanged_three_trial_sample',selected_policy=policy))
with (root/'repaired.log').open('w') as log:subprocess.run([str(python),'scripts/run_repaired_sample_configuration.py','--policy',policy,'--output',str(root/'repaired')],stdout=log,stderr=subprocess.STDOUT,check=True)
command='source .pixi/envs/default/setup.bash\nexport PYTHONPATH=/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/host_reader_deps:"${PYTHONPATH:-}"\n'+str(python)+' scripts/analyze_full_connector_sample_configuration.py '+str(root/'repaired')
with (root/'repaired_analysis.log').open('w') as log:subprocess.run(['bash','-c',command],stdout=log,stderr=subprocess.STDOUT,check=True)
a=json.loads((root/'baseline/analysis_summary.json').read_text());b=json.loads((root/'repaired/analysis_summary.json').read_text())
save(root/'score_comparison.json',dict(first_score=a['total'],second_score=b['total'],maximum=300,first_combined_success=a['strict_physical_insertions'],second_combined_success=b['strict_physical_insertions'],seat_dwell_gate_passed=passed,sample_yaml_unchanged=True,final_scenes_opened=False,RL=False,learned_actor_inference=False,remaining='SC image-derived grasp orientation and any fullsample trace/video failures require targeted diagnosis, not routine training.'))
save(state,dict(stage='both_three_trial_evaluations_complete_needs_trace_video_review',comparison=str(root/'score_comparison.json')))
