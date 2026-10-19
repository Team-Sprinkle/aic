#!/usr/bin/env python3
"""Pilot gate -> four frozen fresh SC starts -> optional full sample rescore."""
import copy,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import yaml
repo=Path.cwd();python=repo/'.pixi/envs/default/bin/python';root=Path('/home/nas/chmin/aic_archives/2026-10-18_three_trial_repair_cycle_v1');state=root/'SC_selected_recovery_followup_state.json'
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
pilot=root/'SC_entry_speed_recovery_pilot_v9/runs/summary.json'
while not pilot.exists():
 save(state,dict(stage='waiting_for_matched8mmpsSCpilot'));time.sleep(30)
d=json.loads(pilot.read_text());rows=d['results'];passed=len(rows)==1 and rows[0]['physical_full_insertion'] and rows[0]['raw_terminal']['no_scored_harm'] and rows[0]['requested_port_locks_passed']
if not passed:
 save(state,dict(stage='pilot_failed_no_expansion_no_promotion',reason='Keep15.5mm/0.5mm/1deg physical gate; inspect saved traces before next repair.'));raise SystemExit(0)
output=root/'SC_fresh_recovery_validation_v1';output.mkdir(exist_ok=False)
sample=repo/'configs/evaluation_protocol/qualification_public_predeadline/sample_config.yaml';assert hashlib.sha256(sample.read_bytes()).hexdigest()=='21f6eb9ac0d71f7dddac3657c67c41b5fa212e5aaa1fd68528b08ec219585c8d';base=yaml.safe_load(sample.read_text());rng=np.random.default_rng(2026101809);cases=[]
for count in [0,5]:
 for port in [0,1]:
  cfg=copy.deepcopy(base);trial=cfg['trials']['trial_3'];cfg['trials']={'trial_3':trial};trial['tasks']['task_1']['target_module_name']=f'sc_port_{port}';board=trial['scene']['task_board']
  for j in range(5):
   board[f'nic_rail_{j}']=dict(entity_present=j<count)
   if j<count:board[f'nic_rail_{j}'].update(entity_name=f'nic_card_{j}',entity_pose=dict(translation=float(rng.uniform(-.01,.01)),roll=0.,pitch=0.,yaw=float(rng.uniform(-.01,.01))))
  for j in range(2):
   existing=board[f'sc_rail_{j}'].get('entity_pose',{}).get('translation',0.)
   board[f'sc_rail_{j}']=dict(entity_present=True,entity_name=f'sc_mount_{j}',entity_pose=dict(translation=float(existing+rng.uniform(-.01,.01)),roll=0.,pitch=0.,yaw=0.))
  for axis in 'xy':board['pose'][axis]+=float(rng.uniform(-.01,.01))
  board['pose']['yaw']+=float(rng.uniform(-.05,.05));angular=rng.normal(size=3);angular*=rng.uniform(0,.01)/np.linalg.norm(angular)
  for j,key in enumerate(['roll','pitch','yaw']):trial['scene']['cables']['cable_1']['pose'][key]+=float(angular[j])
  name=f'fresh_recovery_SC_n{count}_p{port}_seed1809';scene=output/'scenes'/name;scene.mkdir(parents=True);(scene/'eval_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));(scene/'ur_llb_start.xacro').write_bytes((root/'baseline/scene/ur_llb_start.xacro').read_bytes())
  for f in ['world_audit.sdf','bridge_audit.yaml']:(scene/f).write_bytes((repo/'configs/hierarchical_recovery'/f).read_bytes())
  cases.append(dict(case=name,scene=str(scene),family='sc',card_count=count,port=port,requested_card=None,replicate=0,configured_start_joints=cfg['robot']['home_joint_positions'],domain='fresh_sample_derived_SC_recovery_development',split='development',final_scene=False))
save(output/'evaluation_cases.json',dict(cases=cases));variant=json.loads((root/'SC_entry_speed_recovery_pilot_v9/evaluation_variants.json').read_text())['variants'][0];variant['name']='RGB_servo_local_prior_8mmps_sharedJPEG';save(output/'evaluation_variants.json',dict(variants=[variant]));save(output/'coverage_plan.json',dict(seed=2026101809,episodes=4,coverage='BothSCtargets x0/5NICcards; newboardxy±10mm/yaw±0.05rad,SCrail±10mm,NICrail±10mm,graspangleperturbationnorm<=0.01rad. Not full public randomization or sealed final coverage.',models_or_physics_changed=False,preprocessing='Exact sharedJPEG86; same predicted poses. ValidatedSFPbitexact and3/3latency screen.',gate='4/4strict fullinsertion,correctrequestedports,noharmandfreshRGBp95<300ms each; otherwise no fullsample selection.',stopping='One four-case freshvalidation before any additional sample run.'))
save(state,dict(stage='running_four_frozen_fresh_SC_validation_starts'))
with (output/'runner.log').open('w') as log:subprocess.run([str(python),'scripts/run_visual_servo_fresh_SC_recovery_probe.py','--root',str(output),'--output',str(output/'runs')],stdout=log,stderr=subprocess.STDOUT,check=True)
d=json.loads((output/'runs/summary.json').read_text());passed=len(d['results'])==4 and all(r['physical_full_insertion'] and r['raw_terminal']['no_scored_harm'] and r['requested_port_locks_passed'] and r['phase_latency']['fresh_acquisition']['p50_p95_p99_ms'][1]<300 for r in d['results']);save(output/'decision.json',dict(gate_passed=passed,automatic_deployment=False,strict_success=sum(r['physical_full_insertion'] for r in d['results']),episodes=4))
if not passed:
 save(state,dict(stage='fresh_validation_failed_preserve_diagnose_no_sample_promotion'));raise SystemExit(0)
save(state,dict(stage='running_additional_unchanged_three_trial_sample_after_fresh_SC_gate'))
with (root/'validated_recovery_sample.log').open('w') as log:subprocess.run([str(python),'scripts/run_selected_recovery_sample_configuration.py','--policy','RunVisualConnectorServoRecoveryFastV16','--recovery','local','--output',str(root/'validated_recovery_sample')],stdout=log,stderr=subprocess.STDOUT,check=True)
cmd='source .pixi/envs/default/setup.bash\nexport PYTHONPATH=/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/host_reader_deps:"${PYTHONPATH:-}"\n'+str(python)+' scripts/analyze_full_connector_sample_configuration.py '+str(root/'validated_recovery_sample')
with (root/'validated_recovery_sample_analysis.log').open('w') as log:subprocess.run(['bash','-c',cmd],stdout=log,stderr=subprocess.STDOUT,check=True)
d=json.loads((root/'validated_recovery_sample/analysis_summary.json').read_text());save(state,dict(stage='complete_needs_physical_video_review',score=d['total'],combined_success=d['strict_physical_insertions'],automatic_deployment=False))
