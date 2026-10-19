#!/usr/bin/env python3
"""Bounded nominal-centred SC grasp image collection; no transport/BC training."""
import argparse,copy,hashlib,json,os,re,shlex,shutil,subprocess,time
from pathlib import Path
import numpy as np
import yaml
from shared_recovery_eval_environment import environment


def save(path,data):path.write_text(json.dumps(data,indent=2)+'\n')


def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--episode-prefix',default='');p.add_argument('--train-only',action='store_true');p.add_argument('--output',type=Path,required=True);p.add_argument('--wait-for',type=Path);p.add_argument('--resume',action='store_true');p.add_argument('--viewpoint',choices=['home','acquisition','near'],default='home');p.add_argument('--gpu',type=int,default=5);p.add_argument('--domain',default='101');p.add_argument('--seed',type=int,default=2026101781);p.add_argument('--repeats',type=int,default=6);p.add_argument('--card-counts',nargs='+',type=int,default=[0,1,5]);p.add_argument('--qa-count',type=int,default=6);p.add_argument('--qa-file',default='first_six_visual_QA.json');p.add_argument('--prefill',action='store_true',help='After first-six QA, capture future cases in reverse order without owning main progress');a=p.parse_args();repo=Path.cwd()
 if a.prefill and (not a.resume or a.viewpoint!='acquisition'):raise ValueError('Prefill requires existing acquisition manifest')
 a.output.mkdir(parents=True,exist_ok=a.resume)
 if a.resume:
  cases=json.loads((a.output/'plan.json').read_text())['cases']
 else:
  sample=repo/'configs/evaluation_protocol/qualification_public_predeadline/sample_config.yaml';base=yaml.safe_load(sample.read_text());assert hashlib.sha256(sample.read_bytes()).hexdigest()=='21f6eb9ac0d71f7dddac3657c67c41b5fa212e5aaa1fd68528b08ec219585c8d'
  rng=np.random.default_rng(a.seed);cases=[]
  robot=(repo/'aic_description/urdf/ur_gz.urdf.xacro').read_text()
  for key,value in base['robot']['home_joint_positions'].items():robot,n=re.subn(rf'({re.escape(key)}=)(-?[0-9]+\.[0-9]+)',rf'\g<1>{value:.7f}',robot,count=1);assert n==1
  for repeat in range(a.repeats):
   for count in a.card_counts:
    for port in [0,1]:
     cfg=copy.deepcopy(base);trial=cfg['trials']['trial_3'];cfg['trials']={'trial_3':trial};board=trial['scene']['task_board'];task=trial['tasks']['task_1'];task['target_module_name']=f'sc_port_{port}'
     for j in range(5):
      board[f'nic_rail_{j}']=dict(entity_present=j<count)
      if j<count:board[f'nic_rail_{j}'].update(entity_name=f'nic_card_{j}',entity_pose=dict(translation=float(rng.uniform(-.0215,.0234)),roll=0.,pitch=0.,yaw=float(rng.uniform(-.05,.05))))
     for j in range(2):board[f'sc_rail_{j}']=dict(entity_present=True,entity_name=f'sc_mount_{j}',entity_pose=dict(translation=float(rng.uniform(-.06,.055)),roll=0.,pitch=0.,yaw=0.))
     for axis in 'xy':board['pose'][axis]+=float(rng.uniform(-.035,.035))
     board['pose']['yaw']+=float(rng.uniform(-.2,.2))
     cable=trial['scene']['cables']['cable_1']['pose'];angular=rng.normal(size=3);angular*=rng.uniform(.005,.06)/np.linalg.norm(angular);shift=rng.normal(size=3);shift*=rng.uniform(0,.0015)/np.linalg.norm(shift)
     for j,key in enumerate(['roll','pitch','yaw']):cable[key]+=float(angular[j])
     for j,key in enumerate('xyz'):cable['gripper_offset'][key]+=float(shift[j])
     name=f'sc_grasp_r{repeat}_n{count}_p{port}' if a.viewpoint=='home' else f'sc_{a.viewpoint}_grasp_r{repeat}_n{count}_p{port}';name=(a.episode_prefix+'_'+name) if a.episode_prefix else name;scene=a.output/'scenes'/name;scene.mkdir(parents=True)
     (scene/'eval_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));(scene/'ur_llb_start.xacro').write_text(robot)
     for filename in ['world_audit.sdf','bridge_audit.yaml']:shutil.copyfile(repo/'configs/hierarchical_recovery'/filename,scene/filename)
     cases.append(dict(case=name,scene=str(scene),family='sc',card_count=count,port=port,requested_card=None,replicate=repeat,split='validation' if repeat%3==2 and not a.train_only else 'train',domain='sample_derived_perception_only',final_scene=False,BC_expert=False,viewpoint=a.viewpoint))
  save(a.output/'plan.json',dict(cases=cases,planned=36,training=24,validation=12,seed=2026101781,pose_only=True,BC_expert_admission=False,sample_sha256=hashlib.sha256(sample.read_bytes()).hexdigest(),hypothesis='Existing SC demonstrations use a shifted narrow grasp distribution excluding the public default. Capture nominal-centred variation without insertion or route changes.',gate='First six must have valid native image-time labels and stable public pose; otherwise diagnose before continuing. Stop after36, no automatic collection expansion.',reserved_final_scenes_opened=False))
  plan=json.loads((a.output/'plan.json').read_text());plan.update(seed=a.seed,viewpoint=a.viewpoint,physical_gpu=a.gpu,domain=a.domain,planned=len(cases),training=sum(c['split']=='train' for c in cases),validation=sum(c['split']=='validation' for c in cases),first_visual_QA_count=a.qa_count)
  if a.viewpoint=='acquisition':
   plan.update(hypothesis='Only four nominal training resets cover the actual accepted acquisition viewpoint; home holds do not reproduce its appearance. Add36 perception-only acquisition holds before one bounded pose fit.',deployment_gate='On held-out actual-acquisition domains: translation p95<=0.4mm and orientation p95<=1degree. Home errors still reported; no deployment at home.')
  save(a.output/'plan.json',plan)
  save(a.output/'state.json',dict(stage='waiting_for_control_comparison' if a.wait_for else 'prepared',collected=0))
 if a.prepare_only:return
 if a.wait_for:
  while not a.wait_for.exists():time.sleep(15)
 old=Path('/home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1');prior=Path('/home/nas/chmin/aic_archives/2026-10-16_full_sample_insertion_evaluation_v1');spec=json.loads((prior/'frozen/transport_spec.json').read_text());results=json.loads((a.output/'progress.json').read_text())['results'] if a.resume and (a.output/'progress.json').exists() else []
 rejected=[r for r in results if not r['valid']];invalid_cases={r['case']['case'] for r in rejected};results=[r for r in results if r['valid']]
 if rejected and not a.prefill:
  rejection_path=a.output/'rejected_captures.json';previous=json.loads(rejection_path.read_text()) if rejection_path.exists() else [];save(rejection_path,previous+rejected)
 if a.prefill:
  while not (a.output/a.qa_file).exists():time.sleep(10)
  assert json.loads((a.output/a.qa_file).read_text())['passed']
  cases=list(reversed(cases))
  results=[]
 for case in cases:
  if a.prefill:
   plan_cases=json.loads((a.output/'plan.json').read_text())['cases'];indices={c['case']:i for i,c in enumerate(plan_cases)}
   current=json.loads((a.output/'current_job.json').read_text())['case']['case']
   if indices[case['case']]-indices[current]<8:break
   if (a.output/'runs'/case['case']).exists():continue
  if any(x['case']['case']==case['case'] for x in results):continue
  run=a.output/'runs'/case['case'];run.parent.mkdir(exist_ok=True)
  policy=('RunStableGraspObservationCapture' if case.get('stable_hold_required') else 'RunGraspObservationCapture' if case.get('pose_only_reset_hold') else 'RunNearGraspObservationCapture' if a.viewpoint=='near' else 'RunAcquisitionGraspCapture' if a.viewpoint=='acquisition' else 'RunGraspObservationCapture')
  env=environment(repo,old,spec['bundle'],case,policy='aic_example_policies.ros.'+policy,domain=a.domain)
  env.update(AIC_TRANSPORT_EVAL_GPU=str(a.gpu),AIC_TRANSPORT_EVAL_POLICY_GPU=str(a.gpu),AIC_TRANSPORT_EVAL_ORACLE_CONTACTS='false')
  if a.viewpoint in ['acquisition','near']:env['AIC_RULE_TRANSPORT_TEMPLATE']=str(prior/'frozen/transport_spec.json')
  if a.viewpoint=='near' and not case.get('pose_only_reset_hold'):
   actor=Path('/home/nas/chmin/aic_archives/2026-10-17_connector_orientation_and_sfp_oscillation_v1/full_connector_bc_h4/best.pt')
   env=environment(repo,old,spec['bundle'],case,policy='aic_example_policies.ros.'+policy,domain=a.domain,actor=actor,recovery='none')
   aux=json.loads((prior/'frozen/selection.json').read_text())['auxiliary']
   env.update(AIC_TRANSPORT_EVAL_GPU=str(a.gpu),AIC_TRANSPORT_EVAL_POLICY_GPU=str(a.gpu),AIC_TRANSPORT_EVAL_ORACLE_CONTACTS='false',AIC_RULE_TRANSPORT_TEMPLATE=str(prior/'frozen/transport_spec.json'),AIC_LOCKED_GRASP_CHECKPOINT=aux['sc_grasp'],AIC_SFP_GRASP_CHECKPOINT=aux['sfp_grasp'],AIC_SFP_GRASP_APPROVAL=aux['sfp_grasp_approval'],AIC_INSERTION_EXECUTE_STEPS='4')
  launcher='run_chunked_insertion_eval.sh' if a.viewpoint=='near' and not case.get('pose_only_reset_hold') else 'run_shared_recovery_eval.sh'
  cmd=['bash',str(repo/'scripts'/launcher),case['case'],str(repo/'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt'),str(run)]
  save(a.output/('prefill_current_job.json' if a.prefill else 'current_job.json'),dict(case=case,audit=str(run)));save(a.output/(case['case']+'.command.json'),dict(command=cmd,environment={k:v for k,v in env.items() if k.startswith('AIC_')}))
  retained=(run/'runtime_trace.jsonl').exists() and any(json.loads(l)['event']=='grasp_capture_terminal' for l in (run/'runtime_trace.jsonl').open()) and bool(list((run/'results').glob('bag_*/metadata.yaml'))) and case['case'] not in invalid_cases
  teardown_failed=(run/'capture_teardown_issue.json').exists()
  if not retained:
   if run.exists():
    index=1
    while run.with_name(run.name+'_retry'+str(index)).exists():index+=1
    run=run.with_name(run.name+'_retry'+str(index));cmd[-1]=str(run)
   with (a.output/(run.name+'.log')).open('w') as log:finished=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
   teardown_failed=finished.returncode!=0
   retained=(run/'runtime_trace.jsonl').exists() and any(json.loads(l)['event']=='grasp_capture_terminal' for l in (run/'runtime_trace.jsonl').open()) and bool(list((run/'results').glob('bag_*/metadata.yaml')))
   if not retained:raise RuntimeError('Capture lacks complete retained hold/bag; diagnose '+str(run))
   if teardown_failed:save(run/'capture_teardown_issue.json',dict(launcher_returncode=finished.returncode,perception_terminal_retained=True,BC_expert=False,decision='Attempt independent causal label QA; do not count as rollout success'))
  elif not (run/'results/scoring.yaml').exists():
   teardown_failed=True;save(run/'capture_teardown_issue.json',dict(resumed_after_completed_hold=True,official_score_available=False,BC_expert=False))
  prefix='source .pixi/envs/default/setup.bash\nexport PYTHONPATH=/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/host_reader_deps:"${PYTHONPATH:-}"\n'
  cmd=[repo/'.pixi/envs/default/bin/python',repo/'scripts/extract_perception_hold_stream.py',run,run/'scored_stream.json']
  with (run/'extraction.log').open('w') as log:subprocess.run(['bash','-c',prefix+shlex.join(list(map(str,cmd)))],stdout=log,stderr=subprocess.STDOUT,check=True)
  trace=[json.loads(l) for l in (run/'runtime_trace.jsonl').open()];terminal=next(t for t in trace if t['event']=='grasp_capture_terminal');state=np.asarray(terminal['state'])
  # Pose-label QA runs on the first six before further collection.
  verify=[repo/'.pixi/envs/default/bin/python',repo/'scripts/label_sc_grasp_capture.py','--run',run,'--case',a.output/(case['case']+'.case.json')]
  save(a.output/(case['case']+'.case.json'),case)
  with (run/'labels.log').open('w') as log:subprocess.run(['bash','-c',prefix+shlex.join(list(map(str,verify)))],stdout=log,stderr=subprocess.STDOUT,check=True)
  label_report=json.loads((run/'grasp_labels_summary.json').read_text());valid=label_report['samples']>=case.get('minimum_pose_samples',6) and np.linalg.norm(state[26:29])<40
  results.append(dict(case=case,audit=str(run),samples=label_report['samples'],valid=bool(valid),teardown_issue=teardown_failed,official_score_available=(run/'results/scoring.yaml').exists()))
  save(a.output/('prefill_progress.json' if a.prefill else 'progress.json'),dict(completed=len(results),planned=len(cases),results=results))
  if not a.prefill:save(a.output/'state.json',dict(stage='collecting',collected=len(results),planned=len(cases)))
  if not valid:raise RuntimeError('Capture failed pose-label/force QA: '+str(run))
  if len(results)==a.qa_count and not a.prefill:
   save(a.output/'state.json',dict(stage='waiting_for_first_six_visual_QA',collected=len(results)))
   while not (a.output/a.qa_file).exists():time.sleep(10)
   if not json.loads((a.output/a.qa_file).read_text())['passed']:raise RuntimeError('Visual label QA rejected')
 save(a.output/('prefill_summary.json' if a.prefill else 'summary.json'),dict(completed=len(results),results=results,BC_expert=False,pose_only=True,reserved_final_scenes_opened=False))
 if not a.prefill:save(a.output/'state.json',dict(stage='complete',collected=len(results)))


if __name__=='__main__':main()
