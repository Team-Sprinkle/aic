#!/usr/bin/env python3
"""Continue matched BC checkpoints into bounded autonomous development tests.

Wait without GPUs; cooperative launcher locks protect active simulators.
No teacher, ground-truth reset, object TF or oracle contacts guide decisions.
"""
import argparse,hashlib,json,shlex,shutil,subprocess,sys,time
from pathlib import Path
from shared_recovery_eval_environment import environment,effective_environment


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ros_command(repo,command,log):
 prefix='source .pixi/envs/default/setup.bash\nexport PYTHONPATH=/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/host_reader_deps:"${PYTHONPATH:-}"\n'
 with log.open('w') as f:subprocess.run(['bash','-c',prefix+' '.join(shlex.quote(str(x)) for x in command)],cwd=repo,stdout=f,stderr=subprocess.STDOUT,check=True)


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('root','scenes','perception-gate','output'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--fresh-folder',type=Path);p.add_argument('--visual-warm-folder',type=Path)
 p.add_argument('--matched-summary',type=Path)
 p.add_argument('--bc-only',action='store_true',help='Preserve a bounded post-convergence actor comparison before further recovery work')
 a=p.parse_args();repo=Path.cwd();py=repo/'.pixi/envs/default/bin/python'
 if a.output.exists():raise FileExistsError('Preserve prior autonomous comparison')
 a.output.mkdir(parents=True);plan=json.load((a.scenes/'plan.json').open());cases=plan['cases'];bundle=json.load((a.root/'shared_perception_bundle_approved_v1.json').open())
 save(a.output/'plan.json',dict(scene_plan=str(a.scenes/'plan.json'),scene_plan_sha256=digest(a.scenes/'plan.json'),budget='Two matched20-case BC arms then gated no/local/hierarchical comparison;120-case fixed selected coverage only after20-casefull-depthgate.',gate=plan['gates'],final_scenes_opened=False))
 waiting_started=time.monotonic()
 while True:
  gate_path=a.perception_gate/'summary.json';matched=a.matched_summary or a.root/'matched_shared_insertion_BC_summary_v1.json'
  if gate_path.exists() and not json.load(gate_path.open())['passed']:
   save(a.output/'decision.json',dict(status='perception_gate_failed',next='Diagnose bounded corrected acquisition failures before training/actorpromotion',evidence=str(gate_path),final_scenes_opened=False));return
  if gate_path.exists() and matched.exists():break
  if time.monotonic()-waiting_started>12*3600:raise RuntimeError('Prerequisite jobs exceeded12h; inspect without interrupting them')
  time.sleep(30)
 matched=json.load(matched.open());assert matched['matched_updates']
 checkpoints={};training={}
 for arm in ('fresh','visual_warm'):
  folder=(a.fresh_folder if arm=='fresh' else a.visual_warm_folder) or a.root/f'shared_insertion_{arm}_full_v1';summary=json.load((folder/'summary.json').open())
  target=a.output/f'{arm}_frozen_best.pt';shutil.copyfile(folder/'best.pt',target);checkpoints[arm]=target
  training[arm]=dict(updates=summary['updates_completed'],best_step=summary['best_step'],stale_evaluations=summary['stale_evaluations_at_end'],stop_reason=summary['stop_reason'],plateau_evidence=summary['stale_evaluations_at_end']>=40,best_sha256=digest(target))
 save(a.output/'frozen_training_lineage.json',training)
 if not all(x['updates']>=8000 for x in training.values()):raise RuntimeError('Insufficient matched training budget')
 selected_cases=[c for c in cases if c['domain']=='primary' and c['replicate']==0]
 # Both families' integration cases run first; remaining starts are fixed.
 selected_cases.sort(key=lambda c:(c['case'] not in plan['pilot_cases'],c['case']))
 results=[]
 def rollout(c,arm,recovery,actor):
  name=f"{c['case']}__{arm}__{recovery}";run=a.output/'runs'/name;run.parent.mkdir(exist_ok=True)
  if run.exists():raise FileExistsError('Do not duplicate partial rollout')
  casefile=run.with_suffix('.case.json');save(casefile,c)
  env=environment(repo,a.root,bundle,c,policy='aic_example_policies.ros.RunRuleThenSharedRecovery',domain='91',actor=actor,recovery=recovery)
  command=['bash',repo/'scripts/run_shared_recovery_eval.sh',c['case'],repo/'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt',run]
  save(run.with_suffix('.command.json'),dict(command=list(map(str,command)),environment={k:v for k,v in env.items() if k.startswith('AIC_')},actor_sha256=digest(actor)))
  with run.with_suffix('.log').open('w') as f:subprocess.run(list(map(str,command)),env=env,cwd=repo,stdout=f,stderr=subprocess.STDOUT,check=True)
  actual=effective_environment(run,env)
  ros_command(repo,[py,repo/'scripts/extract_moveit_training_stream.py',run,run/'scored_stream.json'],run/'extraction.log')
  ros_command(repo,[py,repo/'scripts/score_shared150mm_rollout.py','--run',run,'--case',casefile,'--arm',arm+'__'+recovery],run/'scoring.log')
  x=json.load((run/'shared150mm_metrics.json').open());x['effective_runtime_settings']=actual;results.append(x)
  # Preserve smooth overview+wrist views for each integration case and every
  # failure. The end is the retained pre-return stamp, not asset teardown.
  if c['case'] in plan['pilot_cases'] or not x['qualified']:
   with (run/'video.log').open('w') as f:subprocess.run([str(py),'scripts/render_transport_five_camera_video.py',str(run),'--fps','20','--grid','--active-only'],stdout=f,stderr=subprocess.STDOUT,check=True)
  save(a.output/'progress.json',dict(completed=len(results),results=results,final_scenes_opened=False))
  print(json.dumps(dict(case=c['case'],arm=arm,recovery=recovery,full=x['physical_full_insertion'],qualified=x['qualified'],actor_ran=x['actor_ran'])),flush=True)
  if x['geometry_runtime_input'] or x['raw_terminal']['no_scored_harm'] is not True or (x['actor_ran'] and not x['requested_port_locks_passed']):
   raise RuntimeError('Unsafe/privileged/wrong-portbranch: preserve and diagnose before continuing')
  return x
 for c in selected_cases:
  for arm in ('fresh','visual_warm'):rollout(c,arm,'none',checkpoints[arm])
 rates={arm:{f:sum(x['qualified'] for x in results if x['arm']==arm+'__none' and x['family']==f) for f in ('sc','sfp')} for arm in checkpoints}
 best=max(checkpoints,key=lambda arm:(min(rates[arm].values()),sum(rates[arm].values())))
 save(a.output/'matched_BC20_summary.json',dict(rates_out_of10_per_family=rates,selected=best,results=results,final_scenes_opened=False))
 if a.bc_only:return
 if not all(training[arm]['plateau_evidence'] for arm in training):
  save(a.output/'training_convergence_review.json',dict(passed=False,lineage=training,next='Preserve matched rollouts; extend both optimizers by a matched bounded budget if their late validation curves still improve. Updatecap is not convergence.'))
 # Do not use recovery to conceal a policy that cannot autonomously insert.
 if min(rates[best].values())<5:
  save(a.output/'decision.json',dict(status='BC_full_insertion_gate_open',rates=rates,next='Bounded phase/frame/command diagnosis using failed nativeRGB videos and pre-return force/geometry. No routine collection/retraining; noRLpromotion.',final_scenes_opened=False));return
 for c in selected_cases:
  for mode in ('local','hierarchical'):rollout(c,best,mode,checkpoints[best])
 recovery_rates={mode:{f:sum(x['qualified'] for x in results if x['arm']==best+'__'+mode and x['family']==f) for f in ('sc','sfp')} for mode in ('none','local','hierarchical')}
 winner=max(recovery_rates,key=lambda m:(min(recovery_rates[m].values()),sum(recovery_rates[m].values())))
 save(a.output/'recovery20_summary.json',dict(selected_actor=best,rates_out_of10_per_family=recovery_rates,selected_recovery=winner,hierarchical_benefit_claimed=(winner=='hierarchical' and sum(recovery_rates[winner].values())>sum(recovery_rates['local'].values()) and sum(recovery_rates[winner].values())>sum(recovery_rates['none'].values())),supervisor_is_RL=False,final_scenes_opened=False))
 if min(recovery_rates[winner].values())<9:
  save(a.output/'decision.json',dict(status='supervised_recovery_gate_open',next='Diagnose phase-specific failures before anyRLpromotion; preserve all matchedarms',rates=recovery_rates,final_scenes_opened=False));return
 for c in cases:
  if c['domain']=='primary' and c['replicate']==0:continue
  rollout(c,best,winner,checkpoints[best])
 save(a.output/'decision.json',dict(status='coverage_completed_requires_independent_review',selected_actor=best,selected_recovery=winner,results=results,next='Review per-cell >=5episode physicaldepth/lateral/force/orientation/latency gates and videos; noautomaticRL orfinalsceneaccess.',final_scenes_opened=False))
if __name__=='__main__':main()
