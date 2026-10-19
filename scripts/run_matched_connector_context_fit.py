#!/usr/bin/env python3
"""Matched context-only BC comparison; preserve the existing visual estimators."""
import argparse,json,os,subprocess,time
from pathlib import Path


def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();repo=Path.cwd();old=Path('/home/nas/chmin/aic_archives/2026-10-17_connector_orientation_and_sfp_oscillation_v1');base=json.loads((old/'full_connector_bc.command.json').read_text())['command'];processes=[]
 state=a.root/'matched_controller_context_fit_state.json'
 if state.exists():raise FileExistsError('Comparison already scheduled')
 save(state,dict(stage='training',hypothesis='Public previous-reference error may preserve compliance compensation and reduce lateral drift. Offline persistence is already strong; promotion requires live advancement/insertion.',dataset=str(old/'full_connector_actor_rows'),new_data=False,matched_initialization=str(old/'full_connector_bc_h4/best.pt')))
 for name,gpu,masked in [('context_masked_current_pose_v1',3,True),('context_public_current_pose_v1',4,False)]:
  # The initial balanced orientation fit retains GPU4 until it converges.
  if gpu==4:
   while not (a.root/'sc_orientation_balanced_v2/summary.json').exists():time.sleep(10)
  cmd=list(base);cmd[1]=str(repo/'scripts/train_full_connector_context_to_plateau.py');cmd.remove('--reset-action-head')
  for flag,value in [('--output',str(a.root/name)),('--min-updates','5000'),('--stale-evaluations','30'),('--updates','100000'),('--minutes','1440'),('--workers','2')]:cmd[cmd.index(flag)+1]=value
  if not masked:cmd.remove('--mask-command-feedback')
  cmd+=['--resume-local',str(old/'full_connector_bc_h4/best.pt')]
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');save(a.root/(name+'.command.json'),dict(command=cmd,physical_gpu=gpu,masked=masked,stopping_rule='At least5000updates, then validation plateaus through LR reductions to1e-6. A ceiling is not convergence.'))
  log=(a.root/(name+'.log')).open('w');processes.append((name,subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT),log))
 results=[]
 for name,process,log in processes:
  code=process.wait();log.close()
  if code:raise RuntimeError(name+' failed; inspect its preserved log')
  d=json.loads((a.root/name/'summary.json').read_text());results.append(dict(name=name,stop_reason=d['stop_reason'],best_step=d['best_step'],updates_completed=d['updates_completed'],selected_validation=d['selected_validation']))
 save(state,dict(stage='complete_needs_live_comparison',results=results,automatic_promotion=False))


if __name__=='__main__':main()
