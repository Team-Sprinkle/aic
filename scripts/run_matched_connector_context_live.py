#!/usr/bin/env python3
"""Follow converged matched fits with frozen autonomous development rollouts."""
import argparse,json,subprocess,time
from pathlib import Path


def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--captures',type=Path,required=True);a=p.parse_args();repo=Path.cwd();output=a.root/'context_live_current_pose_v1';output.mkdir(exist_ok=False)
 save(output/'scheduler_state.json',dict(stage='waiting_for_training_and_capture_GPU_release'))
 names=['context_masked_current_pose_v1','context_public_current_pose_v1']
 while not all((a.root/n/'summary.json').exists() for n in names) or not (a.captures/'summary.json').exists():time.sleep(20)
 for n in names:
  d=json.loads((a.root/n/'summary.json').read_text())
  if d['stop_reason']!='validation_plateau' or d['final_lr']>1.00001e-6:raise RuntimeError('Incomplete convergence; resume before evaluation: '+n)
 cases=json.loads(Path('/home/nas/chmin/aic_archives/2026-10-17_insertion_control_repair_v2/evaluation_cases.json').read_text())['cases']
 save(output/'evaluation_cases.json',dict(cases=cases,scope='Repeated already exposed development configurations, not a new final/generalization split'))
 save(output/'evaluation_variants.json',dict(variants=[dict(name=n,checkpoint=str(a.root/n/'best.pt'),execute=4) for n in names]))
 cmd=[repo/'.pixi/envs/default/bin/python',repo/'scripts/run_consistent_connector_probe.py','--root',output,'--output',output/'runs'];save(output/'command.json',dict(command=list(map(str,cmd)),no_guide=True,recovery='none',scoring_truth_posthoc_only=True))
 save(output/'scheduler_state.json',dict(stage='running_frozen_eight_rollout_comparison',models=names))
 with (output/'runner.log').open('w') as log:subprocess.run(list(map(str,cmd)),stdout=log,stderr=subprocess.STDOUT,check=True)
 d=json.loads((output/'runs/summary.json').read_text());save(output/'scheduler_state.json',dict(stage='complete_need_physical_review',full=d['full'],total=d['completed'],automatic_promotion=False))


if __name__=='__main__':main()
