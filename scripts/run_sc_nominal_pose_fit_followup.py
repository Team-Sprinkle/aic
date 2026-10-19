#!/usr/bin/env python3
"""After capture QA, merge immutable labels and fit both pose candidates to plateau."""
import argparse,json,os,shlex,subprocess,time
from pathlib import Path


def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--captures',type=Path,required=True);a=p.parse_args();repo=Path.cwd();python=repo/'.pixi/envs/default/bin/python'
 state=a.root/'nominal_pose_followup_state.json'
 if state.exists():raise FileExistsError('Do not duplicate the pose followup')
 save(state,dict(stage='waiting_for_verified_captures',capture_root=str(a.captures)))
 while not (a.captures/'summary.json').exists():time.sleep(20)
 source=a.root/'sc_nominal_coverage_v3';cmd=[python,repo/'scripts/merge_sc_grasp_capture_supervision.py','--source',a.root/'sc_grasp_development_coverage_v2','--captures',a.captures,'--output',source]
 save(a.root/'sc_nominal_coverage_v3.command.json',dict(command=list(map(str,cmd))))
 with (a.root/'sc_nominal_coverage_v3.log').open('w') as log:subprocess.run(list(map(str,cmd)),stdout=log,stderr=subprocess.STDOUT,check=True)
 save(state,dict(stage='training_to_lr_floor',source=str(source)))
 processes=[]
 for name,gpu,rotation in [('sc_nominal_joint_calibrated_v3',3,False),('sc_nominal_orientation_calibrated_v3',4,True)]:
  cmd=[python,repo/'scripts/train_connector_orientation_to_plateau.py','--source',source,'--pixels',source/'native_crops.npy','--checkpoint','/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/sc_grasp_joint_geometry_v6_continuation/joint_geometry/best.pt','--output',a.root/name,'--family','sc','--crop-jitter','0','--balance-new-angles','--updates','100000','--minutes','1440','--min-updates','5000','--patience','3000']
  if rotation:cmd.append('--rotation-only')
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');save(a.root/(name+'.command.json'),dict(command=list(map(str,cmd)),physical_gpu=gpu,complete_episode_reset_split=True,stopping_rule='Validation plateau after LR reductions to1e-6; time/update ceilings are incomplete training.'))
  log=(a.root/(name+'.log')).open('w');processes.append((name,subprocess.Popen(list(map(str,cmd)),env=env,stdout=log,stderr=subprocess.STDOUT),log))
 outcomes=[]
 for name,process,log in processes:
  code=process.wait();log.close()
  if code:raise RuntimeError(name+' failed; preserve and diagnose logs')
  d=json.loads((a.root/name/'summary.json').read_text());outcomes.append(dict(name=name,plateau=d['plateau_reached'],metrics=d['metrics'],expanded_validation=d.get('expanded_development_validation'),gate_passed=d['gate_passed']))
 save(state,dict(stage='pose_candidates_complete_need_component_review',results=outcomes,automatic_deployment=False));print(json.dumps(outcomes),flush=True)


if __name__=='__main__':main()
