#!/usr/bin/env python3
"""Snapshot recorded experiment evidence without reading model/training logs."""
import argparse,json
from pathlib import Path


def read(path):return json.loads(path.read_text()) if path.exists() else None


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;fits=[]
 for path in sorted(r.glob('*/summary.json')):
  d=read(path)
  if 'stop_reason' not in d:continue
  fits.append(dict(name=path.parent.name,stop_reason=d['stop_reason'],updates=d.get('updates_completed'),best_step=d.get('best_step'),converged=d['stop_reason']=='validation_plateau' and d.get('final_lr',1)>1e-6-1e-12 and d.get('final_lr',1)<=1.00001e-6,metrics=d.get('metrics'),expanded_validation=d.get('expanded_development_validation'),selected_validation=d.get('selected_validation'),promoted=False))
 captures=Path('/home/nas/chmin/aic_archives/2026-10-17_sc_nominal_grasp_capture_v1');progress=read(captures/'progress.json');rollouts=[]
 for name,path in [('control_v1',r/'runs'),('control_v2',Path('/home/nas/chmin/aic_archives/2026-10-17_insertion_control_repair_v2/runs')),('context_current_pose',r/'context_live_current_pose_v1/runs'),('visual_servo_unwarmed_not_BC',r/'visual_servo_current_pose_v2/runs'),('visual_servo_warmed_not_BC',r/'visual_servo_current_pose_v3/runs')]:
  d=read(path/'summary.json') or read(path/'progress.json')
  if d:rollouts.append(dict(name=name,completed=d['completed'],strict_full=sum(x['physical_full_insertion'] for x in d['results']),finished=(path/'summary.json').exists(),artifacts=str(path),promoted=False))
 result=dict(schema='insertion_repair_evidence/v1',frozen_sample=dict(score=165.418019,maximum=300,strict_full=0,episodes=3,artifact_root='/home/nas/chmin/aic_archives/2026-10-17_full_connector_sample_configuration_v2'),fits=fits,rollouts=rollouts,capture_state=read(captures/'state.json'),capture_rows=sum(x['samples'] for x in progress['results']) if progress else 0,capture_teardown_issues=sum(bool(x.get('teardown_issue')) for x in progress['results']) if progress else 0,perception_capture_not_BC=True,context_training=read(r/'matched_controller_context_fit_state.json'),pose_followup=read(r/'nominal_pose_followup_state.json'),pose_candidate_jobs={n:('finished' if (r/n/'summary.json').exists() else 'training' if (r/n/'plan.json').exists() else 'waiting_for_assigned_GPU' if r.joinpath(n).with_suffix('.converged_trainer_snapshot.py').exists() else 'not_started') for n in ['sc_nominal_joint_calibrated_v3','sc_nominal_orientation_calibrated_v3']},context_live=read(r/'context_live_current_pose_v1/scheduler_state.json'),visual_servo=read(r/'visual_servo_current_pose_v1/scheduler_state.json'),step5_open=True,final_scenes_opened=False,new_RL=False,convergence='Validation plateau after learning-rate reductions to1e-6; an update/time ceiling is incomplete, and convergence is not a global-optimum guarantee.')
 (r/'repair_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(finished_fits=len(fits),rollouts=rollouts,captures=result['capture_state'])),flush=True)


if __name__=='__main__':main()
