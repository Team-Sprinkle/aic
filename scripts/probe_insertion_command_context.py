#!/usr/bin/env python3
"""Offline diagnostic only: does causal controller-error context disambiguate labels?"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from shared_port_tcp_geometry import matrix_from_pose9,matrix_from_xyz_xyzw


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8])
 source=Path('/home/nas/chmin/aic_archives/2026-10-17_connector_orientation_and_sfp_oscillation_v1/full_connector_actor_rows/rows.jsonl')
 features=[];targets=[];splits=[];families=[];near=[]
 for line in source.open():
  r=json.loads(line);state=np.asarray(r['state']);pose=np.asarray(r['predicted_port_tcp_pose9']);tip=np.asarray(r['predicted_tip_port_pose9']);task=np.asarray(r['task'])
  feedback=state[13:19].copy();state=state.copy();state[13:19]=0
  port_base=matrix_from_pose9(pose)[:3,:3]@matrix_from_xyz_xyzw(state[:7])[:3,:3].T
  context=np.r_[feedback,port_base@feedback[:3],port_base@feedback[3:]]
  base=np.r_[state,pose,tip,task]
  features.append(np.r_[base,context]);targets.append((np.asarray(r['teacher_tcp_targets_predicted_port_pose9'])[0,:3]-tip[:3])*1000)
  splits.append(r['split']);families.append(r['family']);near.append(-.04<=tip[2]<=0)
 x=np.asarray(features);y=np.asarray(targets);fit=np.asarray(splits)=='train';family=np.asarray(families);near=np.asarray(near);reports={}
 for name,features in [('masked',x[:,:-12]),('with_controller_error',x)]:
  mean=features[fit].mean(0);scale=features[fit].std(0).clip(.0001)
  z=(features-mean)/scale;z=np.c_[z,np.ones(len(z))]
  regularizer=np.eye(z.shape[1])*10;regularizer[-1,-1]=0
  weights=np.linalg.solve(z[fit].T@z[fit]+regularizer,z[fit].T@y[fit]);pred=z@weights
  reports[name]={}
  for f in ['sc','sfp']:
   ids=(~fit)&(family==f)&near;err=pred[ids]-y[ids]
   reports[name][f]=dict(samples=int(ids.sum()),lateral_command_error_mm_p50_p95=np.percentile(np.linalg.norm(err[:,:2],axis=1),[50,95]).tolist(),position_command_error_mm_p50_p95=np.percentile(np.linalg.norm(err,axis=1),[50,95]).tolist())
 a.output.write_text(json.dumps(dict(reports=reports,source=str(source),diagnostic_only=True,policy_architecture_unchanged=True,split='Existing complete episode/reset train/validation',caveat='Current tcp_error is a public controller observation. Better offline regression is not evidence of closed-loop insertion and does not by itself prove absence of timestamp leakage.'),indent=2)+'\n')
 print(json.dumps(reports),flush=True)


if __name__=='__main__':main()
