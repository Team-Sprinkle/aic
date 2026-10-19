#!/usr/bin/env python3
"""Audit relative pixel/pose loss units and actual landmark errors; no training."""
import cv2
cv2.setNumThreads(2)
import json
from pathlib import Path
import numpy as np
import torch
from sc_spatial_grasp_observation import SCSpatialGraspObservation
from shared_port_tcp_geometry import matrix_from_pose9
from evaluate_sc_native_triangulation import FOCAL
root=Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1');out=root/'spatial_loss_unit_audit';out.mkdir(exist_ok=True)
fit=root/'far_near_spatial_joint';plan=json.loads((fit/'plan.json').read_text());source=Path(plan['source']);rows=json.loads((source/'manifest.json').read_text());pixels=np.load(source/'native_crops.npy',mmap_mode='r');truth=np.stack([matrix_from_pose9(x) for x in np.load(source/'labels.npy')]);model=SCSpatialGraspObservation(fit/'best.pt','cpu');torch.set_num_threads(4);origins=np.stack([np.rint(np.asarray(model.centres[c])-96) for c in model.cameras]);reports={}
for split in ['train','validation']:
 ids=np.array([i for i,x in enumerate(rows) if x['split']==split and x.get('source_role','').startswith('Near25mm')]);pred=[]
 with torch.inference_mode():
  for start in range(0,len(ids),24):
   chosen=ids[start:start+24];x=torch.from_numpy(np.array(pixels[chosen])).permute(0,1,4,2,3).reshape(-1,3,192,192).float()/255;pred.append(model.model(x,camera_id=torch.eye(3).repeat(len(chosen),1)).numpy().reshape(-1,3,5,2)*191+origins[None,:,None,:])
 pred=np.concatenate(pred);cloud=model.cad@truth[ids,:3,:3].transpose(0,2,1)+truth[ids,None,:3,3];target=[]
 for c in model.cameras:
  inv=np.linalg.inv(model.optics[c]);q=cloud@inv[:3,:3].T+inv[:3,3];target.append(q[:,:,:2]/q[:,:,2:]*FOCAL+[576,512])
 target=np.stack(target,1);norm=np.linalg.norm(pred-target,axis=-1);delta=np.abs(pred-target);current=.2*np.mean(np.where(delta<1,.5*delta**2,delta-.5));balanced=np.abs(pred-target)/.5;proposed=.2*np.mean(np.where(balanced<1,.5*balanced**2,balanced-.5));reports[split]=dict(samples=len(ids),episodes=len({rows[int(i)]['episode'] for i in ids}),landmark_error_p50_p95_px=np.percentile(norm,[50,95]).tolist(),actual_current_weighted_pixel_loss=float(current),same_pixels_balanced_half_px_loss=float(proposed))
result=dict(one_px_current_weighted_pixel_loss=.2*.5,one_px_balanced_half_px_loss=.2*1.5,one_degree_pose_loss_approximately=1.,reports=reports,diagnosis='Actualtrainer already multiplies normalizedresidualby191; pixel loss is correctly in nativepixels. Earlier audit used the wrong formula and iswithdrawn. No unitbug demonstrated.',hypothesis_not_measured_improvement=True,training=False,previous_wrong_formula_audit_withdrawn=True,actual_trainer_pixel_contract='smoothL1((prediction-target)*191,0,beta=1)')
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
