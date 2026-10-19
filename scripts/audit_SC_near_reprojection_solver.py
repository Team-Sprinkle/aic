#!/usr/bin/env python3
"""One fixed RGB-landmark solver ablation; never optimize against pose labels."""
from pathlib import Path
import json,time,sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from train_sfp_native_grasp_keypoints import metrics
from shared_port_tcp_geometry import matrix_from_pose9
from evaluate_sc_native_triangulation import FOCAL
root=Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1');output=root/'near_reprojection_solver_audit';output.mkdir(exist_ok=False);reports={}
for name in ['far_near_spatial_joint','compact_far_near_spatial_joint']:
 fit=root/name;plan=json.loads((fit/'plan.json').read_text());rows=json.loads((Path(plan['source'])/'manifest.json').read_text());ids=np.load(fit/'heldout_indices.npy');keep=np.array([j for j,i in enumerate(ids) if rows[int(i)].get('source_role','').startswith('Near25mm')]);pixels=np.load(fit/'heldout_pixel_predictions.npy')[keep];poses=np.load(fit/'heldout_pose_predictions.npy')[keep];truth=np.stack([matrix_from_pose9(x) for x in np.load(Path(plan['source'])/'labels.npy')])[ids[keep]];cad=np.asarray(plan['CAD_reference_points']);inverses=np.linalg.inv(np.stack([plan['tcp_to_optical'][c] for c in ['center','left','right']]));refined=[];diagnostics=[]
 for old,observed in zip(poses,pixels):
  def candidate(x):
   pose=old.copy();pose[:3,3]+=x[:3]*.001;pose[:3,:3]=Rotation.from_rotvec(np.deg2rad(x[3:])).as_matrix()@old[:3,:3];return pose
  def residual(x):
   pose=candidate(x);cloud=cad@pose[:3,:3].T+pose[:3,3];q=np.einsum('cij,kj->cki',inverses[:,:3,:3],cloud)+inverses[:,:3,3][:,None,:];return (q[:,:,:2]/q[:,:,2:]*FOCAL+[576,512]-observed).reshape(-1)
  start=time.perf_counter();before=residual(np.zeros(6));result=least_squares(residual,np.zeros(6),bounds=([-2]*3+[-5]*3,[2]*3+[5]*3),loss='soft_l1',f_scale=1.,max_nfev=50,ftol=1e-9,xtol=1e-9,gtol=1e-9);refined.append(candidate(result.x));diagnostics.append(dict(duration_ms=(time.perf_counter()-start)*1000,success=bool(result.success),nfev=result.nfev,bound_hit=bool(np.max(np.abs(result.x[:3]))>1.99 or np.max(np.abs(result.x[3:]))>4.99),initial_reprojection_rms_px=float(np.sqrt(np.mean(before**2))),final_reprojection_rms_px=float(np.sqrt(np.mean(residual(result.x)**2)))))
 refined=np.stack(refined);reports[name]=dict(samples=len(keep),episodes=len({rows[int(ids[j])]['episode'] for j in keep}),baseline=metrics(poses,truth),refined=metrics(refined,truth),optimization_labels_used=False,optimizer='One predeclared all-five-landmark softL1 reprojection fit,1px scale,translation bounds2mm/rotationbounds5degrees',latency_p50_p95_ms=np.percentile([x['duration_ms'] for x in diagnostics],[50,95]).tolist(),diagnostics=diagnostics);np.save(output/(name+'_refined_poses.npy'),refined)
(output/'summary.json').write_text(json.dumps(dict(results=reports,GT_scoring_only=True,automatic_runtime_change=False),indent=2)+'\n');print(json.dumps({n:dict(baseline=d['baseline'],refined=d['refined']) for n,d in reports.items()}),flush=True)
