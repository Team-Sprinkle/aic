#!/usr/bin/env python3
"""Rank bounded angular retries using training-only frozen-estimator errors.

This is an uncertainty-search prior, not a correction chosen from evaluation
geometry. Labels remain independent; the prior is tied to an estimator hash.
"""
import cv2
cv2.setNumThreads(2)
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from native_connector_pose_observation import NativeConnectorPoseObservation
from shared_port_tcp_geometry import matrix_from_pose9

torch.set_num_threads(4)
root=Path('/home/nas/chmin/aic_archives/2026-10-18_three_trial_repair_cycle_v1');source=Path('/home/nas/chmin/aic_archives/2026-10-17_insertion_control_repair_v1/sc_nominal_coverage_v3');checkpoint=Path('/home/nas/chmin/aic_archives/2026-10-17_connector_orientation_and_sfp_oscillation_v1/sc_orientation_jitter_v2/best.pt');model=NativeConnectorPoseObservation(checkpoint,'cpu');rows=json.loads((source/'manifest.json').read_text());pixels=np.load(source/'native_crops.npy',mmap_mode='r');seen=set();ids=[]
for i,r in enumerate(rows):
 if r['split']=='train' and r['episode'] not in seen:ids.append(i);seen.add(r['episode'])
rotations=[]
with torch.inference_mode():
 for begin in range(0,len(ids),16):
  batch=torch.from_numpy(np.array(pixels[ids[begin:begin+16]],copy=True)).permute(0,1,4,2,3);_,rr=model.model(batch);rotations.extend(rr.numpy())
errors=[];domains=[];records=[]
for i,predicted in zip(ids,rotations):
 row=rows[i];truth=matrix_from_pose9(row['tip_in_tcp_pose9'])[:3,:3]
 e=np.rad2deg(Rotation.from_matrix(predicted.T@truth).as_rotvec());role=row.get('source_role','');domain='nominal_home' if role.startswith('Nominal-centred') else 'nominal_acquisition' if role else 'original';errors.append(e);domains.append(domain);records.append(dict(episode=row['episode'],scene=row['scene'],split=row['split'],domain=domain,rotation_error_in_predicted_tip_axes_deg=e.tolist()))
errors=np.asarray(errors);groupstats={}
for domain in sorted(set(domains)):
 e=errors[np.array(domains)==domain];groupstats[domain]=dict(episodes=len(e),mean_deg=e.mean(0).tolist(),second_moment_deg2=np.mean(e**2,axis=0).tolist())
# Give each intended domain equal influence rather than letting the largest
# legacy dataset hide the newly measured nominal-grasp error distribution.
moments=np.mean([g['second_moment_deg2'] for g in groupstats.values()],axis=0);mean=np.mean([g['mean_deg'] for g in groupstats.values()],axis=0);order=np.argsort(-moments);proposals=[]
for axis in order:
 first=-1 if mean[axis]>0 else 1
 for sign in [first,-first]:
  v=np.zeros(3);v[axis]=sign*2.5;proposals.append(v.tolist())
report=dict(schema='training_only_SC_angular_search_prior/v1',source=str(source),manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),estimator=str(checkpoint),estimator_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),training_episodes=len(ids),validation_or_evaluation_episodes_used=0,records=records,domain_statistics=groupstats,domain_balanced_RMS_deg=np.sqrt(moments).tolist(),domain_balanced_mean_deg=mean.tolist(),axis_order=[int(v) for v in order],proposals_rotvec_deg=proposals,amplitude_deg=2.5,proposal_bound_and_number_unchanged=True,geometry_runtime_input=False,model_predictions_not_ground_truth_labels=True,reason='Prioritize large estimator-error axes/signs from training-only residuals to avoid wasting the task budget. No current episode true geometry selects a retry.',new_actor_training=False)
(root/'SC_orientation_search_prior_training.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['training_episodes','domain_statistics','domain_balanced_RMS_deg','domain_balanced_mean_deg','proposals_rotvec_deg']}),flush=True)
