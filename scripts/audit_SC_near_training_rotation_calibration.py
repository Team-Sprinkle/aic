#!/usr/bin/env python3
"""Bounded old-RGB near calibration baseline: fit TRAIN episodes, score heldout."""
import cv2
cv2.setNumThreads(2)
from pathlib import Path
import json,sys,hashlib
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from native_connector_pose_observation import NativeConnectorPoseObservation
from shared_port_tcp_geometry import matrix_from_pose9
root=Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1');out=root/'near_training_rotation_calibration';out.mkdir(exist_ok=False)
source=root/'far_and_near_pose_dataset';rows=json.loads((source/'manifest.json').read_text());truth=np.stack([matrix_from_pose9(x) for x in np.load(source/'labels.npy')]);ids=np.array([i for i,r in enumerate(rows) if r.get('source_role','').startswith('Near25mm')]);cache=np.load(source/'native_crops.npy',mmap_mode='r');checkpoint=Path('/home/nas/chmin/aic_archives/2026-10-17_connector_orientation_and_sfp_oscillation_v1/sc_orientation_jitter_v2/best.pt');model=NativeConnectorPoseObservation(checkpoint,'cpu');torch.set_num_threads(4)
plan=json.loads((root/'far_near_spatial_joint/plan.json').read_text());assert np.array_equal(model.origins,plan['crop_origins_native']),'Original-image crops must match exactly'
pred=[]
with torch.inference_mode():
 for start in range(0,len(ids),24):
  x=torch.from_numpy(np.array(cache[ids[start:start+24]])).permute(0,1,4,2,3);_,rotation=model.model(x);pred.extend(rotation.numpy())
pred=np.array(pred);train=np.array([j for j,i in enumerate(ids) if rows[int(i)]['split']=='train']);val=np.array([j for j,i in enumerate(ids) if rows[int(i)]['split']=='validation']);residual=[]
for ep in sorted({rows[int(ids[j])]['episode'] for j in train}):
 chosen=np.array([j for j in train if rows[int(ids[j])]['episode']==ep]);residual.append(Rotation.from_matrix(pred[chosen].transpose(0,2,1)@truth[ids[chosen],:3,:3]).mean().as_matrix())
calibration=Rotation.from_matrix(np.stack(residual)).mean().as_matrix();corrected=pred@calibration

def score(r):
 delta=truth[ids[val],:3,:3].transpose(0,2,1)@r;error=np.degrees(Rotation.from_matrix(delta).magnitude());tilt=np.degrees(np.arccos(np.clip(delta[:,2,2],-1,1)));return dict(orientation_p50_p95_deg=np.percentile(error,[50,95]).tolist(),axial_tilt_p50_p95_deg=np.percentile(tilt,[50,95]).tolist())
mean=Rotation.from_matrix(truth[ids[train],:3,:3]).mean().as_matrix();report=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),training_episodes=sorted({rows[int(ids[j])]['episode'] for j in train}),validation_episodes=sorted({rows[int(ids[j])]['episode'] for j in val}),right_rotation_calibration=calibration.tolist(),calibration_rotvec_deg=np.degrees(Rotation.from_matrix(calibration).as_rotvec()).tolist(),fit='Episode-balanced mean of predictedR.T@trueR on8trainingnear episodes only',raw=score(pred[val]),calibrated=score(corrected[val]),constant_near_train_mean=score(np.broadcast_to(mean,(len(val),3,3))),geometry_input_at_runtime=False,final_scenes_opened=False)
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');np.save(out/'predictions.npy',pred);np.save(out/'indices.npy',ids);print(json.dumps(report),flush=True)
