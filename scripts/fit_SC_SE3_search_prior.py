#!/usr/bin/env python3
"""Training-only SE(3) calibration candidates for a frozen RGB grasp pipeline."""
import cv2
cv2.setNumThreads(2)
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from sc_native_grasp_observation import SCNativeGraspObservation
from train_sfp_native_grasp_keypoints import poses_from_pixels
from shared_port_tcp_geometry import matrix_from_pose9

torch.set_num_threads(4)
root=Path('/home/nas/chmin/aic_archives/2026-10-18_three_trial_repair_cycle_v1');prior=json.loads((root/'SC_orientation_search_prior_training.json').read_text());source=Path(prior['source']);rows=json.loads((source/'manifest.json').read_text());pixels=np.load(source/'native_crops.npy',mmap_mode='r');episodes={}
for i,r in enumerate(rows):
 if r['split']=='train' and r['episode'] not in episodes:episodes[r['episode']]=i
records=prior['records'];ids=[episodes[r['episode']] for r in records];selection=json.loads(Path('/home/nas/chmin/aic_archives/2026-10-16_full_sample_insertion_evaluation_v1/frozen/selection.json').read_text())['auxiliary'];checkpoint=Path(selection['sc_grasp']);model=SCNativeGraspObservation(checkpoint,'cpu');origins=np.stack([np.rint(np.asarray(model.centres[c])-96) for c in model.cameras]);coords=[]
reference_origins=np.asarray(json.loads(Path('/home/nas/chmin/aic_archives/2026-10-17_insertion_control_repair_v1/sc_spatial_grasp_pixel_v6/plan.json').read_text())['crop_origins_native']);assert np.array_equal(origins,reference_origins)
with torch.inference_mode():
 for begin in range(0,len(ids),16):
  ii=ids[begin:begin+16];x=torch.from_numpy(np.array(pixels[ii],copy=True)).permute(0,1,4,2,3).reshape(-1,3,192,192).float()/255
  coords.append(model.model(x,camera_id=torch.eye(3).repeat(len(ii),1)).numpy().reshape(-1,3,5,2)*191+origins[None,:,None,:])
translations=poses_from_pixels(np.vstack(coords),model.optics,model.cad)[:,:3,3];grouped={}
for i,record,position in zip(ids,records,translations):
 row=rows[i];assert row['split']=='train';truth=matrix_from_pose9(row['tip_in_tcp_pose9']);e=Rotation.from_rotvec(np.deg2rad(record['rotation_error_in_predicted_tip_axes_deg'])).as_matrix();estimated_rotation=truth[:3,:3]@e.T
 delta=estimated_rotation.T@(truth[:3,3]-position)
 grouped.setdefault(record['domain'],[]).append(delta)
statistics={domain:dict(episodes=len(v),mean_translation_error_predicted_tip_m=np.mean(v,axis=0).tolist(),p95_translation_norm_mm=float(np.percentile(np.linalg.norm(v,axis=1)*1000,95))) for domain,v in grouped.items()}
candidates=[]
for domain in ['nominal_acquisition','nominal_home']:
 rv=-np.array(prior['domain_statistics'][domain]['mean_deg']);rotation=Rotation.from_rotvec(np.deg2rad(rv)).as_matrix();translation=-rotation@np.array(statistics[domain]['mean_translation_error_predicted_tip_m']);candidates.append(dict(rotation_rotvec_deg=rv.tolist(),translation_m=translation.tolist(),source='inverse_training_mean_grasp_SE3_error/'+domain))
for axis in [0,1]:
 for sign in [1,-1]:
  base=candidates[0];translation=np.array(base['translation_m']);translation[axis]+=sign*.0004;candidates.append(dict(rotation_rotvec_deg=base['rotation_rotvec_deg'],translation_m=translation.tolist(),source='acquisition_mean_plus_bounded_0p4mm_lateral_probe'))
assert len(candidates)==6 and all(np.linalg.norm(c['rotation_rotvec_deg'])<=3 and np.linalg.norm(c['translation_m'])<=.002 for c in candidates)
report=dict(schema='SC_training_SE3_uncertainty_search/v1',rotation_estimator_sha256=prior['estimator_sha256'],position_estimator_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),training_episodes=len(ids),validation_or_evaluation_episodes_used=0,statistics=statistics,candidates=candidates,geometry_runtime_input=False,source='Frozen RGB grasp errors evaluated only on complete training episodes. C approximately equals inverse(mean(G_est^-1 G_true)); lateral offsets remain the insertion goal instead of being recentered away after retry.',new_actor_training=False)
(root/'SC_SE3_search_prior_runtime.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
