#!/usr/bin/env python3
"""Near-only fused RGB orientation qualification and prediction-only overlays."""
import cv2
cv2.setNumThreads(2)
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from native_connector_pose_observation import NativeConnectorPoseObservation
from sc_native_grasp_observation import SCNativeGraspObservation
from shared_port_tcp_geometry import matrix_from_pose9
from evaluate_sc_native_triangulation import FOCAL


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
 s=json.loads((a.fit/'summary.json').read_text());source=Path(s['plan']['source']);rows=json.loads((source/'manifest.json').read_text());truth=np.stack([matrix_from_pose9(x) for x in np.load(source/'labels.npy')]);ids=np.load(a.fit/'heldout_indices.npy');pred=np.load(a.fit/'heldout_pose_predictions.npy');reports={};domains={}
 for j,i in enumerate(ids):
  role=rows[int(i)].get('source_role','');domain='near_25mm' if role.startswith('Near25mm') else 'nominal_acquisition' if role.startswith('Predeadline') else 'nominal_home' if role else 'original';domains.setdefault(domain,[]).append(j)
 for domain,selected in domains.items():
  error=np.degrees((Rotation.from_matrix(truth[ids[selected],:3,:3]).inv()*Rotation.from_matrix(pred[selected,:3,:3])).magnitude());counts=Counter(rows[int(ids[j])]['episode'] for j in selected);weights=np.array([1/counts[rows[int(ids[j])]['episode']] for j in selected]);order=np.argsort(error);q=np.interp([.5,.95],np.cumsum(weights[order])/weights.sum(),error[order]);relative=truth[ids[selected],:3,:3].transpose(0,2,1)@pred[selected,:3,:3];tilt=np.degrees(np.arccos(np.clip(relative[:,2,2],-1,1)))
  reports[domain]=dict(episodes=len(counts),samples=len(selected),orientation_p50_p95_deg=np.percentile(error,[50,95]).tolist(),episode_balanced_orientation_p50_p95_deg=q.tolist(),axial_tilt_p50_p95_deg=np.percentile(tilt,[50,95]).tolist())
 model=NativeConnectorPoseObservation(a.fit/'best.pt','cpu');assert model.rotation_only
 translation_checkpoint=Path('/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/sc_grasp_translation_observation_v1.pt');position_model=SCNativeGraspObservation(translation_checkpoint,'cpu');checks=[]
 for episode in sorted({rows[int(ids[j])]['episode'] for j in domains['near_25mm']}):
  selected=[j for j in domains['near_25mm'] if rows[int(ids[j])]['episode']==episode];j=selected[len(selected)//2];row=rows[int(ids[j])];images={c:cv2.cvtColor(cv2.imread(row['images'][c]),cv2.COLOR_BGR2RGB) for c in ['center','left','right']};position=np.asarray(position_model.predict(images,already_training_jpeg=True)['predicted_tip_position_tcp_m']);result=model.predict(images,already_training_jpeg=True,visual_tip_position_tcp_m=position);rotation=np.asarray(result['orientation_for_diagnostic_only']);delta=float(np.degrees(Rotation.from_matrix(rotation.T@pred[j,:3,:3]).magnitude()));panels=[]
  for ci,camera in enumerate(['center','left','right']):
   left,top=model.origins[ci];patch=cv2.warpAffine(cv2.cvtColor(images[camera],cv2.COLOR_RGB2BGR),np.array([[1,0,-left],[0,1,-top]],np.float32),(192,192));cloud=np.vstack([position,position[None]+.008*rotation.T]);inv=np.linalg.inv(position_model.optics[camera]);q=cloud@inv[:3,:3].T+inv[:3,3];points=q[:,:2]/q[:,2:]*FOCAL+[576-left,512-top]
   if np.all(q[:,2]>0):
    start=tuple(np.rint(points[0]).astype(int))
    for k,color in enumerate([(0,0,255),(0,255,0),(255,0,0)]):cv2.arrowedLine(patch,start,tuple(np.rint(points[k+1]).astype(int)),color,1,cv2.LINE_AA,tipLength=.2)
   patch=cv2.copyMakeBorder(patch,22,0,0,0,cv2.BORDER_CONSTANT);cv2.putText(patch,camera+' predicted axes',(3,15),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1);panels.append(patch)
  name=episode+'_prediction_only.png';cv2.imwrite(str(a.output/name),np.hstack(panels));checks.append(dict(episode=episode,rotation_runtime_parity_deg=delta,image=name,GT_markers=False,position_source='Existing RGB prediction for overlay only; deployment retains acquisitionposition'))
 near=reports['near_25mm'];parity=all(x['rotation_runtime_parity_deg']<=.01 for x in checks);gate=bool(s['plateau_reached'] and near['episodes']>=4 and near['orientation_p50_p95_deg'][1]<=1 and near['episode_balanced_orientation_p50_p95_deg'][1]<=1 and parity)
 report=dict(checkpoint_sha256=hashlib.sha256((a.fit/'best.pt').read_bytes()).hexdigest(),retained_translation_checkpoint_sha256=hashlib.sha256(translation_checkpoint.read_bytes()).hexdigest(),pose_gate_passed=gate,prediction_kind='orientation_only',pose_frame_contract='SC_mating_reference_in_gripper_TCP',pose_gate_scope='SC_held_near25mm_only',translation_reestimated=False,untrained_translation_head_not_used=True,visual_inspection_completed=False,metrics_by_domain=reports,runtime_parity_passed=parity,checks=checks,plateau_reached=s['plateau_reached'],live_success_not_asserted=True)
 (a.output/'approval.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(pose_gate_passed=gate,near=near,runtime_parity_passed=parity)),flush=True)


if __name__=='__main__':main()
