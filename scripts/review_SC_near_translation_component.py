#!/usr/bin/env python3
"""Qualify only the RGB tip POSITION component; never approve its rotation."""
import cv2
cv2.setNumThreads(2)
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from sc_spatial_grasp_observation import SCSpatialGraspObservation
from evaluate_sc_native_triangulation import FOCAL
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
s=json.loads((a.fit/'summary.json').read_text());source=Path(s['plan']['source']);rows=json.loads((source/'manifest.json').read_text());ids=np.load(a.fit/'heldout_indices.npy');poses=np.load(a.fit/'heldout_pose_predictions.npy');near=s['metrics_by_domain']['near_25mm'];model=SCSpatialGraspObservation(a.fit/'best.pt','cpu');checks=[];selected=[j for j,i in enumerate(ids) if rows[int(i)].get('source_role','').startswith('Near25mm')]
for episode in sorted({rows[int(ids[j])]['episode'] for j in selected}):
 positions=[j for j in selected if rows[int(ids[j])]['episode']==episode];j=positions[len(positions)//2];row=rows[int(ids[j])];images={c:cv2.cvtColor(cv2.imread(row['images'][c]),cv2.COLOR_BGR2RGB) for c in model.cameras};result=model.predict(images,already_training_jpeg=True);position=np.asarray(result['predicted_tip_position_tcp_m']);delta=float(np.linalg.norm(position-poses[j,:3,3])*1000);panels=[]
 for ci,camera in enumerate(model.cameras):
  left,top=s['plan']['crop_origins_native'][ci];patch=cv2.warpAffine(cv2.cvtColor(images[camera],cv2.COLOR_RGB2BGR),np.array([[1,0,-left],[0,1,-top]],np.float32),(192,192));q=np.linalg.inv(model.optics[camera])@np.r_[position,1.];point=q[:2]/q[2]*FOCAL+[576-left,512-top];cv2.drawMarker(patch,tuple(np.rint(point).astype(int)),(255,255,0),cv2.MARKER_CROSS,10,1,cv2.LINE_AA);patch=cv2.copyMakeBorder(patch,22,0,0,0,cv2.BORDER_CONSTANT);cv2.putText(patch,camera+' predicted POSITION',(2,15),cv2.FONT_HERSHEY_SIMPLEX,.33,(255,255,255),1);panels.append(patch)
 filename=episode+'_position_only.png';cv2.imwrite(str(a.output/filename),np.hstack(panels));checks.append(dict(episode=episode,image=filename,translation_runtime_parity_mm=delta,GT_marker_used=False,orientation_not_used=True))
parity=all(x['translation_runtime_parity_mm']<=.01 for x in checks);gate=bool(s['plateau_reached'] and near['episodes']>=4 and near['translation_p50_p95_mm'][1]<=.4 and near['episode_balanced_translation_p50_p95_mm'][1]<=.4 and parity)
report=dict(schema='sc_near_translation_component_approval/v1',prediction_kind='translation_only_component',pose_gate_scope='SC_held_near25mm_only',translation_gate_passed=gate,checkpoint_sha256=hashlib.sha256((a.fit/'best.pt').read_bytes()).hexdigest(),orientation_approved=False,pose_frame_contract='SC_mating_reference_in_gripper_TCP',metrics=near,checks=checks,runtime_parity_passed=parity,visual_inspection_completed=False,automatic_deployment=False)
(a.output/'approval.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(translation_gate_passed=gate,translation_p95_mm=near['translation_p50_p95_mm'][1],orientation_approved=False)),flush=True)
