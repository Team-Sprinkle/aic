#!/usr/bin/env python3
"""Compare frozen original/new SFP held-out domains and save prediction-only RGB QA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import fcntl
import cv2
cv2.setNumThreads(2)
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from native_connector_pose_observation import NativeConnectorPoseObservation
from train_sc_native_keypoints import PretrainedKeypointRefiner
from train_sfp_native_grasp_keypoints import poses_from_pixels
from evaluate_sc_native_triangulation import FOCAL

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',default='5');p.add_argument('--extra-fit',action='append',type=Path,default=[]);p.add_argument('--output',type=Path);a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu
lock=open('/var/tmp/chmin_aic_gpu'+a.gpu+'.lock','w');fcntl.flock(lock,fcntl.LOCK_EX)
torch.set_num_threads(4)
data=a.root/'SFP_wide_orientation_dataset';out=a.output or a.root/'SFP_pose_candidate_review';out.mkdir(exist_ok=False)
rows=json.loads((data/'manifest.json').read_text());truth=np.load(data/'body_tip_matrices.npy');pixels=np.load(data/'native_crops.npy',mmap_mode='r')
val=np.array([i for i,r in enumerate(rows) if r['split']=='validation']);wide=np.array([bool(rows[i].get('source_role')) for i in val])
keypoint=Path('/home/nas/chmin/aic_archives/2026-10-16_full_sample_insertion_evaluation_v1/frozen/models/sfp_grasp.pt')
model_data=torch.load(keypoint,map_location='cpu',weights_only=False);meta=model_data['plan'];cams=('center','left','right')
origins=np.stack([np.rint(np.array(meta['crop_centres_train_only'][c])-104)+8 for c in cams])
landmark=PretrainedKeypointRefiner(None,False,point_count=5,condition_camera=True)
landmark.backbone.layer4=torch.nn.Identity();landmark.backbone.avgpool=torch.nn.Identity();landmark.backbone.fc=torch.nn.Identity();landmark.load_state_dict(model_data['model']);landmark.cuda().eval()
with torch.inference_mode():
    geometry=[]
    for offset in range(0,len(val),24):
        image=torch.from_numpy(np.array(pixels[val[offset:offset+24],:,8:200,8:200])).permute(0,1,4,2,3).cuda()
        coords=landmark(image.reshape(-1,3,192,192).float()/255,camera_id=torch.eye(3,device='cuda').repeat(len(image),1)).cpu().numpy().reshape(-1,3,5,2)
        geometry.append(poses_from_pixels(coords*191+origins[None,:,None,:],meta['tcp_to_optical'],meta['reference_points']))
geometry=np.concatenate(geometry);np.save(out/'existing_geometry_predictions.npy',geometry)
position_error=np.linalg.norm(geometry[:,:3,3]-truth[val,:3,3],axis=1)*1000
candidates={
 'original':Path('/home/nas/chmin/aic_archives/2026-10-17_connector_orientation_and_sfp_oscillation_v1/sfp_direct_pose/best.pt'),
 'rotation_only':a.root/'SFP_wide_orientation_fit/best.pt',
 'joint_pose':a.root/'SFP_wide_joint_pose_fit/best.pt'}
candidates.update({path.name:path/'best.pt' for path in a.extra_fit})
reports={};predictions={}
for name,checkpoint in candidates.items():
    if name!='original':assert json.loads((checkpoint.parent/'summary.json').read_text())['plateau_reached']
    estimator=NativeConnectorPoseObservation(checkpoint,'cuda')
    assert np.array_equal(estimator.origins,origins)
    prediction=[]
    with torch.inference_mode():
        for offset in range(0,len(val),32):
            image=torch.from_numpy(np.array(pixels[val[offset:offset+32],:,8:200,8:200])).permute(0,1,4,2,3).cuda()
            t,r=estimator.model(image);m=np.broadcast_to(np.eye(4),(len(t),4,4)).copy();m[:,:3,3]=t.cpu().numpy();m[:,:3,:3]=r.cpu().numpy();prediction.append(m)
    prediction=np.concatenate(prediction);predictions[name]=prediction
    angular=np.degrees(Rotation.from_matrix(prediction[:,:3,:3].transpose(0,2,1)@truth[val,:3,:3]).magnitude())
    joint_position=np.linalg.norm(prediction[:,:3,3]-truth[val,:3,3],axis=1)*1000
    domains={}
    for domain,mask in [('original',~wide),('wide',wide)]:
        ids=np.flatnonzero(mask);episode_medians=[]
        for episode in sorted({rows[val[i]]['episode'] for i in ids}):
            episode_medians.append(float(np.median([angular[i] for i in ids if rows[val[i]]['episode']==episode])))
        domains[domain]=dict(samples=len(ids),episodes=len(episode_medians),orientation_p50_p95_deg=np.percentile(angular[ids],[50,95]).tolist(),episode_median_orientation_p50_p95_deg=np.percentile(episode_medians,[50,95]).tolist(),existing_translation_p50_p95_mm=np.percentile(position_error[ids],[50,95]).tolist(),joint_translation_p50_p95_mm=np.percentile(joint_position[ids],[50,95]).tolist())
    reports[name]=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),domains=domains,orientation_gate_passed=all(x['orientation_p50_p95_deg'][1]<=1 and x['episode_median_orientation_p50_p95_deg'][1]<=1 for x in domains.values()),existing_translation_gate_passed=all(x['existing_translation_p50_p95_mm'][1]<=.4 for x in domains.values()),joint_translation_gate_passed=all(x['joint_translation_p50_p95_mm'][1]<=.4 for x in domains.values()))
    np.save(out/(name+'_predictions.npy'),prediction)
    del estimator
usable=[name for name in candidates if name!='original' and reports[name]['orientation_gate_passed'] and (reports[name]['existing_translation_gate_passed'] or (name!='rotation_only' and reports[name]['joint_translation_gate_passed']))]
selected=min(usable,key=lambda name:max(x['orientation_p50_p95_deg'][1] for x in reports[name]['domains'].values())) if usable else None
position_mode='joint_pose' if selected and reports[selected]['joint_translation_gate_passed'] and selected!='rotation_only' else 'existing_geometric'
# Fixed index sampling, not selecting the lowest-error images.
qa=[]
if selected:
    indices=[np.flatnonzero(~wide)[0],*np.flatnonzero(wide)[np.linspace(0,wide.sum()-1,3,dtype=int)]]
    for j,index in enumerate(indices):
        row=rows[val[index]];pose=np.eye(4);pose[:3,3]=predictions[selected][index,:3,3] if position_mode=='joint_pose' else geometry[index,:3,3];pose[:3,:3]=predictions[selected][index,:3,:3]
        points=np.r_[np.zeros((1,3)),np.eye(3)*.010]@pose[:3,:3].T+pose[:3,3]
        panels=[]
        for camera in cams:
            image=cv2.imread(row['images'][camera]);inverse=np.linalg.inv(meta['tcp_to_optical'][camera]);q=points@inverse[:3,:3].T+inverse[:3,3];uv=np.rint(q[:,:2]/q[:,2:]*FOCAL+[576,512]).astype(int)
            cv2.circle(image,tuple(uv[0]),5,(0,255,255),2)
            for axis,color in enumerate([(0,0,255),(0,255,0),(255,0,0)],1):cv2.arrowedLine(image,tuple(uv[0]),tuple(uv[axis]),color,2,tipLength=.2)
            cv2.putText(image,camera+' PREDICTED FRAME ONLY',(15,30),0,.65,(0,255,255),2);panels.append(cv2.resize(image,(576,512)))
        path=out/(f'prediction_only_QA_{j}.jpg');cv2.imwrite(str(path),np.hstack(panels));qa.append(dict(path=str(path),episode=row['episode'],split=row['split'],true_pose_overlay=False))
report=dict(candidates=reports,selected=selected,position_mode=position_mode,prediction_only_QA=qa,visual_inspection_completed=False,automatic_deployment=False,final_scenes_sealed=True)
(out/'review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
