#!/usr/bin/env python3
"""Closed-form pose readout on frozen RGB landmarks, with reset-group CV.

A bounded low-capacity alternative to the overfitted direct-pose heads.
Only train episodes choose regularization; held-out labels enter reporting.
"""
import cv2
cv2.setNumThreads(2)
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from sc_native_grasp_observation import SCNativeGraspObservation
from shared_port_tcp_geometry import matrix_from_pose9
from project_gpu_guard import selected_gpu_reservation


def poses(y):
 x=y[:,3:6];x=x/np.linalg.norm(x,axis=1,keepdims=True)
 v=y[:,6:9];v=v-(v*x).sum(1,keepdims=True)*x;v=v/np.linalg.norm(v,axis=1,keepdims=True)
 p=np.tile(np.eye(4),(len(y),1,1));p[:,:3,3]=y[:,:3];p[:,:3,:3]=np.stack([x,v,np.cross(x,v)],-1);return p


def errors(pred,truth):
 t=np.linalg.norm(pred[:,:3,3]-truth[:,:3,3],axis=1)*1000
 r=np.degrees((Rotation.from_matrix(pred[:,:3,:3]).inv()*Rotation.from_matrix(truth[:,:3,:3])).magnitude())
 return dict(translation_mm_p50_p95=np.percentile(t,[50,95]).tolist(),orientation_deg_p50_p95=np.percentile(r,[50,95]).tolist())


def fit(x,y,alpha):
 xm=x.mean(0);xs=np.maximum(x.std(0),.2);ym=y.mean(0);ys=np.maximum(y.std(0),1e-4)
 xx=(x-xm)/xs;yy=(y-ym)/ys
 weight=np.linalg.solve(xx.T@xx+np.eye(x.shape[1])*alpha,xx.T@yy)
 return dict(feature_mean=xm,feature_scale=xs,target_mean=ym,target_scale=ys,weight=weight)


def predict(m,x):return poses((((x-m['feature_mean'])/m['feature_scale'])@m['weight'])*m['target_scale']+m['target_mean'])


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(4)
 rows=json.loads((a.source/'manifest.json').read_text());images=np.load(a.source/'native_crops.npy',mmap_mode='r');seen=set();ids=[]
 for i,r in enumerate(rows):
  if r['episode'] not in seen:ids.append(i);seen.add(r['episode'])
 device='cuda' if torch.cuda.is_available() else 'cpu';model=SCNativeGraspObservation(a.checkpoint,device);features=[]
 with torch.inference_mode():
  for begin in range(0,len(ids),16):
   ii=ids[begin:begin+16];x=torch.from_numpy(np.array(images[ii],copy=True)).permute(0,1,4,2,3).reshape(-1,3,192,192).to(device).float()/255
   features.append(model.model(x,camera_id=torch.eye(3,device=device).repeat(len(ii),1)).cpu().numpy().reshape(len(ii),30)*191)
 x=np.vstack(features);selected=[rows[i] for i in ids];y=np.array([r['tip_in_tcp_pose9'] for r in selected]);truth=np.array([matrix_from_pose9(v) for v in y]);train=np.array([r['split']=='train' for r in selected]);groups=[r['scene'] for r in selected]
 assert not ({groups[i] for i in np.flatnonzero(train)}&{groups[i] for i in np.flatnonzero(~train)})
 rng=np.random.default_rng(20261018);unique=sorted({groups[i] for i in np.flatnonzero(train)});rng.shuffle(unique);fold={g:i%3 for i,g in enumerate(unique)};records=[]
 for alpha in [1e-4,.01,1.,100.]:
  pp=[];tt=[]
  for k in range(3):
   f=train&np.array([fold.get(g,-1)!=k for g in groups]);v=train&np.array([fold.get(g,-1)==k for g in groups]);m=fit(x[f],y[f],alpha);pp.extend(predict(m,x[v]));tt.extend(truth[v])
  metric=errors(np.array(pp),np.array(tt));score=max(metric['translation_mm_p50_p95'][1]/.4,metric['orientation_deg_p50_p95'][1]);records.append(dict(alpha=alpha,score=score,metrics=metric))
 alpha=min(records,key=lambda d:d['score'])['alpha'];m=fit(x[train],y[train],alpha);pred=predict(m,x);result={}
 for split,mask in [('train',train),('validation',~train)]:
  for domain in ['all','original','nominal_home','nominal_acquisition']:
   roles=[r.get('source_role','') for r in selected];domainmask=np.array([domain=='all' or (domain=='original' and not role) or(domain=='nominal_home' and role.startswith('Nominal-centred')) or(domain=='nominal_acquisition' and role.startswith('Predeadline')) for role in roles]);use=mask&domainmask
   if use.any():result[split+'/'+domain]=dict(episodes=int(use.sum()),**errors(pred[use],truth[use]))
 validation=[v for k,v in result.items() if k.startswith('validation/')];passed=all(v['translation_mm_p50_p95'][1]<=.4 and v['orientation_deg_p50_p95'][1]<=1 for v in validation)
 payload=dict(schema='sc_frozen_landmark_ridge_pose/v1',model=m,alpha=alpha,landmark_checkpoint=str(a.checkpoint),landmark_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),pose_input='three image-predicted native192-crop landmark sets only; no geometry',centres=model.centres,cameras=model.cameras,source=str(a.source),geometry_labels_only=True,orientation_approved=False)
 torch.save(payload,a.output/'candidate.pt');np.savez(a.output/'episode_predictions.npz',indices=ids,predictions=pred,features=x)
 report=dict(source=str(a.source),checkpoint=str(a.checkpoint),training_episodes=int(train.sum()),validation_episodes=int((~train).sum()),CV_reset_groups=len(unique),CV=records,selected_alpha=alpha,metrics=result,gate_passed=passed,model_selection='3fold training-reset-group CV only; complete held-out episodes used afterward for reporting',optimizer='Exact regularized least squares; no iterative training cutoff',automatic_promotion=False,ground_truth_runtime_input=False,final_scenes_opened=False)
 (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'episode_manifest.json').write_text(json.dumps([dict(index=i,episode=r['episode'],scene=r['scene'],split=r['split'],role=r.get('source_role')) for i,r in zip(ids,selected)],indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
 with selected_gpu_reservation():main()
