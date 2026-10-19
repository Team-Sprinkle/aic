#!/usr/bin/env python3
"""Episode-balanced SFP grasp landmark fit from fixed original-image crops.

Training labels are never inference inputs. The saved export authorizes
translation and orientation separately, using held-out physical resets.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from train_sc_native_keypoints import PretrainedKeypointRefiner
from evaluate_sc_native_triangulation import FOCAL

CAMS=('center','left','right')


def poses_from_pixels(pixels, optics, reference):
    """Batched camera-ray intersection followed by rigid registration."""
    batch=len(pixels);lhs=np.zeros((batch,5,3,3));rhs=np.zeros((batch,5,3))
    for ci,c in enumerate(CAMS):
        t=np.asarray(optics[c]);ray=np.concatenate(((pixels[:,ci]-[576,512])/FOCAL,np.ones((batch,5,1))),axis=-1)@t[:3,:3].T
        ray/=np.linalg.norm(ray,axis=-1,keepdims=True)
        projection=np.eye(3)-ray[...,None]*ray[...,None,:]
        lhs+=projection;rhs+=projection@t[:3,3]
    cloud=np.linalg.solve(lhs+1e-9*np.eye(3),rhs[...,None])[...,0]
    centred=cloud-cloud.mean(axis=1,keepdims=True);a=np.asarray(reference)-np.mean(reference,axis=0)
    u,_,vh=np.linalg.svd(np.einsum('ki,nkj->nij',a,centred))
    v=vh.transpose(0,2,1);ut=u.transpose(0,2,1);diagonal=np.broadcast_to(np.eye(3),(batch,3,3)).copy()
    diagonal[:,2,2]=np.linalg.det(v@ut);r=v@diagonal@ut
    result=np.broadcast_to(np.eye(4),(batch,4,4)).copy();result[:,:3,:3]=r
    result[:,:3,3]=cloud.mean(axis=1)-np.einsum('nij,j->ni',r,np.mean(reference,axis=0))
    return result


def metrics(prediction, truth):
    position=np.linalg.norm(prediction[:,:3,3]-truth[:,:3,3],axis=-1)*1000
    angles=(Rotation.from_matrix(truth[:,:3,:3]).inv()*Rotation.from_matrix(prediction[:,:3,:3])).magnitude()*180/np.pi
    return dict(translation_p50_p95_mm=np.percentile(position,[50,95]).tolist(),orientation_p50_p95_deg=np.percentile(angles,[50,95]).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','output','imagenet'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--updates',type=int,default=20000);p.add_argument('--minutes',type=float,default=120)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(2);torch.set_num_threads(4);torch.manual_seed(20261015);rng=np.random.default_rng(20261015)
    source=json.loads((a.source/'summary.json').read_text());rows=json.loads((a.source/'manifest.json').read_text())
    if any(r['family']!='sfp' for r in rows):raise ValueError('SFP-only grasp fit requires SFP labels')
    if not (a.source/'visual_QA_passed.json').exists():raise ValueError('Inspect native crop QA before training')
    fit=np.array([r['split']=='train' for r in rows]);val=np.flatnonzero(~fit)
    if {r['scene'] for r in rows if r['split']=='train'}&{r['scene'] for r in rows if r['split']=='validation'}:raise ValueError('Reset leakage')
    groups={episode:np.array([i for i,r in enumerate(rows) if r['episode']==episode and fit[i]]) for episode in sorted({r['episode'] for r in rows if r['split']=='train'})};groups=list(groups.values())
    pixels=np.load(a.source/'native_crops.npy',mmap_mode='r');target=np.load(a.source/'keypoint_pixels.npy');truth=np.load(a.source/'body_tip_matrices.npy')
    plan=dict(source,family='sfp',source=str(a.source),model='camera-conditioned ImageNet ResNet18 layer3 + dense five-keypoint decoder',
              batch_size_triplets=16,balanced_by_source_episode=True,maximum_updates=a.updates,budget_minutes=a.minutes,
              stop_rule='minimum5000 updates; validation translation p95 plateau2500 updates without0.5% improvement; otherwise update/time cap explicitly reported',
              translation_gate_p95_mm=.4,orientation_gate_p95_deg=1.,
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              manifest_sha256=hashlib.sha256((a.source/'manifest.json').read_bytes()).hexdigest())
    (a.output/'frozen_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    model=PretrainedKeypointRefiner(str(a.imagenet),False,point_count=5,condition_camera=True)
    model.backbone.layer4=nn.Identity();model.backbone.avgpool=nn.Identity();model.backbone.fc=nn.Identity();model.cuda()
    parameters=[x for x in model.parameters() if x.requires_grad];optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=1e-4)
    offsets=np.asarray([np.rint(np.asarray(source['crop_centres_train_only'][c])-104) for c in CAMS])
    def evaluate():
        model.eval();predicted=[]
        with torch.inference_mode():
            for j in range(0,len(val),24):
                image=np.array(pixels[val[j:j+24],:,8:200,8:200,:],copy=True)
                x=torch.from_numpy(image).permute(0,1,4,2,3).reshape(-1,3,192,192).cuda().float()/255
                camera=torch.eye(3,device='cuda').repeat(len(image),1)
                predicted.append(model(x,camera_id=camera).cpu().numpy().reshape(-1,3,5,2)*191+8+offsets[None,:,None,:])
        predicted=np.concatenate(predicted);poses=poses_from_pixels(predicted,source['tcp_to_optical'],source['reference_points'])
        result=metrics(poses,truth[val]);result['pixel_p50_p95']=np.percentile(np.linalg.norm(predicted-(target[val]+offsets[None,:,None,:]),axis=-1),[50,95]).tolist()
        return result,predicted,poses
    best=float('inf');best_update=0;started=time.monotonic();history=[];reason='update_cap'
    for step in range(1,a.updates+1):
        if time.monotonic()-started>a.minutes*60:reason='time_cap';break
        selected=np.array([g[rng.integers(len(g))] for g in [groups[k] for k in rng.integers(len(groups),size=16)]])
        shifts=rng.integers(0,17,size=(len(selected),3,2));images=[];labels=[]
        for i,index in enumerate(selected):
            for ci in range(3):
                sx,sy=shifts[i,ci];images.append(pixels[index,ci,sy:sy+192,sx:sx+192]);labels.append((target[index,ci]-[sx,sy])/191)
        x=torch.from_numpy(np.stack(images)).permute(0,3,1,2).cuda().float()/255;y=torch.from_numpy(np.asarray(labels,np.float32)).cuda().reshape(-1,10)
        model.train();optimizer.zero_grad(set_to_none=True);prediction=model(x,camera_id=torch.eye(3,device='cuda').repeat(len(selected),1))
        loss=nn.functional.smooth_l1_loss((prediction-y)*191,torch.zeros_like(y),beta=1.)
        loss.backward();nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
        if step%250==0:
            metric,_,_=evaluate();score=metric['translation_p50_p95_mm'][1]
            row=dict(update=step,metrics=metric,wall_seconds=time.monotonic()-started);history.append(row)
            if score<best*.995:
                best=score;best_update=step;torch.save(dict(model=model.state_dict(),plan=plan,update=step),a.output/'best.pt')
            (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n');print(json.dumps(row),flush=True)
            if step>=5000 and step-best_update>=2500:reason='validation_plateau';break
    torch.save(dict(model=model.state_dict(),plan=plan,update=step),a.output/'last.pt')
    saved=torch.load(a.output/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);metric,predicted,poses=evaluate()
    np.save(a.output/'heldout_pixel_predictions.npy',predicted);np.save(a.output/'heldout_body_pose_predictions.npy',poses)
    constant=np.broadcast_to(truth[fit].mean(axis=0),(len(val),4,4)).copy()
    constant[:,:3,:3]=Rotation.from_matrix(truth[fit,:3,:3]).mean().as_matrix()
    translation_ok=metric['translation_p50_p95_mm'][1]<=.4;orientation_ok=metric['orientation_p50_p95_deg'][1]<=1
    report=dict(metrics=metric,constant_mean_baseline=metrics(constant,truth[val]),best_update=best_update,
                updates_completed=step,stop_reason=reason,plateau_reached=reason=='validation_plateau',
                translation_approved=translation_ok,orientation_approved=orientation_ok,
                parameters=sum(x.numel() for x in model.parameters()),trainable_parameters=sum(x.numel() for x in parameters),
                source_episodes=source['source_episodes'],reset_groups=source['reset_groups'],samples=len(rows),
                checkpoint_sha256=hashlib.sha256((a.output/'best.pt').read_bytes()).hexdigest(),
                geometry_input=False,final_scenes_opened=False,wall_seconds=time.monotonic()-started)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
