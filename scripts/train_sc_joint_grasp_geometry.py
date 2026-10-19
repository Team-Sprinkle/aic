#!/usr/bin/env python3
"""Matched pixel-only versus joint multiview geometry supervision for grasp."""
import argparse,json,hashlib,time
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from train_sc_native_keypoints import PretrainedKeypointRefiner
from train_sc_grasp_keypoint_probe import CAD,evaluate
from train_shared_port_tcp_bc import rotation_from_pose9
from build_sc_native_pose_labels import FOCAL


def torch_pose(points,projection,cad):
    # All cameras and reference points participate in the same SE3 fit.
    u,v=points[...,0],points[...,1]
    a=u[...,None]*projection[None,:,None,2,:]-projection[None,:,None,0,:]
    b=v[...,None]*projection[None,:,None,2,:]-projection[None,:,None,1,:]
    system=torch.stack((a,b),-2).permute(0,2,1,3,4).reshape(-1,5,6,4)
    _,_,vh=torch.linalg.svd(system,full_matrices=False)
    homogeneous=vh[...,-1,:];cloud=homogeneous[...,:3]/homogeneous[...,3:4]
    centered=cad-cad.mean(0);other=cloud-cloud.mean(1,keepdim=True)
    covariance=centered.T[None]@other
    left,_,right=torch.linalg.svd(covariance)
    diagonal=torch.eye(3,device=points.device)[None].repeat(len(points),1,1)
    diagonal[:,2,2]=torch.linalg.det(right.transpose(1,2)@left.transpose(1,2)).sign()
    rotation=right.transpose(1,2)@diagonal@left.transpose(1,2)
    translation=cloud.mean(1)-(rotation@cad.mean(0)[None,:,None]).squeeze(-1)
    return translation,rotation


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--pixels',type=Path,required=True);p.add_argument('--pixel-labels',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume-comparison',type=Path);p.add_argument('--seed',type=int,default=20261014);a=p.parse_args()
    if a.output.exists():p.error('Preserve comparisons')
    a.output.mkdir();torch.set_num_threads(4);cv2.setNumThreads(2)
    rows=json.loads((a.source/'manifest.json').read_text());fit=np.array([r['split']=='train' for r in rows]);train=np.flatnonzero(fit);val=np.flatnonzero(~fit);labels=np.load(a.source/'labels.npy');pixels=np.load(a.pixels);pixel_labels=np.load(a.pixel_labels);assert pixels.shape==(len(rows),3,192,192,3)
    x=torch.from_numpy(pixels).permute(0,1,4,2,3).contiguous();y=torch.from_numpy(pixel_labels).cuda();truth=torch.from_numpy(labels).cuda();rotation_truth=rotation_from_pose9(truth)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False);assert checkpoint['plan']['condition_camera'];centres=checkpoint['plan']['crop_centres_train_only'];cams=['center','left','right'];cal=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text());optics={c:np.asarray(cal['tcp_to_optical'][c]) for c in cams};intrinsic=np.array([[FOCAL,0,576],[0,FOCAL,512],[0,0,1]],np.float32);projection=torch.as_tensor(np.stack([intrinsic@np.linalg.inv(optics[c])[:3] for c in cams]),device='cuda',dtype=torch.float32);offset=torch.as_tensor(np.stack([np.rint(np.asarray(centres[c])-96) for c in cams]),device='cuda',dtype=torch.float32);cad=torch.as_tensor(CAD,device='cuda',dtype=torch.float32)
    plan={'schema':'aic_matched_grasp_geometry_supervision/v1','hypothesis':'Independent pixel losses leave orientation error despite accurate tip translation. Joint multiview pose loss trains the desired metric without adding privileged inference inputs.','checkpoint':str(a.checkpoint),'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'source_manifest_sha256':hashlib.sha256((a.source/'manifest.json').read_bytes()).hexdigest(),'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'arms':['pixel_control','joint_geometry'],'updates_per_arm':4000,'gpu_hours_per_arm_cap':2,'same_native_pixels':str(a.pixels),'same_pixel_labels':str(a.pixel_labels),'batch_triplets':16,'seed':a.seed,'resume_comparison':str(a.resume_comparison) if a.resume_comparison else None,'continuation_optimizer_reset':bool(a.resume_comparison),'optimizer':'AdamW5e-5 weight_decay1e-4; fresh optimizer in BOTH arms','validation_episodes':21,'train_episodes':83,'selection':'max(translation_p95/0.4,orientation_p95/1), best and last retained; equal fixed update budget, saturation reported separately','gate':{'translation_p95_mm':.4,'orientation_p95_deg':1},'geometry_inference_input':False,'final_scenes_opened':False};(a.output/'frozen_plan.json').write_text(json.dumps(plan,indent=2)+'\n');summary={}
    for arm in plan['arms']:
        out=a.output/arm;out.mkdir();torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed);model=PretrainedKeypointRefiner(None,False,point_count=5,condition_camera=True).cuda();arm_checkpoint=(torch.load(a.resume_comparison/arm/'last.pt',map_location='cpu',weights_only=False) if a.resume_comparison else checkpoint);model.load_state_dict(arm_checkpoint['model']);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5,weight_decay=1e-4);best=float('inf');best_step=0;history=[];start=time.monotonic();plateau_at=None;reason='fixed_update_budget'
        for step in range(1,4001):
            if time.monotonic()-start>7200:reason='time_cap';break
            ids=train[rng.integers(len(train),size=16)];batch=x[ids].reshape(-1,3,192,192).cuda().float()/255;camera=torch.nn.functional.one_hot(torch.arange(len(batch),device='cuda')%3,3).float();model.train();opt.zero_grad(set_to_none=True);pred=model(batch,camera_id=camera).reshape(-1,3,5,2);delta=(pred-y[ids])*191;pixel_loss=nn.functional.smooth_l1_loss(delta,torch.zeros_like(delta),beta=1);loss=pixel_loss
            if arm=='joint_geometry':
                translation,rotation=torch_pose(pred*191+offset[None,:,None,:],projection,cad);position_loss=nn.functional.smooth_l1_loss((translation-truth[ids,:3])/.0004,torch.zeros_like(translation),beta=1);angle_loss=(rotation-rotation_truth[ids]).square().sum((-1,-2)).mean()/(2*(np.pi/180)**2);loss=.2*pixel_loss+position_loss+angle_loss
            if not torch.isfinite(loss):raise ValueError(f'Nonfinite geometry loss at {arm}/{step}; preserve failed run')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            if step%200==0:
                xv=x[val].reshape(-1,3,192,192).cuda().float()/255;metrics,_,_=evaluate(model,xv,pixel_labels[val],labels,val,optics,centres,192);score=max(metrics['translation_p50_p95_mm'][1]/.4,metrics['orientation_p50_p95_deg'][1]);history.append({'step':step,'metrics':metrics,'score':score})
                if score<best*.995:
                    best=score;best_step=step;plateau_at=None;torch.save({'model':model.state_dict(),'plan':checkpoint['plan'],'geometry_plan':plan,'arm':arm,'update':step},out/'best.pt')
                elif step-best_step>=1600 and plateau_at is None:plateau_at=step
                (out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        torch.save({'model':model.state_dict(),'plan':checkpoint['plan'],'geometry_plan':plan,'arm':arm,'update':step},out/'last.pt');selected=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(selected['model']);xv=x[val].reshape(-1,3,192,192).cuda().float()/255;metrics,coordinates,poses=evaluate(model,xv,pixel_labels[val],labels,val,optics,centres,192);np.save(out/'heldout_pixel_predictions.npy',coordinates);np.save(out/'heldout_pose_predictions.npy',poses);summary[arm]={'best_step':best_step,'updates_completed':step,'stop_reason':reason,'last_plateau_at':plateau_at,'metrics':metrics,'passed':metrics['translation_p50_p95_mm'][1]<=.4 and metrics['orientation_p50_p95_deg'][1]<=1,'wall_seconds':time.monotonic()-start};(out/'summary.json').write_text(json.dumps(summary[arm],indent=2)+'\n');print(arm,json.dumps(summary[arm]),flush=True)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
