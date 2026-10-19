#!/usr/bin/env python3
"""Dense original-resolution CAD-reference keypoints -> geometric grasp pose.

Uses the prior matched probe's exact 83/21 split and TF-validated labels.
Fixed crop centres come only from training data. These are CAD reference
points, not claims that every reference point is a textured visible corner.
"""
import argparse,json,time,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from scipy.spatial.transform import Rotation
from train_sc_native_keypoints import PretrainedKeypointRefiner
from shared_port_tcp_geometry import matrix_from_pose9
from build_sc_native_pose_labels import project
from evaluate_sc_native_triangulation import triangulate

ROOT=Path(__file__).resolve().parents[1]
CAMS=('center','left','right')
# Centre of the mating reference and four front-housing reference corners.
# Front housing plane is 2.25mm behind sc_tip_link; constants from SC Plug SDF.
CAD=np.array([[0,0,0],[-.0108,-.00365,-.00225],[-.0108,.00365,-.00225],
              [.0108,-.00365,-.00225],[.0108,.00365,-.00225]],float)

def rigid(points):
    a=CAD-CAD.mean(0);b=points-points.mean(0);u,_,vt=np.linalg.svd(a.T@b)
    r=vt.T@u.T
    if np.linalg.det(r)<0:vt[-1]*=-1;r=vt.T@u.T
    m=np.eye(4);m[:3,:3]=r;m[:3,3]=points.mean(0)-r@CAD.mean(0);return m

def evaluate(model,x,y,labels,indices,optics,centres,roi):
    model.eval();pred=[]
    with torch.inference_mode():
        for start in range(0,len(x),96):
            batch=x[start:start+96]
            camera=(torch.nn.functional.one_hot(torch.arange(start,start+len(batch),device=x.device)%3,3).float()
                    if model.condition_camera else None)
            pred.append(model(batch,camera_id=camera).cpu().numpy().reshape(-1,5,2))
    pred=np.concatenate(pred).reshape(-1,3,5,2);pixel=np.linalg.norm((pred-y)*float(roi-1),axis=-1)
    poses=[];translation=[];angle=[]
    for j,index in enumerate(indices):
        cloud=[]
        for k in range(5):
            points={c:pred[j,ci,k]*(roi-1)+np.rint(np.asarray(centres[c])-roi/2)
                    for ci,c in enumerate(CAMS)}
            cloud.append(triangulate(points,optics))
        estimated=rigid(np.asarray(cloud));truth=matrix_from_pose9(labels[index]);poses.append(estimated)
        translation.append(float(np.linalg.norm(estimated[:3,3]-truth[:3,3])*1000))
        angle.append(float((Rotation.from_matrix(truth[:3,:3]).inv()*Rotation.from_matrix(estimated[:3,:3])).magnitude()*180/np.pi))
    return {'pixel_p50_p95':np.percentile(pixel,[50,95]).tolist(),
            'translation_p50_p95_mm':np.percentile(translation,[50,95]).tolist(),
            'orientation_p50_p95_deg':np.percentile(angle,[50,95]).tolist()},pred,np.asarray(poses)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--imagenet',type=Path,required=True)
    p.add_argument('--updates',type=int,default=6000);p.add_argument('--minutes',type=float,default=120)
    p.add_argument('--native-crops',type=Path,help='Reuse immutable crops with the same manifest ordering and centres')
    p.add_argument('--baseline-p95-mm',type=float,help='Explicit compatible-label translation baseline')
    p.add_argument('--camera-condition',action='store_true',help='Supply known camera identity for ordered CAD landmarks')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous run')
    a.output.mkdir(parents=True);torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(20261014)
    rows=json.loads((a.source/'manifest.json').read_text());prior=json.loads((a.source/'frozen_plan.json').read_text())
    centres=prior['crop_centres_native_pixels_train_only'];labels=np.load(a.source/'labels.npy');roi=192
    fit=np.array([r['split']=='train' for r in rows]);val=np.flatnonzero(~fit)
    optics={c:np.asarray(m) for c,m in json.loads((ROOT/'configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())['tcp_to_optical'].items()}
    plan={'schema':'aic_sc_grasp_dense_keypoints/v1','source':str(a.source),
          'source_manifest_sha256':hashlib.sha256((a.source/'manifest.json').read_bytes()).hexdigest(),
          'encoder_imagenet_sha256':hashlib.sha256(a.imagenet.read_bytes()).hexdigest(),
          'crop_centres_train_only':centres,'roi_original_pixels':roi,'CAD_reference_points':CAD.tolist(),
          'decoder':'5 dense soft-argmax keypoints per camera; triangulation + rigid registration',
          'encoder':'ImageNet ResNet18 early frozen layers; layer3 and spatial decoder trained',
          'comparison_note':'Targeted geometry-decoder repair, not a claim about resolution alone',
          'updates':a.updates,'budget_minutes':a.minutes,'train_episodes':83,'heldout_episodes':21,
          'cached_native_crops':str(a.native_crops) if a.native_crops else None,
          'explicit_baseline_p95_mm':a.baseline_p95_mm,
          'condition_camera':a.camera_condition,
          'gate':{'translation_p95_mm_max':.4,'orientation_p95_deg_max':1.,
                  'minimum_improvement_over_full_probe':.25},'geometry_input':False,'final_scenes_opened':False}
    (a.output/'frozen_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    patches=[];targets=[]
    for i,r in enumerate(rows):
        pose=matrix_from_pose9(labels[i]);one=[];truth=[]
        for c in CAMS:
            left,top=np.rint(np.asarray(centres[c])-roi/2).astype(int)
            if a.native_crops is None:
                im=cv2.cvtColor(cv2.imread(r['images'][c]),cv2.COLOR_BGR2RGB)
                patch=cv2.warpAffine(im,np.array([[1,0,-left],[0,1,-top]],np.float32),(roi,roi));one.append(patch)
            points=[]
            for point in CAD:
                m=pose.copy();m[:3,3]=pose[:3,:3]@point+pose[:3,3]
                pixel=project(m,optics[c])[:2];points.append((np.asarray(pixel)-[left,top])/(roi-1))
            truth.append(points)
        patches.append(one);targets.append(truth)
    patches=(np.load(a.native_crops) if a.native_crops else np.asarray(patches,np.uint8))
    assert patches.shape==(len(rows),3,roi,roi,3)
    targets=np.asarray(targets,np.float32)
    if targets.min()<0 or targets.max()>1:raise ValueError('CAD references fall outside fixed ROI; audit before training')
    np.save(a.output/'keypoint_labels.npy',targets)
    if a.native_crops is None:np.save(a.output/'native_crops.npy',patches)
    x=torch.from_numpy(patches).permute(0,1,4,2,3).reshape(-1,3,roi,roi).cuda().float()/255
    y=torch.from_numpy(targets).reshape(-1,10).cuda();fit_flat=np.repeat(fit,3);val_flat=np.repeat(~fit,3)
    xf,yf=x[fit_flat],y[fit_flat];xv=x[val_flat];yv=targets[~fit]
    model=PretrainedKeypointRefiner(str(a.imagenet),False,point_count=5,condition_camera=a.camera_condition).cuda()
    opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4,weight_decay=1e-4)
    best=float('inf');best_step=0;started=time.monotonic();history=[];reason='update_cap'
    for step in range(1,a.updates+1):
        if time.monotonic()-started>a.minutes*60:reason='time_cap';break
        model.train();ids=torch.randint(len(xf),(48,),device='cuda');opt.zero_grad(set_to_none=True)
        camera=(torch.nn.functional.one_hot(ids%3,3).float() if a.camera_condition else None)
        pred=model(xf[ids],camera_id=camera);delta=(pred-yf[ids])*(roi-1)
        loss=nn.functional.smooth_l1_loss(delta,torch.zeros_like(delta),beta=1.)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%200==0:
            metrics,_,_=evaluate(model,xv,yv,labels,val,optics,centres,roi)
            score=max(metrics['translation_p50_p95_mm'][1]/.4,metrics['orientation_p50_p95_deg'][1])
            history.append({'update':step,'score':score,'metrics':metrics})
            if score<best*.995:
                best=score;best_step=step;torch.save({'model':model.state_dict(),'plan':plan,'update':step},a.output/'best.pt')
            print(step,metrics,flush=True)
            (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
            if step>=2000 and step-best_step>=1600:reason='validation_plateau';break
    checkpoint=torch.load(a.output/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['model'])
    metrics,pixels,poses=evaluate(model,xv,yv,labels,val,optics,centres,roi)
    np.save(a.output/'heldout_pixel_predictions.npy',pixels);np.save(a.output/'heldout_pose_predictions.npy',poses)
    baseline=(a.baseline_p95_mm if a.baseline_p95_mm is not None else
              json.loads((a.source/'summary.json').read_text())['full']['metrics']['translation_p50_p95_mm'][1])
    passed=(metrics['translation_p50_p95_mm'][1]<=.4 and metrics['orientation_p50_p95_deg'][1]<=1 and
            metrics['translation_p50_p95_mm'][1]<=baseline*.75)
    (a.output/'summary.json').write_text(json.dumps({'metrics':metrics,'passed':passed,'best_update':best_step,
        'updates_completed':step,'stop_reason':reason,'train_episodes':83,'heldout_episodes':21,
        'input':'RGB fixed original-resolution crops only','true_geometry_actor_input':False,
        'final_scenes_opened':False},indent=2)+'\n')

if __name__=='__main__':main()
