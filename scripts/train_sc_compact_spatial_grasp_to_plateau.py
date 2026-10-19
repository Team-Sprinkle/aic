#!/usr/bin/env python3
"""Fixed compact native crops with the same calibrated multiview pose supervision.

The pose is solved from predicted pixels. Crop jitter adjusts both pixel labels
and the calibrated crop origin. Simulator geometry supplies labels only.
"""
import cv2
cv2.setNumThreads(2)
import argparse,hashlib,json,os,time
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.nn import functional as F
from train_sc_native_keypoints import PretrainedKeypointRefiner
from train_sc_grasp_keypoint_probe import CAD,rigid
from train_sfp_native_grasp_keypoints import metrics,poses_from_pixels
from shared_port_tcp_geometry import matrix_from_pose9
from evaluate_sc_native_triangulation import FOCAL
from project_gpu_guard import selected_gpu_reservation
from differentiable_rigid_fit import rigid_pose


def save(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def checkpoint(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)


def ray_cloud(native,optical):
 # Least-squares intersection of calibrated rays; no SVD backward pass.
 ray=torch.cat(((native-native.new_tensor([576.,512.]))/FOCAL,torch.ones_like(native[...,:1])),-1)
 direction=torch.einsum('cij,bckj->bcki',optical[:,:3,:3],ray)
 direction=F.normalize(direction,dim=-1)
 projector=torch.eye(3,device=native.device)-direction[...,None]*direction[...,None,:]
 system=projector.sum(1)
 rhs=(projector@optical[None,:,None,:3,3,None]).sum(1).squeeze(-1)
 return torch.linalg.solve(system+torch.eye(3,device=native.device)*1e-7,rhs[...,None]).squeeze(-1)


def planar_pose(cloud,cad):
 x=F.normalize(cloud[:,3]-cloud[:,1]+cloud[:,4]-cloud[:,2],dim=-1)
 y=cloud[:,2]-cloud[:,1]+cloud[:,4]-cloud[:,3]
 y=F.normalize(y-(x*y).sum(-1,keepdim=True)*x,dim=-1)
 rotation=torch.stack((x,y,torch.linalg.cross(x,y)),-1)
 translation=cloud.mean(1)-(rotation@cad.mean(0)[:,None]).squeeze(-1)
 return translation,rotation


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['source','pixels','checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--crop-geometry',type=Path,required=True);p.add_argument('--mode',choices=['pixel','joint'],required=True);p.add_argument('--updates',type=int,default=100000);p.add_argument('--minutes',type=float,default=1440);p.add_argument('--min-updates',type=int,default=5000);p.add_argument('--patience',type=int,default=3000);p.add_argument('--crop-jitter',type=int,default=4);p.add_argument('--resume',action='store_true');p.add_argument('--selection-domains',nargs='+',default=['all','original','nominal_acquisition','nominal_home']);p.add_argument('--pose-solver',choices=['kabsch','planar'],default='kabsch');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=a.resume);torch.set_num_threads(4);torch.manual_seed(20261018);rng=np.random.default_rng(20261018)
 rows=json.loads((a.source/'manifest.json').read_text());fit=np.array([r['split']=='train' for r in rows]);train=np.flatnonzero(fit);val=np.flatnonzero(~fit)
 assert len(train) and len(val) and all(r['split'] in ['train','validation'] for r in rows)
 for key in ['episode','scene']:assert not ({r[key] for r in rows if r['split']=='train'}&{r[key] for r in rows if r['split']=='validation'}),key+' leakage'
 truth=np.stack([matrix_from_pose9(x) for x in np.load(a.source/'labels.npy')]);pixels=np.load(a.pixels,mmap_mode='r');assert pixels.shape==(len(rows),3,192,192,3)
 initial=torch.load(a.checkpoint,map_location='cpu',weights_only=False);centres=initial['plan']['crop_centres_train_only'];cameras=['center','left','right'];geometry=json.loads(a.crop_geometry.read_text());assert geometry['manifest_sha256']==hashlib.sha256((a.source/'manifest.json').read_bytes()).hexdigest();origins=np.array([geometry['regions'][c]['origin_native_xy'] for c in cameras]);native_sizes=np.array([geometry['regions'][c]['native_size'] for c in cameras]);native_steps=native_sizes-1;centres={c:(origins[i]+(native_sizes[i]-1)/2).tolist() for i,c in enumerate(cameras)};assert a.crop_jitter==0,'Compact ablation fixes crop jitter to0';optics={c:np.asarray(x) for c,x in json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())['tcp_to_optical'].items()}
 cloud_truth=np.einsum('nij,kj->nki',truth[:,:3,:3],CAD)+truth[:,None,:3,3]
 labels=[]
 for c in cameras:
  inv=np.linalg.inv(optics[c]);q=cloud_truth@inv[:3,:3].T+inv[:3,3];assert np.min(q[:,:,2])>0;labels.append(q[:,:,:2]/q[:,:,2:]*FOCAL+[576,512])
 native=np.stack(labels,1);local=native-origins[None,:,None,:]
 outside=((local<0)|(local>native_steps[None,:,None,None])).any(-1)
 if outside.any():raise ValueError('Fixed crop omits '+str(int(outside.sum()))+' projected references; diagnose before training')
 groups=defaultdict(list)
 for i in train:groups[rows[i]['episode']].append(i)
 groups=list(groups.values());expanded=[g for g in groups if rows[g[0]].get('source_role')]
 domains={'all':val,'original':np.array([i for i in val if not rows[i].get('source_role')],int),'nominal_acquisition':np.array([i for i in val if rows[i].get('source_role','').startswith('Predeadline')],int),'nominal_home':np.array([i for i in val if rows[i].get('source_role','').startswith('Nominal-centred')],int)}
 domains['near_25mm']=np.array([i for i in val if rows[i].get('source_role','').startswith('Near25mm')],int)
 if any(name not in domains or not len(domains[name]) for name in a.selection_domains):raise ValueError('Every predeclared selection domain requires held-out episodes')
 model=PretrainedKeypointRefiner(None,False,point_count=5,condition_camera=True);model.load_state_dict(initial['model'],strict=True);model.backbone.layer4=nn.Identity();model.backbone.avgpool=nn.Identity();model.backbone.fc=nn.Identity();model.cuda()
 optical=torch.tensor(np.stack([optics[c] for c in cameras]),device='cuda',dtype=torch.float32);cad=torch.tensor(CAD,device='cuda',dtype=torch.float32)
 solve_pose=rigid_pose if a.pose_solver=='kabsch' else planar_pose
 td=torch.tensor(truth[:,:3,3],device='cuda',dtype=torch.float32);rd=torch.tensor(truth[:,:3,:3],device='cuda',dtype=torch.float32);cloud_target=torch.tensor(cloud_truth,device='cuda',dtype=torch.float32)
 # Exact-label geometry preflight, including a nonzero crop-origin shift.
 check=torch.tensor(native[:16],device='cuda',dtype=torch.float32,requires_grad=True);cloud=ray_cloud(check,optical);ct,cr=solve_pose(cloud,cad);terr=float((ct-td[:16]).norm(dim=-1).max()*1000);rerr=float((cr-rd[:16]).abs().max());probe=cloud.square().sum()+ct.square().sum()+cr.square().sum();probe.backward();finite=bool(torch.isfinite(check.grad).all());assert terr<.01 and rerr<1e-4 and finite,(terr,rerr,finite)
 shifted=local[:16]+[3,-2];recovered=shifted+(origins-[3,-2])[None,:,None,:];assert np.max(abs(recovered-native[:16]))<1e-10
 plan=dict(schema='sc_spatial_grasp_training/v1',mode=a.mode,source=str(a.source),pixels=str(a.pixels),initial_checkpoint=str(a.checkpoint),initial_checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256((a.source/'manifest.json').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),crop_origins_native=origins.tolist(),crop_centres=centres,tcp_to_optical={c:optics[c].tolist() for c in cameras},CAD_reference_points=CAD.tolist(),camera_order=cameras,geometry_labels_only=True,GT_runtime_crop_selection=False,crop_jitter=a.crop_jitter,crop_jitter_corrects_pixel_labels_and_origins=True,geometry_preflight=dict(translation_max_mm=terr,rotation_matrix_max_abs=rerr,finite_gradient=finite),sampling='Episode balanced, half batch from added nominal source domains',training_scope='Backbone layer3 plus spatial decoder; earlier ImageNet stages and their BN stats remain frozen',parameters=sum(x.numel() for x in model.parameters()),trainable_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad),split_counts={s:dict(samples=sum(r['split']==s for r in rows),episodes=len({r['episode'] for r in rows if r['split']==s}),resets=len({r['scene'] for r in rows if r['split']==s})) for s in ['train','validation']},physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],min_updates=a.min_updates,patience=a.patience,maximum_updates=a.updates,minutes=a.minutes,stopping_rule='Validation plateau after LR5e-5reductions to1e-6; an update/time ceiling is incomplete',selection='Worst of translation_p95/0.4mm and orientation_p95/1degree across every validation domain, including update0',final_scenes_opened=False)
 plan['native_crop_sizes']={c:int(native_sizes[i]) for i,c in enumerate(cameras)}
 plan['native_crop_origins']={c:origins[i].tolist() for i,c in enumerate(cameras)}
 plan['crop_geometry_sha256']=hashlib.sha256(a.crop_geometry.read_bytes()).hexdigest()
 plan['crop_geometry']=geometry
 plan['selection_domains']=a.selection_domains
 plan['pose_solver']=a.pose_solver
 plan['rigid_fit_source_sha256']=hashlib.sha256(Path(__file__).with_name('differentiable_rigid_fit.py').read_bytes()).hexdigest()
 plan['pose_solver_backward']='SO3 implicit stationarity solve; forward matches deployed rigid SVD' if a.pose_solver=='kabsch' else 'Legacy asymmetric orthonormalized housing edges'
 plan['selection']='Worst translation_p95/0.4mm or orientation_p95/1degree across predeclared deployment domains; all other domains still reported'
 if a.resume:
  previous=json.loads((a.output/'plan.json').read_text());assert all(previous[k]==plan[k] for k in ['manifest_sha256','initial_checkpoint_sha256','mode','crop_jitter','script_sha256','selection_domains','pose_solver','rigid_fit_source_sha256','crop_geometry_sha256'])
 else:
  save(a.output/'plan.json',plan);(a.output/'trainer_snapshot.py').write_bytes(Path(__file__).read_bytes());(a.output/'differentiable_rigid_fit.py').write_bytes(Path(__file__).with_name('differentiable_rigid_fit.py').read_bytes());save(a.output/'split_manifest.json',[dict(index=i,episode=r['episode'],scene=r['scene'],split=r['split'],source_role=r.get('source_role')) for i,r in enumerate(rows)]);np.savez(a.output/'projected_supervision.npz',native_pixels=native,cloud_TCP=cloud_truth,origins=origins)
 def batch(ids,augment=False):
  images=np.array(pixels[ids],copy=True);shift=np.zeros((len(ids),3,2),dtype=int)
  if augment and a.crop_jitter:
   shift=rng.integers(-a.crop_jitter,a.crop_jitter+1,size=shift.shape)
   for j,i in enumerate(ids):
    for camera in range(3):
     moved=local[i,camera]+shift[j,camera]
     if np.min(moved)<0 or np.max(moved)>191:shift[j,camera]=0
     dx,dy=shift[j,camera];images[j,camera]=cv2.warpAffine(images[j,camera],np.array([[1,0,dx],[0,1,dy]],np.float32),(192,192),borderMode=cv2.BORDER_REPLICATE)
  x=torch.from_numpy(images).permute(0,1,4,2,3).reshape(-1,3,192,192).cuda().float()/255
  if augment:x=(x*torch.tensor(rng.uniform(.9,1.1,size=(len(x),1,1,1)),device='cuda',dtype=torch.float32)).clamp(0,1)
  target=torch.tensor((local[ids]+shift[:,:,None,:])/native_steps[None,:,None,None],device='cuda',dtype=torch.float32)
  offset=torch.tensor(origins[None]-shift,device='cuda',dtype=torch.float32)
  return x,target,offset
 def infer(ids):
  model.eval();pred=[]
  with torch.inference_mode():
   for start in range(0,len(ids),24):
    subset=ids[start:start+24];x,_,_=batch(subset);camera=torch.eye(3,device='cuda').repeat(len(subset),1);pred.append(model(x,camera_id=camera).cpu().numpy().reshape(-1,3,5,2)*native_steps[None,:,None,None]+origins[None,:,None,:])
  coords=np.concatenate(pred);return poses_from_pixels(coords,optics,CAD),coords
 def evaluate():
  poses,coords=infer(val);lookup={int(i):j for j,i in enumerate(val)};reports={}
  for name,ids in domains.items():
   if len(ids):
    prediction=poses[[lookup[int(i)] for i in ids]];reports[name]=metrics(prediction,truth[ids])
    counts=Counter(rows[int(i)]['episode'] for i in ids)
    weights=np.array([1/counts[rows[int(i)]['episode']] for i in ids]);translation=np.linalg.norm(prediction[:,:3,3]-truth[ids,:3,3],axis=-1)*1000;angle=np.rad2deg((Rotation.from_matrix(truth[ids,:3,:3]).inv()*Rotation.from_matrix(prediction[:,:3,:3])).magnitude())
    def quantiles(value):
     order=np.argsort(value);cumulative=np.cumsum(weights[order])/weights.sum();return np.interp([.5,.95],cumulative,value[order]).tolist()
    reports[name].update(episodes=len(counts),samples=len(ids),episode_balanced_translation_p50_p95_mm=quantiles(translation),episode_balanced_orientation_p50_p95_deg=quantiles(angle))
    relative=np.swapaxes(truth[ids,:3,:3],1,2)@prediction[:,:3,:3];tilt=np.rad2deg(np.arccos(np.clip(relative[:,2,2],-1,1)));quaternion=Rotation.from_matrix(relative).as_quat();twist=np.abs(np.rad2deg((2*np.arctan2(quaternion[:,2],quaternion[:,3])+np.pi)%(2*np.pi)-np.pi))
    reports[name].update(axial_tilt_p50_p95_deg=np.percentile(tilt,[50,95]).tolist(),axial_twist_p50_p95_deg=np.percentile(twist,[50,95]).tolist())
  score=max(max(reports[name]['translation_p50_p95_mm'][1]/.4,reports[name]['orientation_p50_p95_deg'][1],reports[name]['episode_balanced_translation_p50_p95_mm'][1]/.4,reports[name]['episode_balanced_orientation_p50_p95_deg'][1]) for name in a.selection_domains)
  return reports,float(score),poses,coords
 opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=5e-5,weight_decay=1e-4)
 report,best,_,_=evaluate();best_step=0;last_progress=0;first=1;events=[];history=[dict(step=0,metrics=report,score=best)];started=time.monotonic();reason='update_cap'
 def payload(step):return dict(schema='sc_rgb_grasp_geometry/v1',label_time_contract='causal_image_time_and_same_time_TCP',model=model.state_dict(),crop_centres=centres,native_crop_sizes=plan['native_crop_sizes'],native_crop_origins=plan['native_crop_origins'],tcp_to_optical=plan['tcp_to_optical'],CAD_reference_points=CAD,plan=plan,step=step,orientation_approved=False,optimizer=opt.state_dict(),training_state=dict(best=best,best_step=best_step,last_progress=last_progress,lr_events=events,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state()))
 if a.resume:
  saved=torch.load(a.output/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);state=saved['training_state'];best=state['best'];best_step=state['best_step'];last_progress=state['last_progress'];events=state['lr_events'];rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng']);first=saved['step']+1;history=json.loads((a.output/'history.json').read_text())
 else:checkpoint(a.output/'initial.pt',payload(0));checkpoint(a.output/'best.pt',payload(0))
 for step in range(first,a.updates+1):
  if time.monotonic()-started>a.minutes*60:reason='time_cap';break
  ids=np.array([groups[j][rng.integers(len(groups[j]))] for j in rng.integers(len(groups),size=16)])
  ids[:8]=[expanded[j][rng.integers(len(expanded[j]))] for j in rng.integers(len(expanded),size=8)]
  x,target,offset=batch(ids,True);model.train();opt.zero_grad(set_to_none=True);prediction=model(x,camera_id=torch.eye(3,device='cuda').repeat(len(ids),1)).reshape(-1,3,5,2)
  pixel=F.smooth_l1_loss((prediction-target)*191,torch.zeros_like(target),beta=1);loss=pixel
  if a.mode=='joint':
   cloud=ray_cloud(prediction*torch.tensor(native_steps,device='cuda',dtype=torch.float32)[None,:,None,None]+offset[:,:,None,:],optical);position,rotation=solve_pose(cloud,cad);ploss=F.smooth_l1_loss((position-td[ids])/.0004,torch.zeros_like(position));rloss=(rotation-rd[ids]).square().sum((-1,-2)).mean()/(2*(np.pi/180)**2);closs=F.smooth_l1_loss((cloud-cloud_target[ids])/.0004,torch.zeros_like(cloud));loss=.2*pixel+ploss+rloss+.1*closs
  if not torch.isfinite(loss):raise ValueError('Non-finite loss; resume from last finite checkpoint after diagnosis')
  loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
  if step%200==0:
   report,score,_,_=evaluate();history.append(dict(step=step,metrics=report,score=score,training_loss=float(loss.detach()),wall_seconds=time.monotonic()-started))
   if score<best*.995:best=score;best_step=step;last_progress=step;checkpoint(a.output/'best.pt',payload(step))
   if step>=a.min_updates and step-last_progress>=a.patience:
    rate=opt.param_groups[0]['lr']
    if rate<=1.00001e-6:reason='validation_plateau'
    else:
     selected=torch.load(a.output/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(selected['model']);opt.load_state_dict(selected['optimizer']);new=max(1e-6,rate*.3)
     for group in opt.param_groups:group['lr']=new
     events.append(dict(step=step,previous_lr=rate,new_lr=new,restored_best_step=best_step));last_progress=step
   checkpoint(a.output/'last.pt',payload(step));save(a.output/'history.json',history);save(a.output/'learning_rate_events.json',events);print(json.dumps(dict(step=step,score=score,metrics=report)),flush=True)
   if reason=='validation_plateau':break
 checkpoint(a.output/'last.pt',payload(step));model.load_state_dict(torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)['model']);report,score,poses,coords=evaluate();np.save(a.output/'heldout_pose_predictions.npy',poses);np.save(a.output/'heldout_pixel_predictions.npy',coords);np.save(a.output/'heldout_indices.npy',val)
 save(a.output/'summary.json',dict(plan=plan,metrics_by_domain=report,selection_score=score,best_step=best_step,updates_completed=step,stop_reason=reason,plateau_reached=reason=='validation_plateau',final_lr=opt.param_groups[0]['lr'],learning_rate_events=events,gate_passed=score<=1,orientation_approved=False,automatic_runtime_change=False,wall_seconds=time.monotonic()-started))


if __name__=='__main__':
 with selected_gpu_reservation():main()
