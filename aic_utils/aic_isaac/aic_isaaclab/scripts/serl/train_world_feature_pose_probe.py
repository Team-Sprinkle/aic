#!/usr/bin/env python3
"""Train episode-grouped object-to-target pose probes from Isaac macro replay."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def args():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--train-replay',type=Path,action='append',required=True);p.add_argument('--evaluation-replay',type=Path,action='append',required=True)
 p.add_argument('--scene-manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
 p.add_argument('--updates',type=int,default=5000);p.add_argument('--patience',type=int,default=800);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--ensemble-size',type=int,default=5);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=20260920)
 return p.parse_args()

def qconj(q): return np.array([q[0],-q[1],-q[2],-q[3]],np.float64)
def qmul(a,b):
 w1,x1,y1,z1=a;w2,x2,y2,z2=b
 return np.array([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2],np.float64)
def qrot(q,v): return qmul(qmul(q,np.r_[0.,v]),qconj(q))[1:]
def rotvec(q):
 q=np.asarray(q,np.float64);q=q/max(np.linalg.norm(q),1e-12);q=-q if q[0]<0 else q
 s=np.linalg.norm(q[1:]); return np.zeros(3) if s<1e-9 else q[1:]/s*(2*math.atan2(s,max(q[0],1e-12)))
def rotvec_deg_to_quat(v):
 v=np.asarray(v,np.float64)*math.pi/180.; angle=np.linalg.norm(v,axis=-1,keepdims=True);half=.5*angle
 scale=np.where(angle>1e-12,np.sin(half)/np.maximum(angle,1e-12),.5)
 return np.concatenate((np.cos(half),v*scale),axis=-1)
def geom_row(g):
 body=np.asarray(g['body_world_env0'],np.float64); target=np.asarray(g['target_world_env0'],np.float64)
 qb=np.asarray(g['body_orientation_wxyz_by_env'][0],np.float64); qt=np.asarray(g['target_orientation_wxyz_by_env'][0],np.float64)
 rel_t=qrot(qconj(qt),body-target); rel_q=qmul(qconj(qt),qb)
 axis=qrot(qconj(qt),np.asarray(g['axis_world_env0'],np.float64)); axis=axis/max(np.linalg.norm(axis),1e-12)
 return rel_t,rotvec(rel_q),axis,float(g['target_depth_m_env0']),float(g['signed_depth_m_env0']),float(g['lateral_error_m_env0']),float(g['orientation_error_rad_env0'])
def bodypos(allg,name):
 g=allg.get(name) or {}; v=g.get('world_env0',g.get('body_world_env0')); return None if v is None else np.asarray(v,np.float64)
def file_id(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(8<<20):h.update(b)
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}

def load(paths, allowed):
 rows=[]; audit=Counter()
 for path in paths:
  payload=torch.load(path,map_location='cpu',weights_only=False); transitions=payload['transitions']; audit['transitions_seen']+=len(transitions)
  for idx,t in enumerate(transitions):
   m=t.get('metadata') or {}; ep=(m.get('causal_episode') or {}).get('episode_id')
   if not ep or ep not in allowed: audit['excluded_episode']+=1;continue
   g=m.get('causal_insertion_geometry'); ag=m.get('causal_all_body_insertion_geometry')
   if not g or not ag: audit['missing_causal_geometry']+=1;continue
   term=t.get('terminal_observation') if (m.get('terminated') or m.get('truncated')) else None
   pg=(term or {}).get('insertion_geometry') if term else m.get('post_step_insertion_geometry')
   pag=(term or {}).get('all_body_insertion_geometry') if term else m.get('post_step_all_body_insertion_geometry')
   if not pg or not pag: audit['missing_post_geometry']+=1;continue
   # Reject automatic-reset joins by identity and by the explicit terminal contract.
   if (m.get('terminated') or m.get('truncated')) and term is None: audit['terminal_without_snapshot']+=1;continue
   feature=(t.get('obs') or {}).get('world_feature'); state=(t.get('obs') or {}).get('state')
   if not isinstance(feature,torch.Tensor) or feature.numel()!=384 or not isinstance(state,torch.Tensor):audit['missing_feature_or_state']+=1;continue
   rel,rv,axis,target_depth,depth,lateral,ori=geom_row(g); _,_,_,_,postdepth,_,_=geom_row(pg)
   p0=bodypos(ag,'sfp_tip_link');p1=bodypos(pag,'sfp_tip_link');tcp0=bodypos(ag,'gripper_tcp');tcp1=bodypos(pag,'gripper_tcp')
   if any(v is None for v in (p0,p1,tcp0,tcp1)):audit['missing_motion_body']+=1;continue
   force=np.asarray(m.get('causal_force_xyz_n',state.reshape(-1)[26:29].tolist()),np.float64);fn=float(np.linalg.norm(force));plugmotion=float(np.linalg.norm(p1-p0));tcpmotion=float(np.linalg.norm(tcp1-tcp0))
   contact=fn>=10.; blocked=bool(contact and plugmotion<0.00015)
   phase='contact' if contact else ('approach' if depth < -0.003 else 'alignment')
   motion='lt0p25mm' if plugmotion<0.00025 else ('0p25to1mm' if plugmotion<0.001 else 'ge1mm')
   rows.append({'episode_id':ep,'source_replay':str(path),'transition_index':idx,'feature':feature.reshape(-1).float(),'state':state.reshape(-1).float(),'translation_mm':torch.tensor(rel*1000,dtype=torch.float32),'rotation_deg':torch.tensor(rv*180/math.pi,dtype=torch.float32),'axis_local':axis,'target_depth_m':target_depth,'signed_depth_m':depth,'lateral_m':lateral,'orientation_rad':ori,'contact':float(contact),'blocked':float(blocked),'force_n':fn,'plug_motion_m':plugmotion,'tcp_motion_m':tcpmotion,'phase':phase,'motion_bin':motion,'executed_action':t['action'].reshape(-1).float(),'guide_action':t.get('guide_action'),'model_proposal':torch.tensor(m['model_proposal_24d'],dtype=torch.float32),'model_inference_s':m.get('model_inference_s'),'terminal':bool(term),'post_signed_depth_m':postdepth})
   audit['accepted']+=1
 return rows,dict(audit)

class Probe(nn.Module):
 def __init__(self,n):
  super().__init__();self.net=nn.Sequential(nn.LayerNorm(n),nn.Linear(n,256),nn.GELU(),nn.Linear(256,128),nn.GELU(),nn.Linear(128,11))
 def forward(self,x):return self.net(x)

def train_ensemble(rows,calib,input_key,a,device):
 x=torch.stack([r[input_key] for r in rows]); y=torch.stack([torch.cat((r['translation_mm'],r['rotation_deg'])) for r in rows]); c=torch.tensor([[r['contact'],r['blocked']] for r in rows])
 phase_names=['approach','alignment','contact'];phase_index={v:i for i,v in enumerate(phase_names)};ph=torch.tensor([phase_index[r['phase']] for r in rows],dtype=torch.long)
 episode_counts=Counter(r['episode_id'] for r in rows)
 sample_weights=torch.tensor([1./episode_counts[r['episode_id']] for r in rows],dtype=torch.float32)
 xc=torch.stack([r[input_key] for r in calib]).to(device);yc=torch.stack([torch.cat((r['translation_mm'],r['rotation_deg'])) for r in calib]).to(device);cc=torch.tensor([[r['contact'],r['blocked']] for r in calib],device=device);phc=torch.tensor([phase_index[r['phase']] for r in calib],dtype=torch.long,device=device)
 normalized_weights=sample_weights/sample_weights.sum();ym=(normalized_weights[:,None]*y).sum(0);ys=torch.sqrt((normalized_weights[:,None]*(y-ym).square()).sum(0)).clamp_min(1e-3); models=[]; histories=[]
 for member in range(a.ensemble_size):
  seed=a.seed+member;torch.manual_seed(seed);random.seed(seed);model=Probe(x.shape[1]).to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);best=None;bestloss=float('inf');beststep=0;hist=[]
  gen=torch.Generator().manual_seed(seed)
  for step in range(1,a.updates+1):
   ix=torch.multinomial(sample_weights,a.batch_size,replacement=True,generator=gen);out=model(x[ix].to(device)); reg=F.smooth_l1_loss(out[:,:6],((y[ix]-ym)/ys).to(device)); cls=F.binary_cross_entropy_with_logits(out[:,6:8],c[ix].to(device));phase_loss=F.cross_entropy(out[:,8:11],ph[ix].to(device));loss=reg+0.25*cls+0.15*phase_loss
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
   if step==1 or step%100==0:
    with torch.no_grad():o=model(xc);v=F.smooth_l1_loss(o[:,:6],(yc-ym.to(device))/ys.to(device))+0.25*F.binary_cross_entropy_with_logits(o[:,6:8],cc)+0.15*F.cross_entropy(o[:,8:11],phc)
    val=float(v);hist.append({'step':step,'validation_loss':val})
    if val<bestloss:bestloss=val;beststep=step;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    elif step-beststep>=a.patience:break
  model.load_state_dict(best);models.append(model.cpu());histories.append({'member':member,'seed':seed,'best_step':beststep,'best_validation_loss':bestloss,'completed_step':step,'history':hist})
 return models,ym,ys,histories

def predict(models,rows,key,ym,ys,device):
 x=torch.stack([r[key] for r in rows]).to(device); outs=[]
 for m in models:
  m=m.to(device).eval()
  with torch.no_grad():outs.append(m(x).cpu())
 stack=torch.stack(outs);reg=stack[:,:,:6]*ys+ym;return reg.mean(0),reg.var(0,unbiased=False),torch.sigmoid(stack[:,:,6:8]).mean(0),torch.softmax(stack[:,:,8:11],dim=-1).mean(0)
def qtile(v,q):return float(np.quantile(np.asarray(v),q)) if len(v) else None
def classification(y,p):
 yh=p>=.5;tp=int(((yh==1)&(y==1)).sum());fp=int(((yh==1)&(y==0)).sum());fn=int(((yh==0)&(y==1)).sum())
 return {'count':len(y),'positives':int(y.sum()),'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'accuracy':float((yh==y).mean())}
def metrics(rows,pred,prob,phase_prob,var,residual_var):
 true=np.stack([torch.cat((r['translation_mm'],r['rotation_deg'])).numpy() for r in rows]);pr=pred.numpy();err=pr-true
 trans=np.linalg.norm(err[:,:3],axis=1)
 qtrue=rotvec_deg_to_quat(true[:,3:]);qpred=rotvec_deg_to_quat(pr[:,3:]);qdot=np.abs(np.sum(qtrue*qpred,axis=1)).clip(0.,1.)
 rot=2*np.arccos(qdot)*180/math.pi; axial=[];lat=[];latmag=[];sign=[]
 for r,p in zip(rows,pr):
  axis=r['axis_local']; tpred=p[:3]/1000; trel=r['translation_mm'].numpy()/1000; terr=tpred-trel
  predlatvec=tpred-axis*np.dot(tpred,axis);truelatvec=trel-axis*np.dot(trel,axis)
  axial.append(abs(float(np.dot(terr,axis)))*1000)
  # Vector error is the quantity needed to resolve a submillimetre correction.
  # Also retain scalar corridor-radius error for comparison with the simulator metric.
  lat.append(float(np.linalg.norm(terr-axis*np.dot(terr,axis)))*1000)
  latmag.append(abs(np.linalg.norm(predlatvec)-np.linalg.norm(truelatvec))*1000)
  if np.linalg.norm(truelatvec)>0.00025:sign.append(float(np.dot(predlatvec,truelatvec)>0.0))
 phase_names=['approach','alignment','contact'];phase_true=np.array([phase_names.index(r['phase']) for r in rows]);phase_pred=phase_prob.argmax(1).numpy()
 out={'count':len(rows),'episode_count':len(set(r['episode_id'] for r in rows)),'translation_error_mm':{'mean':float(np.mean(trans)),'median':float(np.median(trans)),'p95':qtile(trans,.95)},'axial_error_mm':{'mean':float(np.mean(axial)),'median':float(np.median(axial)),'p95':qtile(axial,.95)},'lateral_error_mm':{'definition':'norm of predicted-minus-true translation projected onto target lateral plane','mean':float(np.mean(lat)),'median':float(np.median(lat)),'p95':qtile(lat,.95)},'lateral_magnitude_error_mm':{'definition':'absolute error in lateral corridor radius','mean':float(np.mean(latmag)),'median':float(np.median(latmag)),'p95':qtile(latmag,.95)},'orientation_error_deg':{'mean':float(np.mean(rot)),'median':float(np.median(rot)),'p95':qtile(rot,.95)},'lateral_correction_sign_accuracy':float(np.mean(sign)) if sign else None,'contact':classification(np.array([r['contact'] for r in rows]),prob[:,0].numpy()),'blocked':classification(np.array([r['blocked'] for r in rows]),prob[:,1].numpy()),'phase':{'accuracy':float((phase_pred==phase_true).mean()),'support':{name:int((phase_true==i).sum()) for i,name in enumerate(phase_names)},'confusion_rows_true_cols_pred':[[int(((phase_true==i)&(phase_pred==j)).sum()) for j in range(3)] for i in range(3)]}}
 totalvar=var.numpy()+residual_var[None,:];z=np.abs(err)/np.sqrt(np.maximum(totalvar,1e-9));out['uncertainty']={'coverage_68':float((z<=1).mean()),'coverage_95':float((z<=1.96).mean()),'gaussian_nll':float(.5*np.mean(np.log(2*np.pi*np.maximum(totalvar,1e-9))+err**2/np.maximum(totalvar,1e-9)))}
 return out,{'true':true,'pred':pr,'translation_error':trans,'rotation_error':rot,'axial_error':np.asarray(axial),'lateral_error':np.asarray(lat)}
def sliced(rows,pred,prob,phase_prob,var,resvar,key):
 out={}
 for value in sorted(set(r[key] for r in rows)):
  ix=[i for i,r in enumerate(rows) if r[key]==value];out[value]=metrics([rows[i] for i in ix],pred[ix],prob[ix],phase_prob[ix],var[ix],resvar)[0]
 return out

def baseline(rows,train,kind):
 if kind=='constant':
  episodes=sorted(set(r['episode_id'] for r in train)); per_episode_targets=[];per_episode_classes=[]
  for ep in episodes:
   erows=[r for r in train if r['episode_id']==ep];per_episode_targets.append(torch.stack([torch.cat((r['translation_mm'],r['rotation_deg'])) for r in erows]).mean(0));per_episode_classes.append([np.mean([r['contact'] for r in erows]),np.mean([r['blocked'] for r in erows])])
  mean=torch.stack(per_episode_targets).mean(0); pred=mean.repeat(len(rows),1); probs=torch.tensor([np.mean(per_episode_classes,axis=0).tolist()]).repeat(len(rows),1)
  phase_counts=[np.mean([r['phase']==name for r in train]) for name in ['approach','alignment','contact']];return pred,torch.zeros_like(pred),probs,torch.tensor([phase_counts]).repeat(len(rows),1)
 raise ValueError(kind)

def main():
 a=args();device=torch.device(a.device);manifest=json.loads(a.scene_manifest.read_text())
 # Hold out complete reset configurations for calibration, stratified across
 # the four axial distances and both lateral directions in the 4x4 grid.
 calibration_indices={3,5,8,14}
 train_ids=[x['episode_id'] for i,x in enumerate(manifest['train']) if i not in calibration_indices]
 calib_ids=[x['episode_id'] for i,x in enumerate(manifest['train']) if i in calibration_indices]
 eval_ids=[x['episode_id'] for x in manifest['development']]
 alltrain,audit_train=load(a.train_replay,set(train_ids+calib_ids));evaluation,audit_eval=load(a.evaluation_replay,set(eval_ids));fit=[r for r in alltrain if r['episode_id'] in train_ids];calib=[r for r in alltrain if r['episode_id'] in calib_ids]
 if not fit or not calib or not evaluation:raise RuntimeError(f'Empty split fit={len(fit)} calib={len(calib)} eval={len(evaluation)}')
 a.output_dir.mkdir(parents=True,exist_ok=True);config_paths=[Path(x['path']) for x in manifest['train']+manifest['development']];split={'train_episode_ids':train_ids,'calibration_episode_ids':calib_ids,'evaluation_episode_ids':eval_ids,'split_unit':'complete reset configuration / episode_id','sampling':'uniform over training episode_id, then transition','train_rows':len(fit),'calibration_rows':len(calib),'evaluation_rows':len(evaluation),'scene_manifest':file_id(a.scene_manifest),'reset_configuration_files':[file_id(p) for p in config_paths],'source_files':[file_id(p) for p in a.train_replay+a.evaluation_replay],'audit':{'train':audit_train,'evaluation':audit_eval},'terminal_contract':'terminal_observation required; post-reset geometry never joined'};(a.output_dir/'data_manifest.json').write_text(json.dumps(split,indent=2)+'\n')
 report={'data':split,'training_config':{'updates':a.updates,'patience':a.patience,'batch_size':a.batch_size,'learning_rate':a.lr,'ensemble_size':a.ensemble_size,'seed':a.seed,'device':str(device),'probe_architecture':'LayerNorm-input, MLP 256-128-11 GELU','targets':'translation_mm[3], rotation_vector_deg[3], contact, blocked, phase[3]'},'models':{},'slices':{}}
 saved={}
 for key,label in [('feature','world_feature'),('state','state_only')]:
  models,ym,ys,hist=train_ensemble(fit,calib,key,a,device);cp={'input_dim':int(fit[0][key].numel()),'target_mean':ym,'target_std':ys,'members':[m.state_dict() for m in models],'histories':hist};torch.save(cp,a.output_dir/f'{label}_ensemble.pt')
  cpred,cvar,cprob,cphase=predict(models,calib,key,ym,ys,device);ctrue=torch.stack([torch.cat((r['translation_mm'],r['rotation_deg'])) for r in calib]);resvar=((cpred-ctrue)**2).mean(0).numpy().clip(1e-6)
  pred,var,prob,phase_prob=predict(models,evaluation,key,ym,ys,device);overall,raw=metrics(evaluation,pred,prob,phase_prob,var,resvar);report['models'][label]={'overall':overall,'calibration_residual_variance':resvar.tolist(),'training':hist};report['slices'][label]={'phase':sliced(evaluation,pred,prob,phase_prob,var,resvar,'phase'),'motion_size':sliced(evaluation,pred,prob,phase_prob,var,resvar,'motion_bin')};saved[label]=(pred,var,prob,phase_prob,raw,resvar)
 pred,var,prob,phase_prob=baseline(evaluation,fit,'constant');resvar=np.var(np.stack([torch.cat((r['translation_mm'],r['rotation_deg'])).numpy() for r in fit]),axis=0).clip(1e-6);overall,raw=metrics(evaluation,pred,prob,phase_prob,var,resvar);report['models']['constant_mean']={'overall':overall};report['slices']['constant_mean']={'phase':sliced(evaluation,pred,prob,phase_prob,var,resvar,'phase'),'motion_size':sliced(evaluation,pred,prob,phase_prob,var,resvar,'motion_bin')};saved['constant_mean']=(pred,var,prob,phase_prob,raw,resvar)
 report['gate']={'near_port_definition':'signed depth >= -3 mm','required':'lateral median <=0.25 mm, lateral p95 <=0.5 mm, sign accuracy >=0.9','passed':False};near=[i for i,r in enumerate(evaluation) if r['signed_depth_m']>=-.003];wp=saved['world_feature']; nm=metrics([evaluation[i] for i in near],wp[0][near],wp[2][near],wp[3][near],wp[1][near],wp[5])[0] if near else None;report['models']['world_feature']['near_port']=nm;report['gate']['passed']=bool(nm and nm['lateral_error_mm']['median']<=.25 and nm['lateral_error_mm']['p95']<=.5 and (nm['lateral_correction_sign_accuracy'] or 0)>=.9)
 (a.output_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
 with (a.output_dir/'predictions.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['episode_id','phase','motion_bin','signed_depth_mm','lateral_mm','force_n','contact','blocked','true_tx_mm','true_ty_mm','true_tz_mm','pred_tx_mm','pred_ty_mm','pred_tz_mm','translation_error_mm','axial_error_mm','lateral_error_mm','orientation_error_deg'])
  pred,_,_,_,raw,_=saved['world_feature']
  for i,r in enumerate(evaluation):w.writerow([r['episode_id'],r['phase'],r['motion_bin'],r['signed_depth_m']*1000,r['lateral_m']*1000,r['force_n'],r['contact'],r['blocked'],*r['translation_mm'].tolist(),*pred[i,:3].tolist(),raw['translation_error'][i],raw['axial_error'][i],raw['lateral_error'][i],raw['rotation_error'][i]])
 try:
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(1,3,figsize=(12,3.5)); raw=saved['world_feature'][4];ax[0].hist(raw['translation_error'],bins=30);ax[0].set_xlabel('3D translation error (mm)');ax[1].hist(raw['lateral_error'],bins=30);ax[1].axvline(.5,color='r');ax[1].set_xlabel('lateral error (mm)');ax[2].hist(raw['rotation_error'],bins=30);ax[2].set_xlabel('orientation error (deg)');fig.tight_layout();fig.savefig(a.output_dir/'heldout_error_histograms.png',dpi=160);plt.close(fig)
 except Exception as e:(a.output_dir/'plot_error.txt').write_text(repr(e)+'\n')
 print(json.dumps({'output':str(a.output_dir),'gate':report['gate'],'world_feature':report['models']['world_feature']['overall'],'near_port':nm},indent=2))
if __name__=='__main__':main()
