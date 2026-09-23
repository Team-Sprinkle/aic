#!/usr/bin/env python3
"""Calibration-select camera rejection from pretrained landmark heatmap uncertainty."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
def module(name,file):
 spec=importlib.util.spec_from_file_location(name,HERE/file);m=importlib.util.module_from_spec(spec);assert spec.loader is not None;spec.loader.exec_module(m);return m
opening=module("opening_confidence","train_opening_landmark_pose_probe.py");pretrained=module("pretrained_confidence","calibrate_pretrained_opening_landmarks.py")

def landmarks_and_variance(model,rows,coarse,size,device):
 x,_,geometry=opening.native_arrays(rows,coarse,size,jitter=False);model=model.to(device).eval();coords=[];variance=[]
 with torch.no_grad():
  for start in range(0,len(x),96):
   logits=model(x[start:start+96].to(device).float()/255);b,c,h,w=logits.shape;p=torch.softmax(logits.flatten(-2),-1);yy,xx=torch.meshgrid(torch.linspace(0,1,h,device=device),torch.linspace(0,1,w,device=device),indexing="ij");grid=torch.stack((xx.flatten(),yy.flatten()),-1);mean=(p[...,None]*grid).sum(-2);var=(p[...,None]*(grid-mean[...,None,:]).square()).sum((-2,-1));coords.append(mean.cpu());variance.append(var.cpu())
 local=torch.cat(coords);var=torch.cat(variance);points=torch.empty(len(rows),3,6,2);uncertainty=torch.empty(len(rows),3)
 for pred,unc,(row,view,left,top,width,height) in zip(local,var,geometry):
  pixel=pred*(size-1)+torch.tensor([left,top]);points[row,view]=pixel/torch.tensor([width-1,height-1]);uncertainty[row,view]=unc[2:6].mean()
 return points,uncertainty

def relative(points,uncertainty,rows,mode):
 values=[]
 for pts,unc,row in zip(points,uncertainty,rows):
  calibration=opening.camera_row(row);plug=opening.tri.triangulate_one(pts,calibration,0)
  if mode=="all": ids=(0,1,2)
  elif mode=="top2": ids=tuple(torch.argsort(unc)[:2].tolist())
  else: ids=tuple(int(x) for x in mode.split("_")[1:])
  selected_points=torch.stack([pts[i] for i in ids]);selected_cal=[calibration[i] for i in ids]
  target=opening.tri.triangulate_one(selected_points,selected_cal,2);values.append((plug-target)*1000)
 return np.asarray(values)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--train-replay",type=Path,required=True);p.add_argument("--evaluation-replay",type=Path,required=True);p.add_argument("--train-manifest",type=Path,required=True);p.add_argument("--evaluation-manifest",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--crop-size",type=int,default=160);p.add_argument("--device",default="cuda");a=p.parse_args();device=torch.device(a.device)
 tm=json.loads(a.train_manifest.read_text());em=json.loads(a.evaluation_manifest.read_text());ci={3,7,12,16};fit_ids=[x["episode_id"] for i,x in enumerate(tm["train"]) if i not in ci];cal_ids=[x["episode_id"] for i,x in enumerate(tm["train"]) if i in ci];eval_ids=[x["episode_id"] for x in em["development"]]
 train,_=opening.world.load([a.train_replay],set(fit_ids+cal_ids));evaluation,_=opening.world.load([a.evaluation_replay],set(eval_ids));train,_=opening.attach(train,a.train_replay);evaluation,_=opening.attach(evaluation,a.evaluation_replay);fit=[r for r in train if r["episode_id"] in fit_ids];cal=[r for r in train if r["episode_id"] in cal_ids]
 saved=torch.load(a.checkpoint,map_location="cpu",weights_only=False);locator=opening.crop.Locator();locator.load_state_dict(saved["locator"]);model=pretrained.MobileNetLandmarks();model.load_state_dict(saved["landmark"]);fc,_=opening.crop.locator_predict(locator,opening.coarse_rows(fit),device);cc,_=opening.crop.locator_predict(locator,opening.coarse_rows(cal),device);ec,_=opening.crop.locator_predict(locator,opening.coarse_rows(evaluation),device);fl,fu=landmarks_and_variance(model,fit,fc,a.crop_size,device);cl,cu=landmarks_and_variance(model,cal,cc,a.crop_size,device);el,eu=landmarks_and_variance(model,evaluation,ec,a.crop_size,device)
 def points(coarse,land):
  out=torch.empty(len(land),3,4);out[:,:,:2]=coarse[:,:,:2];out[:,:,2:]=land[:,:,2:6].mean(2);return out
 fp,cp,ep=points(fc,fl),points(cc,cl),points(ec,el);target=np.stack([r["translation_mm"].numpy() for r in fit]);candidates=[]
 for mode in ("all","pair_0_1","pair_0_2","pair_1_2","top2"):
  affine=opening.tri.affine_fit(relative(fp,fu,fit,mode),target);translation=np.c_[relative(cp,cu,cal,mode),np.ones(len(cal))]@affine;pred=torch.zeros(len(cal),6);pred[:,:3]=torch.tensor(translation,dtype=torch.float32);near=[i for i,r in enumerate(cal) if r["signed_depth_m"]>=-.003];metric=opening.world.metrics([cal[i] for i in near],pred[near],torch.zeros(len(near),2),torch.full((len(near),3),1/3),torch.zeros(len(near),6),np.ones(6))[0];candidates.append({"mode":mode,"score":metric["lateral_error_mm"]["median"]+metric["lateral_error_mm"]["p95"],"metrics":metric,"affine":affine})
 selected=min(candidates,key=lambda x:x["score"]);translation=np.c_[relative(ep,eu,evaluation,selected["mode"]),np.ones(len(evaluation))]@selected["affine"];pred=torch.zeros(len(evaluation),6);pred[:,:3]=torch.tensor(translation,dtype=torch.float32);near=[i for i,r in enumerate(evaluation) if r["signed_depth_m"]>=-.003];metric=opening.world.metrics([evaluation[i] for i in near],pred[near],torch.zeros(len(near),2),torch.full((len(near),3),1/3),torch.zeros(len(near),6),np.ones(6))[0];passed=bool(metric["lateral_error_mm"]["median"]<=.25 and metric["lateral_error_mm"]["p95"]<=.5 and (metric["lateral_correction_sign_accuracy"] or 0)>=.9)
 result={"schema_version":1,"status":"diagnostic_on_previously_opened_development","representation":"coarse plug; pretrained corners; calibration-selected camera subset or per-frame heatmap-variance top two","selection":{"rule":"minimum calibration near-port lateral median+p95","selected":{k:selected[k] for k in ("mode","score")},"candidates":[{k:c[k] for k in ("mode","score","metrics")} for c in candidates]},"heldout_near_port":metric,"gate":{"required":"near-port lateral median <=0.25 mm, p95 <=0.5 mm, sign >=0.9","passed":passed,"promotion_eligible":False,"reason":"camera rejection proposed after this development split was opened; fresh development required"}}
 a.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"selection":result["selection"]["selected"],"heldout":metric,"gate":result["gate"]},indent=2))
if __name__=="__main__":main()
