#!/usr/bin/env python3
"""Calibration-select a causal world-space filter for pretrained opening corners."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
def module(name,file):
 spec=importlib.util.spec_from_file_location(name,HERE/file);m=importlib.util.module_from_spec(spec);assert spec.loader is not None;spec.loader.exec_module(m);return m
opening=module("opening_pretrained_filter","train_opening_landmark_pose_probe.py")
pretrained=module("pretrained_opening_filter","calibrate_pretrained_opening_landmarks.py")
filters=module("opening_world_filters","evaluate_opening_world_filter.py")

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--train-replay",type=Path,required=True);p.add_argument("--evaluation-replay",type=Path,required=True);p.add_argument("--train-manifest",type=Path,required=True);p.add_argument("--evaluation-manifest",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--auxiliary-checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--crop-size",type=int,default=160);p.add_argument("--device",default="cuda");a=p.parse_args();device=torch.device(a.device)
 tm=json.loads(a.train_manifest.read_text());em=json.loads(a.evaluation_manifest.read_text());cal_indices={3,7,12,16};fit_ids=[x["episode_id"] for i,x in enumerate(tm["train"]) if i not in cal_indices];cal_ids=[x["episode_id"] for i,x in enumerate(tm["train"]) if i in cal_indices];eval_ids=[x["episode_id"] for x in em["development"]]
 train,_=opening.world.load([a.train_replay],set(fit_ids+cal_ids));evaluation,_=opening.world.load([a.evaluation_replay],set(eval_ids));train,_=opening.attach(train,a.train_replay);evaluation,_=opening.attach(evaluation,a.evaluation_replay);fit=[r for r in train if r["episode_id"] in fit_ids];cal=[r for r in train if r["episode_id"] in cal_ids]
 saved=torch.load(a.checkpoint,map_location="cpu",weights_only=False);locator=opening.crop.Locator();locator.load_state_dict(saved["locator"]);landmark=pretrained.MobileNetLandmarks();landmark.load_state_dict(saved["landmark"])
 fitc,calc,evalc=opening.coarse_rows(fit),opening.coarse_rows(cal),opening.coarse_rows(evaluation);fcoarse,_=opening.crop.locator_predict(locator,fitc,device);ccoarse,_=opening.crop.locator_predict(locator,calc,device);ecoarse,_=opening.crop.locator_predict(locator,evalc,device);fland,_=opening.landmark_predict(landmark,fit,fcoarse,a.crop_size,device);cland,_=opening.landmark_predict(landmark,cal,ccoarse,a.crop_size,device);eland,_=opening.landmark_predict(landmark,evaluation,ecoarse,a.crop_size,device)
 fp=opening.pair_points(fcoarse,fland,"coarse","corners");cp=opening.pair_points(ccoarse,cland,"coarse","corners");ep=opening.pair_points(ecoarse,eland,"coarse","corners");fplug,fopen=filters.world_points(fp,fit);cplug,copen=filters.world_points(cp,cal);eplug,eopen=filters.world_points(ep,evaluation);target=np.stack([r["translation_mm"].numpy() for r in fit]);options=[("ema",x) for x in (1.,.75,.5,.25,.1,.05)]+[("mean",0),("median",5),("median",15)]
 candidates=[]
 for mode,param in options:
  frel=(fplug-filters.filter_opening(fopen,fit,mode,param))*1000;affine=opening.tri.affine_fit(frel,target);crel=(cplug-filters.filter_opening(copen,cal,mode,param))*1000;translation=np.c_[crel,np.ones(len(cal))]@affine;pred=torch.zeros(len(cal),6);pred[:,:3]=torch.tensor(translation,dtype=torch.float32);near=[i for i,r in enumerate(cal) if r["signed_depth_m"]>=-.003];metric=opening.world.metrics([cal[i] for i in near],pred[near],torch.zeros(len(near),2),torch.full((len(near),3),1/3),torch.zeros(len(near),6),np.ones(6))[0];candidates.append({"mode":mode,"parameter":param,"score":metric["lateral_error_mm"]["median"]+metric["lateral_error_mm"]["p95"],"metrics":metric,"affine":affine})
 selected=min(candidates,key=lambda x:x["score"]);erel=(eplug-filters.filter_opening(eopen,evaluation,selected["mode"],selected["parameter"]))*1000;translation=np.c_[erel,np.ones(len(evaluation))]@selected["affine"]
 aux=torch.load(a.auxiliary_checkpoint,map_location="cpu",weights_only=False);models=[opening.world.Probe(384) for _ in aux["feature_members"]]
 for model,state in zip(models,aux["feature_members"]):model.load_state_dict(state)
 base,var,prob,phase=opening.world.predict(models,evaluation,"feature",aux["target_mean"],aux["target_std"],device);base[:,:3]=torch.tensor(translation,dtype=torch.float32);cpred,_,_,_=opening.world.predict(models,cal,"feature",aux["target_mean"],aux["target_std"],device);cy=torch.stack([torch.cat((r["translation_mm"],r["rotation_deg"])) for r in cal]);res=((cpred-cy)**2).mean(0).numpy().clip(1e-6);near=[i for i,r in enumerate(evaluation) if r["signed_depth_m"]>=-.003];metric=opening.world.metrics([evaluation[i] for i in near],base[near],prob[near],phase[near],var[near],res)[0];passed=bool(metric["lateral_error_mm"]["median"]<=.25 and metric["lateral_error_mm"]["p95"]<=.5 and (metric["lateral_correction_sign_accuracy"] or 0)>=.9)
 result={"schema_version":1,"status":"diagnostic_on_previously_opened_development","representation":"coarse plug, pretrained corners, causal world-space opening filter","selection":{"rule":"minimum calibration near-port lateral median+p95","selected":{k:selected[k] for k in ("mode","parameter","score")},"candidates":[{k:c[k] for k in ("mode","parameter","score","metrics")} for c in candidates]},"heldout_near_port":metric,"gate":{"required":"near-port lateral median <=0.25 mm, p95 <=0.5 mm, sign >=0.9","passed":passed,"promotion_eligible":False,"reason":"filter proposed after this development split was opened; a fresh split is required"}}
 a.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"selection":result["selection"]["selected"],"heldout":metric,"gate":result["gate"]},indent=2))
if __name__=="__main__":main()
