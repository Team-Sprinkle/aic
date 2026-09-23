#!/usr/bin/env python3
"""Select a causal world-space opening tracker on calibration episodes."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("opening_probe", HERE / "train_opening_landmark_pose_probe.py")
probe = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(probe)


def arguments():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--train-replay",type=Path,required=True)
    p.add_argument("--evaluation-replay",type=Path,required=True);p.add_argument("--scene-manifest",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--crop-size",type=int,default=160);p.add_argument("--device",default="cuda");return p.parse_args()


def world_points(points, rows):
    plug, opening = [], []
    for value, row in zip(points, rows):
        calibration=probe.camera_row(row)
        plug.append(probe.tri.triangulate_one(value,calibration,0))
        opening.append(probe.tri.triangulate_one(value,calibration,2))
    return np.asarray(plug),np.asarray(opening)


def filter_opening(values, rows, mode, parameter):
    output=np.empty_like(values); history={}
    for index,(value,row) in enumerate(zip(values,rows)):
        episode=row["episode_id"]; old=history.setdefault(episode,[]); old.append(value)
        if mode=="ema":
            filtered=value if len(old)==1 else parameter*value+(1-parameter)*output[index-1]
        elif mode=="mean": filtered=np.mean(old,axis=0)
        else: filtered=np.median(old[-int(parameter):],axis=0)
        output[index]=filtered
    return output


def feature_models(saved):
    models=[]
    for state in saved["feature_members"]:
        model=probe.world.Probe(384);model.load_state_dict(state);models.append(model)
    return models


def main():
    a=arguments();device=torch.device(a.device);manifest=json.loads(a.scene_manifest.read_text());calibration_indices={3,7,12,16}
    fit_ids=[x["episode_id"] for i,x in enumerate(manifest["train"]) if i not in calibration_indices]
    cal_ids=[x["episode_id"] for i,x in enumerate(manifest["train"]) if i in calibration_indices]
    eval_ids=[x["episode_id"] for x in manifest["development"]]
    train,_=probe.world.load([a.train_replay],set(fit_ids+cal_ids));evaluation,_=probe.world.load([a.evaluation_replay],set(eval_ids))
    train,_=probe.attach(train,a.train_replay);evaluation,_=probe.attach(evaluation,a.evaluation_replay)
    fit=[r for r in train if r["episode_id"] in fit_ids];calibration=[r for r in train if r["episode_id"] in cal_ids]
    saved=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    locator=probe.crop.Locator();locator.load_state_dict(saved["locator"])
    landmark=probe.LandmarkHeatmaps();landmark.load_state_dict(saved["landmark"])
    fit_c,cal_c,eval_c=probe.coarse_rows(fit),probe.coarse_rows(calibration),probe.coarse_rows(evaluation)
    fit_coarse,_=probe.crop.locator_predict(locator,fit_c,device);cal_coarse,_=probe.crop.locator_predict(locator,cal_c,device);eval_coarse,_=probe.crop.locator_predict(locator,eval_c,device)
    fit_land,_=probe.landmark_predict(landmark,fit,fit_coarse,a.crop_size,device);cal_land,_=probe.landmark_predict(landmark,calibration,cal_coarse,a.crop_size,device);eval_land,_=probe.landmark_predict(landmark,evaluation,eval_coarse,a.crop_size,device)
    models=feature_models(saved);mean=saved["target_mean"];std=saved["target_std"]
    cpred,_,_,_=probe.world.predict(models,calibration,"feature",mean,std,device)
    cy=torch.stack([torch.cat((r["translation_mm"],r["rotation_deg"])) for r in calibration]);residual=((cpred-cy)**2).mean(0).numpy().clip(1e-6)
    base,variance,probability,phase=probe.world.predict(models,evaluation,"feature",mean,std,device)
    target_fit=np.stack([r["translation_mm"].numpy() for r in fit]);filter_options=[("ema",x) for x in (1.,.5,.25,.1,.05)]+[("mean",0),("median",5),("median",15)]
    candidates=[]
    for plug_source in ("coarse","landmark"):
        for port_source in ("coarse","entrance","corners"):
            fp=probe.pair_points(fit_coarse,fit_land,plug_source,port_source);cp=probe.pair_points(cal_coarse,cal_land,plug_source,port_source)
            fplug,fopening=world_points(fp,fit);cplug,copening=world_points(cp,calibration)
            for mode,parameter in filter_options:
                frel=(fplug-filter_opening(fopening,fit,mode,parameter))*1000
                affine=probe.tri.affine_fit(frel,target_fit)
                crel=(cplug-filter_opening(copening,calibration,mode,parameter))*1000
                translation=np.c_[crel,np.ones(len(calibration))]@affine
                prediction=cpred.clone();prediction[:,:3]=torch.tensor(translation,dtype=torch.float32)
                near=[i for i,r in enumerate(calibration) if r["signed_depth_m"]>=-.003]
                metrics=probe.world.metrics([calibration[i] for i in near],prediction[near],torch.zeros(len(near),2),torch.full((len(near),3),1/3),torch.zeros(len(near),6),residual)[0]
                score=metrics["lateral_error_mm"]["median"]+metrics["lateral_error_mm"]["p95"]
                candidates.append({"plug":plug_source,"port":port_source,"mode":mode,"parameter":parameter,"score":score,"metrics":metrics,"affine":affine})
    selected=min(candidates,key=lambda x:x["score"])
    ep=probe.pair_points(eval_coarse,eval_land,selected["plug"],selected["port"]);eplug,eopening=world_points(ep,evaluation)
    erel=(eplug-filter_opening(eopening,evaluation,selected["mode"],selected["parameter"]))*1000
    translation=np.c_[erel,np.ones(len(evaluation))]@selected["affine"];prediction=base.clone();prediction[:,:3]=torch.tensor(translation,dtype=torch.float32)
    near=[i for i,r in enumerate(evaluation) if r["signed_depth_m"]>=-.003]
    metrics=probe.world.metrics([evaluation[i] for i in near],prediction[near],probability[near],phase[near],variance[near],residual)[0]
    passed=bool(metrics["lateral_error_mm"]["median"]<=.25 and metrics["lateral_error_mm"]["p95"]<=.5 and (metrics["lateral_correction_sign_accuracy"] or 0)>=.9)
    result={"schema_version":1,"status":"diagnostic_on_previously_opened_development","selection":{"rule":"minimum calibration near-port lateral median+p95","selected":{k:selected[k] for k in ("plug","port","mode","parameter","score")},"candidates":[{k:c[k] for k in ("plug","port","mode","parameter","score","metrics")} for c in candidates]},"heldout_near_port":metrics,"gate":{"required":"near-port lateral median <=0.25 mm, p95 <=0.5 mm, sign >=0.9","passed":passed,"promotion_eligible":False,"reason":"representation proposed after this development split was opened; fresh development is required"}}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"output":str(a.output),"selection":result["selection"]["selected"],"metrics":metrics,"gate":result["gate"]},indent=2))


if __name__=="__main__":main()
