#!/usr/bin/env python3
"""Train matched PoseDP/RPDP future connector-pose diffusion policies."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rpdp_geometry", HERE / "rpdp_geometry.py")
geo = importlib.util.module_from_spec(spec); sys.modules[spec.name] = geo; spec.loader.exec_module(geo)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--variant", choices=("pose_dp", "rpdp_local", "aic_rpdp"), required=True)
    p.add_argument("--updates", type=int, default=12000)
    p.add_argument("--patience", type=int, default=2400)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--diffusion-steps", type=int, default=100)
    p.add_argument("--inference-steps", type=int, default=20)
    p.add_argument("--policy-mode",choices=("diffusion","direct"),default="diffusion",
                   help="Diffusion BC or a matched deterministic trajectory-regression ablation.")
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--horizon-weights", default="1,1,1,1",
                   help="Comma-separated BC/selection weights for future waypoints")
    return p.parse_args()


def condition_indices(variant, condition_dim=234):
    # Dataset layout: visual180, visibility6, pose3, sigma3, state42.
    if variant == "pose_dp":
        return list(range(180, 192)) + list(range(192, 199)) + list(range(224, 234)) + list(range(234,condition_dim))
    if variant == "rpdp_local":
        return list(range(0, 192)) + list(range(192, 199)) + list(range(224, 234)) + list(range(234,condition_dim))
    return list(range(condition_dim))


def timestep_embedding(t, dim):
    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half-1, 1))
    x = t.float()[:, None] * freq[None]
    return torch.cat((x.sin(), x.cos()), -1)


class TrajectoryDiffusion(nn.Module):
    def __init__(self, condition_dim, horizon=4, width=192, layers=4, *, fusion=False, visual_dim=0):
        super().__init__(); self.horizon = horizon; self.width = width
        self.input = nn.Linear(9, width)
        self.fusion=fusion;self.visual_dim=visual_dim
        if fusion:
            if visual_dim<=0 or visual_dim>=condition_dim:raise ValueError("invalid RPDP visual split")
            self.visual_condition=nn.Sequential(nn.LayerNorm(visual_dim),nn.Linear(visual_dim,width),nn.SiLU(),nn.Linear(width,width))
            pose_dim=condition_dim-visual_dim
            self.pose_condition=nn.Sequential(nn.LayerNorm(pose_dim),nn.Linear(pose_dim,width),nn.SiLU(),nn.Linear(width,width))
            self.visual_gate=nn.Linear(width,width)
        else:
            self.condition = nn.Sequential(nn.LayerNorm(condition_dim), nn.Linear(condition_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.time = nn.Sequential(nn.Linear(64, width), nn.SiLU(), nn.Linear(width, width))
        self.position = nn.Parameter(torch.randn(1, horizon, width)*.01)
        layer = nn.TransformerEncoderLayer(width, 6, width*4, dropout=.05,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 9))

    def forward(self, trajectory, timestep, condition):
        if self.fusion:
            visual=self.visual_condition(condition[:,:self.visual_dim])
            pose=self.pose_condition(condition[:,self.visual_dim:])
            context=pose+torch.sigmoid(self.visual_gate(pose))*visual
        else:
            context=self.condition(condition)
        context = context + self.time(timestep_embedding(timestep, 64))
        x = self.input(trajectory) + self.position + context[:, None]
        return self.output(self.blocks(x))


def schedule(n, device):
    beta = torch.linspace(1e-4, .02, n, device=device)
    alpha = 1-beta
    return beta, alpha, torch.cumprod(alpha, 0)


def tensors(rows, indices):
    return (torch.stack([r["condition"][indices] for r in rows]),
            torch.stack([r["future_pose9"] for r in rows]),
            torch.stack([r["current_pose9"] for r in rows]))


@torch.no_grad()
def sample(model, condition, shape, alpha_bar, steps, seed, prediction_type="sample"):
    generator = torch.Generator(device=condition.device).manual_seed(seed)
    x = torch.randn(shape, generator=generator, device=condition.device)
    timeline = torch.linspace(len(alpha_bar)-1, 0, steps, device=condition.device).round().long().unique_consecutive()
    for i, t in enumerate(timeline):
        tv = torch.full((shape[0],), int(t), dtype=torch.long, device=condition.device)
        prediction = model(x, tv, condition)
        ab = alpha_bar[t]
        if prediction_type == "sample":
            x0=prediction;eps=(x-torch.sqrt(ab)*x0)/torch.sqrt(1-ab).clamp_min(1e-6)
        elif prediction_type == "epsilon":
            eps=prediction;x0=(x-torch.sqrt(1-ab)*eps)/torch.sqrt(ab)
        else:raise ValueError(f"unsupported prediction_type: {prediction_type}")
        if i == len(timeline)-1: x = x0; continue
        previous = timeline[i+1]; ab_prev = alpha_bar[previous]
        # Deterministic DDIM update (eta=0).
        x = torch.sqrt(ab_prev)*x0 + torch.sqrt(1-ab_prev)*eps
    return x


@torch.no_grad()
def infer_trajectory(model,condition,shape,alpha_bar,steps,seed,prediction_type,policy_mode):
    if policy_mode=="direct":
        return model(torch.zeros(shape,device=condition.device),
                     torch.zeros((shape[0],),dtype=torch.long,device=condition.device),condition)
    return sample(model,condition,shape,alpha_bar,steps,seed,prediction_type)


def angle_error_deg(pred6, truth6):
    pq, tq = geo.rot6d_to_quat(pred6), geo.rot6d_to_quat(truth6)
    dq = geo.quat_mul(geo.quat_conjugate(pq), tq)
    return geo.quat_to_rotvec(dq).norm(dim=-1)*180/math.pi


def quantiles(x):
    x=x.float().reshape(-1)
    return {"mean": float(x.mean()), "median": float(x.median()), "p95": float(torch.quantile(x,.95))}


def trajectory_action(pred,current,adapter,trajectory_semantics):
    p0,q0=geo.unpack_pose9(current)
    if trajectory_semantics=="microstep_chunk":
        waypoint_p,waypoint_q=geo.unpack_pose9(pred)
        return adapter.waypoint_chunk(p0,q0,waypoint_p,waypoint_q)
    p1,q1=geo.unpack_pose9(pred[0])
    return adapter.action_chunk(p0,q0,p1,q1)


def metrics(pred, truth, current, rows, adapter,trajectory_semantics="macro_horizon"):
    trans = (pred[..., :3]-truth[..., :3])*1000
    endpoint = trans[:, -1]
    orientation = angle_error_deg(pred[..., 3:], truth[..., 3:])
    action_errors=[]
    for i,row in enumerate(rows):
        supervised_action=row.get("target_action",row["executed_action"])
        action_errors.append(trajectory_action(pred[i],current[i],adapter,trajectory_semantics)-supervised_action)
    action_errors=torch.stack(action_errors)
    report={
      "rows":len(rows),"episodes":len(set(r["episode_id"] for r in rows)),
      "trajectory_translation_error_mm":quantiles(trans.norm(dim=-1)),
      "endpoint_translation_error_mm":quantiles(endpoint.norm(dim=-1)),
      "endpoint_axial_error_mm":quantiles(endpoint[:,2].abs()),
      "endpoint_lateral_error_mm":quantiles(endpoint[:,:2].norm(dim=-1)),
      "trajectory_orientation_error_deg":quantiles(orientation),
      "endpoint_orientation_error_deg":quantiles(orientation[:,-1]),
      "first_waypoint_adapter_vs_target_translation_mm":quantiles(action_errors[...,:3].norm(dim=-1)*1000),
      "first_waypoint_adapter_vs_target_orientation_deg":quantiles(action_errors[...,3:].norm(dim=-1)*180/math.pi),
    }
    by_phase={}
    for phase in sorted(set(r["phase"] for r in rows)):
        ix=torch.tensor([i for i,r in enumerate(rows) if r["phase"]==phase])
        by_phase[phase]={"rows":len(ix),"endpoint_translation_error_mm":quantiles(endpoint[ix].norm(dim=-1)),
                         "endpoint_orientation_error_deg":quantiles(orientation[ix,-1])}
    report["by_phase"]=by_phase
    return report


def first_waypoint_translation_error_mm(pred, current, rows, adapter,trajectory_semantics="macro_horizon"):
    """Mean per-microstep translation error for the chunk used online."""
    errors=[]
    for i,row in enumerate(rows):
        supervised=row.get("target_action",row["executed_action"])
        predicted=trajectory_action(pred[i],current[i],adapter,trajectory_semantics)
        errors.append((predicted[...,:3]-supervised[...,:3]).norm(dim=-1)*1000)
    return float(torch.stack(errors).mean())


def perturb_condition(x, variant, mode, generator):
    y=x.clone()
    # Pose/sigma are after visual+visibility for local/full and first for pose-only.
    if variant == "pose_dp": start=6
    else: start=186
    if mode == "zero_pose": y[:,start:start+6]=0
    elif mode == "shuffle_pose":
        perm=torch.randperm(len(y),generator=generator,device=y.device)
        y[:,start:start+6]=y[perm,start:start+6]
    elif mode == "zero_visual" and variant != "pose_dp": y[:,:180]=0
    return y


def main():
    a=arguments();a.output_dir.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    device=torch.device(a.device);torch.cuda.set_device(device)
    bundle=torch.load(a.dataset,map_location="cpu",weights_only=False)
    trajectory_semantics=str(bundle.get("trajectory_semantics","macro_horizon"))
    condition_dim=int(bundle["splits"]["fit"][0]["condition"].numel())
    indices=condition_indices(a.variant,condition_dim)
    horizon_weights=torch.tensor([float(x) for x in a.horizon_weights.split(",")],dtype=torch.float32)
    if len(horizon_weights)!=int(bundle["horizon"]) or bool((horizon_weights<=0).any()):
        raise ValueError("horizon-weights must contain one positive value per waypoint")
    horizon_weights=horizon_weights/horizon_weights.mean()
    fit,cal,dev=[bundle["splits"][k] for k in ("fit","calibration","development")]
    xf,yf,cf=tensors(fit,indices);xc,yc,cc=tensors(cal,indices);xd,yd,cd=tensors(dev,indices)
    xm, xs=xf.mean(0),xf.std(0).clamp_min(1e-4);ym=yf.mean((0,1));ys=yf.std((0,1)).clamp_min(1e-3)
    def normx(x):return (x-xm)/xs
    def normy(y):return (y-ym)/ys
    fusion=a.variant != "pose_dp";visual_dim=186 if fusion else 0
    model=TrajectoryDiffusion(len(indices),bundle["horizon"],fusion=fusion,visual_dim=visual_dim).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    _,_,abar=schedule(a.diffusion_steps,device)
    episode_counts=Counter(r["episode_id"] for r in fit)
    weights=torch.tensor([1/episode_counts[r["episode_id"]] for r in fit]);weights/=weights.sum()
    generator=torch.Generator().manual_seed(a.seed)
    transform=fit[0]
    adapter=geo.ConnectorTCPAdapter(transform["tcp_to_connector_p"],transform["tcp_to_connector_q"])
    best,best_loss,best_step=None,float("inf"),0;history=[]
    for update in range(1,a.updates+1):
        ix=torch.multinomial(weights,a.batch_size,replacement=True,generator=generator)
        x=normx(xf[ix]).to(device);clean=normy(yf[ix]).to(device)
        if a.policy_mode=="direct":
            t=torch.zeros((a.batch_size,),dtype=torch.long,device=device)
            pred=model(torch.zeros_like(clean),t,x)
        else:
            t=torch.randint(a.diffusion_steps,(a.batch_size,),generator=generator).to(device)
            noise=torch.randn(clean.shape,generator=generator).to(device)
            noisy=abar[t,None,None].sqrt()*clean+(1-abar[t,None,None]).sqrt()*noise
            # Match PoseInsert's released scheduler convention: predict the clean
            # future pose sequence rather than epsilon.
            pred=model(noisy,t,x)
        loss=((pred-clean).square()*horizon_weights.to(device)[None,:,None]).mean()
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if update==1 or update%200==0:
            model.eval()
            with torch.no_grad():
                estimate=infer_trajectory(model,normx(xc).to(device),yc.shape,abar,a.inference_steps,
                                          a.seed+999,"sample",a.policy_mode)
                estimate=estimate*ys.to(device)+ym.to(device)
                element=F.smooth_l1_loss(estimate,yc.to(device),reduction="none")
                val=(element*horizon_weights.to(device)[None,:,None]).mean().item()
                command_val=first_waypoint_translation_error_mm(estimate.cpu(),cc,cal,adapter,trajectory_semantics)
            model.train();history.append({"update":update,"train_clean_pose_mse":float(loss),
                                          "calibration_pose_huber":val,
                                          "calibration_first_waypoint_target_translation_mm_mean":command_val})
            # Select on physical translation error averaged over the four
            # commands that online control executes, rather than on a tiny raw
            # pose Huber value.  The helper and saved field retain their old
            # ``first_waypoint`` name for checkpoint compatibility.
            if command_val<best_loss-1e-6:
                best_loss,best_step=command_val,update;best={k:v.detach().cpu() for k,v in model.state_dict().items()}
            elif update-best_step>=a.patience: break
    model.load_state_dict(best);model.eval()
    reports={}; predictions={}
    for name,rows,x,y,c in (("fit",fit,xf,yf,cf),("calibration",cal,xc,yc,cc),("development",dev,xd,yd,cd)):
        with torch.no_grad():
            pred=infer_trajectory(model,normx(x).to(device),y.shape,abar,a.inference_steps,
                                  a.seed+1000,"sample",a.policy_mode).cpu()*ys+ym
        predictions[name]=pred;reports[name]=metrics(pred,y,c,rows,adapter,trajectory_semantics)
    # Matched persistence/mean baselines on development.
    persistence=cd[:,None,:].expand_as(yd).clone()
    mean_baseline=yf.mean(0)[None].expand_as(yd)
    reports["baselines"]={"persistence":metrics(persistence,yd,cd,dev,adapter,trajectory_semantics),
                          "constant_mean":metrics(mean_baseline,yd,cd,dev,adapter,trajectory_semantics)}
    reliance={};gen=torch.Generator(device=device).manual_seed(a.seed+17)
    for mode in ("zero_pose","shuffle_pose","zero_visual"):
        if mode=="zero_visual" and a.variant=="pose_dp":continue
        pert=perturb_condition(normx(xd).to(device),a.variant,mode,gen)
        with torch.no_grad(): pred=infer_trajectory(model,pert,yd.shape,abar,a.inference_steps,
                                                    a.seed+1000,"sample",a.policy_mode).cpu()*ys+ym
        reliance[mode]=metrics(pred,yd,cd,dev,adapter,trajectory_semantics)
    checkpoint={"schema_version":1,"variant":a.variant,"model_state_dict":best,
                "model":{"condition_dim":len(indices),"horizon":bundle["horizon"],"width":192,"layers":4,
                         "fusion":fusion,"visual_dim":visual_dim,
                         "fusion_contract":"pose/state + sigmoid(pose/state gate) * three-camera visual" if fusion else "pose/state only"},
                "condition_indices":indices,"condition_mean":xm,"condition_std":xs,
                "target_mean":ym,"target_std":ys,"diffusion_steps":a.diffusion_steps,
                "inference_steps":a.inference_steps,"prediction_type":"sample","tcp_to_connector_p":adapter.tcp_to_connector_p,
                "tcp_to_connector_q":adapter.tcp_to_connector_q,"best_update":best_step,
                "horizon_weights":horizon_weights,
                "checkpoint_selection_metric":"calibration_first_waypoint_target_translation_mm_mean",
                "completed_updates":update,"reports":reports,"reliance":reliance,
                "condition_contract":bundle.get("condition_contract"),
                "temporal_context":bool(bundle.get("temporal_context",False)),
                "trajectory_semantics":trajectory_semantics,
                "policy_mode":a.policy_mode,
                "reserved_final_opened":False,"rl_started":False}
    torch.save(checkpoint,a.output_dir/"checkpoint.pt")
    torch.save({"development_prediction":predictions["development"],"development_truth":yd,
                "development_current":cd},a.output_dir/"heldout_predictions.pt")
    summary={k:v for k,v in checkpoint.items() if not isinstance(v,(torch.Tensor,dict))}
    summary.update({"schema_version":1,"variant":a.variant,"parameters":sum(p.numel() for p in model.parameters()),
                    "best_update":best_step,"completed_updates":update,
                    "best_calibration_first_waypoint_target_translation_mm_mean":best_loss,
                    "horizon_weights":[float(x) for x in horizon_weights],
                    "reports":reports,"reliance":reliance,"history":history,"reserved_final_opened":False})
    (a.output_dir/"metrics.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"variant":a.variant,"best_update":best_step,"completed_updates":update,
                      "development":reports["development"],"baselines":reports["baselines"]},indent=2))


if __name__=="__main__":main()
