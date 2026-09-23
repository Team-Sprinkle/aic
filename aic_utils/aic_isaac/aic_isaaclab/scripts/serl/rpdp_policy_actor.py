"""Autonomous RPDP actor: observation-only pose diffusion to TCP deltas."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

from pose_gru_policy_actor import IsaacPoseGRUPolicyActor

HERE=Path(__file__).resolve().parent


def load(name,filename):
    spec=importlib.util.spec_from_file_location(name,HERE/filename)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


training=load("rpdp_training_runtime","train_rpdp_diffusion.py")
dppo=load("rpdp_dppo_runtime","rpdp_dppo.py")
geo=training.geo


class IsaacRPDPPolicyActor(IsaacPoseGRUPolicyActor):
    policy_family="rpdp"

    def __init__(self, *, checkpoint: Path, perception_policy_checkpoint: Path,
                 device: torch.device, dppo_rollout: bool=False,
                 dppo_eta: float=0.1, dppo_retain_last: int=5,
                 dppo_minimum_variance: float=1e-5):
        # Reuse the already validated native-image perception runtime.  Its GRU
        # action head is discarded immediately and never contributes actions.
        super().__init__(checkpoint=perception_policy_checkpoint,
                         variant="pose_conditioned",device=device)
        bundle=torch.load(checkpoint,map_location="cpu",weights_only=False)
        self.rpdp_checkpoint=Path(checkpoint);self.variant=bundle["variant"]
        config=bundle["model"]
        self.diffusion=training.TrajectoryDiffusion(config["condition_dim"],config["horizon"],
                                                    config["width"],config["layers"],
                                                    fusion=config.get("fusion",False),
                                                    visual_dim=config.get("visual_dim",0)).to(self.device)
        self.diffusion.load_state_dict(bundle["model_state_dict"]);self.diffusion.eval().requires_grad_(False)
        del self.head
        self.register_buffer("rpdp_condition_mean",bundle["condition_mean"].float())
        self.register_buffer("rpdp_condition_std",bundle["condition_std"].float())
        self.register_buffer("rpdp_target_mean",bundle["target_mean"].float())
        self.register_buffer("rpdp_target_std",bundle["target_std"].float())
        self.condition_indices=list(bundle["condition_indices"])
        self.diffusion_steps=int(bundle["diffusion_steps"]);self.inference_steps=int(bundle["inference_steps"])
        self.prediction_type=str(bundle.get("prediction_type","epsilon"))
        self.policy_mode=str(bundle.get("policy_mode","diffusion"))
        self.trajectory_semantics=str(bundle.get("trajectory_semantics","macro_horizon"))
        _,_,abar=training.schedule(self.diffusion_steps,self.device);self.register_buffer("alpha_bar",abar)
        self.adapter=geo.ConnectorTCPAdapter(bundle["tcp_to_connector_p"].to(self.device),
                                             bundle["tcp_to_connector_q"].to(self.device))
        # Fixed scene calibration, not per-step simulator geometry.
        self.register_buffer("target_port_q_world",torch.tensor([0.,0.,.9999800324440002,.006321000400930643]))
        self.register_buffer("robot_root_q_world",torch.tensor([0.,0.,0.,1.]))
        target_world=torch.tensor([.27518483996391296,-.3099580407142639,.13223984837532043])
        entrance_world=torch.tensor([.27518483996391296,-.3100590407848358,.14023984968662262])
        self.register_buffer("target_from_entrance_port",geo.quat_apply(
            geo.quat_conjugate(self.target_port_q_world),target_world-entrance_world))
        self._sample_seed=20260922
        self.dppo_rollout=bool(dppo_rollout);self.dppo_eta=float(dppo_eta)
        self.dppo_retain_last=int(dppo_retain_last);self._pending_dppo_chain=None
        self.dppo_minimum_variance=float(dppo_minimum_variance)
        self._active_dppo_chain=None

    def reset(self):
        super().reset();self._sample_seed=20260922
        self._pending_dppo_chain=None;self._active_dppo_chain=None

    def _current_connector_pose(self,state,pose_world_mm):
        target_world=self.target_port_q_world.reshape(1,4).expand(state.shape[0],-1)
        root_world=self.robot_root_q_world.reshape(1,4).expand(state.shape[0],-1)
        # The calibrated visual estimator outputs connector minus seated target
        # center.  RPDP trajectories use the port opening as their origin.
        # Despite the historical ``pose_world_mm`` variable name, the frozen
        # estimator was calibrated against target-frame labels.  It therefore
        # must not be rotated a second time here.
        p=pose_world_mm/1000. + self.target_from_entrance_port
        tcp_q_root=state[:,[6,3,4,5]]
        connector_q_root=geo.quat_mul(tcp_q_root,self.adapter.tcp_to_connector_q.expand_as(tcp_q_root))
        target_q_root=geo.quat_mul(geo.quat_conjugate(root_world),target_world)
        q=geo.quat_mul(geo.quat_conjugate(target_q_root),connector_q_root)
        return p,q

    @torch.inference_mode()
    def _infer(self,obs:dict[str,Any]):
        visual,pose,variance,visibility=self._perception(obs)
        state=obs["state"].to(self.device).float()
        full=torch.cat((visual,visibility,pose/10.,variance.clamp_min(0).sqrt()/10.,state),dim=1)
        if max(self.condition_indices,default=-1)>=234:
            previous_state=state if self._previous_state is None else self._previous_state
            state_delta=state-previous_state
            previous_action=(torch.zeros((state.shape[0],24),device=self.device)
                             if self._previous_action is None else self._previous_action)
            force=state[:,26:29]
            phase=torch.stack([torch.tensor(
                [1.,0.,0.,0.] if float(torch.linalg.norm(pose[i]))>3. else
                [0.,1.,0.,0.] if float(torch.linalg.norm(pose[i]))>.75 else
                [0.,0.,0.,1.] if float(torch.linalg.norm(force[i]))>=5. else
                [0.,0.,1.,0.],device=self.device) for i in range(state.shape[0])])
            # Contact has priority over distance, matching the offline builder.
            contact=torch.linalg.norm(force,dim=1)>=5.
            phase[contact]=torch.tensor([0.,0.,0.,1.],device=self.device)
            full=torch.cat((full,state_delta,previous_action,phase),dim=1)
        selected=full[:,self.condition_indices]
        condition=(selected-self.rpdp_condition_mean)/self.rpdp_condition_std
        if self.dppo_rollout:
            generator=torch.Generator(device=self.device).manual_seed(self._sample_seed);self._sample_seed+=1
            normalized,chain=dppo.sample_chain(
                self.diffusion,condition,(state.shape[0],self.action_horizon,9),self.alpha_bar,
                self.inference_steps,generator,eta=self.dppo_eta,
                retain_last=self.dppo_retain_last,prediction_type=self.prediction_type,
                minimum_variance=self.dppo_minimum_variance)
            self._pending_dppo_chain=chain
        elif self.policy_mode=="direct":
            normalized=self.diffusion(torch.zeros((state.shape[0],self.action_horizon,9),device=self.device),
                                      torch.zeros((state.shape[0],),dtype=torch.long,device=self.device),condition)
        else:
            # Frozen BC evaluation uses a fixed DDIM initial noise sample,
            # matching offline checkpoint selection.
            normalized=training.sample(self.diffusion,condition,
                                       (state.shape[0],self.action_horizon,9),self.alpha_bar,
                                       self.inference_steps,self._sample_seed,self.prediction_type)
        trajectory=normalized*self.rpdp_target_std+self.rpdp_target_mean
        current_p,current_q=self._current_connector_pose(state,pose)
        if self.trajectory_semantics=="microstep_chunk":
            next_p,next_q=geo.unpack_pose9(trajectory)
            action=self.adapter.waypoint_chunk(current_p,current_q,next_p,next_q).reshape(state.shape[0],24)
        else:
            next_p,next_q=geo.unpack_pose9(trajectory[:,0])
            action=self.adapter.action_chunk(current_p,current_q,next_p,next_q).reshape(state.shape[0],24)
        feature=torch.nn.functional.pad(condition,(0,max(0,384-condition.shape[1])))[:,:384]
        obs["world_feature"]=feature
        self._previous_state=state.detach().clone()
        return action,feature

    def action_components(self,obs):
        components=super().action_components(obs)
        self._active_dppo_chain=self._pending_dppo_chain;self._pending_dppo_chain=None
        return components

    def export_active_dppo_chain(self):
        if self._active_dppo_chain is None:return None
        return [{"noisy":item.noisy.detach().cpu(),"denoised":item.denoised.detach().cpu(),
                 "timestep":item.timestep.detach().cpu(),
                 "previous_timestep":item.previous_timestep.detach().cpu(),
                 "condition":item.condition.detach().cpu(),
                 "old_log_prob":item.old_log_prob.detach().cpu()}
                for item in self._active_dppo_chain]
