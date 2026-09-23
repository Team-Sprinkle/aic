"""Runtime for matched action-only and pose-conditioned GRU policies."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from world_policy_actor import IsaacWorldPolicyActor

HERE = Path(__file__).resolve().parent
CAMERA_KEYS = (
    "observation.images.center_camera",
    "observation.images.left_camera",
    "observation.images.right_camera",
)


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


trainer_module = load_module("pose_gru_training_runtime", "train_pose_conditioned_gru_policy.py")
visibility = trainer_module.visibility
opening = trainer_module.opening
pretrained = trainer_module.pretrained


class RuntimeNormalizer:
    def __init__(self, bundle: dict[str, Any], device: torch.device):
        self.state_mean = torch.zeros(1, 32, device=device)
        self.state_std = torch.ones(1, 32, device=device)
        action_mean = bundle["normalization"]["action_mean"].reshape(4, 6).mean(0)
        action_std = bundle["normalization"]["action_std"].reshape(4, 6).mean(0)
        self.action_mean = action_mean.to(device).reshape(1, 6)
        self.action_std = action_std.to(device).reshape(1, 6)

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return state

    @staticmethod
    def normalize_image(_key: str, image: torch.Tensor) -> torch.Tensor:
        return image


class IsaacPoseGRUPolicyActor(IsaacWorldPolicyActor):
    """Frozen perception plus a supervised causal GRU action head."""

    actor_mode = "world_policy"  # reuse the validated four-command macro transport
    policy_family = "pose_gru"
    state_dim = 42
    action_dim = 24
    action_horizon = 4
    single_action_dim = 6
    adapter_delta_clip = None
    tcp_translation_action_clip = None
    tcp_rotation_action_clip = None
    action_clip = None
    requires_native_camera_observation = True
    requires_camera_calibration = True

    def __init__(self, *, checkpoint: Path, variant: str, device: torch.device):
        nn.Module.__init__(self)
        if variant not in {"action_only", "pose_conditioned"}:
            raise ValueError(f"Unsupported pose-GRU variant: {variant}")
        bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.variant = variant
        self.conditioned = variant == "pose_conditioned"
        self.device = torch.device(device)
        self.checkpoint = Path(checkpoint)
        self.act_torchscript_path = self.checkpoint
        self.history = int(bundle["history"])
        self.normalization = {key: value.to(self.device) for key, value in bundle["normalization"].items()}
        input_dim = int(bundle["dimensions"]["base"] + bundle["dimensions"]["conditioning"])
        self.head = trainer_module.PoseConditionedGRUPolicy(input_dim, int(bundle["hidden_dim"])).to(self.device)
        self.head.load_state_dict(bundle["models"][variant]); self.head.eval()
        self.act_normalizer = RuntimeNormalizer(bundle, self.device)

        pose_checkpoint = torch.load(Path(bundle["pose_checkpoint"]), map_location="cpu", weights_only=False)
        self.locator = opening.crop.Locator().to(self.device)
        self.locator.load_state_dict(pose_checkpoint["locator"]); self.locator.eval().requires_grad_(False)
        self.landmark = pretrained.MobileNetLandmarks().to(self.device)
        self.landmark.load_state_dict(pose_checkpoint["landmark"]); self.landmark.eval().requires_grad_(False)
        self.visibility_head = visibility.VisibilityHead(pose_checkpoint["visibility_input_dim"]).to(self.device)
        self.visibility_head.load_state_dict(pose_checkpoint["visibility_head"])
        self.visibility_head.eval().requires_grad_(False)
        self.selected_landmarks = pose_checkpoint["selected_landmarks"]
        self.register_buffer("pose_affine", torch.tensor(pose_checkpoint["affines"]["predicted_visibility"], dtype=torch.float32))
        saved = pose_checkpoint["current"]
        self.residuals = nn.ModuleList()
        for state in saved["state_dicts"]:
            model = visibility.ResidualPose(int(saved["input_mean"].numel())).to(self.device)
            model.load_state_dict(state); model.eval().requires_grad_(False); self.residuals.append(model)
        self.register_buffer("residual_input_mean", saved["input_mean"].float())
        self.register_buffer("residual_input_std", saved["input_std"].float())
        self.register_buffer("residual_target_mean", saved["target_mean"].float())
        self.register_buffer("residual_target_std", saved["target_std"].float())
        self.register_buffer("_diagnostic_log_std", torch.tensor([-2.0]), persistent=False)
        self._inputs: list[torch.Tensor] = []
        self._previous_state: torch.Tensor | None = None
        self._previous_action: torch.Tensor | None = None
        self._prefetched: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def log_std(self) -> torch.Tensor:
        return self._diagnostic_log_std

    def reset(self) -> None:
        self._inputs.clear(); self._previous_state = None; self._previous_action = None; self._prefetched = None

    @staticmethod
    def _crop(image: torch.Tensor, center: torch.Tensor, size: int = 160) -> tuple[torch.Tensor, int, int]:
        _, _, height, width = image.shape
        left = int(round(float(center[0])) - size // 2)
        top = int(round(float(center[1])) - size // 2)
        pad_left=max(0,-left); pad_top=max(0,-top); pad_right=max(0,left+size-width); pad_bottom=max(0,top+size-height)
        padded=F.pad(image,(pad_left,pad_right,pad_top,pad_bottom))
        x=left+pad_left; y=top+pad_top
        return padded[:, :, y:y+size, x:x+size], left, top

    def _perception(self, obs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        images = obs.get("native_images") or obs["images"]
        native = [images[key].to(self.device).float() for key in CAMERA_KEYS]
        if any(image.shape[0] != 1 for image in native):
            raise ValueError("Pose-GRU runtime currently supports one environment")
        locator_batch = torch.cat([F.interpolate(image, size=(256,288), mode="bilinear", align_corners=False)
                                   for image in native], dim=0)
        coarse = self.locator(locator_batch).reshape(1,3,4)
        crops=[]; geometry=[]
        for view_index, (image, prediction) in enumerate(zip(native, coarse[0])):
            _,_,height,width=image.shape
            scale=torch.tensor([width-1,height-1,width-1,height-1],device=self.device)
            center=(prediction*scale).reshape(2,2).mean(0)
            crop,left,top=self._crop(image,center)
            crops.append(crop); geometry.append((view_index,left,top,width,height))
        crop_batch=torch.cat(crops,dim=0)
        value=(crop_batch-self.landmark.mean)/self.landmark.std
        taps={}
        for index,layer in enumerate(self.landmark.features):
            value=layer(value)
            if index in (1,3,8): taps[index]=value
        fused=self.landmark.lateral40(taps[1])
        fused=fused+F.interpolate(self.landmark.lateral20(taps[3]),size=fused.shape[-2:],mode="bilinear",align_corners=False)
        fused=fused+F.interpolate(self.landmark.lateral10(taps[8]),size=fused.shape[-2:],mode="bilinear",align_corners=False)
        logits=self.landmark.head(fused)
        local=opening.decode(logits)
        probability_map=torch.softmax(logits.flatten(-2),-1)
        maximum=probability_map.max(-1).values
        entropy=-(probability_map*probability_map.clamp_min(1e-9).log()).sum(-1)/math.log(probability_map.shape[-1])
        visual=torch.cat((taps[8].mean((-2,-1)),maximum,entropy),dim=1).reshape(1,3,-1)
        landmarks=torch.empty(1,3,6,2,device=self.device)
        for prediction,(view_index,left,top,width,height) in zip(local,geometry):
            pixels=prediction*159+torch.tensor([left,top],device=self.device)
            landmarks[0,view_index]=pixels/torch.tensor([width-1,height-1],device=self.device)
        vis_probability=torch.sigmoid(self.visibility_head(visual.reshape(-1,visual.shape[-1]))).reshape(1,3,2)
        points=torch.empty((1,3,4),device=self.device)
        points[:,:,:2]=coarse[:,:,:2] if self.selected_landmarks["plug"]=="coarse" else landmarks[:,:,0]
        if self.selected_landmarks["port"]=="coarse": points[:,:,2:]=coarse[:,:,2:]
        elif self.selected_landmarks["port"]=="entrance": points[:,:,2:]=landmarks[:,:,1]
        else: points[:,:,2:]=landmarks[:,:,2:6].mean(2)
        calibrations=obs.get("camera_calibration")
        if calibrations is None: raise RuntimeError("Pose-GRU observation lacks camera calibration")
        row={"highres":[]}
        for key,image in zip(CAMERA_KEYS,native):
            item=calibrations[key]; _,_,height,width=image.shape
            row["highres"].append({"intrinsic_matrix":item["K"][0].detach().cpu().numpy(),
                                   "camera_position_world":item["position"][0].detach().cpu().numpy(),
                                   "camera_orientation_wxyz_ros":item["quat"][0].detach().cpu().numpy(),
                                   "width":width,"height":height})
        raw=visibility.relative_weighted(points.detach().cpu(),[row],vis_probability.detach().cpu().numpy())
        affine_raw=torch.tensor(np.c_[raw,np.ones(1)],device=self.device,dtype=torch.float32)@self.pose_affine
        residual_feature=visibility.residual_features(coarse.detach().cpu(),landmarks.detach().cpu(),
                                                       affine_raw.detach().cpu().numpy(),vis_probability.detach().cpu()).to(self.device)
        normalized=(residual_feature-self.residual_input_mean)/self.residual_input_std
        predictions=[]
        for model in self.residuals:
            predictions.append(affine_raw+model(normalized)*self.residual_target_std+self.residual_target_mean)
        stack=torch.stack(predictions)
        return visual.reshape(1,-1), stack.mean(0), stack.var(0,unbiased=False), vis_probability.reshape(1,-1)

    def _infer(self, obs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        visual,pose,variance,vis=self._perception(obs)
        state=obs["state"].to(self.device).float()
        previous_state=torch.zeros_like(state) if self._previous_state is None else self._previous_state
        state_delta=state-previous_state if self._previous_state is not None else torch.zeros_like(state)
        previous_action=torch.zeros((1,24),device=self.device) if self._previous_action is None else self._previous_action
        force=state[:,26:29]
        base=torch.cat((visual,state,state_delta,force,previous_action,vis),dim=1)
        phase=trainer_module.phase_from_observation(pose[0].detach().cpu(),force[0].detach().cpu()).to(self.device).reshape(1,4)
        conditioning=torch.cat((pose,variance.clamp_min(0).sqrt(),phase),dim=1)
        base=(base-self.normalization["base_mean"])/self.normalization["base_std"]
        condition=(conditioning-self.normalization["conditioning_mean"])/self.normalization["conditioning_std"]
        if not self.conditioned: condition=torch.zeros_like(condition)
        value=torch.cat((base,condition),dim=1)
        self._inputs.append(value.detach())
        self._inputs=self._inputs[-self.history:]
        sequence=[torch.zeros_like(value) for _ in range(self.history-len(self._inputs))]+self._inputs
        normalized_action=self.head(torch.stack(sequence,dim=1))
        action=normalized_action*self.normalization["action_std"]+self.normalization["action_mean"]
        self._previous_state=state.detach().clone()
        feature=F.pad(value,(0,max(0,384-value.shape[1])))[:,:384].detach()
        obs["world_feature"]=feature
        return action,feature

    def _components(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        zero=torch.zeros_like(action)
        return {"base_action":action,"raw_delta_action":zero,"delta_action":zero,
                "unclipped_final_action":action,"translation_clipped_action":action,
                "rotation_clipped_action":action,"final_action":action}

    def action_components(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        if self._prefetched is not None:
            action,feature=self._prefetched; self._prefetched=None; obs["world_feature"]=feature
        else: action,feature=self._infer(obs)
        return self._components(action)

    def record_executed_chunk(self, action: torch.Tensor) -> None:
        self._previous_action=action.detach().to(self.device).reshape(1,24)

    @torch.inference_mode()
    def prefetch_next(self, obs: dict[str, Any]) -> None:
        self._prefetched=self._infer(obs)

    def mean_action(self, obs: dict[str, Any]) -> torch.Tensor:
        return self._infer(obs)[0]
