"""Runtime for the frozen nominal GRU plus explicit predicted-pose correction."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from pose_gru_policy_actor import IsaacPoseGRUPolicyActor, trainer_module


HERE = Path(__file__).resolve().parent


def load_correction_module():
    path = HERE / "train_explicit_pose_correction.py"
    spec = importlib.util.spec_from_file_location("explicit_pose_correction_training_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


correction_module = load_correction_module()


class IsaacExplicitPoseCorrectionActor(IsaacPoseGRUPolicyActor):
    """Action-only nominal controller plus an odd-symmetric pose residual."""

    policy_family = "explicit_pose_correction"

    def __init__(self, *, checkpoint: Path, device: torch.device):
        bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
        nominal_path = Path(bundle["nominal_checkpoint"])
        super().__init__(checkpoint=nominal_path, variant="action_only", device=device)
        if str(self.checkpoint) != str(nominal_path):
            raise RuntimeError("Nominal checkpoint lineage mismatch")
        self.correction_checkpoint = Path(checkpoint)
        self.correction = correction_module.ExplicitPoseCorrection(
            int(bundle["context_dim"]), int(bundle["correction_hidden_dim"])
        ).to(self.device)
        self.correction.load_state_dict(bundle["correction_state_dict"])
        self.correction.eval().requires_grad_(False)
        self._correction_contexts: list[torch.Tensor] = []
        self._pose_history: list[torch.Tensor] = []

    def reset(self) -> None:
        super().reset()
        self._correction_contexts.clear()
        self._pose_history.clear()

    def _infer(self, obs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        visual, pose, variance, visibility = self._perception(obs)
        state = obs["state"].to(self.device).float()
        state_delta = torch.zeros_like(state) if self._previous_state is None else state-self._previous_state
        previous_action = (
            torch.zeros((1,24),device=self.device) if self._previous_action is None else self._previous_action
        )
        force = state[:,26:29]
        base = torch.cat((visual,state,state_delta,force,previous_action,visibility),dim=1)
        phase = trainer_module.phase_from_observation(
            pose[0].detach().cpu(), force[0].detach().cpu()
        ).to(self.device).reshape(1,4)
        condition = torch.cat((pose,variance.clamp_min(0).sqrt(),phase),dim=1)
        base_normalized=(base-self.normalization["base_mean"])/self.normalization["base_std"]
        condition_normalized=(condition-self.normalization["conditioning_mean"])/self.normalization["conditioning_std"]

        nominal_value=torch.cat((base_normalized,torch.zeros_like(condition_normalized)),dim=1)
        context=torch.cat((base_normalized[:,222:297],condition_normalized[:,3:]),dim=1)
        self._inputs=(self._inputs+[nominal_value.detach()])[-self.history:]
        self._correction_contexts=(self._correction_contexts+[context.detach()])[-self.history:]
        self._pose_history=(self._pose_history+[pose.detach()])[-self.history:]
        nominal_sequence=[torch.zeros_like(nominal_value) for _ in range(self.history-len(self._inputs))]+self._inputs
        context_sequence=[torch.zeros_like(context) for _ in range(self.history-len(self._correction_contexts))]+self._correction_contexts
        pose_sequence=[torch.zeros_like(pose) for _ in range(self.history-len(self._pose_history))]+self._pose_history
        nominal_normalized=self.head(torch.stack(nominal_sequence,dim=1))
        nominal=nominal_normalized*self.normalization["action_std"]+self.normalization["action_mean"]
        residual,_=self.correction(torch.stack(pose_sequence,dim=1),torch.stack(context_sequence,dim=1))
        action=nominal+residual
        self._previous_state=state.detach().clone()
        feature=F.pad(torch.cat((base_normalized,condition_normalized),dim=1),
                      (0,max(0,384-base_normalized.shape[1]-condition_normalized.shape[1])))[:,:384].detach()
        obs["world_feature"]=feature
        return action,feature
