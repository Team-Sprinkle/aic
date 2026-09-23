import importlib.util
import sys
from pathlib import Path

import torch


PATH=Path(__file__).parents[1]/"aic_isaaclab/scripts/serl/train_rpdp_diffusion.py"
spec=importlib.util.spec_from_file_location("train_rpdp_diffusion",PATH)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)


def test_pose_gated_visual_fusion_forward_and_gradient():
    model=module.TrajectoryDiffusion(209,horizon=4,width=48,layers=1,fusion=True,visual_dim=186)
    trajectory=torch.randn(3,4,9);condition=torch.randn(3,209);timestep=torch.tensor([1,2,3])
    output=model(trajectory,timestep,condition)
    assert output.shape==(3,4,9)
    output.square().mean().backward()
    assert model.visual_gate.weight.grad is not None
    assert torch.isfinite(model.visual_gate.weight.grad).all()


def test_temporal_condition_indices_preserve_new_causal_tail():
    assert module.condition_indices("aic_rpdp",304)==list(range(304))
    assert module.condition_indices("rpdp_local",304)[-70:]==list(range(234,304))
    assert module.condition_indices("pose_dp",304)[-70:]==list(range(234,304))
