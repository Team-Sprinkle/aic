import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn


PATH=Path(__file__).parents[1]/"aic_isaaclab/scripts/serl/rpdp_dppo.py"
spec=importlib.util.spec_from_file_location("rpdp_dppo",PATH)
d=importlib.util.module_from_spec(spec);sys.modules[spec.name]=d;spec.loader.exec_module(d)


class Tiny(nn.Module):
    def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(.1))
    def forward(self,x,t,c):return self.scale*x+c[:,:1,None]


def test_chain_retains_exact_finite_likelihoods():
    model=Tiny();alpha_bar=torch.cumprod(1-torch.linspace(1e-4,.02,100),0)
    condition=torch.zeros(3,2);generator=torch.Generator().manual_seed(3)
    _,chain=d.sample_chain(model,condition,(3,4,9),alpha_bar,20,generator,retain_last=5)
    assert len(chain)==5
    for transition in chain:
        new=d.transition_log_prob(model,transition,alpha_bar)
        assert torch.isfinite(new).all()
        assert torch.allclose(new,transition.old_log_prob,atol=2e-4)


def test_unchanged_policy_has_unit_probability_ratio():
    model=Tiny();alpha_bar=torch.cumprod(1-torch.linspace(1e-4,.02,100),0)
    condition=torch.zeros(4,2);generator=torch.Generator().manual_seed(4)
    _,chain=d.sample_chain(model,condition,(4,4,9),alpha_bar,20,generator,retain_last=3)
    loss,ratio=d.clipped_dppo_loss(model,chain,torch.tensor([1.,2.,3.,4.]),alpha_bar)
    assert torch.allclose(ratio,torch.ones_like(ratio),atol=2e-4)
    assert torch.isfinite(loss)


def test_serialized_chain_keeps_unit_probability_ratio(tmp_path):
    model=Tiny();alpha_bar=torch.cumprod(1-torch.linspace(1e-4,.02,100),0)
    condition=torch.zeros(4,2);generator=torch.Generator().manual_seed(5)
    _,chain=d.sample_chain(model,condition,(4,4,9),alpha_bar,20,generator,
                           retain_last=5,minimum_variance=1e-5)
    path=tmp_path/"chain.pt";torch.save(chain,path)
    restored=torch.load(path,weights_only=False)
    for transition in restored:
        new=d.transition_log_prob(model,transition,alpha_bar,minimum_variance=1e-5)
        assert torch.allclose(new,transition.old_log_prob,atol=3e-3)


def test_dppo_disables_mha_inference_fastpath():
    if hasattr(torch.backends,"mha"):
        torch.backends.mha.set_fastpath_enabled(True)
        d.disable_mha_fastpath()
        assert not torch.backends.mha.get_fastpath_enabled()
