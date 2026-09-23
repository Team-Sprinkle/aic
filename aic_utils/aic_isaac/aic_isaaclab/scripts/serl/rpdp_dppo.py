"""DPPO denoising-chain storage and likelihoods for RPDP.

This module contains no environment or actor update loop.  It makes the
stochastic reverse diffusion transitions explicit so rollout storage can retain
the exact states, noise and old log probabilities required by PPO.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


def disable_mha_fastpath():
    """Use one attention kernel for no-grad rollout and gradient updates.

    PyTorch's eval plus no-grad MultiheadAttention fast path is numerically
    different from the gradient-capable path.  Tiny mean changes are amplified
    by diffusion transition likelihoods and create a non-unit PPO ratio before
    any update.  DPPO needs the two paths to use the same kernel.
    """
    backend=getattr(torch.backends,"mha",None)
    if backend is not None and hasattr(backend,"set_fastpath_enabled"):
        backend.set_fastpath_enabled(False)


@dataclass
class DenoisingTransition:
    noisy: torch.Tensor
    denoised: torch.Tensor
    timestep: torch.Tensor
    previous_timestep: torch.Tensor
    condition: torch.Tensor
    old_log_prob: torch.Tensor


def transition_parameters(model, noisy, timestep, previous_timestep,
                          condition, alpha_bar, *, eta=1.0, prediction_type="sample"):
    """Generalized stochastic DDIM transition parameters."""
    disable_mha_fastpath()
    prediction = model(noisy, timestep, condition)
    ab_t = alpha_bar[timestep].reshape(-1, 1, 1)
    valid = previous_timestep >= 0
    safe_previous = previous_timestep.clamp_min(0)
    ab_previous = alpha_bar[safe_previous].reshape(-1, 1, 1)
    ab_previous = torch.where(valid[:, None, None], ab_previous, torch.ones_like(ab_previous))
    if prediction_type == "sample":
        clean = prediction
        epsilon = (noisy-torch.sqrt(ab_t)*clean)/torch.sqrt(1-ab_t).clamp_min(1e-6)
    elif prediction_type == "epsilon":
        epsilon = prediction
        clean = (noisy - torch.sqrt(1-ab_t)*epsilon) / torch.sqrt(ab_t)
    else:
        raise ValueError(f"unsupported prediction_type: {prediction_type}")
    variance = (eta**2) * (1-ab_previous)/(1-ab_t) * (1-ab_t/ab_previous)
    variance = variance.clamp_min(0)
    mean = torch.sqrt(ab_previous)*clean + torch.sqrt((1-ab_previous-variance).clamp_min(0))*epsilon
    return mean, variance


def gaussian_log_prob(value, mean, variance, minimum_variance=1e-8):
    variance = variance.clamp_min(minimum_variance)
    element = -.5*((value-mean).square()/variance + torch.log(2*math.pi*variance))
    return element.flatten(1).sum(-1)


def transition_log_prob(model, transition, alpha_bar, *, eta=1.0, prediction_type="sample",
                        minimum_variance=1e-5):
    mean, variance = transition_parameters(model, transition.noisy,
                                           transition.timestep,
                                           transition.previous_timestep,
                                           transition.condition, alpha_bar, eta=eta,
                                           prediction_type=prediction_type)
    return gaussian_log_prob(transition.denoised, mean, variance,
                             minimum_variance=minimum_variance)


@torch.no_grad()
def sample_chain(model, condition, shape, alpha_bar, inference_steps, generator,
                 *, eta=1.0, retain_last=10, prediction_type="sample",
                 minimum_variance=1e-5):
    disable_mha_fastpath()
    noisy = torch.randn(shape, generator=generator, device=condition.device)
    timeline = torch.linspace(len(alpha_bar)-1, 0, inference_steps,
                              device=condition.device).round().long().unique_consecutive()
    retained = []
    for index, timestep_scalar in enumerate(timeline):
        previous_scalar = timeline[index+1] if index+1 < len(timeline) else torch.tensor(-1,device=condition.device)
        timestep = torch.full((shape[0],),int(timestep_scalar),device=condition.device,dtype=torch.long)
        previous = torch.full_like(timestep,int(previous_scalar))
        mean, variance = transition_parameters(model,noisy,timestep,previous,condition,alpha_bar,eta=eta,
                                               prediction_type=prediction_type)
        if int(previous_scalar) >= 0 and float(variance.max()) > 0:
            # The final DDIM transitions have vanishing analytical variance.
            # Sampling with that variance makes a float32 serialized trajectory
            # insufficient to reproduce its likelihood for PPO.  Use the same
            # explicit floor for sampling and likelihood evaluation.
            effective_variance=variance.clamp_min(minimum_variance)
            noise=torch.randn(shape,generator=generator,device=condition.device)
            denoised=mean+torch.sqrt(effective_variance)*noise
            log_prob=gaussian_log_prob(denoised,mean,effective_variance,
                                       minimum_variance=minimum_variance)
        else:
            denoised=mean;log_prob=torch.zeros(shape[0],device=condition.device)
        # DPPO fine tunes transitions closest to the clean action.
        if index >= max(0,len(timeline)-1-retain_last) and int(previous_scalar)>=0:
            retained.append(DenoisingTransition(noisy.clone(),denoised.clone(),timestep,
                                                previous,condition.clone(),log_prob.clone()))
        noisy=denoised
    return noisy, retained


def clipped_dppo_loss(model, transitions, advantages, alpha_bar, *,
                      clip_ratio=.01, eta=1.0, prediction_type="sample",
                      minimum_variance=1e-5):
    """PPO surrogate over retained denoising transitions."""
    losses=[];ratios=[]
    normalized=(advantages-advantages.mean())/advantages.std().clamp_min(1e-6)
    for transition in transitions:
        log_prob=transition_log_prob(model,transition,alpha_bar,eta=eta,
                                     prediction_type=prediction_type,
                                     minimum_variance=minimum_variance)
        ratio=torch.exp((log_prob-transition.old_log_prob).clamp(-20,20))
        unclipped=ratio*normalized
        clipped=ratio.clamp(1-clip_ratio,1+clip_ratio)*normalized
        losses.append(-torch.minimum(unclipped,clipped).mean());ratios.append(ratio.detach())
    if not losses: raise ValueError("No stochastic denoising transitions were retained")
    return torch.stack(losses).mean(), torch.cat(ratios)
