# Agentic Full-Insertion Plan v2 - 2026-05-29

## Current Control State

All active training / rollout jobs were stopped or verified absent before this plan was written.

Recent no-guard reward-only runs did not achieve strict insertion:

| Run | Best s mm | r mm | theta rad | module s mm | final axial consistency error mm | Strict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v492 conservative no-guard | 3.141 | 0.507 | 0.05789 | -20.478 | 42.652 | false |
| v493 wide-gate axial discovery | 2.027 | 0.785 | 0.05570 | -21.597 | 43.772 | false |
| v494 soft guide loss, guide not executed | 2.672 | 0.791 | 0.05065 | -20.955 | 43.130 | false |

The best recent true shallow/final transient remains v483: `s=17.372 mm`, `r=0.445 mm`, `theta=0.05138 rad`, `module_s=-6.250 mm`, strict false. The current checkpoints did not reproduce it reliably.

## Immediate Diagnosis

The no-guard policy does not learn enough axial insertion from the current shallow/final reward alone. Small reward changes mainly produce shallow tip motion while module/body consistency remains far behind. The observed v483 transient was real but not reproducibly captured by actor-only checkpoint resume because online critics/replay/optimizer state are fresh on resume.

Reset/randomization is also a concrete risk. A zero-action 40 mm axial / 10 mm lateral reset-settle ablation on `full_depth_start40x10_v413_fixed_lat_ypos` produced:

| Randomization | step | s mm | r mm | theta rad | final consistency error mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 8 | -38.090 | 10.992 | 0.04754 | 83.893 |
| light | 8 | -31.276 | 11.817 | 0.04664 | 77.078 |

The light profile shifts effective axial start by about `6.8 mm` on this probe. That is large enough to confound curriculum labels such as 40/10 versus 30/10, so reset/randomization must be controlled in ablations.

## Time-Boxed Work Allocation

Each family is capped at 7 hours of wall-clock work before either promotion, rejection, or blocker documentation.

| Family | Max time | Purpose | Promotion gate |
| --- | ---: | --- | --- |
| Reset/randomization ablation | 7 h | Verify start geometry for 40/10, 20/4, 10/2, 2/0 under `none`, `light`, and if needed `heavy` randomization. | Settled starts match requested `s/r/theta` within documented tolerance and do not silently start in contact. |
| Reward/curriculum phasing | 7 h | Build a staged schedule: approach, near-gate, shallow insertion, retention/final seating. | Improves strict metrics, not reward alone; no increase in tip-depth false positives. |
| Training-state reproducibility | 7 h | Add or validate full-state SERL checkpoint/resume, or otherwise preserve transient good behavior. | Resume reproduces a candidate trajectory better than actor-only checkpoints. |
| Architecture ablation | 7 h | Test available `actor_state_history_steps`, `critic_state_history_steps`, and `critic_image_encoder=convnext_tiny_imagenet`. | Improves 40/10 approach or shallow/final strict metrics without breaking runtime policy paths. |
| Imitation/residual data | 7 h | Collect privileged/guarded trajectories as offline supervised targets, then train no-hard-override actor. | Policy-only rollout inherits axial progress without executed guide overrides. |
| Literature-driven experiments | 7 h | Convert relevant SERL, HIL-SERL, Eureka, RL-VLM-F, and contact-rich assembly ideas into concrete ablations. | At least one new experiment family or justified blocker. |

## Experiment Order

1. Finish reset/randomization ablations first.
   Use deterministic `AIC_ISAAC_RANDOMIZATION_PROFILE=none` for short experiments until the requested start geometry is proven. Reintroduce randomization only after policy behavior is interpretable.

2. Add full-state or trajectory-preserving checkpointing.
   The v483 transient is the closest shallow/final evidence. Repeating actor-only resumes has failed, so preserving replay/critic/optimizer state or full near-success action traces is higher value than another reward-only rerun.

3. Run architecture ablations from the best partial checkpoint.
   Existing flags already support `actor_state_history_steps`, `critic_state_history_steps`, and ConvNeXt critic encoders. Start with low-cost smoke runs:
   - actor history 4 or 8,
   - critic history 4,
   - `critic_image_encoder=convnext_tiny_imagenet` if memory allows.

4. Rebuild phased curriculum after reset validation.
   Start from 2/0 and 10/2 only after deterministic settle is confirmed, then mix 20/4 and 40/10 back in. Avoid far-start mixed training until final-window insertion can combine depth, lateral, theta, and module consistency.

5. Use soft imitation before hard servo.
   Guarded trajectories are useful as teachers or diagnostics. They should not be executed during the target policy rollout unless a minimal safety fallback is explicitly being tested.

## Literature Hooks

- SERL emphasizes sample-efficient off-policy robot RL with careful reset, rewards, and controller engineering, not reward hacking alone. This supports spending time on reset validation and controller-aware data collection before long online runs. Source: https://arxiv.org/abs/2401.16013
- HIL-SERL shows interventions/corrections can improve precise manipulation beyond plain SERL. For this repo, the analogue is privileged/guarded correction data distilled into the actor, not silently overriding actions at evaluation. Source: https://hil-serl.github.io/
- Eureka demonstrates LLM-generated reward programs in Isaac tasks, but the lesson here is to audit reward code against strict geometry and false positives, not to trust reward return. Source: https://arxiv.org/abs/2310.12931
- RL-VLM-F uses VLM feedback to generate rewards from visual observations. For this insertion task, visual sanity checks can be used as audit labels or failure classifiers, but strict success must still come from semantic tip/body geometry. Source: https://arxiv.org/abs/2402.03681
- Contact-rich assembly work repeatedly decomposes insertion into reaching/alignment/insertion phases and uses visual-force or residual control. This supports phased curriculum, contact/force failure labels, and residual imitation rather than monolithic far-start RL. Example source: https://arxiv.org/abs/2305.17110

## Next Concrete Commands

Reset-settle ablation examples:

```bash
python aic_utils/aic_isaac/scripts/validate_serl_reset_settle.py \
  --episode-config-dir outputs/agentic_reward_curriculum_20260529/generated_episode_configs/full_depth_start40x10_v413_fixed_lat_ypos \
  --output-root outputs/agentic_reward_curriculum_20260529/reset_ablation_runs \
  --run-name resetcheck_40x10_ypos_none_steps8 \
  --num-envs 1 --steps 8 --seed 57001 --max-wall-time-minutes 10 \
  --randomization-profile none --max-lateral-m 0.020 --max-theta-rad 0.20
```

```bash
python aic_utils/aic_isaac/scripts/validate_serl_reset_settle.py \
  --episode-config-dir outputs/agentic_reward_curriculum_20260529/generated_episode_configs/full_depth_start40x10_v413_fixed_lat_ypos \
  --output-root outputs/agentic_reward_curriculum_20260529/reset_ablation_runs \
  --run-name resetcheck_40x10_ypos_light_steps8 \
  --num-envs 1 --steps 8 --seed 57001 --max-wall-time-minutes 10 \
  --randomization-profile light --max-lateral-m 0.020 --max-theta-rad 0.20
```

## Implementation Change Made

`aic_utils/aic_isaac/scripts/validate_serl_reset_settle.py` now supports:

```bash
--randomization-profile {none,light,heavy}
```

This keeps reset ablations config-driven and reproducible.

## Reset Ablation Update

After fixing `validate_serl_reset_settle.py` to use a positive episode length during settle probes, deterministic `none` randomization results were:

| Config | step | s mm | r mm | theta rad | final consistency error mm | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 40x10 original | 8 | -38.198 | 11.440 | 0.04753 | 84.000 | Interpretable pre-contact start; lateral slightly high. |
| 20x4 original | 8 | -21.589 | 3.277 | 0.04724 | 67.391 | Close enough for 20/4 curriculum. |
| 10x2 original | 8 | -0.886 | 6.949 | 0.04953 | 46.685 | Not a 10/2 start after settle. |
| 2x0 original | 8 | -0.903 | 8.555 | 0.05078 | 46.699 | Not a 2/0 start after settle. |

Rejected repair attempts:

| Config | step | s mm | r mm | theta rad | Reason rejected |
| --- | ---: | ---: | ---: | ---: | --- |
| 20x4 v495 semantic-tip repair | 8 | 3.587 | 29.248 | 0.06394 | Moved into/beside port, large lateral error. |
| 10x2 v495 semantic-tip repair | 8 | 18.375 | 27.555 | 0.06383 | Severe lateral bypass. |
| 2x0 v495 semantic-tip repair | 8 | 16.784 | 27.393 | 0.06296 | Severe lateral bypass. |
| 10x2 v497 measured 2D shift | 8 | 1.510 | 31.196 | 0.06656 | Nonlinear/contact response; worse lateral error. |
| 2x0 v497 measured 2D shift | 8 | -0.947 | 9.166 | 0.04473 | Did not fix lateral error. |

Conclusion: do not train new policies on the current shallow 10/2 or 2/0 configs as if they are valid near-gate/final-window starts. The next reset fix should regenerate shallow starts from the current 40/10 or 20/4 calibrated reference, then validate with positive episode-length reset-settle probes before use.
