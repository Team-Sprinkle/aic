# Phased Full-Insertion Plan - 2026-05-30 v2

This plan supersedes shallow-only continuation. All long training remains stopped until reset validation and short
ablation smokes justify it.

## Current State

- No active AIC/Isaac insertion training or rollout process was found on host or inside `isaac-lab-base`.
- v699 no-guard randomized training improved axial motion but not strict insertion:
  - best tip `s` mean/max `4.668 / 13.252 mm`,
  - tip `r` mean/max `0.325 / 0.385 mm`,
  - theta mean/max `0.0328 / 0.0506 rad`,
  - module `s` mean/max `-18.955 / -10.351 mm`,
  - strict success false.
- The immediate blocker is module/body consistency plus theta retention, not lateral alignment alone.

## Reset Fix

`build_randomized_near_gate_curriculum.py` had a concrete reset-generation bug:

- old generated curricula shifted `body_start_position_world` and `reset_body_offset_from_reference_world`,
  but left `reference_reward_body_start_position_world` stale;
- old curricula also allowed rotation perturbations without a measured tip transform, so orientation randomization could
  create uncontrolled semantic-tip sweep.

Patch applied:

- shifted semantic reference fields are now written back to the YAML;
- pure translation preserves the calibrated gripper-to-tip offset instead of adding the world shift into the offset;
- a regression test checks `body_start - reference == reset_body_offset_from_reference_world`.

Validation:

- `aic_utils/aic_isaac/test/test_randomized_near_gate_curriculum.py`
- `aic_utils/aic_isaac/test/test_insertion_reward_geometry.py`
- both passed: `32 passed`.

New generated curriculum:

- `outputs/agentic_reward_curriculum_20260529/generated_episode_configs/randomized_curriculum_v701_offsetfix_from_v642`
- static audit: `outputs/agentic_reward_curriculum_20260529/reset_audits/audit_v702_v701_offsetfix_train_mixed_static_distribution`
- dynamic validation v703: accepted 10/12 from the interleaved mixed bucket.

## Literature Priors

- SERL emphasizes that implementation details, reset quality, reward computation, and controllers are central for
  sample-efficient robot RL; its reported tasks include PCB assembly and cable routing with real-world learning in
  tens of minutes under well-engineered resets/controllers. Source: https://arxiv.org/abs/2401.16013
- HIL-SERL shows precision assembly can benefit from demonstrations/corrections and efficient off-policy RL, with
  reported 1-2.5 hour training windows. Our analogue should be privileged/teacher correction or accepted-reset replay
  rather than pure blind online RL. Source: https://arxiv.org/abs/2410.21845
- Eureka supports LLM/code-driven reward iteration, but its lesson for this task is not to trust reward return:
  reward candidates must be evaluated by strict geometric success and visual sanity. Source: https://arxiv.org/abs/2310.12931
- RL-VLM-F and real-world offline VLM-feedback variants suggest visual preference checks can help label progress, but
  for SFP insertion the VLM should be an auxiliary sanity filter because metric false positives are already known.
  Sources: https://arxiv.org/abs/2402.03681 and https://arxiv.org/abs/2411.05273

## Time-Boxed Work Packages

Each package is capped at 7 hours wall-clock. Stop early if strict evidence rejects the path.

| package | cap | goal | first experiments | promotion gate |
|---|---:|---|---|---|
| Reset/contact fixes | 7h | eliminate stale gripper-tip randomization and unstable reset distributions | v701/v703 offset-fixed reset validation; held-out 40x10 validation | accepted resets with post-step `s/r/theta/module` in intended buckets |
| Reward/curriculum ablations | 7h | recover module-consistent axial progress without hard servo | depth-gated module-following weight; theta-retention penalty after `s > 0`; reduced axial reward when theta drifts | better module `s`, final consistency error, and theta without worse lateral bypass |
| Architecture ablations | 7h | test whether current-observation/weak-vision is limiting | `actor_state_history_steps=4`; `critic_state_history_steps=4`; `critic_image_encoder_override=convnext_tiny_imagenet`; optional ACT ResNet50 offline path | stable smoke plus improved held-out rollout metrics |
| Literature-derived variants | 7h | test ideas from SERL/HIL-SERL/Eureka/VLM feedback | teacher/privileged correction replay; reward audit from near-success rows; visual sanity labeling | improves strict metrics, not reward alone |
| Evaluation/reporting | 7h | keep evidence reproducible | policy-only rollouts from promoted checkpoints on shallow, near-gate, held-out 40x10, with videos | strict success or blocker with closest gap |

## Immediate Queue

1. Use v703 accepted episodes as the reset-fixed training/eval substrate.
2. Run a short no-guard smoke from v699/v692 lineage with:
   - offset-fixed accepted resets,
   - `actor_state_history_steps=4`,
   - `critic_state_history_steps=4`,
   - `critic_image_encoder_override=convnext_tiny_imagenet`,
   - conservative updates and no hard action overrides.
3. If architecture smoke fails due checkpoint shape mismatch, run the same reset-fixed smoke with actor history `1` and
   critic history/ConvNeXt only.
4. Compare against v699 and reject any path that improves tip `s` while worsening theta/module consistency.
