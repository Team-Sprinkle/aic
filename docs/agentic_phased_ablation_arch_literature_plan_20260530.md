# Agentic Phased Ablation, Architecture, and Literature Plan - 2026-05-30

## Current Stop State

All active Isaac/SERL training and rollout processes were checked on the host and in the `isaac-lab-base` container.
No active training or rollout job was running at the start of this plan.

Recent evidence:

- v640 policy-only rollout from v639 checkpoint 100: best centered positive depth `s=6.393 mm`, `r=0.500 mm`,
  `theta=0.05572 rad`, module `s=-17.230 mm`, strict success false.
- v643 no-guard randomized continuation from v639 checkpoint 100: best centered positive depth `s=8.233 mm`,
  `r=0.401 mm`, `theta=0.05790 rad`, final axial consistency error `37.564 mm`, strict success false.
- v642 low-theta reset generation fixed the selected-env local/world conversion bug, but zero-action validation still
  accepted `0/15`; theta and module consistency drift remain unresolved.

Conclusion: the current best "8-10 mm insertion" checkpoint is useful as a partial-depth baseline, but not a success
candidate. It should be used only as a seed for controlled ablations, with strict module/body consistency checks.

## Time Allocation

Each track is capped at 7 hours of wall-clock experiment time before it must either promote a candidate, write a
negative result, or move to the next track.

| Track | Cap | Goal | First promotion metric |
|---|---:|---|---|
| A. Reset and initialization audit | 7 h | Verify gripper/reset-body/tip-frame randomization and eliminate invalid reset distributions | zero-action validation improves theta/r/module consistency without already-inserted artifacts |
| B. Phased reward/curriculum ablations | 7 h | Test phase-specific reward gates without hard servo overrides | improves strict metrics or near-success metrics without tip-depth false positives |
| C. Controller/teacher trajectory generation | 7 h | Produce module-consistent low-theta final-window trajectories for imitation/residual learning | at least one trajectory with high module consistency and theta trending below current floor |
| D. Model architecture ablations | 7 h | Test stronger vision/history settings and determine whether architecture is blocking learning | stable training/eval improves s/r/theta/module over v640/v643 |
| E. Literature-derived ideas | 7 h | Convert public literature into concrete ablations | at least two implementable ideas with commands and success criteria |

## Track A: Reset and Initialization Audit

Risk being tested:

- Episode YAML generation may mix `gripper_tcp`, `wrist_3_link`, and semantic `sfp_tip_link` assumptions.
- The reset event supports `reset_body_name`, but one reset batch cannot mix reset bodies.
- The generator allows semantic tip offsets and reset-body offsets; bugs here can create a systematic mismatch between
  intended tip starts and realized tip/module starts.

Relevant code:

- `aic_utils/aic_isaac/scripts/isaac_episode_configs.py`
  - `_apply_start_near_gate()` computes reference semantic-tip position and reset-body pose.
  - It rejects negative `axial_distance_m`, so positive-depth final-window starts need calibrated body shifts rather
    than vanilla request generation.
- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/events.py`
  - `reset_robot_tcp_to_episode_start()` reads per-episode `reset_mode`, `reset_body_name`, pose, and env origin.
  - It raises if a reset batch mixes reset body names.
- `aic_utils/aic_isaac/scripts/build_lowtheta_reset_from_metric.py`
  - now fixed to subtract the selected clone origin when writing episode-local coordinates.

Planned checks:

1. Audit all promoted episode dirs for reset-body uniformity and intended/reference semantic tip pose.
2. Compare intended `reference_tip_center_position_world` to post-step `sfp_tip_link` and `sfp_module_link`.
3. Reject curricula where zero-action starts are in contact, already inserted without module consistency, or have
   repeated one-YAML starts.

Initial Track A result:

- Added reset-body/reference warnings to `audit_reset_curriculum_distribution.py`.
- v622 accepted low-r audit:
  - `reset_body_name`: `wrist_3_link` for all 35 episodes.
  - `reference_reward_body_name`: `sfp_tip_link` for all 35 episodes.
  - reset body differs from reward body in all episodes, with reset-body/reference offset around `0.253-0.256 m`.
  - This is expected for wrist IK plus semantic tip compensation, but it means post-step validation is mandatory; the
    YAML metadata alone cannot prove tip placement.
  - Decision hint remains reject for strict-orientation curriculum because validation theta is `0.048-0.083 rad`.
- v642 low-theta audit:
  - `reset_body_name`: `gripper_tcp`, `reference_reward_body_name`: `sfp_tip_link`.
  - reset body/reference offset is much smaller, `0.0596 m`.
  - Step-1 depth/lateral are close for some envs, but theta remains `0.0389-0.0478 rad`; accepted `0/15`.

## Track B: Phased Reward/Curriculum Ablations

Use the existing multiplicative gate:

`G_insert = G_lateral * G_orientation * G_action_axis`

Do not reward forward/tip-depth progress when lateral, orientation, or module gates fail.

First ablations:

1. Phase A/B only: approach and near-gate alignment with no positive-depth reward.
2. Phase C shallow insertion: small gated axial reward plus strong bypass penalty.
3. Phase D final-window retention: module-depth progress/loss only after tight r and bounded theta.

Starting checkpoints:

- partial-depth seed: v639 checkpoint 100 / v640 rollout baseline.
- older near-depth seed: v483 checkpoint 400, because it remains competitive on v622 policy-only checks.

Promotion:

- Any promoted run must improve at least two of `s`, `r`, `theta`, module consistency without worsening false positives.
- A run with larger tip `s` but module consistency error above `5 mm` is rejected.

## Track C: Controller/Teacher Trajectory Generation

Reason:

- Reward-only no-guard training repeatedly increases tip signed depth while module/body consistency lags.
- Literature and local evidence both point toward contact-aware compliance/teacher data for tight insertion.

First experiments:

1. Run a teacher collection from v622/v642 final-window starts with tiny axial steps and no hard success claim.
2. Collect state/action/metric rows only when module consistency improves or stays bounded.
3. Train a conservative imitation/residual branch from the accepted teacher rows.

Reject if:

- theta only improves after retreat,
- module `s` remains far behind tip `s`,
- realized motion shows lateral sweep or bypass.

## Track D: Model Architecture Ablations

Current repo support:

- `configs/train/act.yaml` uses ResNet18.
- `configs/train/act_resnet50.yaml` exists and uses ImageNet ResNet50.
- `configs/train/vision_offline_serl_convnext_history.yaml` exists and uses `convnext_tiny_imagenet` for the critic.
- Isaac online SERL supports:
  - `--critic_image_encoder_override convnext_tiny_imagenet`
  - `--critic_state_history_steps`
  - `--actor_state_history_steps`

Important constraint:

- Resuming older ACT-adapter checkpoints with `actor_state_history_steps > 1` changes adapter input dimension. Treat
  actor-history as a new-architecture branch, not a direct continuation of v639 unless checkpoint loading proves
  compatible.

First ablations:

1. Critic-only architecture: ConvNeXt Tiny ImageNet + critic state history 4, seeded from v483/v639.
2. Actor-history branch: initialize from ACT TorchScript with a zero adapter and actor state history 4, then train short
   randomized no-guard smoke. Compare to v640/v643.
3. Offline stronger ACT: ResNet50 ACT or ConvNeXt-based ACT if local trainer supports it; only promote after policy-only
   Isaac rollout improves strict metrics.

Initial Track D result:

- v644 architecture smoke used `--act_only`, `--actor_state_history_steps 4`,
  `--critic_state_history_steps 4`, and `--critic_image_encoder_override convnext_tiny_imagenet`.
- The Isaac online path initialized and stepped successfully with one update, so actor-history and ConvNeXt critic wiring
  are available for bounded experiments.
- v644 is not an insertion improvement: best row was `s=4.012 mm`, `r=0.227 mm`, `theta=0.05798 rad`, final axial
  consistency error `41.791 mm`, strict success false.
- Because actor history changes adapter input dimensionality, v644 is a new-architecture branch rather than a direct
  continuation of v639/v643.

Track D reproducibility update:

- Added `aic_utils/aic_isaac/scripts/build_architecture_ablation_commands.py`.
- The script reads a preserved `train_config.json`, replaces architecture flags exactly, and writes reproducible command
  files plus `architecture_ablation_manifest.json`.
- It now emits container-relative output paths when `--docker-container` is used; an earlier v674 smoke wrote to the
  container-local absolute `/data1/...` path and had to be copied back to the host output tree.

Generated command set:

- manifest:
  `outputs/agentic_reward_curriculum_20260529/commands/architecture_v674_from_v643/architecture_ablation_manifest.json`
- base:
  `outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-30_07-27-45_train_v643_noguard_randomized_v622_continue_from_v639_ckpt100/train_config.json`
- variants:
  - `critic_convnext_hist4`
  - `actorcritic_hist4_convnext`
  - `critic_resnet18_imagenet_hist4`

v674 critic-only ConvNeXt/history smoke:

- command:
  `outputs/agentic_reward_curriculum_20260529/commands/architecture_v674_from_v643/critic_convnext_hist4.sh`
- copied-back run:
  `outputs/agentic_reward_curriculum_20260529/architecture_ablation_runs/2026-05-30_09-57-22_arch_critic_convnext_hist4_smoke`
- setup: critic image encoder `convnext_tiny_imagenet`, critic state history `4`, actor state history `1`, 4 envs,
  120 steps, strict success unchanged.
- result: strict success `0`.
- best centered-depth row: step 113/env1, `s=4.378 mm`, `r=0.224 mm`, theta `0.05621 rad`, final axial consistency
  error `41.435 mm`, consistency gate `0`.
- final step 120 distribution: `s=-6.726..0.842 mm`, `r=0.350..3.430 mm`, theta `0.051..0.069 rad`, consistency
  gate `0`.

Decision:

- Reject v674 as an insertion improvement.
- Do not extend architecture training on the current reset family. The architecture path initializes and trains, but it
  does not address the persistent reset/contact/theta/module-consistency blocker.
- Keep the generated architecture commands for later reuse after a valid randomized shallow/final reset source exists.

## Track E: Literature-Derived Ideas

Initial public literature scan:

- SERL emphasizes sample-efficient off-policy real-robot RL with image observations, demonstrations, and robust system
  engineering: https://serl-robot.github.io/ and https://huggingface.co/papers/2401.16013
- HIL-SERL reports precise manipulation through demonstrations, intervention data, and binary reward classifiers:
  https://hil-serl.github.io/ and https://huggingface.co/papers/2410.21845
- Eureka uses LLM-generated reward code and evolutionary feedback for reward design:
  https://arxiv.org/abs/2310.12931
- RL-VLM-F learns rewards from VLM preference feedback over visual observations instead of raw scalar VLM rewards:
  https://arxiv.org/abs/2402.03681
- REvolve evolves rewards using LLMs and human feedback:
  https://arxiv.org/abs/2406.01309
- Contact-rich assembly literature repeatedly emphasizes compliance/force feedback, curriculum, and sim2real/domain
  randomization:
  https://arxiv.org/abs/2008.10224,
  https://arxiv.org/abs/2305.17110,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10590057/,
  https://www.sciencedirect.com/science/article/pii/S0736584522001995

Literature refresh on 2026-05-30:

- SERL's paper/project page emphasizes a complete real-robot RL system: off-policy RL, reward/reset methods, and a
  high-quality robot controller. Local implication: our blocker is aligned with the controller/reset part of SERL, not
  just the neural architecture.
- HIL-SERL emphasizes demonstrations, interventions, and human-in-the-loop correction for precise manipulation. Local
  implication: a binary intervention/success classifier over strict geometry + visual sanity is more appropriate than
  trusting dense insertion reward alone.
- Eureka and REvolve-style reward evolution are relevant only under hard evaluator constraints. Local implication:
  candidate rewards must be rejected if they rank tip-depth-only states above module-consistent states.
- RL-VLM-F and real-world offline VLM-feedback work are useful as visual audit/preference labelers, not as direct
  success checkers. Local implication: use VLM-style visual review on saved center/left/right frames to flag bypass or
  ambiguous seating, while strict success remains geometry/module based.
- IndustReal is the closest contact-rich assembly analogue: it combines SDF-style rewards, sampling curricula,
  simulation-aware policy updates, and policy-level action integration. Local implication: the next code path should
  prioritize contact/controller action integration or compliant residual control before more SERL reward sweeps.
- Peg-in-hole compliance papers repeatedly point to admittance/variable compliance or force-conditioned control for
  tight-clearance insertion. Local implication: add a bounded compliance/action-integrator diagnostic around the final
  seating phase instead of increasing rotation authority or axial reward.

Immediate ideas to test:

1. HIL-SERL-style binary classifier/audit label for strict visual/module consistency, used as an auxiliary reward rather
   than trusting dense reward return.
2. IndustReal/contact-rich-style residual action integration and compliance/force gates before additional reward tuning.
3. Eureka/REvolve-style reward evolution, but with hard evaluator constraints: candidates are rejected if reward prefers
   tip-depth-only states over module-consistent states.
4. RL-VLM-F-style visual preference audit on saved center/left/right frames to detect false positives; this is an
   evaluator/audit aid, not a success criterion by itself.

## Execution Order

1. Track A reset/randomization audit.
2. Track C teacher/controller trajectory generation, because current reward-only training lacks positive examples of
   module-consistent full-depth motion.
3. Track D architecture smoke tests, starting with critic-only ConvNeXt/history because it is already supported by CLI.
4. Track B reward/curriculum ablations only after reset/teacher evidence identifies which phase is actually learnable.
5. Track E literature-derived additions are folded into the above tracks as specific ablations, not run as vague long
   training.

## Completion Criteria

The broader goal remains incomplete until either:

- strict full insertion is demonstrated with post-step strict success and visual sanity, or
- the bounded tracks above are exhausted and the blocker report identifies the remaining controller, contact, reset,
  architecture, or reward/curriculum limitation with run evidence.

## Track C Update: v645-v648 Final-Window Teacher/Replay Tests

I ran a deeper Track C sequence because reward-only no-guard training was repeatedly producing shallow tip-depth false
positives:

| run | purpose | best relevant post-step row | strict_success | decision |
|---|---|---|---:|---|
| v645 | privileged target-tip teacher from low-theta v642 starts | step20/env2: `s=45.845 mm`, `r=0.083 mm`, theta `0.04221 rad`, module `s=22.210 mm`, module `r=0.885 mm`, consistency error `0.015 mm` | 0 | useful near-full module-consistent evidence, but theta/contact blocked |
| v646 | robot-joint reset replay from v645 with zero velocities | step1 mean `s=40.680 mm`, mean `r=0.401 mm`, mean theta `0.04446 rad`, mean consistency error `5.932 mm`; step10 degrades | 0 | reject as stable reset source |
| v647 | robot-joint reset replay from v645 with recorded velocities | step1 best `s=47.872 mm`, `r=0.530 mm`, theta `0.03938 rad`, module `s=24.231 mm`, consistency error `0.009 mm`; step10 degrades | 0 | immediate replay improved, but not stable enough for training |
| v648 | conservative final-orientation teacher from v647 recorded-velocity resets | step2/env1: `s=49.055 mm`, `r=0.299 mm`, theta `0.04007 rad`, module `s=25.416 mm`, consistency error `0.011 mm`; videos show visual ambiguity | 0 | reject as strict success; classify as near-full orientation/contact blocked |

Interpretation:

- There are now reproducible near-full metric states with module/tip axial consistency, which is an improvement over the
  shallow no-guard runs.
- The remaining blocker is not scalar reward. The final-window state is contact-sensitive: zero-action replay backs out,
  recorded-velocity replay is only briefly near-full, and the conservative orientation trim does not reduce theta below
  the strict `0.030 rad` threshold.
- v648 also confirms the false-positive risk: high tip signed depth plus good axial consistency is still visually
  ambiguous and must not be counted as insertion unless the strict checker and center/left/right visual sanity agree.

Next Track C action:

1. Stop adding more reward-only axial pressure.
2. Add a tighter final-window diagnostic/classifier for `near_full_orientation_contact_blocked` and separate it from
   generic module-depth blockage.
3. If time remains in the Track C cap, test one bounded final-window orientation/contact variant from v647:
   orientation-only or micro-backoff/micro-reinsert, with force gating and no over-depth promotion.

Classifier update:

- `aic_utils/aic_isaac/scripts/extract_privileged_residual_targets.py` now iterates all envs in multi-env metrics
  instead of silently collapsing most rows to env0.
- It adds `near_full_orientation_blocked` and `near_full_orientation_contact_blocked` labels when tip/module axial
  consistency is near target but theta remains above strict threshold.
- Re-extracting v645+v648 produced 1360 labeled rows:
  - `tip_depth_false_positive`: 886
  - `contact_spike`: 444
  - `lateral_bypass`: 15
  - `near_full_orientation_blocked`: 6
  - `near_full_orientation_contact_blocked`: 4
  - `prejump_realization_mismatch`: 5
- Validation after this code change: `python -m py_compile` passed and insertion reward geometry tests passed
  (`31 passed`).

Micro-backoff probe:

- v649 tested the bounded final-window variant from v647 with lower target depth (`45.8 mm`), 3 um axial steps,
  30 um lateral steps, 0.00004 rad rotation steps, tighter lateral gate, and realized-r recovery.
- Strict success remained false.
- Best near-full row: step21/env6, `s=49.741 mm`, `r=0.384 mm`, theta `0.03890 rad`, module `s=26.104 mm`,
  module `r=0.399 mm`, consistency error `0.013 mm`, force `27.99 N`.
- Lowest-theta near-full-ish row: step1/env1, `s=48.642 mm`, `r=0.395 mm`, theta `0.03740 rad`, module
  `s=25.000 mm`, consistency error `0.007 mm`, force `32.20 N`.
- Decision: reject as success and reject as training data. It nudges theta slightly lower only in high-force or visually
  ambiguous states and still misses the strict `0.030 rad` threshold.
