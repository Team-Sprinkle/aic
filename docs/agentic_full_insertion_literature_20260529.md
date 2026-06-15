# Literature-Driven Full-Insertion Experiment Notes - 2026-05-29

Scope: extract testable ideas for strict SFP-to-NIC insertion in Isaac. Success remains strict post-step geometry plus visual sanity, not reward return.

## Sources Read

- SERL: https://arxiv.org/abs/2401.16013 and https://serl-robot.github.io/
- HIL-SERL: https://hil-serl.github.io/ and https://hil-serl.github.io/static/hil-serl-paper.pdf
- Eureka reward synthesis: https://arxiv.org/abs/2310.12931
- RL-VLM-F: https://arxiv.org/abs/2402.03681
- Real-world offline RL from VLM feedback: https://arxiv.org/abs/2411.05273
- IndustReal contact-rich assembly: https://arxiv.org/abs/2305.17110
- Tactile/force peg-in-hole insertion: https://arxiv.org/abs/2309.15681
- Contact-rich imitation survey: https://arxiv.org/abs/2506.13498

## Actionable Takeaways

1. SERL/HIL-SERL: do not expect off-policy RL to fix a bad reset/servo target.
   - In this repo, promote only guide/reset settings that improve post-step `s/r/theta/module` before long training.
   - Prefer conservative actor updates, guide distillation, and residual behavior around the controller.

2. Eureka-style reward synthesis: constrain reward iteration with geometry audits.
   - Candidate reward changes must be rejected if positive axial motion scores high while `r`, `theta`, or module consistency are bad.
   - Use the existing multiplicative insertion gate as the base, not a new additive score.

3. RL-VLM-F-style visual feedback: use visual preferences as audit labels, not success.
   - Center/left/right frames can classify "visually inserted", "bypass", "ambiguous", or "not contacted".
   - The strict checker remains authoritative; visual labels catch false positives and asset/camera artifacts.

4. Contact-rich insertion literature: split approach and final contact.
   - Approach can be learned/imitated from vision and state.
   - Final insertion should be a guarded residual/compliance-like servo using semantic tip/body frames, force/contact spikes, realized motion history, and module consistency.

5. Partial observability is real for this task.
   - A current-frame actor cannot directly infer whether the last rotation command caused lateral sweep or whether contact is building.
   - The lowest-risk test is state/action history in the adapter or residual head before replacing the ACT runtime path.

6. HIL-SERL suggests the missing ingredient is likely useful intervention data, not just a scalar reward.
   - HIL-SERL explicitly combines demonstrations, negative/positive samples, and interventions during online RL.
   - In this repo, the closest analogue should be policy-owned rollouts with non-executed privileged guide/imitation targets or replay from successful guide-only final insertion, not hard action overrides during RL.

7. VLM-feedback work is most relevant as a visual audit source.
   - RL-VLM-F and follow-up offline VLM-feedback work learn reward/preferences from visual observations, but this task already has privileged geometry.
   - Use VLM/manual labels to catch "looks inserted", "bypass", and "ambiguous contact" frame classes; keep strict geometry as the success gate.

## Candidate Experiments

| ID | Idea | Repo-level test | Promote if |
| --- | --- | --- | --- |
| L1 | Residual guarded final insertion | Guide-only two-stage final servo with realized command diagnostics, no actor updates | Realized module/tip axial delta follows commanded tiny axial steps while `r/theta` stay gated |
| L2 | Imitation-heavy residual policy | Train from a guide trajectory only after L1 produces non-false-positive progress | Held-out 40/10 rollout improves strict metrics without actor drift |
| L3 | History-aware adapter | Use `--actor_state_history_steps 4` and `--critic_state_history_steps 4` with conservative updates | Better 40/10 post-step `r/theta/module` than the same guide/checkpoint without history |
| L4 | Stronger critic vision | Use `--critic_image_encoder_override convnext_tiny_imagenet` only with a stable guide | Improves training stability or candidate ranking without increasing false positives |
| L5 | Visual audit classifier | Save 1-second snapshots and label best/last frames manually or with a simple script | Flags bypass/ambiguous states that metrics might overrate |
| L6 | Reward synthesis loop | Sweep phase-conditioned gates and bypass penalties through offline reward audit first | Forward motion outside lateral/orientation/module gates stays low/negative |
| L7 | No-execute guide imitation | Use privileged guide as actor loss with `collect_blend=0` and no action guard | Policy-owned rollout improves `s/r/theta/module` versus same checkpoint without guide |

## Current Recommendation

Do not spend the next 7-hour block on reward-only or architecture-only training. The best use of the next block is intervention-quality data: either make L1 produce a repeatable module-consistent guide-only final insertion, or make L7 strong enough to change policy-owned behavior without executing the guide. Once a guide-only or no-execute-imitation trajectory improves module axial consistency under full-depth metrics, run L2/L3 training from the best shallow/warm checkpoint with periodic 40/10 rollouts.

## Update - 2026-05-29 v527

L7 was smoke-tested as `train_v527_noguard_shallow2x0_guideimitation_noexecute_from_v483_ckpt400`. The guide target was computed, `target_action_guide_weight=0.04`, `collect_blend=0.0`, and the action guard stayed disabled, so this was policy-owned execution with privileged guide loss only. The loss was active (`target_action_guide_loss=2.03e-4` at step 320) but the rollout remained at the entrance: best `s=-0.566 mm`, `r=0.272 mm`, `theta=0.0510 rad`, module `s=-24.19 mm`, strict false.

Implication: the HIL-SERL-style intervention idea is still sound, but a weak always-on guide imitation loss is insufficient. Next L7 variant should be phase-specific and stronger, or it should consume actual successful/near-success final-window guide data. If that data cannot be generated, the blocker is intervention data quality rather than vision backbone capacity.

## Update - 2026-05-29 v528

The stronger no-execute L7 variant used `target_action_guide_weight=0.40` and lower actor-Q pressure. It was stable and the weighted guide loss increased to `8.99e-5`, but it still failed strict insertion: best `s=-0.843 mm`, `r=0.769 mm`, `theta=0.0476 rad`, module `s=-24.47 mm`, strict false.

Implication: simply increasing always-on guide imitation is not enough. The literature-aligned next step is closer to HIL-SERL intervention data quality: generate or mine final-window examples that are actually module-consistent, then use those as sparse/phase-specific intervention targets. Architecture-only changes such as stronger critics or history should remain secondary because the current ConvNeXt/history critic already exists and actor-history ablation regressed from older checkpoints.

## Update - 2026-05-29 v529-v530

The phase-specific version of L7 was implemented and tested. `v529` trained a no-guard policy from v483 checkpoint 400 with guide loss only on centered/high-theta final-window samples. It was stable and the phase selector was active, but the best post-step depth stayed at `s=-0.564 mm` with module `s=-24.19 mm`; strict success remained false.

`v530` then removed guide imitation and increased gated axial/action-axis reward while preserving module/bypass penalties. It was stopped after checkpoint 300 because it also stayed at the entrance (`best s=-0.568 mm`, `theta=0.0481 rad`, module `s=-24.19 mm`, strict false).

Literature interpretation: the HIL-SERL/Eureka-style lesson is not just "add a stronger reward." The agent needs useful intervention examples or a curriculum distribution containing solvable module-consistent final insertion states. In the current setup, the policy repeatedly sees centered pre-insertion hover states, so both reward-only training and no-execute imitation optimize hover/trim behavior instead of the missing module-following axial motion.
