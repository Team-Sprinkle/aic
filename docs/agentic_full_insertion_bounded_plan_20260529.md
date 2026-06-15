# Agentic Full-Insertion Bounded Plan - 2026-05-29

Objective: achieve strict full SFP-to-NIC insertion from a non-contact near-gate start, initially centered on 40 mm axial / 10 mm lateral semantic tip offset. Strict success remains the full-depth checker with post-step axial depth, lateral error, semantic tip orientation, module/body consistency, and visual sanity.

## Current Evidence

- No AIC Isaac training or rollout process is currently running.
- The strongest current 40/10 runs still plateau around `s=-27.3 mm`, `r=0.06 mm`, `theta=0.054 rad`, module lateral about `1.25 mm`, with strict success false.
- The corrected seated target is full cage-depth scale, not the old 8 mm shallow criterion. The old 8-10 mm model/checkpoints remain useful warm starts, not success evidence.
- Current failures are not reward-return failures. They are coupled controller/contact/semantic consistency failures: the guard can center the tip, but realized module/body axial progress and strict orientation do not follow.
- The near-gate episode generator previously sampled a random lateral direction for `axial_distance_m/lateral_distance_m`. This is reasonable for broad training but weakens fixed 40/10 ablations. A config-driven `scene.start_near_gate.lateral_direction_world` hook was added so ablations can pin the start direction without removing randomized curricula.

## Time Allocation

Each task family is capped at 7 hours of active work before promotion, rejection, or blocker documentation.

| Family | Cap | Stop/Promote Gate |
| --- | ---: | --- |
| Reset/randomization audit and fixes | 7 h | Fixed 40/10 starts reproduce semantic tip `s/r/theta` within tolerance across seeds, or blocker identifies reset/body mismatch. |
| Controller/servo realization diagnostics | 7 h | A guide-only probe improves combined post-step `s/r/theta/module`, or evidence proves guarded commands are not realized. |
| Reward/curriculum ablations | 7 h | Offline reward audit rejects false forward progress and at least one short eval improves strict metrics without bypass. |
| Model architecture ablations | 7 h | History/vision variants run without replay/checkpoint breakage and beat the same guide baseline on held-out 40/10 eval. |
| Literature-review-based ideas | 7 h | Convert papers into 2-4 concrete repo-level candidate experiments; reject ideas that conflict with strict geometry/video gates. |
| Training from promoted candidates | 7 h per candidate family | Periodic 40/10 rollouts every about 30 minutes show improving post-step strict metrics; stop early on actor drift or false depth. |

## Experiment Order

1. Fixed-direction reset ablations.
   - Generate 40/10 variants with pinned `lateral_direction_world`, existing `wrist_3_link` repaired reset, and semantic-tip reset where feasible.
   - Run `validate_serl_reset_settle.py` for 2-5 zero/hold steps before policy learning.
   - Promote only starts whose post-settle `s/r/theta` match the requested non-contact semantic start.

2. Controller/servo realization before long training.
   - Use guide-only runs from the best current checkpoint/config.
   - Log commanded final-window axial step, realized tip axial delta, realized module axial delta, realized lateral delta, and guard branch override state.
   - If axial commands open but module/tip do not advance, prioritize controller/guard fixes over reward tuning.

3. Reward/curriculum ablations.
   - Keep insertion credit conjunctive: lateral gate * orientation gate * action-axis gate * module consistency gate.
   - Add or verify penalties for positive tip depth with poor module consistency and for forward motion outside lateral/orientation gates.
   - Bias phases toward final millimeter first, then mix 40/10 approach back in.

4. Architecture ablations.
   - First use existing low-risk switches: `--actor_state_history_steps`, `--critic_state_history_steps`, and `--critic_image_encoder_override convnext_tiny[_imagenet]`.
   - Do not replace ACT/runtime policy paths globally.
   - If history/ConvNeXt only helps Q but actor still ignores recent contact state, evaluate a residual/adapter training target rather than a broad model rewrite.

5. Literature-driven candidates.
   - SERL/HIL-SERL support conservative off-policy learning from strong controller/reset/intervention data; apply this as imitation-heavy residual training after guide-only behavior is sane.
   - Eureka-style reward synthesis is useful only inside strict audits; generated/tuned reward variants must be rejected if they pay tip-depth false positives.
   - RL-VLM-F-style visual feedback is useful as an auxiliary frame/video sanity classifier, not as the success label.
   - Contact-rich insertion work favors phase decomposition, force/contact feedback, compliance/residual policies, and guarded safety filters.

6. Training cadence.
   - Start from the best shallow/old warm checkpoint only after the corresponding guide rollout is non-contact and measured under the full-depth checker.
   - Train in bounded windows.
   - Every roughly 30 minutes or major checkpoint, run a fixed 40/10 held-out rollout with center/left/right videos and 1-second snapshots.
   - Continue only if post-step strict metrics improve without lateral bypass or module inconsistency.

## Commands

Focused unit validation after reset-generator changes:

```bash
python -m py_compile aic_utils/aic_isaac/scripts/isaac_episode_configs.py
.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_isaac_online_serl.py -k 'near_gate'
```

Reset-settle probe template:

```bash
.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/validate_serl_reset_settle.py \
  --episode-config-dir outputs/agentic_reward_curriculum_20260529/generated_episode_configs/<candidate>/episodes \
  --output-root outputs/agentic_reward_curriculum_20260529/reset_settle_validation \
  --run-name <candidate> \
  --num-envs 1 \
  --steps 5 \
  --max-lateral-m 0.012 \
  --max-theta-rad 0.080
```

Guide-only/eval runs should continue to use `train.py` with `--act_only`, `--debug_audit_steps` or no-learning update settings, strict post-step metrics, `--save_step_images`, and the current high-quality separate camera video mode.

## Literature Sources

- SERL: https://arxiv.org/abs/2401.16013 and https://serl-robot.github.io/
- HIL-SERL: https://hil-serl.github.io/ and https://hil-serl.github.io/static/hil-serl-paper.pdf
- Eureka reward synthesis: https://arxiv.org/abs/2310.12931
- RL-VLM-F: https://arxiv.org/abs/2402.03681
- Contact-rich insertion examples/surveys: https://arxiv.org/abs/2305.17110, https://arxiv.org/abs/2309.15681, https://arxiv.org/abs/2506.13498

