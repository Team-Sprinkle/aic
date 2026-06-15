# Agentic Full-Insertion Timeboxed Plan - 2026-05-29

## Objective

Reach strict full-depth SFP-to-NIC insertion in Isaac, or produce an evidence-backed blocker after bounded reward/curriculum/reset/architecture/literature-driven iterations. Strict success remains post-step full-depth insertion with tight lateral error, semantic tip orientation, module/body consistency, and visual sanity. Tip-depth-only progress is a false positive.

## Current State

All Isaac training/eval jobs were stopped before starting this plan.

Best current partials:

- v483 transient: `s=17.372 mm`, `r=0.445 mm`, `theta=0.05138 rad`, `module_s=-6.250 mm`; strict false and not reproducibly captured by actor-only checkpoint resume.
- v485 mixed bridge: best depth `s=13.702 mm`, `r=0.132 mm`, `theta=0.05277 rad`; best theta `0.02746 rad` only at shallow `s=1.652 mm`; strict false.
- v514 shallow continuation: `s=7.87 mm`, `r=0.321 mm`, `theta=0.0201 rad`, but module lag/final axial consistency error remained large; strict false.

Latest failure diagnosis:

- Hard failure termination protects against lateral bypass but can end 10/2 and 20/4 bridge episodes before useful learning.
- Disabling or delaying termination produces axial-looking false positives with huge lateral error.
- Small reset-quaternion sweeps did not bring post-step reset theta below strict threshold.
- Existing online SERL resume preserves actor weights but refreshes critic/replay/optimizer state, so rare v483-like transients are not reproducibly preserved.

## Time Allocation

Each task family is capped at 7 hours of active work:

1. Reset/randomization and semantic tip/body initialization audit: 7 h max.
2. Architecture/history/vision ablations: 7 h max.
3. Reward/curriculum ablations from best 8-17 mm partial checkpoints: 7 h max.
4. Literature-review-driven experiment ideas and paper framing: 7 h max.
5. Implementation of the best next structural change: 7 h max.

## Phase A: Reset/Randomization Audit

Questions:

- Is any randomization still moving the gripper/wrist relative to `sfp_tip_link` after `start_near_gate` metadata is generated?
- Does `reset_body_name=wrist_3_link` plus calibrated semantic tip transform produce the requested post-step tip `s/r/theta`?
- Are fixed lateral directions and sampled lateral directions being mixed in a way that makes 40/10, 20/4, 10/2 incomparable?

Planned checks:

- Compare generated YAML metadata with post-step `post_step_insertion_geometry` for 2/0, 10/2, 20/4, 40/10.
- Inspect `events.py` reset ordering and randomization profile interactions.
- If needed, add a reset audit script that logs reset-body pose, semantic tip pose, module pose, target gate pose, and post-step drift in one JSON.

## Phase B: Architecture/History/Vision Ablations

Existing hooks to use before adding new architecture:

- ACT BC config supports `vision_backbone: resnet50` via `configs/train/act_resnet50.yaml`.
- Online SERL supports `--actor_state_history_steps` and `--critic_state_history_steps`.
- Online/offline SERL critic supports `--critic_image_encoder_override convnext_tiny_imagenet`.
- `configs/train/vision_offline_serl_convnext_history.yaml` already exists for stronger critic vision/history-style training.

Bounded ablations:

- A1: v485/v514 checkpoint with actor history 4 and critic history 4, no guard, shallow 2/0.
- A2: same with ConvNeXt critic override.
- A3: same with 10/2 bridge and strict termination unchanged.
- A4: if offline dataset/checkpoint paths are available, train a ResNet50 or ConvNeXt-based ACT/SERL adapter warm start and eval only after compile/preflight passes.

Promotion gate:

- Improve at least two of `s`, `theta`, `module_s/final_err` without increasing lateral false positives.
- No promotion from reward alone.

## Phase C: Reward/Curriculum Ablations

Start from partial checkpoints, not from scratch:

- v483/v485/v514 shallow partials for final insertion.
- 10/2 only after final-window policy combines `s > 8 mm`, `r < 0.5 mm`, `theta < 0.03 rad`, and improving module depth.

Families:

- C1: module-consistency-first shallow curriculum: reward/terminate on module lag while keeping axial reward gated.
- C2: two-stage curriculum: freeze/low actor Q for first 100-200 steps to evaluate actor behavior, then small updates.
- C3: failure termination schedule with strict early lateral gate but longer timeout only when `r < 1 mm`.

## Phase D: Literature Review

Scope:

- SERL/HIL-SERL and sample-efficient robot RL.
- LLM/reward design systems such as Eureka.
- VLM/reward learning such as RL-VLM-F.
- Contact-rich peg/cable insertion and residual/compliance policies.
- History/visual representation choices for manipulation policies.

Deliverable:

- Update or replace `docs/agentic_full_insertion_literature_20260529.md` with sources, claims, and concrete experiment ideas.

## Phase E: Structural Improvement

Current leading hypothesis:

Reward-only online SERL is not enough because rare insertion transients are not reproducibly preserved and module-consistent insertion requires temporal/contact context. The likely next implementation is one of:

- full-state online SERL checkpoint/resume for actor, critics, optimizers, replay, normalizers, and history state;
- offline residual target extraction from guarded/v483-like rollouts followed by imitation-heavy adapter training;
- history-aware adapter/critic training from the best partial checkpoint with ConvNeXt critic and conservative actor Q;
- a module-consistency auxiliary target that predicts/fits `module_s` and final axial consistency from observations/history.

First implementation target should be the smallest change that directly tests one of these hypotheses while keeping old ACT/offline SERL/online SERL/Gazebo/runtime paths intact.

