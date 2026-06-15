# Agentic Phased Ablation, Architecture, and Literature Plan - 2026-05-29

## Current State

All host/container Isaac, SERL, train, rollout, eval, and Kit jobs were stopped and the process table was clean before this plan was written.

The latest evidence rejects continuing single-reset shallow training. Runs v529/v530 and earlier v519/v520/v524/v526 repeatedly used `full_depth_start2x0_v464_settle_centered_from_v462` with `num_envs=1`, which taught entrance hover/alignment but not module-consistent full insertion. Randomized v543 curriculum is physically plausible, but its reset distribution has a theta floor around `0.05 rad`; v544/v551 did not produce any strict-theta samples below `0.03 rad`.

Best recent partials to rebase from:

| candidate | evidence | limitation |
|---|---|---|
| v514 checkpoint family | best low-theta partial: `s=7.87 mm`, `r=0.321 mm`, `theta=0.0201 rad` | not reproducible, module_s `-15.78 mm`, final axial consistency error `37.96 mm` |
| v513 | best s `8.41 mm`, `r=0.223 mm`, `theta=0.0522` | theta not strict and module still behind |
| v516 | best s `28.14 mm` | lateral bypass: `r=57.3 mm`, module_r `54.8 mm`; false positive |
| v543/v544/v551 | randomized/no-guard curriculum | no strict theta samples and no module-consistent insertion |

## Time Allocation

Each work package is capped at 7 hours before it must either promote a candidate, reject the family, or produce a blocker note.

| package | cap | stop/promote criterion |
|---|---:|---|
| Reset and geometry randomization fixes | 7 h | validated randomized reset buckets include shallow/final, near-gate, bridge, and held-out 40/10 with post-step theta including `<0.03 rad` cases and no impossible lateral offsets |
| Reward/curriculum ablations | 7 h | at least one short run improves strict-near metrics without tip-only/module false positives |
| Architecture/history ablations | 7 h | compare current actor/critic against history and ConvNeXt critic variants using same curriculum/eval |
| Literature-derived experiments | 7 h | implement one concrete idea from papers, not just notes; e.g. automated reward audit/selection, preference-style visual sanity scoring, or intervention/imitation schedule |
| Reporting and blocker synthesis | 7 h | reproducible tables, commands, metrics, and next-code-change recommendation |

## Literature Priors

SERL emphasizes sample-efficient off-policy robotic RL with controllers, reset mechanisms, rewards, and image observations as part of one system, so our experiments should treat reset/servo/reward/controller mismatch as first-class variables rather than only tuning scalar reward. Source: SERL paper, arXiv 2401.16013, https://arxiv.org/abs/2401.16013.

HIL-SERL suggests that precise manipulation benefits from intervention data when autonomous exploration stalls. We cannot silently add human intervention here, but the analogous Isaac path is privileged guide/intervention trajectories that are logged, filtered, and imitated without hard action override at execution time. Source: HIL-SERL project/paper page, https://hil-serl.github.io/.

Eureka shows LLM-generated reward code can be iteratively improved with environment feedback, but its relevance here is bounded by strict metrics: candidate rewards must be rejected if they pay tip-depth-only progress or visual false positives. Source: Eureka, arXiv 2310.12931, https://arxiv.org/abs/2310.12931.

RL-VLM-F learns rewards from VLM preferences over image pairs instead of raw VLM scores. For this project, the practical adaptation is not to replace the strict geometry checker, but to add a visual sanity preference/audit over center/left/right frames to flag bypass, already-contact starts, and ambiguous near-seat states. Source: RL-VLM-F, arXiv 2402.03681, https://arxiv.org/abs/2402.03681.

## Experiment Families

### A. Reset/Geometry Ablations

Goal: fix the reset distribution before more long training.

1. Validate wrist-to-tip reset calibration explicitly:
   - compare requested `start_near_gate` axial/lateral/theta against post-step `s/r/theta` for wrist reset, `gripper_tcp`, and semantic `sfp_tip_link` metrics;
   - log `body_start_position_world`, `reference_tip_center_position_world`, `reset_body_offset_from_reference_world`, measured tip pose in reset body, and post-step realized tip pose.
2. Build a low-theta reset sweep from the v514 best-row geometry or measured strict-theta correction, but only promote if validation proves post-step theta < `0.03 rad` without lateral/module regressions.
3. Keep existing fixed 40/10 configs as held-out eval; reject synthetic 40/10 generated from shallow bases until validated.

### B. Reward/Curriculum Ablations

Use v514/v513 as partial-depth seeds where appropriate, but evaluate on strict post-step metrics.

1. Rebase short no-guard runs from v514 checkpoint 200/400 and v513/v485 candidates.
2. Sweep only safe reward changes:
   - stronger depth-gated module consistency after shallow positive s;
   - no forward credit outside lateral/orientation/action gates;
   - tighter bypass penalty for tip s improving while module_s stays behind;
   - shallow/final-window probability increased only after resets validate.
3. Reject any run where best s improves only with lateral/module false positives.

### C. Architecture/History Ablations

The current trainer already supports config-only architecture switches:

| option | flag |
|---|---|
| actor low-dimensional state history | `--actor_state_history_steps N` |
| critic low-dimensional state history | `--critic_state_history_steps N` |
| stronger critic vision encoder | `--critic_image_encoder_override convnext_tiny_imagenet` |

Initial ablation matrix:

| candidate | actor history | critic history | critic vision | checkpoint |
|---|---:|---:|---|---|
| arch_base | 1 | 1 | current/small_conv | v514 or v483 |
| arch_critic_hist | 1 | 4 | current/small_conv | same |
| arch_actor_critic_hist | 4 | 4 | current/small_conv | same |
| arch_convnext_hist | 4 | 4 | `convnext_tiny_imagenet` | same |

Training should be short first, with `num_envs=8` only if reset validation is stable. Held-out evals remain fixed 40/10, randomized near-gate, and randomized shallow/final-window.

### D. Literature-Derived Agentic Loop

1. Use Eureka-style iteration only as a proposal generator, with strict automated rejection rules:
   - reject if reward surface pays forward insertion when r/theta/module gates fail;
   - reject if run improves reward but not strict metrics.
2. Use RL-VLM-F-style visual preference only as an audit layer:
   - compare before/after center/left/right frames;
   - label likely bypass/already-contact/ambiguous insertion;
   - never override strict geometry.
3. Use HIL-SERL-style intervention analog:
   - collect privileged guide/controller trajectories;
   - filter to rows with low r/theta and improving module consistency;
   - train as residual/imitation data only after validation, not hard action replacement.

## Immediate Next Actions

1. Add a reset-calibration diagnostic that measures wrist/gripper/tip initialization error for episode folders and summarizes post-step reset distributions.
2. Run it on v464, v543, v514-related configs, and fixed 40/10.
3. Generate architecture ablation commands using existing config-only flags, but do not launch long training until the reset diagnostic identifies a valid curriculum.
4. If a low-theta randomized reset cannot be validated, classify blocker as reset/controller geometry rather than reward or model capacity.

## Update: Low-Theta Metric Reset Attempt v552

I added `aic_utils/aic_isaac/scripts/build_lowtheta_reset_from_metric.py`, which takes the v514 measured low-theta partial (`step=232`, `s=7.87 mm`, `r=0.321 mm`, `theta=0.0201 rad`) and the calibrated semantic-tip-in-wrist transform from v464, then solves a wrist pose intended to place the semantic tip at requested shallow/final-window depths.

Validation rejected this direct reset strategy:

| run | result |
|---|---|
| `validate_v552_lowtheta_metric_from_v514_step232` | post-step tip `s=-247.6..-87.4 mm`, `r=38.2..59.9 mm`, `theta=2.96..3.11 rad`, force saturated at `35 N` |

The reset IK did solve the requested wrist pose, but that pose put the wrist/cable into an impossible flipped configuration. This means the v514 low-theta pose cannot be naively converted into a start reset by solving only the wrist pose from a static tip transform. The low-theta event appears to be a dynamically reached configuration, not a directly resettable wrist pose under the current initialization path.

Decision: reject v552 for training. The next reset fix should use a controller/settle calibration sweep around the existing valid wrist orientation, or reset from robot joint state snapshots if joint-state logging is enabled, rather than directly imposing the measured semantic-tip quaternion.

## Update: Randomized/Robot-State Curriculum Attempts v553-v558

I added two small reset/curriculum utilities:

- `aic_utils/aic_isaac/scripts/audit_reset_curriculum_distribution.py`
- `aic_utils/aic_isaac/scripts/build_robot_joint_reset_from_metrics.py`

I also added `--log_robot_state_every` to `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py` so promising articulated states can be materialized as `reset_mode: robot_joint_state` episodes instead of reconstructing a wrist pose from semantic-tip geometry.

Validation:

| check | result |
|---|---|
| `python -m py_compile .../serl/train.py .../build_robot_joint_reset_from_metrics.py` | passed |
| `.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py` | `31 passed` |

Recent decisions:

| run/config | result | decision |
|---|---|---|
| v553 randomized mixed + actor history/ConvNeXt | best theta `0.0488 rad`, best s `1.95 mm` with theta `0.058 rad`, zero samples <= `0.03 rad` | reject architecture/history as immediate bottleneck |
| v554 recapture v514 recipe with robot-state logging | robot state logged every step, but best theta only `0.0495 rad`, no positive s | reject old v514 metric as reproducible low-theta source |
| v555 policy-only from v514 checkpoint 200 | best theta `0.0478 rad`, no positive s | reject checkpoint-only rebase as reproducible low-theta source |
| v556 robot-joint reset validation | stable reset: step-1 mean `s=-1.84 mm`, `r=0.059 mm`, `theta=0.0505 rad`, module_s `-25.47 mm`; zero-action settle stays plausible | accept robot-joint reset mechanism, but theta is not strict |
| v557 no-guard training on v556 final-window starts | best theta improved to `0.0399 rad` with `r=0.507 mm`, but s stayed negative; no samples <= `0.03 rad` | partial improvement, not promoted |
| v558 axial-boost from v557 checkpoint 100 | produced positive tip s in 16 samples; best s `0.315 mm`, but theta `0.0537 rad` and module_s `-23.31 mm`; best theta still `0.0399 rad` at negative s | reject as tip-depth false-positive risk |
| v559 orientation-first from v557 checkpoint 100 | best theta `0.0396 rad` with `r=0.536 mm`, no positive s, no samples <= `0.03 rad`; reward had large penalty spikes | reject as orientation plateau, not progress |

Current blocker classification:

- Primary blocker: final-window orientation remains around `0.04-0.05 rad`; strict target is `<0.03 rad`.
- Secondary blocker: increasing axial incentive before strict theta causes shallow tip-depth progress without module-consistent insertion.
- Orientation-first reward shaping alone did not break the `~0.04 rad` plateau.
- Not currently supported by evidence: model capacity alone. v553 did not improve strict-theta metrics.
- Useful new capability: robot-joint-state resets are valid and reproducible, so further curriculum can start from physically plausible articulated states rather than unstable wrist-pose reconstructions.

Next recommended code/experiment change:

Reward-only tuning has now plateaued around `0.04 rad`. The next change should collect privileged low-theta orientation-refinement trajectories with robot-state logging and use them as imitation/residual data, not hard action override. If privileged refinement also cannot produce theta `<0.03 rad` while preserving r/module consistency, classify the blocker as controller/IK/contact realization rather than reward/curriculum.
