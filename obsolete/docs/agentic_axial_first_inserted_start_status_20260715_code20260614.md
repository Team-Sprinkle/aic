# Axial-First Inserted-Start Reward/Curriculum Status

> Archived on 2026-09-17 from `docs/agentic_axial_first_inserted_start_status_20260715_code20260614.md`.
> See the [maintained documentation](../../docs/README.md) for current status and workflows.
> Status, recommendations, and commands below describe the original work.

Saved: 2026-07-15T19:30Z UTC  
Referenced code latest update: 2026-06-14T14:58Z UTC  
Main run covered here: `/tmp/aic_axial_first_inserted_start_40depth_lateral_curriculum_20260614_145937`

## Objective

Implement and iterate a stateful, phase-gated insertion reward/curriculum where:

- lateral alignment is solved before axial insertion gets meaningful credit;
- axial insertion credit is only available in the aligned axial phase;
- axial-phase reward encourages nearly pure motion toward the target, with minimal lateral/rotational motion and no orientation/lateral worsening;
- training/eval are actor-only: no guide, no guard, no hard-coded rollout override;
- promotion-gated curriculum trains/evals every 10 episodes and only advances on success.

## Files Updated

The relevant local code/config changes were last updated on 2026-06-14:

- `configs/axial_first_inserted_start_curriculum.yaml`  
  Latest mtime: 2026-06-14T14:57:34Z. The final curriculum used here fixes axial depth at 40 mm pre-entrance and gradually reintroduces lateral error.

- `tools/train_axial_first_inserted_start.sh`  
  Latest mtime: 2026-06-14T14:58:05Z. Sets actor-only reward/training defaults for this experiment, including reduced exploration and strict axial action purity defaults.

- `tools/train_stateful_promotion_gated_insertion.sh`  
  Latest mtime: 2026-06-14T14:40:59Z. Generic promotion-gated train/eval wrapper. Added pass-throughs for strict action lateral sigma and action-radius schedule.

- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/insertion_geometry.py`  
  Latest mtime: 2026-06-14T14:12:56Z. Contains the stateful reward implementation and phase-specific totals.

- `aic_utils/aic_isaac/scripts/audit_axial_first_inserted_start_reward.py`  
  Latest mtime: 2026-06-14T14:14:12Z. Reward audit script adjusted to match the stricter off-axis/axial-action penalties.

`git status --short --untracked-files=all` was clean when this note was written.

## Reward Design State

The reward implementation is structurally sequential:

- Lateral phase activates when lateral error exceeds `lateral_enter_threshold_m`.
- Orientation phase activates after lateral alignment but before orientation is inside threshold.
- Axial phase activates only when lateral and orientation thresholds are both satisfied.

In axial phase, positive axial credit is gated by:

- insertion/alignment gate;
- non-worsening lateral and orientation gate;
- action direction gate;
- rotation quiet gate;
- forward action gate.

Axial phase also penalizes:

- lateral action;
- rotational action;
- missing forward action when applicable;
- alignment loss;
- impure action.

For the final run, axial purity was made stricter by setting:

- `AIC_STATEFUL_ACTION_LATERAL_SIGMA=0.00002`
- `AIC_STATEFUL_ACTION_LATERAL_SIGMA_FAR=0.00002`
- `AIC_STATEFUL_ACTION_MIN_FORWARD=0.0`

This was done because the previous far-field lateral action sigma (`0.00030`) allowed too much lateral motion to retain axial reward gate credit.

## Curriculum Iterations

Several curriculum designs were tried:

1. Original near-entrance axial ramp (`3-20 mm`) was too contact-loaded.
2. Level 2 was moved farther away after observing it was too close to target depth.
3. Aligned levels were tried at 20, 30, 35, 37.5, 40, and 42.5 mm pre-entrance.
4. Short reset probes showed:
   - 20/30 mm had strong contact preload and rapid lateral/orientation drift.
   - 40 mm aligned with reduced/no exploration was the least bad tested reset.
   - 42.5 mm was worse than 40 mm.
   - longer reset settle reduced force but let lateral alignment decay before training began.
   - zero settle preserved alignment better but kept contact force high.

Final curriculum saved in `configs/axial_first_inserted_start_curriculum.yaml`:

```yaml
level_count: 6
reintroduction_schedule:
  - {name: preentrance_40mm_aligned, signed_depth_m: -0.0400, lateral_m: 0.0, theta_rad: 0.0}
  - {name: preentrance_40mm_lat1mm, signed_depth_m: -0.0400, lateral_m: 0.001, theta_rad: 0.0}
  - {name: preentrance_40mm_lat2mm, signed_depth_m: -0.0400, lateral_m: 0.002, theta_rad: 0.0}
  - {name: preentrance_40mm_lat5mm, signed_depth_m: -0.0400, lateral_m: 0.005, theta_rad: 0.0}
  - {name: preentrance_40mm_lat8mm, signed_depth_m: -0.0400, lateral_m: 0.008, theta_rad: 0.0}
  - {name: preentrance_40mm_lat10mm, signed_depth_m: -0.0400, lateral_m: 0.010, theta_rad: 0.0}
```

This means the run did not keep a depth-ramp curriculum. It switched to target axial depth first, then lateral reintroduction.

## Training Defaults Used

Final axial-first launcher defaults:

- actor-only mode: `act_direct`
- guide disabled: `--target_action_guide_weight 0.0`
- guard disabled: `--no-insertion_action_guard`
- actor initial axial bias disabled: `AIC_STATEFUL_ACTOR_INITIAL_AXIAL_BIAS_M=0.0`
- reset settle: `AIC_STATEFUL_RESET_SETTLE_STEPS=10`
- exploration noise reduced:
  - `AIC_STATEFUL_ACTOR_EXPLORATION_NOISE_STD=0.000005`
  - `AIC_STATEFUL_ACTOR_EXPLORATION_PHASE_LATERAL_MULTIPLIER=1.0`
  - `AIC_STATEFUL_ACTOR_EXPLORATION_PHASE_ORIENTATION_STD=0.0`
- adapter learning rate: `3e-6`
- eval every 10 episodes
- total requested episodes: 2000

## Important Probes and Findings

Representative short probes before the final run:

- `-40 mm`, zero settle, no exploration:
  - step 1: `force_norm_mean ~13.7 N`, `lat_gate ~0.862`, `g_insert ~0.294`, reward around `-2.68`
  - actions were very small, so this reflected reset/physics more than policy exploration.

- `-40 mm`, default reduced exploration:
  - step 1-5: action lateral stayed around `1-4 um`
  - `g_insert` stayed around `0.36-0.42`
  - rewards were no longer catastrophically negative.

- `-40 mm`, longer settle 50:
  - force dropped to around `5-6 N`
  - but `lat_gate` was already around `0.35` at step 1, so settle allowed drift before training.

Conclusion from probes: reset/contact realization remains a major blocker. Reward gating is strict enough that bad axial/lateral behavior is punished, but reset physics can put the policy into a deteriorating state before useful axial learning begins.

## Final Long Run Outcome

Run root:

`/tmp/aic_axial_first_inserted_start_40depth_lateral_curriculum_20260614_145937`

Timeline:

- Started: 2026-06-14T14:59:37Z
- Finished: 2026-06-19T13:32:34Z
- Cycles: 200
- Episodes used: 2000
- Final level: 0
- Promotion: never promoted
- Strict success: 0 in every eval
- Final checkpoint:
  `/tmp/aic_axial_first_inserted_start_40depth_lateral_curriculum_20260614_145937/train_cycle0200_level000/2026-06-19_12-57-20_stateful_train_cycle0200_level000/checkpoint_latest.pt`

The wrapper emitted repeated `assessment_due_no_promotion` events roughly every 2 hours, but no automated reward/hyperparameter change was made during this finished run.

Final event:

```json
{"time":"2026-06-19T13:32:34Z","event":"done","cycle":200,"level":0,"episodes_used":2000,"checkpoint":"/tmp/aic_axial_first_inserted_start_40depth_lateral_curriculum_20260614_145937/train_cycle0200_level000/2026-06-19_12-57-20_stateful_train_cycle0200_level000/checkpoint_latest.pt"}
```

## Eval Evidence

The run stayed actor-only in eval summaries:

- `guide_action_any: false`
- `guide_blend_max: 0.0`
- `guard_max: 0.0`
- `executed_minus_actor_max: 0.0`

Late eval summaries still reported:

- `strict_success: false`
- `strict_success_count: 0`
- no promotion beyond level 0

Examples from summaries:

- `eval_cycle0038_level000_summary.json`
  - best lateral: `0.000464 m`, but depth/orientation not solved at the same time.
  - best orientation: `0.0324 rad`, but lateral around `0.00265 m`.

- `eval_cycle0094_level000_summary.json`
  - best depth only around `0.00516 m`, lateral `0.00854 m`, orientation `0.102 rad`.
  - strict success false.

The final summaries show the same pattern: individual partial metrics improve in isolation but do not combine into strict success.

## Current Runtime State

When this file was written on 2026-07-15:

- No active `train_stateful`, `train_axial_first`, or `serl/train.py` processes were found in the container for this task.
- The long run had already completed all 2000 episodes.
- The goal is not complete because training did not achieve strict success and never progressed past level 0.

## Recommended Next Steps

The next work should not simply run the same curriculum again. Evidence points to reset/contact and early realized dynamics as the limiting issue.

Recommended next investigations:

1. Audit reset realization at `40 mm / 0 lateral` with full geometry snapshots and force metrics immediately after reset and after 1-10 zero-action steps.
2. Compare reset pose/body used by the reward (`sfp_tip_link`) against the IK reset body (`gripper_tcp`) to see whether the reset target leaves the tip preloaded.
3. Add or tune reset compensation/collision geometry so `40 mm / 0 lateral` starts with low force while preserving `lat_gate`.
4. Only after reset is physically stable, resume actor-only training with the strict stateful reward.
5. If reward changes are needed after reset is stable, keep the high-level phase design: lateral first, orientation/alignment next, pure axial insertion last.

Do not re-enable guide, guard, or actor axial bias unless the user explicitly changes the training constraint.
