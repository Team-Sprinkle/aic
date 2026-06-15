# Agentic Randomized Curriculum Plan Update - 2026-05-30

## Current Plan Change

Stop continuing the single-reset shallow branch. The no-guard reward-only policy was repeatedly trained on
`full_depth_start2x0_v464_settle_centered_from_v462` with `num_envs=1`, which made the policy see the same
shallow entrance state over and over. This explains the observed entrance hover/alignment behavior without
robust axial/module-consistent insertion.

The active training path is now:

1. Validate randomized reset buckets before long training.
2. Train only on many-episode randomized curricula, with hard action/servo overrides disabled.
3. Keep strict full-insertion evaluation unchanged: full target depth around `46.864 mm`, lateral threshold
   around `0.5 mm`, theta threshold `0.030 rad`, module consistency, contact/force sanity, and center/left/right
   visual sanity.
4. Reject reward-only improvements that produce positive tip `s` without module-consistent insertion.

## Recent Single-Reset Evidence

| run | config | envs | first post-step reset | best post-step metric | decision |
|---|---|---:|---|---|---|
| v529 | `full_depth_start2x0_v464_settle_centered_from_v462` | 1 | `s=-2.053 mm`, `r=0.082 mm`, `theta=0.0507 rad`, module `s=-25.678 mm`, module `r=1.010 mm` | best `s=-0.564 mm`, strict false | reject: hover/alignment only |
| v530 | `full_depth_start2x0_v464_settle_centered_from_v462` | 1 | `s=-2.053 mm`, `r=0.082 mm`, `theta=0.0507 rad`, module `s=-25.678 mm`, module `r=1.010 mm` | best `s=-0.568 mm`, strict false | reject: stronger axial reward did not insert |

Earlier v519/v520/v524/v526 used the same single shallow reset family and showed the same pattern.

## Randomized Curriculum State

Accepted as a distribution-building baseline, but not promoted for long training:

- Config: `outputs/agentic_reward_curriculum_20260529/generated_episode_configs/randomized_curriculum_v565_interleaved_robotfinal_v543_bridge`
- Audit: `outputs/agentic_reward_curriculum_20260529/reset_audits/v565_interleaved_robotfinal_v543_bridge_current.md`
- Validation run: `outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-30_00-02-50_validate_v566_reset_mixed_v565_constant0_20`

Post-step reset validation:

| step | s min/mean/max mm | r min/mean/max mm | theta min/mean/max rad | module_s min/mean/max mm | module_r min/mean/max mm | decision |
|---|---:|---:|---:|---:|---:|---|
| step 1 | `-20.08 / -6.31 / -1.80` | `0.043 / 1.200 / 4.054` | `0.0505 / 0.0563 / 0.0849` | `-43.70 / -29.94 / -25.42` | `0.880 / 1.485 / 3.850` | plausible spread, not strict-theta-ready |
| step 20 | `-2.03 / -1.42 / -0.85` | `0.043 / 0.360 / 0.792` | `0.0495 / 0.0502 / 0.0507` | `-25.66 / -25.04 / -24.48` | `1.026 / 1.229 / 1.571` | reject for strict-orientation curriculum until low-theta starts are added |

This validates the user's diagnosis that single-reset training was insufficient, but it also shows that the
current randomized resets still settle to a theta floor around `0.05 rad`. More long training on this exact
distribution is unlikely to create strict insertion.

## Next Experiments

1. Generate low-theta randomized reset buckets that are physically valid after settle:
   - shallow/final: `s≈-3..+2 mm`, lateral `0..1 mm`, theta target `<0.030 rad`.
   - near-gate: `2/4/6/10 mm` outside, lateral `0.25..2 mm`, theta small/medium.
   - bridge: `10/20 mm` outside, lateral `1..4 mm`, theta medium.
   - held-out eval remains fixed `40 mm / 10 mm`.
2. Validate each bucket with `validate_serl_reset_settle.py`; reject starts that contact immediately, start already
   inserted, or drift to the `0.05 rad` orientation floor.
3. Only after at least one low-theta shallow/final bucket validates, run short no-guard reward-only training from
   the best partial checkpoint. Candidate checkpoint remains the v514/v483 lineage because it produced the best
   strict-metric partial row: `s=+7.867 mm`, `r=0.321 mm`, `theta=0.0201 rad`, but module `s=-15.781 mm`, strict false.
4. Held-out evals after meaningful checkpoints:
   - fixed `40 mm / 10 mm`,
   - randomized near-gate,
   - randomized shallow/final-window.

## Commands

Current audit command:

```bash
.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/audit_reset_curriculum_distribution.py \
  --config-dir outputs/agentic_reward_curriculum_20260529/generated_episode_configs/randomized_curriculum_v565_interleaved_robotfinal_v543_bridge \
  --name v565_interleaved_robotfinal_v543_bridge_current \
  --validation-run outputs/agentic_reward_curriculum_20260529/policy_train_runs/2026-05-30_00-02-50_validate_v566_reset_mixed_v565_constant0_20 \
  --output-dir outputs/agentic_reward_curriculum_20260529/reset_audits
```

Validation already run after the parser patch:

```bash
python -m py_compile aic_utils/aic_isaac/scripts/audit_reset_curriculum_distribution.py
.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py
```

Result: `31 passed`.

## Decision

Do not continue `full_depth_start2x0_v464_settle_centered_from_v462` training. Do not launch a long run on `v565`
until the low-theta reset/controller realization gap is addressed. The next code/experiment target is a physically
valid low-theta randomized reset generator or compensated orientation diagnostic that preserves semantic tip
lateral position and module consistency after settle.

## Update - v578 to v581

I found and fixed one concrete low-theta reset-builder bug:

- Previous low-theta metric resets used `post_step_insertion_geometry.body_orientation_wxyz_by_env` as if it were the
  raw semantic-tip world quaternion.
- The metrics also contain the actual world quaternion under
  `post_step_body_frame_offsets.world_quat_wxyz_by_env.sfp_tip_link`.
- `build_lowtheta_reset_from_metric.py` now prefers the explicit body-frame world quaternion and can preserve the
  measured reset-body pose relation from the metrics row with `--pose-source metric_reset_body`.

Generated candidates:

- `lowtheta_metric_v578_from_v514_step232_worldquat`
- `lowtheta_metric_v579_from_v514_step232_metricbody`

Validation:

| run | command intent | result | decision |
|---|---|---|---|
| v578 | validate low-theta starts with corrected tip world quaternion | rejected; step 1 already had `r≈21.6-26.4 mm`, `theta≈0.063-0.065 rad`; step 20 still `r≈23-40 mm`, `theta≈0.071-0.086 rad` | wrist-only reset reconstruction invalid |
| v580 | rerun v579 with new `validate_serl_reset_settle.py --zero-action` to remove ACT motion from reset validation | rejected; zero-action step 20 still had `r≈25.9-31.2 mm`, `theta≈0.071-0.078 rad` | failure is reset/cable realization, not policy motion |
| v581 | reproduce v514 for 260 steps with `--log_robot_state_every 1` to harvest replayable robot-joint starts | did not reproduce the low-theta positive-depth transient; best `s=-0.569 mm`, best theta `0.0486 rad`, strict false | no robot-joint low-theta starts harvested |

Additional validator fix:

- `validate_serl_reset_settle.py` now supports `--zero-action`. This matters because `--tcp_translation_action_clip 0.0`
  and `--tcp_rotation_action_clip 0.0` mean "no clipping", not "zero action"; without `--zero-action`, reset validation
  includes ACT policy motion.

Current diagnosis after v581:

- The v514 low-theta positive-depth row remains the best partial evidence, but it was transient and did not log robot
  joint state.
- Reconstructing that state from wrist pose is not reliable because `sfp_tip_link` is cable/articulated and does not
  preserve a fixed wrist-to-tip transform after reset/settle.
- Reproducing v514 from the available checkpoint did not recover the transient in 260 steps, so the immediate
  train-from-single-reset branch remains rejected.

Next recommended experiment:

1. Run a bounded reproduction/collection branch specifically designed to recover low-theta positive-depth states while
   logging robot state from the start:
   - start from the v483/v514 lineage,
   - keep randomized or at least multi-start shallow/near-gate distribution,
   - log robot state every step,
   - stop/promote only if rows satisfy `theta <= 0.030 rad`, `r <= 0.5 mm`, and nontrivial positive `s`.
2. If no rows appear quickly, switch to a controller/asset diagnostic: reset-body IK alone is not enough to place the
   semantic tip, so a replayable full articulated cable/module reset or explicit cable-state initialization may be
   required before randomized final-window training can be made valid.

Validation after code changes:

```bash
python -m py_compile \
  aic_utils/aic_isaac/scripts/audit_reset_curriculum_distribution.py \
  aic_utils/aic_isaac/scripts/build_lowtheta_reset_from_metric.py \
  aic_utils/aic_isaac/scripts/validate_serl_reset_settle.py
.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py
```

Result: `31 passed`.
