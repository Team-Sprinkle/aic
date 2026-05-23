# Isaac Insertion Experiment Summary - 2026-05-23

This is an index and concise summary of the Isaac Lab SFP-to-NIC insertion work
done on branch `feat/hybrid-train`. Detailed raw commands, metrics, and run
paths remain in the linked files below.

## Detailed Records

- `docs/training_debug_findings_20260513.md`
  - Early Isaac online SERL debugging, near-gate curriculum checks, adapter
    saturation, reset quality, and multi-GPU launcher notes.
- `docs/isaac_near_gate_experiments_20260515.md`
  - Near-gate reward and rollout experiments, `s/r/theta` definitions,
    start-near-gate sign convention, ACT/checkpoint behavior, and early
    cheatcode-style reward validation.
- `docs/cheatcode_insertion_reward_validation_20260515.md`
  - Formula-level validation of `cheatcode_insertion_v1` and
    `cheatcode_alignment_v1`, phase reward tests, audit plots, and short
    ACT-only/alignment-only rollouts.
- `docs/one_day_insertion_pipeline_20260515.md`
  - First autonomous one-day multi-GPU experiment pipeline results, guide
    sweeps, scoring tables, selected runs, and exact commands.
- `docs/curriculum_insertion_pipeline_20260515.md`
  - Staged and mixed curriculum runs, scheduled lateral/orientation tolerance,
    long mixed episode experiments, checkpoints, and later reward/action-axis
    trials.
- `docs/isaac_tip_orientation_reward_findings_20260515.md`
  - Tip-axis orientation reward change. Key point: reward orientation must be
    measured on the plug/tip semantic body, not the gripper.
- `docs/action_axis_reward_iteration_20260516.md`
  - Action-axis gated reward attempts, semantic progress/consistency trials,
    orientation/controller debugging, visual inspection notes, and the final
    closed-loop rotation guard probes from 2026-05-19.

## What Was Tried

### 1. Initial Online SERL Near-Gate Training

Started from the 175k ACT checkpoint and attempted online SERL near the port.
The initial reward was too much like a Euclidean closeness blob: policies could
move near the gate but did not reliably insert.

Main findings:

- The ACT checkpoint can place the plug near the port but does not solve final
  insertion.
- Online actor updates were unstable unless adapter deltas and action clips were
  kept very small.
- Starting exactly at 6 mm axial / 6 mm lateral is semantically correct but
  contact-prone and hard for RL exploration.

### 2. Reward Geometry Redesign

Implemented insertion geometry using semantic quantities:

- `s`: signed axial depth relative to port entrance. Negative means outside,
  zero means at the entrance plane, positive means past the entrance.
- `r`: lateral distance from the port centerline.
- `theta`: tip orientation error relative to the port axis/orientation.

Added and validated:

- `cheatcode_insertion_v1`
- `cheatcode_alignment_v1`
- strict success checks requiring axial, lateral, orientation, and consistency
  body criteria
- `cheatcode_phase_summary.json`
- formula/audit plots for several orientation errors

Main findings:

- The reward formula now behaves correctly in isolation.
- Inward motion while laterally or orientationally misaligned is penalized.
- Positive `s` alone is not success and can be a false positive if the module
  body remains outside.

### 3. Tip vs Gripper Orientation

Changed orientation logic to use semantic tip orientation/axis rather than
gripper orientation.

Main findings:

- The relevant orientation is the tip axis relative to the port axis.
- Full gripper quaternion alignment is not the right insertion metric.
- This applies to both SFP-to-NIC and SC-to-SC; the reward body/tip body must be
  task-specific.

### 4. Multi-GPU Independent Experiment Pipeline

Extended the one-day pipeline to use independent GPU workers rather than a
synchronized central learner.

Tried:

- zero-action reset/contact audits
- guide-only alignment sweeps
- alignment imitation
- weak insertion learning
- staged A/B/C/D curriculum
- mixed single-stage curricula with 10/20/30/40 mm axial starts and varying
  lateral starts

Main findings:

- Independent worker sweeps are practical; synchronized multi-GPU learning was
  not needed.
- Stage/curriculum generation helped produce controlled starts, but staged
  training did not by itself solve final insertion.
- Single mixed curriculum runs were more useful than exhaustive stage sweeps
  under the time budget.

### 5. Scheduled Lateral and Orientation Tolerances

Tried reward schedules where the lateral reward radius and orientation tolerance
are loose farther from the entrance and tighten near the port.

Main findings:

- The concept is correct: far away, allow a wider funnel; near the entrance,
  require tight alignment.
- Scheduled tolerance helped shape approach behavior, but controller realization
  still caused lateral drift near insertion.

### 6. Multiplicative Action-Axis Reward

Changed the reward idea from additive terms to a multiplicative gate:

- high reward only when the tip is laterally aligned,
- the tip is correctly oriented,
- and the executed action points roughly along the insertion axis.

Main findings:

- This better matches the desired behavior than additive reward.
- It prevents paying for forward motion that is not aligned with the port axis.
- It still cannot fully overcome controller/IK behavior if wrist rotations push
  the tip sideways.

### 7. Consistency Body Checks

Added/used `sfp_module_link` consistency checks so `sfp_tip_link` depth cannot
claim success by itself.

Main findings:

- Several runs showed shallow or even large positive `s` for the tip while the
  module remained visibly outside or bypassed the port.
- Strict success must include module consistency.
- The best runs should be judged by `s`, `r`, `theta`, and consistency together,
  not reward total or tip depth alone.

### 8. Orientation and Controller Debugging

Validated that the guide rotation direction was mathematically correct, then
checked whether Isaac realized the intended tip motion.

Tried:

- direct IK on `sfp_tip_link`
- wrist IK with rotation compensation
- separate clipping for translation and rotation-induced compensation
- projected compensation to remove positive axial compensation while off-center
- visual inspection from center/left/right cameras

Main findings:

- Direct IK on `sfp_tip_link` was worse and caused large lateral errors.
- Wrist IK is safer, but wrist rotation can physically sweep the tip off-axis.
- Some environments plateaued around `theta ~= 0.05 rad`.
- Reward tuning alone is unlikely to fix this controller/semantic-frame mismatch.

### 9. Closed-Loop Rotation Guard

Added a guard that zeros rotational TCP commands while the tip is laterally
off-center, plus a final orientation-hold phase near the entrance.

Best no-learning probe:

```text
outputs/one_day_insertion_pipeline/orientation_calibration_20260519/runs/2026-05-19_23-22-46_guard_final_orientation_hold_full_2026-05-19_23-22-00
```

Final metrics:

- env0: `s=-0.18 mm`, `r=0.95 mm`, `theta=0.052 rad`
- env1: `s=0.09 mm`, `r=0.95 mm`, `theta=0.030 rad`
- no strict success

Main findings:

- The guard fixed the most severe lateral blow-up.
- The plug reached a stable near-port pose.
- Visual inspection still showed the module outside the cage, so this is not a
  solved insertion policy.

## Main Learnings

1. The reward geometry is mostly correct now; the remaining blocker is not just
   scalar reward tuning.
2. Positive `s` for `sfp_tip_link` can be misleading. Always check `r`, `theta`,
   module consistency, and video.
3. The policy must align before insertion; relaxed thresholds produce bypass.
4. Tip orientation must be measured on the tip/plug semantic body, not the
   gripper.
5. The controller frame and insertion semantic frame do not match perfectly.
   Wrist rotations can improve orientation metrics while moving the tip
   laterally.
6. Guide/action guards are currently more reliable than pure online RL.
7. The safest current fallback is ACT plus `cheatcode_transform` guide with the
   closed-loop rotation guard and final orientation hold.

## Current Best Practical Recommendation

Use the guided fallback from:

```text
outputs/one_day_insertion_pipeline/orientation_calibration_20260519/runs/2026-05-19_23-22-46_guard_final_orientation_hold_full_2026-05-19_23-22-00
```

Treat it as a conservative near-gate alignment policy, not a successful final
insertion policy.

The next engineering step should be a retention/servo phase after `r < 1 mm`:

- keep lateral correction active,
- hold or trim orientation,
- allow only very small positive axial steps,
- stop/recover on force spikes or consistency-body disagreement,
- never loosen success to tip depth alone.
