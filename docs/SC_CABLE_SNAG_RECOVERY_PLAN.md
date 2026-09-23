# SC-to-SC cable-snag recovery plan

Status: reprioritized by observed failures; SC mechanics gate remains open, 2026-09-23

## Decision and motivation

Continue with perception, port-relative supervised control, and gated online
RL. Do not restart the parked world-model branch for this experiment.

The [ranked failure audit](experiments/2026-09-23-ranked-failure-scenarios.md)
now controls execution order. Alignment drift and local axial blockage are
Priority 1; large approach blockage is Priority 2. Cable snag is Priority 3
because it remains plausible but has not been reproduced causally in a valid
current scene. Do not label ordinary five-card failures as cable snags.

The latest SFP-to-NIC video rerun shows that the current measured-path recovery
can plausibly release a millimeter-scale port-lip contact. It does not establish
recovery from a cable wrapped around cards or blocked by another object. A
multi-card SC-to-SC scene can require a different behavior: move away far
enough to release cable tension, route around the obstruction, and only then
resume alignment and insertion.

The current selected Isaac RGB pose estimator and RPDP/SERL controller have
only been validated in SFP-to-NIC near-port scenes. The verified mixed Gazebo
collection contains 21 SC episodes, but those episodes did not train or validate
the selected Isaac RPDP checkpoint. SC support must therefore pass its own
perception and BC gates before RL.

## September 23 mechanics-gate result

Steps 1--4 were entered, but step 2 exposed a scene-contract problem that must
be fixed before the discovery set can be interpreted. The audit corrected the
SC port mount, reversed cable topology, fixed-joint transforms, reset ordering,
and the unreachable center-of-collider target. In the corrected zero-card
scene, a 14 mm retreat and positive 4 mm lateral retry reached 0.099 mm final
lateral and 0.008 mm axial error.

The first five-card failure was not a verified cable snag. Disabling cable and
plug collisions did not change it, while the contact sensor measured about
128 N between the gripper base and a NIC card. The reachable Isaac grasp makes
the gripper housing cross the card. The source reversed Gazebo grasp avoids
that route but is not reachable in the current Isaac robot/base/board layout.

A clearly labeled diagnostic asset with the ten gripper colliders disabled was
used only to test the routing hypothesis. The direct high approach ended in the
component gate in 1/3 seeds; a coherent 100 mm route around the card edge did so
in 3/3 seeds. Those privileged IK runs support large routed recovery, but they
are not autonomous evidence and must not enter BC or RL replay. Rendering also
changed the recorded seed-1 outcome, so videos and metrics-only runs remain
separate evidence.

The next required result is a collision-faithful and reachable SC grasp/scene
contract. Then repeat scripted insertion for every card count 0--5 before
continuing the bounded discovery set. See the
[mechanics and routing record](experiments/2026-09-23-sc-mechanics-and-routing.md).

A subsequent ten-trial Gazebo suite added three exact controls, two five-card
runs to each SC rail, and three bounded grasp repeats. Exact and grasp groups
were 3/3 full. Rail-1 five-card runs were both near-aligned partial insertions;
rail-0 runs both had large tracking failures. The cable remained visibly clear
of the cards. These episodes support local axial and large approach recovery
curricula, but still provide no accepted cable-snag incident.

The official reversed Gazebo cable includes explicit palm-clearance collision
exceptions: two endpoint collision groups are removed and the first rope
collider is shortened and shifted. The Isaac builder now matches those source
exceptions and wraps the equivalent wrist solution into its valid range. A
full-scene hold still became unstable, while removing the board, port, and self
collision yielded low force but left a 44.27 mm reset miss after physical
interpolation. Treat this as a remaining transform/placement/actuation fault;
it does not authorize the discovery set.

## What the latest SFP experiment established

- The selected deterministic BC and tight-trust RL actor can perform local
  SFP-to-NIC insertion, but recording-enabled results vary between reruns.
- Most recorded failures are lateral divergence or bypass rather than cable
  snagging.
- The current recovery commands at most `0.25 mm` per control step, requires at
  least `1 mm` clearance, and then constrains 12 steps of retry motion near the
  plane perpendicular to the blocked direction.
- A 1 mm retreat can matter for a port-lip contact because the success corridor
  is only about 0.5 mm. It may be too small for a multi-card cable obstruction.
- The video rerun did not retain compact per-step force and recovery telemetry,
  so the two extra recovery-arm successes cannot be causally attributed to the
  hard-coded backtracking state machine.

See the [matched SERL experiment](experiments/2026-09-22-serl-mixture-recovery.md)
and [video failure analysis](experiments/2026-09-23-serl-video-failure-analysis.md).

## Proposed controller structure

Use a hierarchical controller with three observation-driven parts:

```text
three RGB cameras + robot state + force history
                       |
                       v
        connector/port pose and visibility estimate
                       |
                       v
        task-conditioned port-relative BC policy
                       |
             normal action proposal
                       |
                       v
       contact and progress recovery supervisor
          |                         |
          | local lip contact       | persistent cable/card blockage
          v                         v
   short measured-path       larger retreat and observed
   retreat + lateral retry   obstacle-routing recovery option
          |                         |
          +------------+------------+
                       v
              force-safe execution
```

The simulator may supply pose, cable, card, and contact labels during training.
Those labels must not be actor inputs or directly select recovery directions at
autonomous evaluation. Evaluation uses RGB, robot state, measured motion, force,
and learned predictions.

## Ordered execution plan

### 1. Freeze the current SFP evidence

Preserve the selected BC/RL checkpoints, eight development configurations,
video rerun, failure table, and exact recovery settings. Do not reinterpret the
3/8 recovery-arm rerun as a promoted result.

**Pass condition:** hashes, configs, outcomes, diagnostic sheets, and the
committed analysis are available.

### 2. Audit Isaac SC-to-SC support before collecting data

Confirm that the Isaac scene can instantiate the SC source plug, SC target port,
all intended intervening-card counts, cable collision geometry, force sensing,
three cameras, reset identity, and deterministic seeds. Verify the physical SC
insertion depth and frame definitions rather than copying the SFP 8 mm surrogate.

**Pass condition:** one scripted reset and one CheatCode insertion for every
card-count stratum have valid geometry and terminal accounting.

### 3. Define cable snag separately from port-lip contact

Use a causal detector based on commanded motion, measured TCP and plug motion,
force persistence, contact location if available, and cable displacement.

- **Port-lip contact:** plug is near the target opening and axial progress stalls.
- **Cable/card snag:** plug may remain free, but cable motion or tension blocks
  TCP progress away from the opening or against an intervening card.
- **Simple misalignment:** lateral or orientation error grows without evidence
  of a blocked cable.

Predeclare thresholds and manually inspect a sample before using these labels
for training or reward.

**Pass condition:** reviewed examples show that normal insertion load is not
systematically labeled as snagging.

### 4. Run a bounded SC discovery set

Run complete autonomous or CheatCode-assisted SC-to-SC episodes across the
available intervening-card counts and cable seeds. Start with approximately ten
episodes per stratum, capped at 50 episodes, and retain every outcome. Do not
stop at the first visually interesting failure.

Record three RGB cameras plus compact 20 Hz telemetry: episode/reset identity,
commands, executed actions, TCP and plug motion, estimated pose, force, contact,
recovery mode, backtrack count, retreat distance, and terminal observation.

**Pass condition:** either at least ten reviewed natural snag incidents are
captured across at least two layouts, or the bounded run establishes that the
current simulator setup does not reproduce the expected failure.

### 5. Build reproducible short incident episodes

For each reviewed snag, save a reset 1--2 seconds before the blocking contact.
The reset must include robot, plug, cable, cards, target, controller history,
and random seed. If exact physics snapshots are not reproducible, generate a
family of nearby seeded resets and report their variation.

Split by original full episode, layout, and seed. All short episodes from one
incident remain in one split.

**Pass condition:** replaying a held-out incident reproduces the same snag class
often enough to compare recovery methods; report that reproduction rate.

### 6. Audit SC perception

Test the frozen SFP pose estimator on SC plug and opening labels without tuning
on final scenes. Report 3D translation, axial/lateral error, orientation,
visibility, and error by card count and occlusion.

If it fails, retain the current multiview structure but train an SC connector
and opening head with simulator labels. Start from shared ImageNet/SFP visual
features only if that improves a fixed SC validation split. Keep a task or
connector-type token and allow separate small output heads because SFP and SC
geometry differ.

**Pass condition:** held-out near-port lateral error resolves the required SC
insertion corridor and complete inference p95 remains below 300 ms.

### 7. Collect SC supervised demonstrations

Use the SC CheatCode target actions, converted into the selected port frame, as
the action labels. Collect successful normal transport, cable routing around
intervening cards, alignment, and insertion trajectories across card counts.
Do not train on blended executed actions as if they were teacher targets.

The existing 21 verified SC Gazebo episodes may be audited and used when their
images, calibration, action frame, target commands, and terminal success are
compatible. They are not sufficient coverage by themselves.

**Pass condition:** episode-grouped train/validation manifests cover every
development card-count stratum and contain verified full SC insertion labels.

### 8. Train the SC BC warm start

Train the same full-trajectory port-relative actor family used by RPDP, with
pose and visibility provided through explicit conditioning. Predict complete
actions rather than residual corrections. First train SC alone; then compare a
task-conditioned joint SFP+SC model under an identical SC validation split.

Evaluate near-port SC insertion first, then normal full-start episodes. Preserve
the SC-only model if joint training causes negative transfer.

**Pass condition:** autonomous BC succeeds on new episode-grouped SC development
starts with acceptable force, including multi-card layouts. Do not begin online
actor updates before BC provides a useful warm start.

### 9. Establish matched recovery baselines on short incidents

Compare the same frozen BC actor with:

1. no recovery;
2. current local recovery: 1 mm minimum retreat and short lateral retry;
3. multi-scale retreat: predeclared 1, 3, 5, and 10 mm clearances;
4. retreat plus larger lateral displacement;
5. an observation-driven detour option that routes around a detected card or
   cable obstruction before returning to the target.

Larger lateral motion must be bounded by observed free space and force, because
blindly moving farther sideways can tighten the cable or hit another card.
Sweep on development incidents and freeze settings before the held-out incident
comparison.

**Pass condition:** a method improves held-out snag clearance and eventual
insertion without increasing peak force or ordinary-port failures.

### 10. Train a recovery option before changing the whole actor

Represent recovery as a short temporally coherent option: retreat, detour, and
reapproach. Warm-start it with successful scripted/teacher recoveries and train
it on short incident episodes. Keep the normal BC actor frozen initially so a
bad recovery update cannot destroy transport and insertion behavior.

Inputs may include recent RGB features, predicted plug/port pose, TCP/force
history, previous actions, and recovery phase. True simulator geometry remains
a label only.

**Pass condition:** the learned option beats fixed backtracking on held-out
incident groups and returns control to BC from a valid, low-force state.

### 11. Build episode-grouped offline RL replay

Include successful demonstrations, natural failures, failed recoveries, and
successful scripted recoveries. Preserve terminal observations and distinguish
actor-owned actions from hard-coded recovery actions. Train and validate twin
critics on complete held-out incidents. Require sensible ranking of successful
clearance/insertion over persistent snag and unsafe-force outcomes.

**Pass condition:** critic ordering and calibration hold on more than a token
one-success/one-failure validation set.

### 12. Run bounded online SERL on short incidents

Start from the frozen BC actor and validated recovery option. Use balanced
prior/online replay, full-mixture behavior anchoring, and explicit learning-rate
restoration. Rewards should distinguish:

- reduction in excessive force;
- measured retreat and clearance;
- safe lateral/detour progress;
- return toward the target after clearance;
- final alignment and insertion;
- repeated pushing into the same obstruction.

Compare against the frozen BC and fixed-recovery baselines with identical
incident resets and budgets.

**Pass condition:** autonomous held-out incident recovery improves without mode
collapse or degradation on no-snag insertion controls.

### 13. Evaluate complete SC-to-SC episodes

Run normal starts across held-out card counts, layouts, target ports, and cable
seeds. Report task completion, snag incidence, recovery attempts, successful
clearance, final insertion, peak/integrated force, time, path length, repeated
contacts, and inference p50/p95/p99.

Include no-recovery BC, fixed recovery, and learned recovery. Keep reserved final
scenes sealed until one development candidate passes the predefined gate.

### 14. Recheck SFP-to-NIC regression

Run the selected combined controller on new SFP starts, including ordinary lip
contacts. A cable-routing recovery must not turn small SFP alignment errors into
large detours.

**Pass condition:** SFP performance and force remain within the declared
noninferiority margin while SC recovery improves.

### 15. Consider a world model only if history remains the bottleneck

If the observation/history policy cannot predict cable recoil or persistent
occlusion after the supervised and model-free recovery stages, test a compact
pretrained temporal representation or feature-space dynamics auxiliary loss.
Require improvement in held-out snag-state prediction and recovery decisions;
RGB next-frame quality alone is not a promotion criterion.

## Resource and evaluation rules

- Use rootless Docker and existing GPU allocations.
- Begin each experiment on one GPU and never exceed four concurrent GPUs.
- Keep complete live inference p95 below 300 ms.
- Split by complete episode, incident, layout, and reset seed.
- Preserve failures and invalid runs as well as selected checkpoints.
- Never pass true simulator plug, port, cable, or card geometry to the deployed
  actor or use it to choose an autonomous crop or detour.
- Keep the existing reserved final configurations sealed until the relevant
  development gate passes.

## Recommended immediate scope

Execute steps 1--5 first. They determine whether multi-card SC snagging is
reproducible and whether exact short-incident resets are possible. Do not choose
a larger backtrack distance or begin SC RL before that evidence exists.
